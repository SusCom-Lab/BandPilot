"""集群状态管理模块。

该模块实现了多租户场景下的集群状态管理，包括资源分配、争用检测和带宽预测。
"""
from __future__ import annotations

import numpy as np
import copy
import logging
from typing import List, Dict, Tuple, Optional, Set, Callable
from pathlib import Path

from core.bandwidth import prepare_model_inputs, calculate_bandwidth_values, SwitchBandwidthConfig
from training.evaluator import predict_with_model

logger = logging.getLogger(__name__)


def create_bandwidth_predictor(
    if_real_data: bool,
    total_gpu: int,
    gpu_bw_dict_list: List,
    switch_config: SwitchBandwidthConfig,
    data_path: str,
    model=None,
    device=None,
    artifact_dir: Optional[Path] = None,
) -> Callable[[np.ndarray], float]:
    """创建带宽预测函数工厂。
    
    根据 if_real_data 标志创建对应的带宽预测函数。这个函数将作为参数传入
    ClusterStateManager，使其不绑定特定的评估模式。
    
    Args:
        if_real_data: 如果为 True，使用真实数据计算带宽；如果为 False，使用模型预测
        total_gpu: 集群总GPU数量
        gpu_bw_dict_list: GPU带宽字典列表
        switch_config: 交换机配置
        data_path: 带宽数据文件路径（用于真实数据模式）
        model: PyTorch模型（用于模型预测模式）
        device: PyTorch设备（用于模型预测模式）
        artifact_dir: 模型和scaler文件所在目录（用于模型预测模式）
    
    Returns:
        一个函数，接受 np.ndarray (GPU组合) 作为参数，返回 float (带宽值)
    
    Raises:
        ValueError: 如果 if_real_data=False 但缺少必要的模型参数
    """
    if if_real_data:
        # 真实数据模式：使用 calculate_bandwidth_values
        def predictor(combo: np.ndarray) -> float:
            if np.sum(combo) == 0:
                return 0.0
            bw, _, _ = calculate_bandwidth_values(
                combo, total_gpu, gpu_bw_dict_list, switch_config, data_path
            )
            return float(bw)
        
        return predictor
    else:
        # 模型预测模式：使用 predict_with_model
        if model is None or device is None or artifact_dir is None:
            raise ValueError(
                "模型预测模式需要提供 model、device 和 artifact_dir 参数"
            )
        
        def predictor(combo: np.ndarray) -> float:
            if np.sum(combo) == 0:
                return 0.0
            part_bws, node_counts, total_counts = prepare_model_inputs(
                np.array([combo]), total_gpu, gpu_bw_dict_list, switch_config, data_path
            )
            bw = predict_with_model(model, part_bws, node_counts, total_counts, device, artifact_dir)
            return float(bw[0])
        
        return predictor

class ClusterStateManager:
    """集群状态管理器。
    
    管理多租户场景下的GPU资源分配和争用检测。
    
    设计原则：
    - ClusterStateManager 是一个通用的状态管理器，不绑定特定的评估模式
    - 通过传入带宽预测函数（bandwidth_predictor），使其能够支持不同的评估模式
    - if_real_data 是算法级别的选择，不应该硬编码在 ClusterStateManager 中
    - 使用 create_bandwidth_predictor() 工厂函数创建预测函数，确保一致性
    
    使用示例：
        # 创建预测函数
        predictor = create_bandwidth_predictor(
            if_real_data=False,
            total_gpu=32,
            gpu_bw_dict_list=gpu_bw_dict_list,
            switch_config=switch_config,
            data_path=data_path,
            model=model,
            device=device,
            artifact_dir=artifact_dir,
        )
        
        # 创建状态管理器
        manager = ClusterStateManager(
            total_gpu=32,
            bandwidth_predictor=predictor,
        )
    """
    
    def __init__(
        self, 
        total_gpu: int,
        bandwidth_predictor: Callable[[np.ndarray], float],
    ):
        """初始化集群状态管理器。
        
        Args:
            total_gpu: 集群总GPU数量
            bandwidth_predictor: 带宽预测函数，接受 np.ndarray (GPU组合) 返回 float (带宽值)
        """
        self.total_gpu = total_gpu
        self.bandwidth_predictor = bandwidth_predictor

        # 0: idle, 1: busy
        self.allocated_gpu_mask = np.zeros(total_gpu, dtype=int)
        
        # 存储活跃任务信息
        # Format: {
        #   'job_id': int, 
        #   'combo': np.array, 
        #   'standalone_bw': float, # 独占时的理论带宽
        #   'current_bw': float,    # 当前考虑争用后的带宽
        #   'history': List[float]  # 带宽变化历史
        # }
        self.active_jobs: List[Dict] = []
        
        # 节点大小（假设8卡一机）
        self.node_size = 8

    def _validate_combo_request(
        self,
        combo: np.ndarray,
        enforce_availability: bool = True,
    ) -> np.ndarray:
        """校验候选GPU组合的基本合法性。

        - 形状需与 total_gpu 一致
        - 元素必须为 {0,1}
        - 任意节点上的GPU数量不能超过 node_size
        - 当 enforce_availability=True 时，不允许与已分配GPU重叠
        """
        combo_arr = np.asarray(combo)
        if combo_arr.ndim != 1 or combo_arr.size != self.total_gpu:
            raise ValueError(
                f"非法 combo 长度：期望 {self.total_gpu}, 实际 {combo_arr.size}"
            )

        if not np.all((combo_arr == 0) | (combo_arr == 1)):
            raise ValueError("combo 中存在非 {0,1} 元素，无法映射到具体GPU")

        combo_int = combo_arr.astype(int, copy=False)

        # 单节点容量校验：出现超过 node_size 的需求视为非法
        reshaped = combo_int.reshape(-1, self.node_size)
        node_usage = reshaped.sum(axis=1)
        if np.any(node_usage > self.node_size):
            raise ValueError(
                f"combo 触发单节点超额：节点需求 {node_usage.tolist()}，单节点容量 {self.node_size}"
            )

        if enforce_availability:
            conflicted = (self.allocated_gpu_mask > 0) & (combo_int > 0)
            if np.any(conflicted):
                conflict_indices = np.where(conflicted)[0][:5]
                raise ValueError(
                    f"combo 包含已分配GPU（示例索引: {conflict_indices.tolist()}）"
                )

        return combo_int

    def get_available_gpus(self) -> List[int]:
        """返回当前未被占用的GPU索引列表"""
        return list(np.where(self.allocated_gpu_mask == 0)[0])

    def _predict_combo_bandwidth(self, combo: np.ndarray) -> float:
        """基础预测函数：预测给定组合的带宽（不考虑外部争用）。
        
        使用传入的 bandwidth_predictor 函数进行预测，使 ClusterStateManager
        不绑定特定的评估模式。
        
        Args:
            combo: GPU组合（0/1向量）
        
        Returns:
            预测的带宽值
        """
        return self.bandwidth_predictor(combo)

    def _is_cross_node_combo(self, combo: np.ndarray) -> bool:
        """判断GPU组合是否跨多个节点。
        
        根据需求，只有跨节点的任务才会相互干扰。单节点内的任务不会与其他任务产生争用。
        
        Args:
            combo: GPU组合（0/1向量）
        
        Returns:
            如果组合的GPU分布在多个节点上，返回 True；否则返回 False
        """
        involved_nodes = set()
        for i in range(len(combo)):
            if combo[i] == 1:
                involved_nodes.add(i // self.node_size)
        
        # 如果涉及多个节点，则是跨节点组合
        return len(involved_nodes) > 1

    def _get_nodes_for_combo(self, combo: np.ndarray) -> set:
        """获取combo涉及的节点集合。
        
        Args:
            combo: GPU组合（0/1向量）
        
        Returns:
            涉及的节点索引集合
        """
        nodes = set()
        for i in range(len(combo)):
            if combo[i] == 1:
                nodes.add(i // self.node_size)
        return nodes

    def _project_combo_to_nodes(self, combo: np.ndarray, target_nodes: set) -> np.ndarray:
        """将combo投影到指定的节点集合上。
        
        Args:
            combo: GPU组合（0/1向量）
            target_nodes: 目标节点集合
        
        Returns:
            投影后的GPU组合（只保留目标节点上的GPU）
        """
        projected = np.zeros_like(combo)
        for i in range(len(combo)):
            if combo[i] == 1 and (i // self.node_size) in target_nodes:
                projected[i] = 1
        return projected

    def _canonicalize_combo(self, combo: np.ndarray) -> np.ndarray:
        """将任意GPU组合标准化为每节点前count个GPU置1的0/1向量。
        
        super combo 中可能出现同一GPU位置数值>1或非连贯的取值，这会导致带宽查表失败。
        该函数按节点统计使用数量，并在对应节点的前count个槽位标记为1，确保符合查表格式。
        """
        canonical = np.zeros_like(combo, dtype=int)
        num_nodes = len(combo) // self.node_size
        for node_idx in range(num_nodes):
            start = node_idx * self.node_size
            end = start + self.node_size
            node_slice = combo[start:end]
            count = int(np.sum(node_slice))
            if count <= 0:
                continue
            if count > self.node_size:
                # logger.warning(
                #     "节点 %s super combo 需求 %s 超过节点容量 %s，已按容量截断",
                #     node_idx,
                #     count,
                #     self.node_size,
                # )
                count = self.node_size
            canonical[start : start + count] = 1
        return canonical

    def _calculate_dual_super_combo_capacity(
        self, 
        combo1: np.ndarray, 
        nodes1: set, 
        combo2: np.ndarray, 
        nodes2: set
    ) -> float:
        """计算两个跨节点任务之间的双向super combo容量，取最小值。
        
        对于两个跨节点任务，计算：
        1. 从combo1视角：combo1 + project(combo2, nodes1)
        2. 从combo2视角：combo2 + project(combo1, nodes2)
        取这两个super combo的带宽预测值的最小值。
        
        示例：
            Job X = [6,4,0,2] (节点{0,1,3}), Job Y = [0,2,2,2] (节点{1,2,3})
            共享节点：{1,3}
            
            从X视角：
                project(Y, {0,1,3}) = [0,2,0,2]
                super_X = [6,4,0,2] + [0,2,0,2] = [6,6,0,4]
                capacity1 = f([6,6,0,4])
            
            从Y视角：
                project(X, {1,2,3}) = [0,4,0,2]
                super_Y = [0,2,2,2] + [0,4,0,2] = [0,6,2,4]
                capacity2 = f([0,6,2,4])
            
            返回：min(capacity1, capacity2)
        
        Args:
            combo1: 第一个任务的GPU组合
            nodes1: 第一个任务涉及的节点集合
            combo2: 第二个任务的GPU组合
            nodes2: 第二个任务涉及的节点集合
        
        Returns:
            双向super combo容量的最小值
        """
        # 从combo1视角：combo1 + project(combo2, nodes1)
        projection1 = self._project_combo_to_nodes(combo2, nodes1)
        super_combo1 = self._canonicalize_combo(combo1 + projection1)
        capacity1 = self._predict_combo_bandwidth(super_combo1)
        
        # 从combo2视角：combo2 + project(combo1, nodes2)
        projection2 = self._project_combo_to_nodes(combo1, nodes2)
        super_combo2 = self._canonicalize_combo(combo2 + projection2)
        capacity2 = self._predict_combo_bandwidth(super_combo2)
        
        # 取最小值
        return min(capacity1, capacity2)

    def predict_with_contention(self, candidate_combo: np.ndarray) -> float:
        """
        Probe接口：预测如果加入该候选组合，它能获得的带宽。
        实现逻辑：
        1. 判断候选组合是否跨节点。
        2. 如果候选组合不是跨节点的，直接返回独立带宽（单节点任务不与其他任务争用）。
        3. 如果候选组合是跨节点的，只考虑其他跨节点的活跃任务进行争用计算。
        4. 对于每个与候选任务有共享节点的跨节点任务，计算双向super combo容量：
           - 从候选任务视角：candidate + project(job, candidate_nodes)
           - 从已有任务视角：job + project(candidate, job_nodes)
           - 取这两个容量的最小值
        5. 最终的瓶颈容量 = 所有双向容量的最小值
        6. 总需求 = Candidate_Standalone + 所有相关任务的完整 Standalone_BW
        7. 若总需求 > 瓶颈，按比例瓜分。
        """
        candidate_combo = self._validate_combo_request(candidate_combo, enforce_availability=True)

        # 计算候选者的独立带宽
        candidate_standalone = self._predict_combo_bandwidth(candidate_combo)
        
        # 1. 判断候选组合是否跨节点
        if not self._is_cross_node_combo(candidate_combo):
            # 单节点任务不与其他任务争用，直接返回独立带宽
            return candidate_standalone
        
        # 2. 候选组合是跨节点的，需要检查与其他跨节点任务的争用
        # 确定候选任务涉及的节点索引
        candidate_nodes = self._get_nodes_for_combo(candidate_combo)
        
        if not candidate_nodes:
            return 0.0

        # 3. 收集所有与候选任务有共享节点的跨节点任务，并计算双向super combo容量
        # 对于每个相关任务，计算候选任务与该任务的双向容量，取最小值
        dual_capacities = []
        existing_demands = 0.0
        
        # 只考虑跨节点的活跃任务
        for job in self.active_jobs:
            job_combo = job['combo']
            # 只处理跨节点的任务
            if not self._is_cross_node_combo(job_combo):
                continue
            
            # 检查该任务是否与候选任务有共享节点
            job_nodes = self._get_nodes_for_combo(job_combo)
            
            # 如果有共享节点，则参与争用计算
            if candidate_nodes & job_nodes:  # 集合交集不为空
                # 计算候选任务与该任务的双向super combo容量
                dual_cap = self._calculate_dual_super_combo_capacity(
                    candidate_combo, candidate_nodes,
                    job_combo, job_nodes
                )
                dual_capacities.append(dual_cap)
                # 使用完整任务的独立带宽
                existing_demands += job['standalone_bw']

        # 4. 计算瓶颈带宽：取所有双向容量的最小值
        if not dual_capacities:
            # 没有其他相关任务，直接返回独立带宽
            return candidate_standalone
        
        max_capacity = min(dual_capacities)
        
        # 5. 争用判定
        total_demand = candidate_standalone + existing_demands
        
        if total_demand <= max_capacity:
            # 未达到瓶颈
            return candidate_standalone
        else:
            # 达到瓶颈，按比例瓜分
            # 候选者获得的比例
            ratio = candidate_standalone / total_demand
            allocated_bw = max_capacity * ratio
            return allocated_bw

    def allocate_job(self, job_id: int, combo: np.ndarray) -> float:
        """
        Commit接口：正式分配GPU给任务，更新状态，并修正受影响的历史任务带宽。
        
        根据需求，只有跨节点的任务才会相互干扰。单节点任务不会影响其他任务的带宽。
        """
        combo = self._validate_combo_request(combo, enforce_availability=True)

        # 1. 注册新任务
        standalone_bw = self._predict_combo_bandwidth(combo)
        self.allocated_gpu_mask += combo
        
        new_job = {
            'job_id': job_id,
            'combo': combo,
            'standalone_bw': standalone_bw,
            'current_bw': standalone_bw, # 初始值，马上会更新
            'history': []
        }
        self.active_jobs.append(new_job)
        
        # 2. 判断新任务是否跨节点
        if not self._is_cross_node_combo(combo):
            # 单节点任务不与其他任务争用，直接设置带宽为独立带宽
            # 不更新其他任务，记录历史后返回
            for job in self.active_jobs:
                job['history'].append(job['current_bw'])
            return standalone_bw
        
        # 3. 新任务是跨节点的，需要检查与其他跨节点任务的争用
        # 全局重算（处理争用）
        # 这里的逻辑：新任务的加入可能导致与其有共享节点的其他跨节点任务带宽下降（修正）。
        # 根据需求：修正值 = min(原记录值, 新的瓜分值)
        
        # 确定新任务涉及的节点索引
        new_job_nodes = self._get_nodes_for_combo(combo)
                
        if not new_job_nodes:
            # 记录历史后返回
            for job in self.active_jobs:
                job['history'].append(job['current_bw'])
            return 0.0
            
        # 收集所有与新任务有共享节点的跨节点任务（包括刚加入的自己）
        # 对于每个相关任务，计算新任务与该任务的双向super combo容量
        clashing_jobs = [] # List of (job_dict, standalone_bw)
        dual_capacities = []
        
        for job in self.active_jobs:
            job_combo = job['combo']
            # 只处理跨节点的任务
            if not self._is_cross_node_combo(job_combo):
                continue
            
            # 检查该任务是否与新任务有共享节点
            job_nodes = self._get_nodes_for_combo(job_combo)
            
            # 如果有共享节点，则参与争用计算
            if new_job_nodes & job_nodes:  # 集合交集不为空
                # 计算新任务与该任务的双向super combo容量
                dual_cap = self._calculate_dual_super_combo_capacity(
                    combo, new_job_nodes,
                    job_combo, job_nodes
                )
                dual_capacities.append(dual_cap)
                # 使用完整任务的独立带宽
                clashing_jobs.append((job, job['standalone_bw']))
        
        # 计算瓶颈：取所有双向容量的最小值
        if not dual_capacities:
            # 没有其他相关任务，新任务维持独立带宽
            for job in self.active_jobs:
                job['history'].append(job['current_bw'])
            return standalone_bw
        
        max_capacity = min(dual_capacities)
        total_demand = sum(item[1] for item in clashing_jobs)
        
        # 更新带宽
        if total_demand > max_capacity:
            # print(f"带宽争用检测: 节点 {new_job_nodes} 需求 {total_demand:.2f} > 容量 {max_capacity:.2f}")
            for job, job_standalone_bw in clashing_jobs:
                ratio = job_standalone_bw / total_demand
                allocated_part = max_capacity * ratio
                
                # 核心逻辑：
                # 如果是新任务，它的带宽就是 allocated_part
                # 如果是旧任务，它的新带宽 = min(它之前的 current_bw, allocated_part)
                # 这符合 "修正后的...还会与之前记录的...对比，各自选择最小的那个"
                
                if job['job_id'] == job_id:
                    job['current_bw'] = allocated_part
                else:
                    old_bw = job['current_bw']
                    new_bw = min(old_bw, allocated_part)
                    if new_bw < old_bw:
                        # print(f"Job {job['job_id']} 带宽被压缩: {old_bw:.2f} -> {new_bw:.2f}")
                        job['current_bw'] = new_bw
                    
        else:
            # 如果没有瓶颈，新任务维持 standalone_bw，旧任务不受影响（维持原状）
            # 但为了保险，新任务这里确认设为 standalone
            self.active_jobs[-1]['current_bw'] = self.active_jobs[-1]['standalone_bw']

        # 记录历史
        for job in self.active_jobs:
            job['history'].append(job['current_bw'])
            
        return self.active_jobs[-1]['current_bw']

    def get_state_summary(self):
        """获取当前集群状态摘要"""
        return [
            {
                'job_id': j['job_id'], 
                'bw': j['current_bw'], 
                'gpu_count': int(np.sum(j['combo']))
            } 
            for j in self.active_jobs
        ]