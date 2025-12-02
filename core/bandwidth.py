"""带宽计算核心模块。

该模块封装了带宽查表、GPU配置统计以及与模型输入相关的核心逻辑。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

import pickle

from data.preprocessing import preprocess_gpu_data, find_matching_bandwidth, analyze_gpu_pattern

logger = logging.getLogger(__name__)


@dataclass
class SwitchBandwidthConfig:
    """交换机带宽配置类。"""

    num_machines: int
    cluster_type: str | None = None
    bw_matrix: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.bw_matrix = np.zeros((self.num_machines, self.num_machines), dtype=float)

    def set_bandwidth(self, i: int, j: int, bw: float) -> None:
        """设置两个机器组之间的带宽。"""
        if 0 <= i < self.num_machines and 0 <= j < self.num_machines:
            self.bw_matrix[i, j] = bw
            self.bw_matrix[j, i] = bw

    def get_bandwidth(self, i: int, j: int) -> float:
        """获取两个机器组之间的带宽。"""
        if 0 <= i < self.num_machines and 0 <= j < self.num_machines:
            return float(self.bw_matrix[i, j])
        return 0.0

    def get_path_bandwidth(self, path: Sequence[int]) -> float:
        """返回路径上的最小带宽。"""
        if len(path) < 2:
            return float("inf")
        bandwidths = [
            self.get_bandwidth(path[idx], path[idx + 1]) for idx in range(len(path) - 1)
            if self.get_bandwidth(path[idx], path[idx + 1]) > 0
        ]
        return min(bandwidths) if bandwidths else 0.0


class BandwidthLookupCache:
    """负责缓存查找表，避免重复读取CSV。"""

    _lookup_table: Dict | None = None

    @classmethod
    def ensure_loaded(cls, data_path: Path) -> Dict:
        if cls._lookup_table is None:
            logger.info("加载带宽查找表: %s", data_path)
            cls._lookup_table = preprocess_gpu_data(str(data_path))
            if cls._lookup_table is None:
                raise RuntimeError(f"无法加载带宽查找表: {data_path}")
        return cls._lookup_table


def load_gpu_bw_dict(file_path: Path) -> Dict:
    """从pickle文件中加载GPU带宽字典。"""
    if not file_path.exists():
        raise FileNotFoundError(f"Bandwidth dictionary not found: {file_path}")
    with file_path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"File {file_path} did not contain a dictionary.")
    return data


CUSTOM_CLUSTER_NODE_TYPES = {
    "Het-4Mix": ["4090", "V100", "A6000","A800"],
}


def _expand_gpu_types_for_nodes(node_types: Sequence[str], repeat: int) -> List[str]:
    if repeat <= 0 or not node_types:
        return []
    cycles = math.ceil(repeat / len(node_types))
    ordered = list(node_types) * cycles
    return [f"{gpu}_gpu_bw_dict.pkl" for gpu in ordered[:repeat]]


def get_gpu_dict_files(cluster_type: str, repeat: int) -> List[str]:
    """根据集群类型列举需要的带宽字典文件。"""
    if cluster_type in CUSTOM_CLUSTER_NODE_TYPES:
        node_types = CUSTOM_CLUSTER_NODE_TYPES[cluster_type]
        return _expand_gpu_types_for_nodes(node_types, repeat)

    known_gpu_types = ["4090", "V100", "A6000", "A800", "H100_26", "H100_27", "H100_28", "H100_29"]
    gpu_types = [gpu for gpu in known_gpu_types if gpu in cluster_type]
    if not gpu_types:
        logger.warning("cluster_type中未发现已知GPU类型: %s", cluster_type)
        return []
    return _expand_gpu_types_for_nodes(gpu_types, repeat)


def get_gpu_counts_for_model(gpu_config: np.ndarray, total_gpu: int) -> Tuple[List[int], int]:
    """计算每个节点的活跃GPU数及总数。"""
    if total_gpu % 8 != 0:
        raise ValueError("total_gpu must be divisible by 8")
    num_machines = total_gpu // 8
    per_node_counts = [
        int(np.sum(gpu_config[idx * 8 : (idx + 1) * 8])) for idx in range(num_machines)
    ]
    total_active = int(np.sum(per_node_counts))
    return per_node_counts, total_active


def calculate_bandwidth_values(
    gpu: Sequence[int],
    total_gpu: int,
    gpu_bw_dict_list: Sequence[Dict[Tuple[int, ...], float]],
    switch_config: SwitchBandwidthConfig | float | None,
    data_path: str,
) -> Tuple[float, List[float], SwitchBandwidthConfig | float | None]:
    """根据GPU配置查表获取带宽值。"""
    if total_gpu % 8 != 0:
        raise ValueError("total_gpu must be a multiple of 8")
    if len(gpu) != total_gpu:
        raise ValueError("gpu配置长度必须等于total_gpu")

    # 确保gpu是标准的Python列表格式，而不是np.ndarray
    # 这对于后续的切片和类型转换很重要
    if isinstance(gpu, np.ndarray):
        gpu = gpu.tolist()
    else:
        gpu = list(gpu)

    # 早期返回：如果gpu_sum为0或1，直接返回0带宽
    # 单GPU配置通常不在带宽表中，避免不必要的查表和警告
    gpu_sum = sum(gpu)
    if gpu_sum == 0:
        # 空配置，返回0带宽
        parts = [tuple(int(x) for x in gpu[idx : idx + 8]) for idx in range(0, total_gpu, 8)]
        part_bandwidths = [0.0] * len(parts)
        return 0.0, part_bandwidths, switch_config
    elif gpu_sum == 1:
        # 单GPU配置，直接返回0带宽，避免查表失败产生警告
        parts = [tuple(int(x) for x in gpu[idx : idx + 8]) for idx in range(0, total_gpu, 8)]
        part_bandwidths = [0.0] * len(parts)
        return 0.0, part_bandwidths, switch_config

    lookup_table = BandwidthLookupCache.ensure_loaded(Path(data_path))
    # 确保每个node配置都是标准的Python列表，元素都是int类型
    nodes_config = []
    for idx in range(0, total_gpu, 8):
        node_slice = gpu[idx : idx + 8]
        # 确保是列表格式，且元素都是int类型
        node_list = [int(x) for x in node_slice]
        # 如果长度不足8，补齐为0
        if len(node_list) < 8:
            node_list.extend([0] * (8 - len(node_list)))
        nodes_config.append(node_list)
    
    result = find_matching_bandwidth(nodes_config, lookup_table)
    if result is not None:
        _, bandwidth = result
        final_bandwidth = float(bandwidth)  # 确保带宽是浮点数
    else:
        # 如果找不到匹配项，记录调试信息并返回0带宽
        key = analyze_gpu_pattern(nodes_config)
        logger.warning(
            f"在带宽表中未找到匹配的GPU配置: "
            f"nodes_config={nodes_config}, key={key}, "
            f"total_gpu={total_gpu}, gpu_sum={sum(gpu)}"
        )
        final_bandwidth = 0.0

    parts = [tuple(int(x) for x in gpu[idx : idx + 8]) for idx in range(0, total_gpu, 8)]
    part_bandwidths: List[float] = []
    for idx, part_tuple in enumerate(parts):
        current_dict = gpu_bw_dict_list[idx]
        part_bandwidths.append(float(round(current_dict.get(part_tuple, 0.0), 2)))

    cluster_label = getattr(switch_config, "cluster_type", None)
    if cluster_label in CUSTOM_CLUSTER_NODE_TYPES:
        active_bws = [
            part_bandwidths[idx]
            for idx, part in enumerate(parts)
            if any(part)
        ]
        if active_bws:
            intra_bottleneck = min(active_bws)
            final_bandwidth = float(min(final_bandwidth, intra_bottleneck))

    return final_bandwidth, part_bandwidths, switch_config


def config_to_bandwidth(
    gpu_config_list: Iterable[Sequence[int]],
    total_gpu: int,
    gpu_bw_dict_list: Sequence[Dict[Tuple[int, ...], float]],
    switch_config: SwitchBandwidthConfig | float | None,
    data_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """将一组GPU配置转换为带宽值及分组带宽。"""
    bandwidths: List[float] = []
    part_bandwidths: List[List[float]] = []
    for gpu_config in gpu_config_list:
        final_bw, part_bws, _ = calculate_bandwidth_values(
            gpu_config, total_gpu, gpu_bw_dict_list, switch_config, data_path
        )
        bandwidths.append(final_bw)
        part_bandwidths.append(part_bws)
    return np.array(bandwidths), np.array(part_bandwidths)


def prepare_model_inputs(
    gpu_config_list: Sequence[Sequence[int]],
    total_gpu: int,
    gpu_bw_dict_list: Sequence[Dict[Tuple[int, ...], float]],
    switch_config: SwitchBandwidthConfig | float | None,
    data_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """为模型构造part带宽、节点计数及总计数。"""
    part_bws_list: List[List[float]] = []
    node_counts_list: List[List[int]] = []
    total_counts_list: List[int] = []

    for gpu_config in gpu_config_list:
        _, part_bws, _ = calculate_bandwidth_values(
            gpu_config, total_gpu, gpu_bw_dict_list, switch_config, data_path
        )
        node_counts, total_counts = get_gpu_counts_for_model(np.array(gpu_config), total_gpu)

        part_bws_list.append(part_bws)
        node_counts_list.append(node_counts)
        total_counts_list.append(total_counts)

    return (
        np.array(part_bws_list),
        np.array(node_counts_list),
        np.array(total_counts_list).reshape(-1, 1),
    )

