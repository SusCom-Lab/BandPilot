"""拓扑结构相关函数。"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def parse_topo_matrix(filepath: str | Path) -> pd.DataFrame:
    """解析单节点拓扑结构文件（如A6000_topo.txt）。"""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"拓扑文件未找到: {filepath}")

    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    header = lines[0].split("|")[2:-1]
    data_lines = [line.split("|")[1:-1] for line in lines[2:] if not line.startswith("|--")]
    row_labels = [row[0].strip() for row in data_lines]
    matrix = [[cell.strip() for cell in row[1:]] for row in data_lines]

    if len(row_labels) != len(matrix):
        raise ValueError("拓扑文件的行数与标签不匹配")
    if matrix and len(matrix[0]) != len(header):
        raise ValueError("拓扑文件的列数与标签不匹配")

    return pd.DataFrame(matrix, index=row_labels, columns=header)


def build_composite_topo_matrix(node_configs: Sequence[Tuple[str, int]]) -> Tuple[pd.DataFrame, int]:
    """构建复合拓扑矩阵。"""
    parsed: Dict[str, pd.DataFrame] = {}
    total_gpu = 0
    offsets = [0]

    for topo_file, gpus_on_node in node_configs:
        if topo_file not in parsed:
            parsed[topo_file] = parse_topo_matrix(topo_file)
            if parsed[topo_file].shape[0] != gpus_on_node:
                raise ValueError(f"{topo_file} 的GPU数量与配置不一致")
        total_gpu += gpus_on_node
        offsets.append(total_gpu)

    if total_gpu == 0:
        return pd.DataFrame(), 0

    composite = pd.DataFrame(
        "INTER",
        index=[f"GPU{i}" for i in range(total_gpu)],
        columns=[f"GPU{i}" for i in range(total_gpu)],
    )

    for idx, (topo_file, gpus_on_node) in enumerate(node_configs):
        start, end = offsets[idx], offsets[idx + 1]
        composite.iloc[start:end, start:end] = parsed[topo_file].values

    for diag in range(total_gpu):
        composite.iloc[diag, diag] = "X"

    return composite, total_gpu


def get_link_weight(link_type: str) -> int:
    """根据连接类型返回权重。
    
    支持的连接类型：
    - X, INTER: 权重 0（无连接或跨节点）
    - SYS: 权重 1
    - PIX: 权重 1.5
    - PXB: 权重 2
    - NV<N>: NVLink 连接，根据版本号返回权重（NV16+ -> 6, NV8+ -> 5, NV4+ -> 4, NV1+ -> 3）
    """
    mapping = {"X": 0, "INTER": 0, "SYS": 1, "PIX": 1.5, "PXB": 2}
    link_type = link_type.strip().upper()
    # 首先检查是否在预定义映射中
    if link_type in mapping:
        return mapping[link_type]
    # 处理 NVLink 类型（如 NV16, NV8, NV4 等）
    if link_type.startswith("NV"):
        # 修复：在原始字符串中应使用 \d+ 而不是 \\d+
        match = re.match(r"NV(\d+)", link_type)
        if match:
            num = int(match.group(1))
            # 根据 NVLink 版本号返回不同权重
            if num >= 16:
                return 6
            if num >= 8:
                return 5
            if num >= 4:
                return 4
            if num >= 1:
                return 3
    # 如果无法识别连接类型，记录警告并返回 0
    logger.warning("未知连接类型 %s，按0处理", link_type)
    return 0


def calculate_connectivity_score(gpu_indices: Sequence[int], topo_matrix: pd.DataFrame) -> float:
    """计算给定GPU集合的连接权重得分。"""
    score = 0.0
    valid_indices = [idx for idx in sorted(gpu_indices) if 0 <= idx < topo_matrix.shape[0]]
    if len(valid_indices) != len(gpu_indices):
        logger.warning("GPU索引存在越界: all=%s, valid=%s", gpu_indices, valid_indices)

    for i in range(len(valid_indices)):
        for j in range(i + 1, len(valid_indices)):
            link = topo_matrix.iloc[valid_indices[i], valid_indices[j]]
            score += get_link_weight(link)
    return score


def convert_cluster_type_to_node_configs(cluster_type: str, gpu_num: int) -> List[Tuple[str, int]]:
    """根据cluster_type生成节点配置。"""
    gpu_models = ["4090", "V100", "A6000", "A800", "H100_26", "H100_27", "H100_28", "H100_29"]
    extracted = [model for model in gpu_models if model in cluster_type]

    node_configs: List[Tuple[str, int]] = []
    if "H100_26" in extracted:
        node_configs = [("Data/H100_Real/H100_topo.txt", 8) for _ in extracted]
    else:
        for model in extracted:
            node_configs.append((f"data/{model}_topo.txt", 8))

    total = sum(count for _, count in node_configs)
    idx = 0
    while total < gpu_num and extracted:
        model = extracted[idx % len(extracted)]
        node_configs.append((f"data/{model}_topo.txt", 8))
        total += 8
        idx += 1

    return node_configs


def create_gpu_to_node_map(node_configs: Sequence[Tuple[str, int]]) -> Dict[int, int]:
    """创建全局GPU索引到节点索引的映射。"""
    mapping: Dict[int, int] = {}
    start = 0
    for node_idx, (_, gpu_count) in enumerate(node_configs):
        for local_idx in range(gpu_count):
            mapping[start + local_idx] = node_idx
        start += gpu_count
    return mapping

