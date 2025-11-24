"""带宽数据预处理与查找。"""
from __future__ import annotations

import ast
import logging
import time
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

AnalysisKey = Tuple[int, int, Tuple[int, ...]]
LookupTable = Dict[AnalysisKey, List[Tuple[str, float]]]


def analyze_gpu_pattern(pattern: Iterable[Iterable[int]]) -> AnalysisKey | Tuple[None, None, None]:
    """根据节点GPU活动情况生成查找键。"""
    total_active = 0
    active_counts: List[int] = []
    try:
        for node in pattern:
            node_active = sum(int(gpu) for gpu in node if int(gpu) == 1)
            if node_active > 0:
                active_counts.append(node_active)
            total_active += node_active
    except (ValueError, TypeError):
        return (None, None, None)

    if total_active == 0:
        return (0, 0, tuple())
    return total_active, len(active_counts), tuple(sorted(active_counts))


def preprocess_gpu_data(file_path: str) -> LookupTable | None:
    """读取CSV并构建查找表。"""
    logger.info("开始预处理带宽数据: %s", file_path)
    print(f"开始预处理带宽数据文件: {file_path}...")
    start = time.time()
    lookup: DefaultDict[AnalysisKey, List[Tuple[str, float]]] = defaultdict(list)

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error("CSV文件未找到: %s", file_path)
        return None

    required_cols = ["GPU_Mapping_Across_Nodes", "Bandwidth(GB/s)"]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logger.error("CSV缺少必要列: %s", missing)
        return None

    processed = 0
    for _, row in df.iterrows():
        mapping_str = row["GPU_Mapping_Across_Nodes"]
        bandwidth = row["Bandwidth(GB/s)"]
        if pd.isna(mapping_str) or pd.isna(bandwidth):
            continue

        try:
            pattern = ast.literal_eval(str(mapping_str))
            key = analyze_gpu_pattern(pattern)
            if key != (None, None, None):
                lookup[key].append((str(mapping_str), float(bandwidth)))
                processed += 1
        except (ValueError, SyntaxError, TypeError):
            continue

    duration = time.time() - start
    logger.info("带宽数据预处理完成，成功处理 %s 行，耗时 %.2fs", processed, duration)
    print(f"带宽数据预处理完成，耗时 {duration:.2f} 秒。")
    print(f"  成功处理 {processed} 个有效行。")
    return dict(lookup)


def find_matching_bandwidth(
    test_data: Sequence[Sequence[int]],
    lookup_table: LookupTable,
) -> Optional[Tuple[str, float]]:
    """在查找表中寻找与输入配置匹配的带宽。"""
    if lookup_table is None:
        logger.error("查找表不可用")
        return None

    # 确保test_data是标准的Python列表格式
    # 将每个node转换为标准的列表格式，元素都是int类型
    normalized_data = []
    for node in test_data:
        if isinstance(node, np.ndarray):
            normalized_node = [int(x) for x in node.tolist()]
        else:
            normalized_node = [int(x) for x in node]
        normalized_data.append(normalized_node)

    key = analyze_gpu_pattern(normalized_data)
    if key == (None, None, None):
        logger.error("输入的GPU配置格式不合法: test_data=%s, normalized_data=%s", test_data, normalized_data)
        return None

    matches = lookup_table.get(key, [])
    if not matches:
        # 记录查找失败时的调试信息
        logger.debug(
            f"带宽表查找失败: key={key}, normalized_data={normalized_data}, "
            f"lookup_table_keys_count={len(lookup_table)}"
        )
        return None
    return matches[0]

