"""通用工具函数。"""
from __future__ import annotations

from pathlib import Path


def ensure_directory(path: Path) -> None:
    """确保目录存在。"""
    path.mkdir(parents=True, exist_ok=True)

