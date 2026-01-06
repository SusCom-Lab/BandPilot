"""IO-related helper functions."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict


def load_pickle_dict(file_path: Path) -> Dict:
    """Load a dictionary from a pickle file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")
    with file_path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"File {file_path} does not contain a dict")
    return data


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

