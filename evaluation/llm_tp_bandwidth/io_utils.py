"""I/O helpers for the tensor-parallel bandwidth sidecar.

The helpers centralize JSON, JSONL, CSV, manifest, and directory management so
runner, worker, analyzer, and report builder use the same artifact layout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def ensure_directory(path: Path) -> Path:
    """Create a directory if needed and return the same path."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Dict[str, Any]) -> Path:
    """Write UTF-8 pretty JSON and return the output path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def append_jsonl(path: Path, payload: Dict[str, Any]) -> Path:
    """Append one payload to a JSONL file.

    Step metrics, profiler events, and worker records use append-only writes so
    partial runs remain inspectable.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False))
        handle.write("\n")
    return path


def load_json(path: Path) -> Dict[str, Any]:
    """Load a UTF-8 JSON file into a dictionary."""

    return json.loads(path.read_text(encoding="utf-8"))
