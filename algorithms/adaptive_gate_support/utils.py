"""Shared coercion and CSV helpers for BandPilot adaptive replay support.

The public artifact keeps only the utility functions required by the main
single-contention and scalability paths. These helpers intentionally avoid
pandas so runtime decisions and unit tests can run in lightweight
environments.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import List, Mapping, Sequence


def _as_bool(value: object) -> bool:
    """Return a stable boolean interpretation for YAML/CSV-style values."""

    if isinstance(value, bool):
        return value
    if value in ("", None):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_int(value: object) -> int:
    """Return an integer for YAML/CSV-style values, treating empty values as zero."""

    if value in ("", None):
        return 0
    return int(value)


def _as_float(value: object) -> float:
    """Return a float for YAML/CSV-style values, treating empty values as zero."""

    if value in ("", None):
        return 0.0
    return float(value)


def _as_json_list(value: object) -> List[object]:
    """Decode a JSON-list field while accepting already materialized lists."""

    if value in ("", None):
        return []
    if isinstance(value, list):
        return list(value)
    return list(json.loads(str(value)))


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean, or zero for an empty sequence."""

    return float(statistics.mean(values)) if values else 0.0


def _percentile(values: Sequence[float], q: float) -> float:
    """Return a linearly interpolated percentile without requiring pandas."""

    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(float(value) for value in values)
    position = (len(sorted_values) - 1) * max(0.0, min(1.0, float(q)))
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write row dictionaries to CSV with deterministic first-seen column order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
