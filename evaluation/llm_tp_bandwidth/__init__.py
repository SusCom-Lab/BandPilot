"""Two-GPU tensor-parallel bandwidth sidecar package.

The package provides runner, worker, analysis, plotting, and report-building
utilities for local measured sidecar experiments.
"""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = PACKAGE_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIG_ROOT / "default.yaml"
ARTIFACT_ROOT = PACKAGE_ROOT / "artifacts"
LATEST_ROOT = ARTIFACT_ROOT / "latest"
