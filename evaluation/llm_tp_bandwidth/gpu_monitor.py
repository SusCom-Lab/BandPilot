"""GPU discovery and exclusivity helpers for the TP sidecar.

The helpers parse `nvidia-smi` output, classify GPU-pair topology, and check
that selected GPUs are free before the runner launches worker processes.
"""

from __future__ import annotations

import csv
import io
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple


class GPUExclusiveViolation(RuntimeError):
    """Raised when an external compute process occupies a requested GPU."""


@dataclass(frozen=True)
class GPUInfo:
    """Static GPU identity information from `nvidia-smi`."""

    index: int
    name: str
    uuid: str


@dataclass(frozen=True)
class GPUProcess:
    """One compute process reported by `nvidia-smi`."""

    gpu_uuid: str
    pid: int
    process_name: str
    used_memory_mb: int


def _run_text_command(command: Sequence[str]) -> str:
    """Run a command and return stdout as text."""

    result = subprocess.run(
        list(command),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout


def list_gpus() -> List[GPUInfo]:
    """Parse `nvidia-smi -L` into GPU index, name, and UUID records.

    The parser expects the standard `nvidia-smi -L` format and sorts by GPU
    index for stable downstream selection.
    """

    output = _run_text_command(["nvidia-smi", "-L"])
    gpus: List[GPUInfo] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or not line.startswith("GPU "):
            continue
        # Example: GPU 0: NVIDIA RTX A6000 (UUID: GPU-xxxx)
        prefix, uuid_part = line.split("(UUID:")
        index_part, name_part = prefix.split(":", maxsplit=1)
        index = int(index_part.replace("GPU", "").strip())
        name = name_part.strip()
        uuid = uuid_part.replace(")", "").strip()
        gpus.append(GPUInfo(index=index, name=name, uuid=uuid))
    return sorted(gpus, key=lambda item: item.index)


def topology_matrix() -> Dict[Tuple[int, int], str]:
    """Parse `nvidia-smi topo -m` into `(gpu_a, gpu_b) -> raw label`.

    CPU affinity columns are ignored; only GPU-to-GPU topology entries are kept.
    """

    output = _run_text_command(["nvidia-smi", "topo", "-m"])
    ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
    lines = [ansi_escape.sub("", line).rstrip() for line in output.splitlines() if line.strip()]
    if not lines:
        return {}

    header_tokens = [token.strip() for token in lines[0].split("\t") if token.strip()]
    gpu_headers = [token for token in header_tokens if re.fullmatch(r"GPU\d+", token)]
    matrix: Dict[Tuple[int, int], str] = {}
    for raw_line in lines[1:]:
        tokens = [token.strip() for token in raw_line.split("\t") if token.strip()]
        if not tokens or not re.fullmatch(r"GPU\d+", tokens[0]):
            continue
        row_gpu = int(tokens[0].replace("GPU", ""))
        for column_offset, header in enumerate(gpu_headers, start=1):
            if column_offset >= len(tokens):
                break
            col_gpu = int(header.replace("GPU", ""))
            matrix[(row_gpu, col_gpu)] = tokens[column_offset]
    return matrix


def normalize_bridge_type(raw_label: str) -> str:
    """Normalize an `nvidia-smi topo -m` label to a bridge type.

    `NV*` labels map to `nvlink`; all non-self GPU links map to `pcie`; `X` is a
    self-link and is rejected.
    """

    if raw_label.startswith("NV"):
        return "nvlink"
    if raw_label == "X":
        raise ValueError("Self link label `X` cannot be normalized as a bridge type.")
    return "pcie"


def describe_gpu_pair(gpu_pair: Sequence[int]) -> Dict[str, object]:
    """Describe GPU names, UUIDs, and topology for one GPU pair."""

    if len(gpu_pair) != 2:
        raise ValueError(f"gpu_pair must contain exactly two indices, got {gpu_pair}")

    ordered_pair = tuple(int(item) for item in gpu_pair)
    topo = topology_matrix()
    raw_label = topo.get((ordered_pair[0], ordered_pair[1]))
    if raw_label is None:
        raise ValueError(f"Cannot find topology label for GPU pair {ordered_pair}")

    info_map = {gpu.index: gpu for gpu in list_gpus()}
    if ordered_pair[0] not in info_map or ordered_pair[1] not in info_map:
        raise ValueError(f"GPU pair {ordered_pair} is not fully present on this machine.")

    return {
        "gpu_pair": list(ordered_pair),
        "gpu_names": [info_map[index].name for index in ordered_pair],
        "gpu_uuids": [info_map[index].uuid for index in ordered_pair],
        "raw_topology_label": raw_label,
        "bridge_type": normalize_bridge_type(raw_label),
    }


def query_compute_processes() -> List[GPUProcess]:
    """Query currently active GPU compute processes.

    Returns an empty list when `nvidia-smi` reports no compute processes.
    """

    output = _run_text_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ]
    ).strip()
    if not output:
        return []

    reader = csv.reader(io.StringIO(output))
    processes: List[GPUProcess] = []
    for row in reader:
        if len(row) != 4:
            continue
        gpu_uuid = row[0].strip()
        pid = int(row[1].strip())
        process_name = row[2].strip()
        memory_text = row[3].strip().replace(" MiB", "")
        processes.append(
            GPUProcess(
                gpu_uuid=gpu_uuid,
                pid=pid,
                process_name=process_name,
                used_memory_mb=int(memory_text),
            )
        )
    return processes


def list_processes_for_gpu_pair(gpu_pair: Sequence[int]) -> List[GPUProcess]:
    """List compute processes that occupy either GPU in a pair."""

    pair_desc = describe_gpu_pair(gpu_pair)
    pair_uuid_set = set(pair_desc["gpu_uuids"])
    return [process for process in query_compute_processes() if process.gpu_uuid in pair_uuid_set]


def assert_exclusive(
    gpu_pair: Sequence[int],
    allowed_pids: Iterable[int],
) -> List[GPUProcess]:
    """Assert that a GPU pair is used only by allowed PIDs.

    Raises `GPUExclusiveViolation` when another process is present.
    """

    allowed_pid_set: Set[int] = {int(pid) for pid in allowed_pids}
    processes = list_processes_for_gpu_pair(gpu_pair)
    violators = [process for process in processes if process.pid not in allowed_pid_set]
    if violators:
        pretty = ", ".join(
            f"pid={proc.pid} name={proc.process_name} mem={proc.used_memory_mb}MB"
            for proc in violators
        )
        raise GPUExclusiveViolation(
            f"External compute process detected on GPU pair {list(gpu_pair)}: {pretty}"
        )
    return processes
