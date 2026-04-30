"""Worker process for two-GPU tensor-parallel sidecar runs.

The worker is launched by `torchrun`, supports preflight and full modes, records
per-step timing and memory metrics, and writes rank-zero JSONL records for the
runner's analysis phase.
"""

from __future__ import annotations

import argparse
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Replicate

# Direct `torchrun worker.py` execution needs the repository root on sys.path.
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.llm_tp_bandwidth.dataset import iterate_blocks, load_token_blocks
from evaluation.llm_tp_bandwidth.gpu_monitor import (
    GPUExclusiveViolation,
    assert_exclusive,
    list_processes_for_gpu_pair,
)
from evaluation.llm_tp_bandwidth.io_utils import append_jsonl, load_json, write_json


def parse_args() -> argparse.Namespace:
    """Parse worker arguments passed by `runner.py`.

    Arguments are intentionally explicit so worker logs remain self-contained.
    """

    parser = argparse.ArgumentParser(description="Two-GPU TP training worker")
    parser.add_argument("--run-dir", type=Path, required=True, help="Artifact directory for this run")
    parser.add_argument(
        "--run-mode",
        type=str,
        choices=["preflight", "full"],
        required=True,
        help="Worker mode: preflight or full",
    )
    parser.add_argument("--model-path", type=str, required=True, help="Local HF model directory")
    parser.add_argument("--model-name", type=str, required=True, help="Human-readable model label")
    parser.add_argument("--token-cache-path", type=Path, help="Prepared token cache for full run")
    parser.add_argument("--seq-len", type=int, required=True, help="Sequence length for each training step")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Warmup step count")
    parser.add_argument("--measured-steps", type=int, default=0, help="Measured step count")
    parser.add_argument("--profile-steps", type=int, default=0, help="Profiler sidecar step count")
    parser.add_argument("--micro-batch-size", type=int, default=1, help="Micro batch size; current harness expects 1")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument(
        "--gradient-checkpointing",
        type=str,
        default="true",
        help="Whether to enable gradient checkpointing (`true` / `false`)",
    )
    parser.add_argument(
        "--physical-gpu-pair",
        type=str,
        required=True,
        help="Physical GPU indices, e.g. `0,1`",
    )
    parser.add_argument("--requested-bridge-type", type=str, required=True, help="Requested bridge type")
    parser.add_argument("--detected-bridge-type", type=str, required=True, help="Detected bridge type")
    parser.add_argument(
        "--detected-topology-label",
        type=str,
        required=True,
        help="Raw topology label from `nvidia-smi topo -m`",
    )
    return parser.parse_args()


def _parse_bool(value: str) -> bool:
    """Parse a permissive boolean string."""

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value from: {value}")


def _parse_gpu_pair(raw_pair: str) -> List[int]:
    """Parse a physical GPU pair such as `0,1`."""

    pair = [int(item.strip()) for item in raw_pair.split(",") if item.strip()]
    if len(pair) != 2:
        raise ValueError(f"physical-gpu-pair must contain exactly two indices, got {raw_pair}")
    return pair


def _rank() -> int:
    """Return the global distributed rank."""

    return int(os.environ.get("RANK", "0"))


def _local_rank() -> int:
    """Return the local rank within the node."""

    return int(os.environ.get("LOCAL_RANK", "0"))


def _world_size() -> int:
    """Return the tensor-parallel world size."""

    return int(os.environ.get("WORLD_SIZE", "1"))


def _is_rank0() -> bool:
    """Return whether the current process is rank zero."""

    return _rank() == 0


def _dist_ready() -> bool:
    """Return whether torch.distributed is initialized."""

    return dist.is_available() and dist.is_initialized()


def _init_distributed() -> None:
    """Initialize NCCL distributed state and set the CUDA device."""

    torch.cuda.set_device(_local_rank())
    if not _dist_ready():
        dist.init_process_group(backend="nccl")


def _cleanup_distributed() -> None:
    """Destroy the distributed process group when initialized."""

    if _dist_ready():
        dist.destroy_process_group()


def _all_rank_pids() -> Set[int]:
    """Gather all TP worker PIDs for GPU exclusivity checks."""

    current_pid = os.getpid()
    if not _dist_ready():
        return {current_pid}

    gathered: List[int] = [0 for _ in range(_world_size())]
    dist.all_gather_object(gathered, current_pid)
    return {int(pid) for pid in gathered}


def _reduce_scalar(value: float, op: str) -> float:
    """Reduce one scalar across ranks.

    `op=max` is used for wall time, phase time, and memory. `op=mean` is used
    for loss.
    """

    if not _dist_ready():
        return float(value)

    tensor = torch.tensor(float(value), dtype=torch.float64, device=torch.cuda.current_device())
    if op == "max":
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return float(tensor.item())
    if op == "mean":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= float(_world_size())
        return float(tensor.item())
    raise ValueError(f"Unsupported reduce op: {op}")


def _metadata_path(run_dir: Path) -> Path:
    """Return the metadata path for a run directory."""

    return run_dir / "run_metadata.json"


def _update_metadata(run_dir: Path, **fields: object) -> Dict[str, object]:
    """Update run metadata from rank zero.

    The helper performs a load-merge-write cycle so the runner can observe
    progress and failure state.
    """

    if not _is_rank0():
        return {}

    path = _metadata_path(run_dir)
    payload: Dict[str, object] = load_json(path) if path.exists() else {}
    payload.update(fields)
    write_json(path, payload)
    return payload


def _record_exclusivity_snapshot(
    *,
    raw_path: Path,
    gpu_pair: Sequence[int],
    allowed_pids: Iterable[int],
    step_idx: int,
    phase: str,
    stage: str,
) -> None:
    """Record a GPU-pair compute-process snapshot.

    Only rank zero writes the event to avoid duplicate records.
    """

    if not _is_rank0():
        return

    allowed_pid_set = {int(pid) for pid in allowed_pids}
    processes = list_processes_for_gpu_pair(gpu_pair)
    append_jsonl(
        raw_path,
        {
            "timestamp": time.time(),
            "phase": phase,
            "stage": stage,
            "step_idx": int(step_idx),
            "gpu_pair": list(gpu_pair),
            "allowed_pids": sorted(allowed_pid_set),
            "processes": [
                {
                    "gpu_uuid": process.gpu_uuid,
                    "pid": int(process.pid),
                    "process_name": process.process_name,
                    "used_memory_mb": int(process.used_memory_mb),
                    "is_allowed": int(process.pid) in allowed_pid_set,
                }
                for process in processes
            ],
        },
    )


def _assert_exclusive_with_logging(
    *,
    gpu_pair: Sequence[int],
    allowed_pids: Iterable[int],
    raw_event_path: Path,
    step_idx: int,
    phase: str,
    stage: str,
) -> None:
    """Assert exclusivity and log a snapshot before re-raising violations."""

    try:
        assert_exclusive(gpu_pair, allowed_pids)
    except GPUExclusiveViolation:
        _record_exclusivity_snapshot(
            raw_path=raw_event_path,
            gpu_pair=gpu_pair,
            allowed_pids=allowed_pids,
            step_idx=step_idx,
            phase=phase,
            stage=stage,
        )
        raise


def _build_model(model_path: str, enable_gradient_checkpointing: bool) -> torch.nn.Module:
    """Build a HuggingFace causal LM with the automatic TP plan."""

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        tp_plan="auto",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if enable_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    return model


def _device() -> torch.device:
    """Return this rank's CUDA device."""

    return torch.device("cuda", _local_rank())


def _loss_value(loss: torch.Tensor) -> float:
    """Convert a loss tensor to a Python float."""

    return float(loss.detach().float().item())


def _run_train_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_input_ids: torch.Tensor,
) -> Dict[str, float]:
    """Run one training step and record forward/backward/optimizer timings.

    CUDA events measure phase time while host wall time captures end-to-end
    synchronization overhead.
    """

    batch_input_ids = batch_input_ids.to(device=_device(), non_blocking=False)
    torch.cuda.reset_peak_memory_stats(_device())
    optimizer.zero_grad(set_to_none=True)

    # Synchronize before timing so each step starts from a clean CUDA boundary.
    torch.cuda.synchronize(_device())
    step_start_wall = time.perf_counter()

    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    backward_start = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)
    optimizer_start = torch.cuda.Event(enable_timing=True)
    optimizer_end = torch.cuda.Event(enable_timing=True)

    forward_start.record()
    # Avoid `model(..., labels=...)` because the HF auto-TP plan can keep
    # `lm_head.weight` as a DTensor while base-model hidden states are local.
    # Running the base model and lm_head explicitly avoids mixed Tensor/DTensor
    # loss-path issues.
    base_outputs = model.model(input_ids=batch_input_ids, use_cache=False)
    hidden_states = base_outputs.last_hidden_state
    if not isinstance(hidden_states, DTensor):
        hidden_states = DTensor.from_local(hidden_states, model._device_mesh, [Replicate()], run_check=False)
    logits = model.lm_head(hidden_states)
    if isinstance(logits, DTensor):
        logits = logits.to_local()

    # Standard causal-LM shift-one-token cross entropy.
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = batch_input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
    )
    forward_end.record()

    backward_start.record()
    loss.backward()
    backward_end.record()

    optimizer_start.record()
    optimizer.step()
    optimizer_end.record()

    # Synchronize before collecting host wall time and CUDA events.
    torch.cuda.synchronize(_device())
    step_wall_ms = (time.perf_counter() - step_start_wall) * 1000.0

    metrics = {
        "loss": _loss_value(loss),
        "host_step_wall_time_ms": float(step_wall_ms),
        "forward_ms": float(forward_start.elapsed_time(forward_end)),
        "backward_ms": float(backward_start.elapsed_time(backward_end)),
        "optimizer_ms": float(optimizer_start.elapsed_time(optimizer_end)),
        "max_memory_allocated_mb": float(torch.cuda.max_memory_allocated(_device()) / (1024.0 ** 2)),
        "max_memory_reserved_mb": float(torch.cuda.max_memory_reserved(_device()) / (1024.0 ** 2)),
    }
    return metrics


def _aggregate_step_metrics(step_metrics: Dict[str, float], tokens_per_step: int) -> Dict[str, float]:
    """Aggregate rank-local metrics into one TP-step record."""

    total_ms = _reduce_scalar(step_metrics["host_step_wall_time_ms"], op="max")
    forward_ms = _reduce_scalar(step_metrics["forward_ms"], op="max")
    backward_ms = _reduce_scalar(step_metrics["backward_ms"], op="max")
    optimizer_ms = _reduce_scalar(step_metrics["optimizer_ms"], op="max")
    loss_mean = _reduce_scalar(step_metrics["loss"], op="mean")
    memory_allocated_mb = _reduce_scalar(step_metrics["max_memory_allocated_mb"], op="max")
    memory_reserved_mb = _reduce_scalar(step_metrics["max_memory_reserved_mb"], op="max")

    tokens_per_sec = 0.0
    if total_ms > 0:
        # TP shards one model rather than data-parallel batches, so token count
        # is not multiplied by world size.
        tokens_per_sec = float(tokens_per_step) / (float(total_ms) / 1000.0)

    return {
        "loss": float(loss_mean),
        "host_step_wall_time_ms": float(total_ms),
        "forward_ms": float(forward_ms),
        "backward_ms": float(backward_ms),
        "optimizer_ms": float(optimizer_ms),
        "tokens_per_sec": float(tokens_per_sec),
        "max_memory_allocated_mb": float(memory_allocated_mb),
        "max_memory_reserved_mb": float(memory_reserved_mb),
    }


def _profiler_nccl_summary(profiler: torch.profiler.profile) -> Dict[str, object]:
    """Summarize NCCL CUDA activity from a profiler sidecar step.

    The profiler is intentionally short; it explains communication time without
    making the main measured run pay profiler overhead.
    """

    total_cuda_time_us = 0.0
    kernel_count = 0
    event_names: List[str] = []
    for event in profiler.key_averages():
        key = str(getattr(event, "key", "") or "")
        if "nccl" not in key.lower():
            continue
        device_time = getattr(event, "device_time_total", None)
        if device_time is None:
            device_time = getattr(event, "cuda_time_total", 0.0)
        total_cuda_time_us += float(device_time or 0.0)
        kernel_count += int(getattr(event, "count", 0) or 0)
        event_names.append(key)

    max_nccl_ms = _reduce_scalar(total_cuda_time_us / 1000.0, op="max")
    max_kernel_count = int(round(_reduce_scalar(float(kernel_count), op="max")))
    merged_names = sorted(set(event_names))
    return {
        "nccl_cuda_time_ms": float(max_nccl_ms),
        "nccl_kernel_count": int(max_kernel_count),
        "nccl_event_names": merged_names,
    }


def _run_profile_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_input_ids: torch.Tensor,
) -> Dict[str, object]:
    """Run one profiled training step and merge NCCL summary fields."""

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        profile_memory=False,
        record_shapes=False,
        with_stack=False,
    ) as profiler:
        step_metrics = _run_train_step(
            model=model,
            optimizer=optimizer,
            batch_input_ids=batch_input_ids,
        )

    profile_summary = _profiler_nccl_summary(profiler)
    profile_summary.update(step_metrics)
    return profile_summary


def _write_preflight_result(
    *,
    run_dir: Path,
    payload: Dict[str, object],
) -> None:
    """Write the preflight result from rank zero."""

    if _is_rank0():
        write_json(run_dir / "preflight_result.json", payload)


def _full_phase_plan(args: argparse.Namespace) -> List[str]:
    """Build the warmup, measured, and profile phase plan for a full run."""

    phases: List[str] = []
    phases.extend(["warmup"] * int(args.warmup_steps))
    phases.extend(["measured"] * int(args.measured_steps))
    phases.extend(["profile"] * int(args.profile_steps))
    return phases


def main() -> int:
    """Worker entry point."""

    args = parse_args()
    if int(args.micro_batch_size) != 1:
        raise ValueError("Current harness only supports micro_batch_size=1 for strict comparability.")

    physical_gpu_pair = _parse_gpu_pair(args.physical_gpu_pair)
    raw_dir = args.run_dir / "raw"
    step_metrics_path = raw_dir / "step_metrics.jsonl"
    profile_metrics_path = raw_dir / "profiler_steps.jsonl"
    exclusivity_events_path = raw_dir / "exclusivity_events.jsonl"

    try:
        _init_distributed()
        whitelist_pids = _all_rank_pids()

        if _is_rank0():
            from transformers import __version__ as transformers_version

            _update_metadata(
                args.run_dir,
                status="running_preflight" if args.run_mode == "preflight" else "running_full",
                worker_hostname=socket.gethostname(),
                worker_rank0_pid=int(os.getpid()),
                worker_all_pids=sorted(int(pid) for pid in whitelist_pids),
                world_size=int(_world_size()),
                local_rank_count=int(_world_size()),
                python_version=platform.python_version(),
                torch_version=torch.__version__,
                transformers_version=transformers_version,
            )

        _assert_exclusive_with_logging(
            gpu_pair=physical_gpu_pair,
            allowed_pids=whitelist_pids,
            raw_event_path=exclusivity_events_path,
            step_idx=-1,
            phase=args.run_mode,
            stage="before_model_load",
        )

        model = _build_model(
            model_path=args.model_path,
            enable_gradient_checkpointing=_parse_bool(args.gradient_checkpointing),
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )

        if args.run_mode == "preflight":
            batch = torch.zeros((1, int(args.seq_len)), dtype=torch.long)
            step_metrics = _run_train_step(
                model=model,
                optimizer=optimizer,
                batch_input_ids=batch,
            )
            aggregated = _aggregate_step_metrics(
                step_metrics,
                tokens_per_step=int(args.seq_len),
            )
            if _is_rank0():
                payload = {
                    "status": "passed",
                    "model_name": args.model_name,
                    "model_path": args.model_path,
                    "requested_bridge_type": args.requested_bridge_type,
                    "detected_bridge_type": args.detected_bridge_type,
                    "detected_topology_label": args.detected_topology_label,
                    "gpu_pair": physical_gpu_pair,
                    **aggregated,
                }
                _write_preflight_result(run_dir=args.run_dir, payload=payload)
                _update_metadata(
                    args.run_dir,
                    status="preflight_passed",
                    preflight_result=payload,
                )
            return 0

        if args.token_cache_path is None or not args.token_cache_path.exists():
            raise FileNotFoundError(f"Token cache path is missing for full run: {args.token_cache_path}")

        token_blocks = load_token_blocks(args.token_cache_path)
        total_steps = len(_full_phase_plan(args))
        block_iterator = iter(iterate_blocks(token_blocks, total_steps=total_steps))

        global_step = 0
        measured_step_count = 0
        profile_step_count = 0
        for phase in _full_phase_plan(args):
            _assert_exclusive_with_logging(
                gpu_pair=physical_gpu_pair,
                allowed_pids=whitelist_pids,
                raw_event_path=exclusivity_events_path,
                step_idx=global_step,
                phase=phase,
                stage="before_step",
            )

            batch = next(block_iterator).unsqueeze(0)
            if phase == "profile":
                step_metrics = _run_profile_step(
                    model=model,
                    optimizer=optimizer,
                    batch_input_ids=batch,
                )
                aggregated = _aggregate_step_metrics(step_metrics, tokens_per_step=int(args.seq_len))
                aggregated.update(
                    {
                        "nccl_cuda_time_ms": float(step_metrics["nccl_cuda_time_ms"]),
                        "nccl_kernel_count": int(step_metrics["nccl_kernel_count"]),
                        "nccl_event_names": list(step_metrics["nccl_event_names"]),
                    }
                )
            else:
                step_metrics = _run_train_step(
                    model=model,
                    optimizer=optimizer,
                    batch_input_ids=batch,
                )
                aggregated = _aggregate_step_metrics(step_metrics, tokens_per_step=int(args.seq_len))

            _assert_exclusive_with_logging(
                gpu_pair=physical_gpu_pair,
                allowed_pids=whitelist_pids,
                raw_event_path=exclusivity_events_path,
                step_idx=global_step,
                phase=phase,
                stage="after_step",
            )

            if phase == "measured":
                measured_step_count += 1
            if phase == "profile":
                profile_step_count += 1

            if _is_rank0():
                base_record = {
                    "timestamp": time.time(),
                    "step_idx": int(global_step),
                    "phase": phase,
                    "model_name": args.model_name,
                    "model_path": args.model_path,
                    "requested_bridge_type": args.requested_bridge_type,
                    "detected_bridge_type": args.detected_bridge_type,
                    "detected_topology_label": args.detected_topology_label,
                    "gpu_pair": physical_gpu_pair,
                    "world_size": int(_world_size()),
                    **aggregated,
                }
                append_jsonl(step_metrics_path, base_record)
                if phase == "profile":
                    append_jsonl(profile_metrics_path, base_record)
                _update_metadata(
                    args.run_dir,
                    status="running_full",
                    completed_total_steps=int(global_step + 1),
                    completed_measured_steps=int(measured_step_count),
                    completed_profile_steps=int(profile_step_count),
                    last_phase=str(phase),
                )
            global_step += 1

        if _is_rank0():
            _update_metadata(
                args.run_dir,
                status="worker_completed",
                completed_total_steps=int(global_step),
                completed_measured_steps=int(measured_step_count),
                completed_profile_steps=int(profile_step_count),
            )
        return 0

    except GPUExclusiveViolation as exc:
        if _is_rank0():
            _update_metadata(
                args.run_dir,
                status="aborted_external_process",
                error_type=exc.__class__.__name__,
                error_message=str(exc),
            )
            if args.run_mode == "preflight":
                _write_preflight_result(
                    run_dir=args.run_dir,
                    payload={
                        "status": "failed",
                        "model_name": args.model_name,
                        "model_path": args.model_path,
                        "error_type": exc.__class__.__name__,
                        "error_message": str(exc),
                    },
                )
        return 2
    except torch.cuda.OutOfMemoryError as exc:
        if _is_rank0():
            _update_metadata(
                args.run_dir,
                status="failed_oom",
                error_type=exc.__class__.__name__,
                error_message=str(exc),
            )
            if args.run_mode == "preflight":
                _write_preflight_result(
                    run_dir=args.run_dir,
                    payload={
                        "status": "failed",
                        "model_name": args.model_name,
                        "model_path": args.model_path,
                        "error_type": exc.__class__.__name__,
                        "error_message": str(exc),
                    },
                )
        return 3
    except Exception as exc:  # noqa: BLE001
        if _is_rank0():
            _update_metadata(
                args.run_dir,
                status="failed",
                error_type=exc.__class__.__name__,
                error_message=str(exc),
            )
            if args.run_mode == "preflight":
                _write_preflight_result(
                    run_dir=args.run_dir,
                    payload={
                        "status": "failed",
                        "model_name": args.model_name,
                        "model_path": args.model_path,
                        "error_type": exc.__class__.__name__,
                        "error_message": str(exc),
                    },
                )
        raise
    finally:
        _cleanup_distributed()


if __name__ == "__main__":
    raise SystemExit(main())
