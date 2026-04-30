"""Launch two-GPU tensor-parallel sidecar experiments.

The runner parses YAML and CLI overrides, validates GPU availability, prepares
token caches, invokes the worker through `torchrun`, and then triggers analysis,
plotting, report generation, and latest-manifest updates.
"""

from __future__ import annotations

import argparse
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import yaml

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.llm_tp_bandwidth import ARTIFACT_ROOT, DEFAULT_CONFIG_PATH, LATEST_ROOT
from evaluation.llm_tp_bandwidth.analyze import build_summary_artifacts
from evaluation.llm_tp_bandwidth.dataset import prepare_wikitext_token_cache
from evaluation.llm_tp_bandwidth.gpu_monitor import describe_gpu_pair, list_processes_for_gpu_pair
from evaluation.llm_tp_bandwidth.io_utils import append_jsonl, ensure_directory, load_json, write_json
from evaluation.llm_tp_bandwidth.report_builder import build_report


MODEL_REGISTRY = {
    "qwen2.5-7b": "/home/apps/LLM_models/qwen2.5-7B",
    "llama-3.2-3b": "/home/apps/LLM_models/Llama-3.2-3b",
}


def parse_args() -> argparse.Namespace:
    """Parse the runner CLI."""

    parser = argparse.ArgumentParser(description="Run two-GPU TP bandwidth experiment")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="YAML config path")
    parser.add_argument("--gpu-pair", type=str, help="Override physical GPU pair, e.g. `0,1`")
    parser.add_argument(
        "--bridge-type",
        type=str,
        choices=["auto", "nvlink", "pcie"],
        help="Requested bridge type label",
    )
    parser.add_argument(
        "--model-choice",
        type=str,
        choices=["auto", "qwen2.5-7b", "llama-3.2-3b"],
        help="Model selection strategy",
    )
    parser.add_argument("--warmup-steps", type=int, help="Override warmup step count")
    parser.add_argument("--measured-steps", type=int, help="Override measured step count")
    parser.add_argument("--profile-steps", type=int, help="Override profiler step count")
    parser.add_argument("--run-tag", type=str, help="Optional stable run tag")
    return parser.parse_args()


def _load_config(config_path: Path) -> Dict[str, object]:
    """Load a YAML config file."""

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _parse_gpu_pair(raw_pair: str) -> List[int]:
    """Parse a GPU pair from CLI or YAML text."""

    return [int(item.strip()) for item in str(raw_pair).split(",") if item.strip()]


def _candidate_models(model_choice: str) -> List[str]:
    """Return candidate model names in fallback order."""

    if model_choice == "auto":
        return ["qwen2.5-7b", "llama-3.2-3b"]
    return [model_choice]


def _build_run_tag(
    *,
    gpu_pair: Sequence[int],
    detected_bridge_type: str,
    model_choice: str,
    manual_run_tag: str | None,
) -> str:
    """Build a run tag from GPU pair, bridge type, model, and timestamp."""

    if manual_run_tag:
        return manual_run_tag
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    gpu_tag = "-".join(str(index) for index in gpu_pair)
    return f"tp2_a6000_g{gpu_tag}_{detected_bridge_type}_{model_choice}_{timestamp}"


def _prepare_run_dirs(run_dir: Path) -> Dict[str, Path]:
    """Create the run directory layout."""

    raw_dir = ensure_directory(run_dir / "raw")
    ensure_directory(run_dir / "summary")
    ensure_directory(run_dir / "figures")
    report_dir = ensure_directory(run_dir / "reports")
    log_dir = ensure_directory(run_dir / "logs")
    ensure_directory(run_dir / "preflight")
    return {
        "run_dir": run_dir,
        "raw_dir": raw_dir,
        "report_dir": report_dir,
        "log_dir": log_dir,
    }


def _initial_metadata(
    *,
    args: argparse.Namespace,
    config: Dict[str, object],
    run_tag: str,
    run_dir: Path,
    pair_desc: Dict[str, object],
    requested_bridge_type: str,
    model_choice: str,
) -> Dict[str, object]:
    """Build the initial `run_metadata.json` payload."""

    experiment_cfg = dict(config.get("experiment", {}))
    data_cfg = dict(config.get("data", {}))
    return {
        "run_tag": run_tag,
        "run_dir": str(run_dir),
        "status": "preparing",
        "evidence_type": "measured",
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "config_path": str(args.config),
        "gpu_pair": list(pair_desc["gpu_pair"]),
        "gpu_names": list(pair_desc["gpu_names"]),
        "gpu_uuids": list(pair_desc["gpu_uuids"]),
        "requested_bridge_type": requested_bridge_type,
        "detected_bridge_type": pair_desc["bridge_type"],
        "detected_topology_label": pair_desc["raw_topology_label"],
        "model_choice": model_choice,
        "model_attempts": [],
        "warmup_steps": int(experiment_cfg["warmup_steps"]),
        "measured_steps": int(experiment_cfg["measured_steps"]),
        "profile_steps": int(experiment_cfg["profile_steps"]),
        "seq_len": int(experiment_cfg["seq_len"]),
        "micro_batch_size": int(experiment_cfg["micro_batch_size"]),
        "learning_rate": float(experiment_cfg["learning_rate"]),
        "weight_decay": float(experiment_cfg["weight_decay"]),
        "gradient_checkpointing": bool(experiment_cfg["gradient_checkpointing"]),
        "dataset_name": str(data_cfg["dataset_name"]),
        "dataset_config_name": str(data_cfg["dataset_config_name"]),
        "dataset_split": str(data_cfg["split"]),
        "dataset_cache_dir": str(data_cfg["cache_dir"]),
    }


def _validate_pair(pair_desc: Dict[str, object], requested_bridge_type: str) -> None:
    """Validate that the selected GPU pair matches hardware and bridge constraints."""

    gpu_names = list(pair_desc["gpu_names"])
    if not all("A6000" in str(name) for name in gpu_names):
        raise ValueError(f"Selected GPU pair is not fixed A6000: {gpu_names}")

    if requested_bridge_type != "auto" and pair_desc["bridge_type"] != requested_bridge_type:
        raise ValueError(
            "Requested bridge type does not match actual topology: "
            f"requested={requested_bridge_type}, detected={pair_desc['bridge_type']}, "
            f"raw_label={pair_desc['raw_topology_label']}"
        )


def _check_initial_exclusivity(run_dir: Path, gpu_pair: Sequence[int]) -> None:
    """Record and enforce GPU-pair exclusivity before launching workers."""

    processes = list_processes_for_gpu_pair(gpu_pair)
    append_jsonl(
        run_dir / "raw" / "exclusivity_events.jsonl",
        {
            "timestamp": time.time(),
            "phase": "startup",
            "stage": "before_preflight",
            "step_idx": -1,
            "gpu_pair": list(gpu_pair),
            "processes": [
                {
                    "gpu_uuid": process.gpu_uuid,
                    "pid": int(process.pid),
                    "process_name": process.process_name,
                    "used_memory_mb": int(process.used_memory_mb),
                    "is_allowed": False,
                }
                for process in processes
            ],
        },
    )
    if processes:
        pretty = ", ".join(
            f"pid={process.pid} name={process.process_name} mem={process.used_memory_mb}MB"
            for process in processes
        )
        raise RuntimeError(f"GPU pair {list(gpu_pair)} is not exclusive before launch: {pretty}")


def _worker_base_args(
    *,
    run_dir: Path,
    run_mode: str,
    model_name: str,
    model_path: str,
    token_cache_path: Path | None,
    metadata: Dict[str, object],
) -> List[str]:
    """Build the common `worker.py` argument list."""

    args = [
        str(CURRENT_DIR / "worker.py"),
        "--run-dir",
        str(run_dir),
        "--run-mode",
        run_mode,
        "--model-name",
        model_name,
        "--model-path",
        model_path,
        "--seq-len",
        str(metadata["seq_len"]),
        "--warmup-steps",
        str(metadata["warmup_steps"]),
        "--measured-steps",
        str(metadata["measured_steps"]),
        "--profile-steps",
        str(metadata["profile_steps"]),
        "--micro-batch-size",
        str(metadata["micro_batch_size"]),
        "--learning-rate",
        str(metadata["learning_rate"]),
        "--weight-decay",
        str(metadata["weight_decay"]),
        "--gradient-checkpointing",
        "true" if metadata["gradient_checkpointing"] else "false",
        "--physical-gpu-pair",
        ",".join(str(item) for item in metadata["gpu_pair"]),
        "--requested-bridge-type",
        str(metadata["requested_bridge_type"]),
        "--detected-bridge-type",
        str(metadata["detected_bridge_type"]),
        "--detected-topology-label",
        str(metadata["detected_topology_label"]),
    ]
    if token_cache_path is not None:
        args.extend(["--token-cache-path", str(token_cache_path)])
    return args


def _launch_torchrun(
    *,
    worker_args: List[str],
    visible_gpu_pair: Sequence[int],
    log_path: Path,
) -> int:
    """Launch `torchrun` through the current Python executable."""

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=2",
        *worker_args,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in visible_gpu_pair)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["OMP_NUM_THREADS"] = "1"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("# Command\n")
        handle.write(shlex.join(command))
        handle.write("\n\n# Output\n")
        handle.flush()
        process = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=str(REPO_ROOT),
            check=False,
        )
    return int(process.returncode)


def _run_preflight(
    *,
    run_dir: Path,
    metadata: Dict[str, object],
    model_name: str,
    model_path: str,
) -> Dict[str, object]:
    """Run a lightweight preflight for one candidate model."""

    preflight_dir = ensure_directory(run_dir / "preflight" / model_name)
    preflight_metadata = {
        **metadata,
        "status": "preflight_preparing",
        "selected_model": model_name,
        "selected_model_path": model_path,
    }
    write_json(preflight_dir / "run_metadata.json", preflight_metadata)

    return_code = _launch_torchrun(
        worker_args=_worker_base_args(
            run_dir=preflight_dir,
            run_mode="preflight",
            model_name=model_name,
            model_path=model_path,
            token_cache_path=None,
            metadata=metadata,
        ),
        visible_gpu_pair=metadata["gpu_pair"],
        log_path=preflight_dir / "preflight.log",
    )
    result_path = preflight_dir / "preflight_result.json"
    result = load_json(result_path) if result_path.exists() else {"status": "failed", "error_type": "MissingResult"}
    result["return_code"] = int(return_code)
    result["preflight_dir"] = str(preflight_dir)
    return result


def _prepare_token_cache_for_model(
    *,
    config: Dict[str, object],
    model_path: str,
    total_steps: int,
) -> Path:
    """Prepare the token cache required by one model run."""

    experiment_cfg = dict(config.get("experiment", {}))
    data_cfg = dict(config.get("data", {}))
    return prepare_wikitext_token_cache(
        model_path=model_path,
        cache_dir=Path(str(data_cfg["cache_dir"])),
        seq_len=int(experiment_cfg["seq_len"]),
        split=str(data_cfg["split"]),
        min_blocks=max(int(total_steps) + 8, 64),
    )


def _write_latest_manifest(run_dir: Path, figure_paths: Dict[str, Path], report_paths: Dict[str, Path]) -> Path:
    """Write the stable latest manifest for downstream inspection."""

    LATEST_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_dir": str(run_dir),
        "run_summary_json": str(run_dir / "summary" / "run_summary.json"),
        "iteration_csv": str(run_dir / "summary" / "iteration_metrics.csv"),
        "phase_csv": str(run_dir / "summary" / "phase_summary.csv"),
        "profiler_csv": str(run_dir / "summary" / "profiler_summary.csv"),
        "report_md": str(report_paths["report_md"]),
        "latest_report_md": str(report_paths["latest_report_md"]),
        "figure_paths": {key: str(path) for key, path in figure_paths.items()},
    }
    manifest_path = LATEST_ROOT / "latest_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def main() -> int:
    """Run the full tensor-parallel sidecar workflow."""

    args = parse_args()
    config = _load_config(args.config)

    experiment_cfg = dict(config.get("experiment", {}))
    if args.gpu_pair:
        experiment_cfg["gpu_pair"] = _parse_gpu_pair(args.gpu_pair)
    if args.bridge_type:
        experiment_cfg["bridge_type"] = args.bridge_type
    if args.model_choice:
        experiment_cfg["model_choice"] = args.model_choice
    if args.warmup_steps is not None:
        experiment_cfg["warmup_steps"] = int(args.warmup_steps)
    if args.measured_steps is not None:
        experiment_cfg["measured_steps"] = int(args.measured_steps)
    if args.profile_steps is not None:
        experiment_cfg["profile_steps"] = int(args.profile_steps)
    config["experiment"] = experiment_cfg

    gpu_pair = list(experiment_cfg["gpu_pair"])
    requested_bridge_type = str(experiment_cfg["bridge_type"])
    model_choice = str(experiment_cfg["model_choice"])
    pair_desc = describe_gpu_pair(gpu_pair)
    _validate_pair(pair_desc, requested_bridge_type)

    run_tag = _build_run_tag(
        gpu_pair=gpu_pair,
        detected_bridge_type=str(pair_desc["bridge_type"]),
        model_choice=model_choice,
        manual_run_tag=args.run_tag,
    )
    output_root = Path(str(experiment_cfg.get("output_dir", ARTIFACT_ROOT)))
    run_dir = output_root / run_tag
    _prepare_run_dirs(run_dir)

    metadata = _initial_metadata(
        args=args,
        config=config,
        run_tag=run_tag,
        run_dir=run_dir,
        pair_desc=pair_desc,
        requested_bridge_type=requested_bridge_type,
        model_choice=model_choice,
    )
    write_json(run_dir / "run_metadata.json", metadata)

    try:
        _check_initial_exclusivity(run_dir, gpu_pair)
    except Exception as exc:  # noqa: BLE001
        metadata.update(
            {
                "status": "blocked_external_process",
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
            }
        )
        write_json(run_dir / "run_metadata.json", metadata)
        raise

    model_attempts: List[Dict[str, object]] = []
    selected_model_name = None
    selected_model_path = None
    for model_name in _candidate_models(model_choice):
        model_path = MODEL_REGISTRY[model_name]
        attempt = _run_preflight(
            run_dir=run_dir,
            metadata=metadata,
            model_name=model_name,
            model_path=model_path,
        )
        model_attempts.append(attempt)
        metadata["model_attempts"] = model_attempts
        write_json(run_dir / "run_metadata.json", metadata)
        if attempt.get("status") == "passed" and int(attempt.get("return_code", 1)) == 0:
            selected_model_name = model_name
            selected_model_path = model_path
            break

    if selected_model_name is None or selected_model_path is None:
        metadata.update(
            {
                "status": "model_selection_failed",
                "error_type": "PreflightFailed",
                "error_message": "All model candidates failed preflight.",
            }
        )
        write_json(run_dir / "run_metadata.json", metadata)
        return 4

    metadata.update(
        {
            "status": "preflight_passed",
            "selected_model": selected_model_name,
            "selected_model_path": selected_model_path,
        }
    )
    write_json(run_dir / "run_metadata.json", metadata)

    total_steps = int(metadata["warmup_steps"]) + int(metadata["measured_steps"]) + int(metadata["profile_steps"])
    token_cache_path = _prepare_token_cache_for_model(
        config=config,
        model_path=selected_model_path,
        total_steps=total_steps,
    )
    metadata["token_cache_path"] = str(token_cache_path)
    metadata["status"] = "token_cache_ready"
    write_json(run_dir / "run_metadata.json", metadata)

    worker_return_code = _launch_torchrun(
        worker_args=_worker_base_args(
            run_dir=run_dir,
            run_mode="full",
            model_name=selected_model_name,
            model_path=selected_model_path,
            token_cache_path=token_cache_path,
            metadata=metadata,
        ),
        visible_gpu_pair=gpu_pair,
        log_path=run_dir / "logs" / "worker_full.log",
    )

    final_metadata = load_json(run_dir / "run_metadata.json")
    final_metadata["worker_return_code"] = int(worker_return_code)
    if worker_return_code == 0:
        final_metadata["status"] = "completed"
    elif final_metadata.get("status") == "worker_completed":
        final_metadata["status"] = "completed"
    else:
        final_metadata["status"] = str(final_metadata.get("status", "failed"))
    write_json(run_dir / "run_metadata.json", final_metadata)

    figure_paths: Dict[str, Path] = {}
    report_paths: Dict[str, Path] = {}
    step_metrics_path = run_dir / "raw" / "step_metrics.jsonl"
    if step_metrics_path.exists() and step_metrics_path.stat().st_size > 0:
        build_summary_artifacts(run_dir)
        try:
            from evaluation.llm_tp_bandwidth.plot import build_all_figures

            figure_paths = build_all_figures(run_dir)
        except ModuleNotFoundError as exc:
            final_metadata["plot_skipped_reason"] = str(exc)
            write_json(run_dir / "run_metadata.json", final_metadata)
        report_paths = build_report(run_dir, figure_paths)
        _write_latest_manifest(run_dir, figure_paths, report_paths)
    elif worker_return_code != 0:
        final_metadata["status"] = f"no_steps_{final_metadata.get('status', 'failed')}"
        write_json(run_dir / "run_metadata.json", final_metadata)
    return int(worker_return_code)


if __name__ == "__main__":
    raise SystemExit(main())
