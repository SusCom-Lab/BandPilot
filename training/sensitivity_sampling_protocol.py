"""Shared sampling-protocol helpers for sensitivity experiments.

These helpers implement the nested/cumulative protocol used by predictor-level
and dispatch-level sensitivity studies. A mother pool of size
`max(sample_sizes)` is generated per `(cluster, strategy, seed)`, and each
smaller sample budget is a deterministic prefix or quota-preserving subset of
that pool. CSV and JSON manifests record membership for reviewer traceability.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from utils.helpers import ensure_directory

NUM_STRATIFIED_BINS = 8


def _set_sampling_seed(seed: int) -> None:
    """Reset Python / NumPy RNG so mother-pool generation is reproducible."""

    random.seed(seed)
    np.random.seed(seed)


def _build_stratified_bin_quotas(num_samples: int, num_bins: int = NUM_STRATIFIED_BINS) -> List[int]:
    """Return the per-bin quota rule used by the stratified protocol.

    Remainders are distributed round-robin to the first few bins so that the
    quota rule matches the existing sensitivity study wording.
    """

    per_bin = num_samples // num_bins
    remainder = num_samples % num_bins
    return [per_bin + (1 if index < remainder else 0) for index in range(num_bins)]


def _density_from_config(config: np.ndarray, total_gpu: int) -> float:
    """Compute active-GPU density for a binary GPU config."""

    return float(np.sum(config)) / float(total_gpu)


def _density_bin_from_ratio(density: float, num_bins: int = NUM_STRATIFIED_BINS) -> int:
    """Map an actual density ratio to the reviewer-facing stratified density bins."""

    edges = np.linspace(0.1, 0.9, num_bins + 1)
    clamped_density = min(max(float(density), float(edges[0])), float(edges[-1]))
    index = int(np.searchsorted(edges, clamped_density, side="right") - 1)
    return max(0, min(num_bins - 1, index))


def _active_gpu_indices(config: np.ndarray) -> str:
    """Serialize active GPU indices to a compact manifest string."""

    active = np.flatnonzero(config).tolist()
    return ",".join(str(index) for index in active)


def _deserialize_gpu_config(serialized: str, total_gpu: int) -> np.ndarray:
    """Restore a binary GPU config from the manifest's index string."""

    config = np.zeros(total_gpu, dtype=int)
    if not serialized:
        return config
    for token in serialized.split(","):
        token = token.strip()
        if not token:
            continue
        config[int(token)] = 1
    return config


def _gen_stratified_mother_pool_with_metadata(
    *,
    num_samples: int,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config,
    training_data_path: str,
    compute_bandwidths_fn: Callable[[Sequence[Sequence[int]], int, object, object, str], List[float]],
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    """Generate a stratified mother pool while preserving explicit bin metadata.

    This function mirrors `data_process.dataset.get_stratified_train_dataset`, but
    additionally records each sample's intended density bin and whether it came
    from the balanced or random half. The explicit metadata is necessary because
    actual active-GPU density is quantized by integer GPU counts, so inferring the
    intended bin purely from the final config can drift at bin boundaries.
    """

    gpu_configs: List[np.ndarray] = []
    seen: set = set()
    metadata: List[Dict[str, object]] = []
    num_parts = total_gpu // 8
    bin_edges = np.linspace(0.1, 0.9, NUM_STRATIFIED_BINS + 1)
    bin_quotas = _build_stratified_bin_quotas(num_samples)

    for bin_idx in range(NUM_STRATIFIED_BINS):
        low, high = float(bin_edges[bin_idx]), float(bin_edges[bin_idx + 1])
        quota = int(bin_quotas[bin_idx])
        balanced_count = quota // 2 if num_parts >= 2 else 0
        random_count = quota - balanced_count

        # Balanced half: preserve the original cross-node balanced-allocation bias.
        for _ in range(balanced_count):
            attempts = 0
            while attempts < 500:
                attempts += 1
                density = float(np.random.uniform(low, high))
                gpus_to_allocate = max(2, int(total_gpu * density))
                nodes_to_use = random.randint(2, num_parts) if num_parts >= 2 else 1
                base_alloc = gpus_to_allocate // nodes_to_use
                remainder = gpus_to_allocate % nodes_to_use
                if base_alloc + (1 if remainder else 0) > 8:
                    continue
                allocation_plan = [base_alloc + 1] * remainder + [base_alloc] * (nodes_to_use - remainder)
                chosen_nodes = random.sample(range(num_parts), nodes_to_use)
                config = np.zeros(total_gpu, dtype=int)
                for node_pos, node_id in enumerate(chosen_nodes):
                    num_to_take = allocation_plan[node_pos]
                    node_indices = range(node_id * 8, (node_id + 1) * 8)
                    selected = random.sample(list(node_indices), num_to_take)
                    config[selected] = 1
                key = tuple(config.tolist())
                if key in seen:
                    continue
                seen.add(key)
                gpu_configs.append(config)
                metadata.append(
                    {
                        "density_bin": int(bin_idx),
                        "sample_source": "balanced",
                        "draw_density": density,
                        "active_gpu_count": int(np.sum(config)),
                    }
                )
                break

        # Random half: preserve the original cross-node random coverage.
        for _ in range(random_count):
            attempts = 0
            while attempts < 500:
                attempts += 1
                density = float(np.random.uniform(low, high))
                gpus_to_allocate = max(2, int(total_gpu * density))
                config = np.zeros(total_gpu, dtype=int)
                active_indices = np.random.choice(total_gpu, gpus_to_allocate, replace=False)
                config[active_indices] = 1
                if num_parts >= 2:
                    parts = [int(np.sum(config[i * 8 : (i + 1) * 8])) for i in range(num_parts)]
                    if sum(1 for part in parts if part > 0) < 2:
                        continue
                key = tuple(config.tolist())
                if key in seen:
                    continue
                seen.add(key)
                gpu_configs.append(config)
                metadata.append(
                    {
                        "density_bin": int(bin_idx),
                        "sample_source": "random",
                        "draw_density": density,
                        "active_gpu_count": int(np.sum(config)),
                    }
                )
                break

    bandwidths = compute_bandwidths_fn(
        gpu_configs,
        total_gpu,
        gpu_bw_dict_list,
        switch_config,
        training_data_path,
    )
    return np.array(gpu_configs), np.array(bandwidths), metadata


def _generate_unique_extension_pool(
    *,
    generator_fn: Callable[..., Tuple[np.ndarray, np.ndarray]],
    target_new_count: int,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config,
    training_data_path: str,
    existing_configs: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """Extend non-stratified mother pools while preserving uniqueness.

    For cumulative budgets such as `100/250/500`, grow one mother pool while
    keeping all configs unique. Random and worst-case strategies do not require
    per-bin quotas, so this helper only enforces uniqueness.
    """

    _set_sampling_seed(seed + 1000003)
    seen = {tuple(int(value) for value in config.tolist()) for config in existing_configs}
    collected: List[np.ndarray] = []
    metadata: List[Dict[str, object]] = []
    batch_size = max(64, target_new_count)
    max_rounds = 64

    for _ in range(max_rounds):
        if len(collected) >= target_new_count:
            break
        batch_configs, _ = generator_fn(
            num_samples=batch_size,
            total_gpu=total_gpu,
            gpu_bw_dict_list=gpu_bw_dict_list,
            switch_config=switch_config,
            training_data_path=training_data_path,
        )
        for config in batch_configs:
            key = tuple(int(value) for value in config.tolist())
            if key in seen:
                continue
            seen.add(key)
            config_np = np.array(config, dtype=int)
            collected.append(config_np)
            metadata.append(
                {
                    "density_bin": _density_bin_from_ratio(_density_from_config(config_np, total_gpu)),
                    "sample_source": "extended_mother_pool",
                }
            )
            if len(collected) >= target_new_count:
                break
        batch_size = max(32, target_new_count - len(collected))

    if len(collected) != target_new_count:
        raise ValueError(
            f"Failed to extend mother pool: need {target_new_count} new configs, "
            f"but only generated {len(collected)} unique configs."
        )

    return np.array(collected, dtype=int), metadata


def _extend_stratified_mother_pool_with_metadata(
    *,
    existing_configs: np.ndarray,
    existing_metadata: Sequence[Mapping[str, object]],
    target_mother_pool_size: int,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config,
    training_data_path: str,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """Extend a stratified mother pool to a larger total size without moving old samples.

          `100/250/500`       ,     `1000`       tail.
            `1000`   bin quota      density bin    ,   deficit,
               `1000`   .
    """

    num_parts = total_gpu // 8
    bin_edges = np.linspace(0.1, 0.9, NUM_STRATIFIED_BINS + 1)
    seen = {tuple(int(value) for value in config.tolist()) for config in existing_configs}

    current_balanced = {bin_idx: 0 for bin_idx in range(NUM_STRATIFIED_BINS)}
    current_random = {bin_idx: 0 for bin_idx in range(NUM_STRATIFIED_BINS)}
    for row in existing_metadata:
        bin_idx = int(row["density_bin"])
        sample_source = str(row.get("sample_source", "random"))
        if sample_source == "balanced":
            current_balanced[bin_idx] += 1
        else:
            current_random[bin_idx] += 1

    gpu_configs: List[np.ndarray] = []
    metadata: List[Dict[str, object]] = []

    target_quotas = _build_stratified_bin_quotas(target_mother_pool_size)
    for bin_idx, quota in enumerate(target_quotas):
        low, high = float(bin_edges[bin_idx]), float(bin_edges[bin_idx + 1])
        target_balanced = quota // 2 if num_parts >= 2 else 0
        target_random = quota - target_balanced
        need_balanced = max(0, target_balanced - current_balanced[bin_idx])
        need_random = max(0, target_random - current_random[bin_idx])

        for _ in range(need_balanced):
            attempts = 0
            while attempts < 1000:
                attempts += 1
                density = float(np.random.uniform(low, high))
                gpus_to_allocate = max(2, int(total_gpu * density))
                nodes_to_use = random.randint(2, num_parts) if num_parts >= 2 else 1
                base_alloc = gpus_to_allocate // nodes_to_use
                remainder = gpus_to_allocate % nodes_to_use
                if base_alloc + (1 if remainder else 0) > 8:
                    continue
                allocation_plan = [base_alloc + 1] * remainder + [base_alloc] * (nodes_to_use - remainder)
                chosen_nodes = random.sample(range(num_parts), nodes_to_use)
                config = np.zeros(total_gpu, dtype=int)
                for node_pos, node_id in enumerate(chosen_nodes):
                    num_to_take = allocation_plan[node_pos]
                    node_indices = range(node_id * 8, (node_id + 1) * 8)
                    selected = random.sample(list(node_indices), num_to_take)
                    config[selected] = 1
                key = tuple(int(value) for value in config.tolist())
                if key in seen:
                    continue
                seen.add(key)
                gpu_configs.append(config)
                metadata.append(
                    {
                        "density_bin": int(bin_idx),
                        "sample_source": "balanced",
                    }
                )
                break
            else:
                raise ValueError(f"Failed to extend stratified balanced bin={bin_idx}.")

        for _ in range(need_random):
            attempts = 0
            while attempts < 1000:
                attempts += 1
                density = float(np.random.uniform(low, high))
                gpus_to_allocate = max(2, int(total_gpu * density))
                config = np.zeros(total_gpu, dtype=int)
                active_indices = np.random.choice(total_gpu, gpus_to_allocate, replace=False)
                config[active_indices] = 1
                if num_parts >= 2:
                    parts = [int(np.sum(config[i * 8 : (i + 1) * 8])) for i in range(num_parts)]
                    if sum(1 for part in parts if part > 0) < 2:
                        continue
                key = tuple(int(value) for value in config.tolist())
                if key in seen:
                    continue
                seen.add(key)
                gpu_configs.append(config)
                metadata.append(
                    {
                        "density_bin": int(bin_idx),
                        "sample_source": "random",
                    }
                )
                break
            else:
                raise ValueError(f"Failed to extend stratified random bin={bin_idx}.")

    expected_new_count = target_mother_pool_size - int(len(existing_configs))
    if len(gpu_configs) != expected_new_count:
        raise ValueError(
            f"Stratified extension count mismatch: expected {expected_new_count}, got {len(gpu_configs)}."
        )

    return np.array(gpu_configs, dtype=int), metadata


def _load_existing_nested_manifest(
    *,
    existing_manifest_root: Path,
    cluster_type: str,
    strategy_name: str,
    seed: int,
    total_gpu: int,
) -> Tuple[np.ndarray, List[Dict[str, object]], Dict[int, np.ndarray], int]:
    """Load an existing nested manifest and reconstruct the old mother pool."""

    strategy_slug = strategy_name.lower().replace(" ", "-")
    manifest_path = existing_manifest_root / cluster_type / strategy_slug / f"seed{seed}_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing nested manifest: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    if not rows:
        raise ValueError(f"Nested manifest is empty: {manifest_path}")

    mother_pool_size = int(rows[0]["mother_pool_size"])
    configs = np.array(
        [_deserialize_gpu_config(row["selected_gpu_indices"], total_gpu) for row in rows],
        dtype=int,
    )
    if len(configs) != mother_pool_size:
        raise ValueError(
            f"Manifest mother pool size mismatch: summary={mother_pool_size}, rows={len(configs)}"
        )

    sample_sizes = sorted(
        int(column[len("included_n") :])
        for column in rows[0].keys()
        if column.startswith("included_n")
    )
    selected_indices_by_size = {
        sample_size: np.array(
            [row_index for row_index, row in enumerate(rows) if int(row[f"included_n{sample_size}"]) == 1],
            dtype=int,
        )
        for sample_size in sample_sizes
    }
    metadata = [
        {
            "density_bin": int(row["density_bin"]),
            "sample_source": str(row.get("sample_source", "mother_pool")),
        }
        for row in rows
    ]
    return configs, metadata, selected_indices_by_size, mother_pool_size


def _build_prefix_index_map(
    *,
    mother_pool_size: int,
    sample_sizes: Sequence[int],
    seed: int,
) -> Dict[int, np.ndarray]:
    """Build nested prefix subsets for non-stratified strategies."""

    ordered_sizes = sorted({int(sample_size) for sample_size in sample_sizes})
    rng = np.random.default_rng(seed + 1701)
    permutation = rng.permutation(mother_pool_size)
    return {
        sample_size: np.sort(permutation[:sample_size].copy())
        for sample_size in ordered_sizes
    }


def _build_stratified_index_map(
    *,
    mother_metadata: Sequence[Mapping[str, object]],
    sample_sizes: Sequence[int],
    seed: int,
) -> Dict[int, np.ndarray]:
    """Build nested, quota-preserving subsets for the stratified protocol."""

    ordered_sizes = sorted({int(sample_size) for sample_size in sample_sizes})
    indices_by_bin: MutableMapping[int, List[int]] = {bin_idx: [] for bin_idx in range(NUM_STRATIFIED_BINS)}
    for mother_index, row in enumerate(mother_metadata):
        bin_idx = int(row["density_bin"])
        indices_by_bin.setdefault(bin_idx, []).append(mother_index)

    # Use a fixed within-bin ordering so that smaller budgets are strict prefixes
    # of larger budgets inside each density bin.
    ordered_indices_by_bin: Dict[int, List[int]] = {}
    for bin_idx, mother_indices in indices_by_bin.items():
        rng = np.random.default_rng(seed + 9109 + bin_idx)
        ordered_indices = list(mother_indices)
        rng.shuffle(ordered_indices)
        ordered_indices_by_bin[bin_idx] = ordered_indices

    index_map: Dict[int, np.ndarray] = {}
    previous_selection: set[int] = set()
    for sample_size in ordered_sizes:
        quotas = _build_stratified_bin_quotas(sample_size)
        selected_indices: List[int] = []
        for bin_idx, quota in enumerate(quotas):
            candidates = ordered_indices_by_bin.get(bin_idx, [])
            if len(candidates) < quota:
                raise ValueError(
                    f"Stratified mother pool is insufficient for bin={bin_idx}: "
                    f"have {len(candidates)}, need {quota}"
                )
            selected_indices.extend(candidates[:quota])
        selection = set(selected_indices)
        if previous_selection and not previous_selection.issubset(selection):
            raise ValueError("Nested stratified subsets violate inclusion ordering.")
        previous_selection = selection
        index_map[int(sample_size)] = np.array(sorted(selection), dtype=int)
    return index_map


def _write_nested_manifest(
    *,
    manifest_root: Path,
    cluster_type: str,
    strategy_name: str,
    seed: int,
    mother_configs: np.ndarray,
    mother_metadata: Sequence[Mapping[str, object]],
    sample_sizes: Sequence[int],
    selected_indices_by_size: Mapping[int, np.ndarray],
    protocol_name: str,
    mother_pool_size: int,
    total_gpu: int,
) -> None:
    """Persist CSV + JSON summaries for reviewer-traceable nested subsets."""

    strategy_slug = strategy_name.lower().replace(" ", "-")
    target_dir = manifest_root / cluster_type / strategy_slug
    ensure_directory(target_dir)

    ordered_sizes = sorted({int(sample_size) for sample_size in sample_sizes})
    inclusion_sets = {
        sample_size: set(int(index) for index in indices.tolist())
        for sample_size, indices in selected_indices_by_size.items()
    }

    rows: List[Dict[str, object]] = []
    for mother_index, config in enumerate(mother_configs):
        metadata = dict(mother_metadata[mother_index])
        density = _density_from_config(config, total_gpu)
        row: Dict[str, object] = {
            "cluster_type": cluster_type,
            "strategy": strategy_name,
            "seed": int(seed),
            "sampling_protocol": protocol_name,
            "mother_pool_size": int(mother_pool_size),
            "mother_index": int(mother_index),
            "active_gpu_count": int(np.sum(config)),
            "density_ratio": density,
            "density_bin": int(metadata.get("density_bin", _density_bin_from_ratio(density))),
            "sample_source": str(metadata.get("sample_source", "mother_pool")),
            "selected_gpu_indices": _active_gpu_indices(config),
        }
        for sample_size in ordered_sizes:
            row[f"included_n{sample_size}"] = int(mother_index in inclusion_sets[sample_size])
        rows.append(row)

    csv_path = target_dir / f"seed{seed}_manifest.csv"
    import csv

    with csv_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_payload = {
        "cluster_type": cluster_type,
        "strategy": strategy_name,
        "seed": int(seed),
        "sampling_protocol": protocol_name,
        "mother_pool_size": int(mother_pool_size),
        "sample_sizes": ordered_sizes,
        "bin_counts": {
            str(bin_idx): int(sum(1 for row in rows if int(row["density_bin"]) == bin_idx))
            for bin_idx in range(NUM_STRATIFIED_BINS)
        },
        "subset_sizes": {
            str(sample_size): int(len(selected_indices_by_size[sample_size]))
            for sample_size in ordered_sizes
        },
    }
    summary_path = target_dir / f"seed{seed}_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_nested_dataset_family(
    *,
    cluster_type: str,
    strategy_name: str,
    sample_sizes: Sequence[int],
    seed: int,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config,
    training_data_path: str,
    generator_fn: Callable[..., Tuple[np.ndarray, np.ndarray]],
    compute_bandwidths_fn: Callable[[Sequence[Sequence[int]], int, object, object, str], List[float]],
    manifest_root: Path | None = None,
    mother_pool_size: int | None = None,
    protocol_name: str = "nested",
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Generate a cumulative training-set family for one `(cluster, strategy, seed)`.

    Returns:
        A dict mapping each requested sample size to `(gpu_configs, bandwidths)`.
    """

    ordered_sizes = sorted({int(sample_size) for sample_size in sample_sizes})
    if not ordered_sizes:
        raise ValueError("sample_sizes must be non-empty for nested protocol.")

    resolved_mother_pool_size = int(mother_pool_size or ordered_sizes[-1])
    if resolved_mother_pool_size < ordered_sizes[-1]:
        raise ValueError(
            f"mother_pool_size={resolved_mother_pool_size} is smaller than "
            f"max(sample_sizes)={ordered_sizes[-1]}"
        )

    _set_sampling_seed(seed)
    if strategy_name == "Stratified":
        mother_configs, mother_bandwidths, mother_metadata = _gen_stratified_mother_pool_with_metadata(
            num_samples=resolved_mother_pool_size,
            total_gpu=total_gpu,
            gpu_bw_dict_list=gpu_bw_dict_list,
            switch_config=switch_config,
            training_data_path=training_data_path,
            compute_bandwidths_fn=compute_bandwidths_fn,
        )
        selected_indices_by_size = _build_stratified_index_map(
            mother_metadata=mother_metadata,
            sample_sizes=ordered_sizes,
            seed=seed,
        )
    else:
        mother_configs, mother_bandwidths = generator_fn(
            num_samples=resolved_mother_pool_size,
            total_gpu=total_gpu,
            gpu_bw_dict_list=gpu_bw_dict_list,
            switch_config=switch_config,
            training_data_path=training_data_path,
        )
        mother_metadata = [
            {
                "density_bin": _density_bin_from_ratio(_density_from_config(config, total_gpu)),
                "sample_source": "mother_pool",
            }
            for config in mother_configs
        ]
        selected_indices_by_size = _build_prefix_index_map(
            mother_pool_size=resolved_mother_pool_size,
            sample_sizes=ordered_sizes,
            seed=seed,
        )

    family = {
        sample_size: (
            mother_configs[selected_indices].copy(),
            mother_bandwidths[selected_indices].copy(),
        )
        for sample_size, selected_indices in selected_indices_by_size.items()
    }

    if manifest_root is not None:
        _write_nested_manifest(
            manifest_root=manifest_root,
            cluster_type=cluster_type,
            strategy_name=strategy_name,
            seed=seed,
            mother_configs=mother_configs,
            mother_metadata=mother_metadata,
            sample_sizes=ordered_sizes,
            selected_indices_by_size=selected_indices_by_size,
            protocol_name=protocol_name,
            mother_pool_size=resolved_mother_pool_size,
            total_gpu=total_gpu,
        )

    return family


def extend_nested_dataset_family_from_manifest(
    *,
    cluster_type: str,
    strategy_name: str,
    sample_sizes: Sequence[int],
    seed: int,
    total_gpu: int,
    gpu_bw_dict_list,
    switch_config,
    training_data_path: str,
    generator_fn: Callable[..., Tuple[np.ndarray, np.ndarray]],
    compute_bandwidths_fn: Callable[[Sequence[Sequence[int]], int, object, object, str], List[float]],
    existing_manifest_root: Path,
    manifest_root: Path | None = None,
    mother_pool_size: int | None = None,
    protocol_name: str = "nested",
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Extend an existing nested family while preserving old subsets exactly.

    This is used when an old artifact already contains smaller nested subsets
    such as `100/250/500`, and a larger budget such as `1000` must be added
    without changing any previously published subset membership.
    """

    ordered_sizes = sorted({int(sample_size) for sample_size in sample_sizes})
    if not ordered_sizes:
        raise ValueError("sample_sizes must be non-empty for nested extension.")

    existing_configs, existing_metadata, selected_indices_by_size, old_mother_pool_size = _load_existing_nested_manifest(
        existing_manifest_root=existing_manifest_root,
        cluster_type=cluster_type,
        strategy_name=strategy_name,
        seed=seed,
        total_gpu=total_gpu,
    )

    resolved_mother_pool_size = int(mother_pool_size or ordered_sizes[-1])
    if resolved_mother_pool_size < old_mother_pool_size:
        raise ValueError(
            f"mother_pool_size={resolved_mother_pool_size} is smaller than existing mother pool "
            f"{old_mother_pool_size}."
        )

    new_sizes = [size for size in ordered_sizes if size > old_mother_pool_size]
    if len(new_sizes) > 1 or (new_sizes and new_sizes[0] != resolved_mother_pool_size):
        raise ValueError(
            "Nested extension currently supports at most one new sample size, "
            "and it must equal the final mother_pool_size."
        )

    missing_old_sizes = [size for size in ordered_sizes if size <= old_mother_pool_size and size not in selected_indices_by_size]
    if missing_old_sizes:
        raise ValueError(
            f"Requested old sample sizes are missing from the existing manifest: {missing_old_sizes}"
        )

    if resolved_mother_pool_size == old_mother_pool_size:
        combined_configs = existing_configs.copy()
        combined_metadata = list(existing_metadata)
    else:
        additional_needed = resolved_mother_pool_size - old_mother_pool_size
        if strategy_name == "Stratified":
            _set_sampling_seed(seed + 2000003)
            new_configs, new_metadata = _extend_stratified_mother_pool_with_metadata(
                existing_configs=existing_configs,
                existing_metadata=existing_metadata,
                target_mother_pool_size=resolved_mother_pool_size,
                total_gpu=total_gpu,
                gpu_bw_dict_list=gpu_bw_dict_list,
                switch_config=switch_config,
                training_data_path=training_data_path,
            )
        else:
            new_configs, new_metadata = _generate_unique_extension_pool(
                generator_fn=generator_fn,
                target_new_count=additional_needed,
                total_gpu=total_gpu,
                gpu_bw_dict_list=gpu_bw_dict_list,
                switch_config=switch_config,
                training_data_path=training_data_path,
                existing_configs=existing_configs,
                seed=seed,
            )

        combined_configs = np.concatenate([existing_configs, new_configs], axis=0)
        combined_metadata = list(existing_metadata) + list(new_metadata)
        if len(combined_configs) != resolved_mother_pool_size:
            raise ValueError(
                f"Extended mother pool has unexpected size: {len(combined_configs)} "
                f"(expected {resolved_mother_pool_size})."
            )

    combined_bandwidths = np.array(
        compute_bandwidths_fn(
            combined_configs,
            total_gpu,
            gpu_bw_dict_list,
            switch_config,
            training_data_path,
        )
    )

    family: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for sample_size in ordered_sizes:
        if sample_size in selected_indices_by_size:
            indices = selected_indices_by_size[sample_size]
        elif sample_size == resolved_mother_pool_size:
            indices = np.arange(resolved_mother_pool_size, dtype=int)
            selected_indices_by_size[sample_size] = indices
        else:
            raise ValueError(
                f"Unsupported new sample size {sample_size}. "
                f"Only the final extended mother_pool_size={resolved_mother_pool_size} "
                "can be added incrementally."
            )
        family[sample_size] = (
            combined_configs[indices].copy(),
            combined_bandwidths[indices].copy(),
        )

    if manifest_root is not None:
        _write_nested_manifest(
            manifest_root=manifest_root,
            cluster_type=cluster_type,
            strategy_name=strategy_name,
            seed=seed,
            mother_configs=combined_configs,
            mother_metadata=combined_metadata,
            sample_sizes=ordered_sizes,
            selected_indices_by_size=selected_indices_by_size,
            protocol_name=protocol_name,
            mother_pool_size=resolved_mother_pool_size,
            total_gpu=total_gpu,
        )

    return family


__all__ = [
    "NUM_STRATIFIED_BINS",
    "build_nested_dataset_family",
    "extend_nested_dataset_family_from_manifest",
]
