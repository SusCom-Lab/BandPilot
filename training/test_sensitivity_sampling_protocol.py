"""Unit tests for nested/cumulative sensitivity sampling."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.sensitivity_sampling_protocol import (
    _build_prefix_index_map,
    _build_stratified_bin_quotas,
    _build_stratified_index_map,
    build_nested_dataset_family,
    extend_nested_dataset_family_from_manifest,
)


def _dummy_generator(num_samples, total_gpu, gpu_bw_dict_list, switch_config, training_data_path):
    """Generate deterministic dummy configs without touching CSV loaders."""

    configs = np.zeros((num_samples, total_gpu), dtype=int)
    for row_index in range(num_samples):
        active_count = 2 + (row_index % max(1, total_gpu - 2))
        offset = (row_index * 3) % total_gpu
        active_indices = [(offset + step) % total_gpu for step in range(active_count)]
        configs[row_index, active_indices] = 1
    bandwidths = np.array([float(np.sum(config)) for config in configs], dtype=float)
    return configs, bandwidths


def _dummy_compute_bandwidths(gpu_configs, total_gpu, gpu_bw_dict_list, switch_config, training_data_path):
    """Use active GPU count as a deterministic dummy bandwidth."""

    return [float(np.sum(config)) for config in gpu_configs]


class SensitivitySamplingProtocolTest(unittest.TestCase):
    """Cover inclusion, quota, and manifest invariants of nested sampling."""

    def test_build_prefix_index_map_is_nested(self) -> None:
        """Random/Worst-Case style prefix subsets must satisfy strict inclusion."""

        index_map = _build_prefix_index_map(
            mother_pool_size=500,
            sample_sizes=[100, 250, 500],
            seed=12345,
        )
        subset_100 = set(index_map[100].tolist())
        subset_250 = set(index_map[250].tolist())
        subset_500 = set(index_map[500].tolist())

        self.assertEqual(len(subset_100), 100)
        self.assertEqual(len(subset_250), 250)
        self.assertEqual(len(subset_500), 500)
        self.assertTrue(subset_100.issubset(subset_250))
        self.assertTrue(subset_250.issubset(subset_500))

    def test_build_stratified_index_map_preserves_quota_and_inclusion(self) -> None:
        """Stratified nested subsets must preserve both inclusion and per-bin quotas."""

        mother_metadata = []
        mother_bin_counts = _build_stratified_bin_quotas(500)
        for bin_idx, count in enumerate(mother_bin_counts):
            for _ in range(count):
                mother_metadata.append({"density_bin": bin_idx, "sample_source": "balanced"})

        index_map = _build_stratified_index_map(
            mother_metadata=mother_metadata,
            sample_sizes=[100, 250, 500],
            seed=7,
        )

        subset_100 = set(index_map[100].tolist())
        subset_250 = set(index_map[250].tolist())
        subset_500 = set(index_map[500].tolist())
        self.assertTrue(subset_100.issubset(subset_250))
        self.assertTrue(subset_250.issubset(subset_500))

        for sample_size in [100, 250, 500]:
            quotas = _build_stratified_bin_quotas(sample_size)
            observed_counts = {bin_idx: 0 for bin_idx in range(8)}
            for mother_index in index_map[sample_size]:
                observed_counts[int(mother_metadata[int(mother_index)]["density_bin"])] += 1
            self.assertEqual(
                [observed_counts[bin_idx] for bin_idx in range(8)],
                quotas,
            )

    def test_build_nested_dataset_family_writes_manifest(self) -> None:
        """The exported manifest must describe the same nested family returned in memory."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_root = Path(tmp_dir) / "nested_manifests"
            family = build_nested_dataset_family(
                cluster_type="DummyCluster",
                strategy_name="Random",
                sample_sizes=[100, 250, 500],
                seed=99,
                total_gpu=32,
                gpu_bw_dict_list=[],
                switch_config=None,
                training_data_path="unused.csv",
                generator_fn=_dummy_generator,
                compute_bandwidths_fn=_dummy_compute_bandwidths,
                manifest_root=manifest_root,
                protocol_name="nested",
            )

            self.assertEqual(family[100][0].shape[0], 100)
            self.assertEqual(family[250][0].shape[0], 250)
            self.assertEqual(family[500][0].shape[0], 500)

            summary_path = manifest_root / "DummyCluster" / "random" / "seed99_summary.json"
            manifest_path = manifest_root / "DummyCluster" / "random" / "seed99_manifest.csv"
            self.assertTrue(summary_path.exists())
            self.assertTrue(manifest_path.exists())

            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["sampling_protocol"], "nested")
            self.assertEqual(payload["subset_sizes"]["100"], 100)
            self.assertEqual(payload["subset_sizes"]["250"], 250)
            self.assertEqual(payload["subset_sizes"]["500"], 500)

    def test_extend_nested_dataset_family_preserves_old_subsets_and_adds_new_budget(self) -> None:
        """Incremental extension must keep old nested subsets unchanged."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            base_root = Path(tmp_dir) / "base"
            ext_root = Path(tmp_dir) / "extended"

            base_family = build_nested_dataset_family(
                cluster_type="DummyCluster",
                strategy_name="Random",
                sample_sizes=[10, 20, 30],
                seed=123,
                total_gpu=32,
                gpu_bw_dict_list=[],
                switch_config=None,
                training_data_path="unused.csv",
                generator_fn=_dummy_generator,
                compute_bandwidths_fn=_dummy_compute_bandwidths,
                manifest_root=base_root / "nested_manifests",
                protocol_name="nested",
            )

            extended_family = extend_nested_dataset_family_from_manifest(
                cluster_type="DummyCluster",
                strategy_name="Random",
                sample_sizes=[10, 20, 30, 40],
                seed=123,
                total_gpu=32,
                gpu_bw_dict_list=[],
                switch_config=None,
                training_data_path="unused.csv",
                generator_fn=_dummy_generator,
                compute_bandwidths_fn=_dummy_compute_bandwidths,
                existing_manifest_root=base_root / "nested_manifests",
                manifest_root=ext_root / "nested_manifests",
                mother_pool_size=40,
                protocol_name="nested",
            )

            for sample_size in [10, 20, 30]:
                base_set = {tuple(row.tolist()) for row in base_family[sample_size][0]}
                ext_set = {tuple(row.tolist()) for row in extended_family[sample_size][0]}
                self.assertEqual(base_set, ext_set)

            self.assertEqual(extended_family[40][0].shape[0], 40)
            summary_path = ext_root / "nested_manifests" / "DummyCluster" / "random" / "seed123_summary.json"
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["subset_sizes"]["10"], 10)
            self.assertEqual(payload["subset_sizes"]["20"], 20)
            self.assertEqual(payload["subset_sizes"]["30"], 30)
            self.assertEqual(payload["subset_sizes"]["40"], 40)


if __name__ == "__main__":
    unittest.main()
