"""Unit tests for scalability report-builder audit behavior."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from evaluation.scalability.report_builder import (
    BenchmarkSpec,
    _determine_report_scope,
    _default_spec,
    _build_expected_row_map,
    _filter_representative_display_context,
    _select_diagnostic_algorithm,
    build_artifact_audit,
    validate_artifact_audit,
)


class ScalabilityReportBuilderTest(unittest.TestCase):
    """Validate report-builder audit and representative-scope behavior."""

    def _make_spec(self) -> BenchmarkSpec:
        """Create a compact benchmark spec with public-repeat settings."""

        return BenchmarkSpec(
            cluster_total_gpu=32,
            cluster_types=["ClusterA", "ClusterB"],
            algorithms=["EHA", "BandPilot"],
            public_algorithms=["BandPilot"],
            representative_contention_mode="common",
            representative_avail_ratio=0.7,
            representative_inter_pod_factor=0.7,
            representative_k_values=[8],
            tier1_k_values=[8],
            tier1_contention_modes=["common"],
            tier1_repeat_num=2,
            tier2_gpu_counts=[64, 128],
            tier2_k_values=[8, 16],
            tier2_avail_ratios=[0.7],
            tier2_contention_modes=["common"],
            tier2_inter_pod_factors=[0.7],
            tier2_repeat_num=3,
            tier2_public_repeat_num=5,
            predictor_node_counts=[4, 8],
            predictor_inference_repeats=7,
            tier4_target_gpu_counts=[128, 256],
        )

    def _write_csv(self, path: Path, header: list[str], row_count: int) -> None:
        """Write a synthetic CSV with a fixed header and row count."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for idx in range(row_count):
                writer.writerow([idx] * len(header))

    def test_expected_row_map_accounts_for_public_repeat_slice(self) -> None:
        """Expected rows account for public-repeat representative slices."""

        spec = self._make_spec()
        expected = _build_expected_row_map(spec)

        # Tier 1: 1(k) * 1(mode) * 2(repeat) * 2(algorithms) = 4
        self.assertEqual(expected["tier1"], 4)

        # Tier 2 keeps one public-repeat group per scale. The representative
        # group contains k=8 and k=16, each repeated five times.
        self.assertEqual(expected["tier2"], 40)

        # Predictor profile has one row per node_count.
        self.assertEqual(expected["predictor_profile"], 2)

        # Tier 4 maps both 128 and 256 targets to the observed 128-GPU scale.
        self.assertEqual(expected["tier4"], 40)

    def test_build_artifact_audit_marks_complete_partial_missing(self) -> None:
        """Artifact audit marks complete, partial, and missing tiers."""

        spec = self._make_spec()
        expected = _build_expected_row_map(spec)

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_dir = Path(tmp_dir)

            # ClusterA has complete Tier 1, partial Tier 2, missing predictor, and complete Tier 4.
            self._write_csv(artifact_dir / "tier1_ClusterA.csv", ["dummy"], expected["tier1"])
            self._write_csv(artifact_dir / "scaled_search_ClusterA.csv", ["dummy"], expected["tier2"] - 2)
            self._write_csv(artifact_dir / "synthesized_dispatch_latency_ClusterA.csv", ["dummy"], expected["tier4"])

            # ClusterB has no artifacts.
            audit = build_artifact_audit(artifact_dir, spec)

            cluster_a_tier1 = audit[(audit["cluster_type"] == "ClusterA") & (audit["tier"] == "tier1")].iloc[0]
            cluster_a_tier2 = audit[(audit["cluster_type"] == "ClusterA") & (audit["tier"] == "tier2")].iloc[0]
            cluster_a_predictor = audit[(audit["cluster_type"] == "ClusterA") & (audit["tier"] == "predictor_profile")].iloc[0]
            cluster_b_tier1 = audit[(audit["cluster_type"] == "ClusterB") & (audit["tier"] == "tier1")].iloc[0]

            self.assertEqual(cluster_a_tier1["status"], "complete")
            self.assertEqual(cluster_a_tier2["status"], "partial")
            self.assertEqual(cluster_a_predictor["status"], "missing")
            self.assertEqual(cluster_b_tier1["status"], "missing")

    def test_validate_artifact_audit_enforces_mode_policy(self) -> None:
        """Audit validation enforces strict and partial policies."""

        spec = self._make_spec()
        expected = _build_expected_row_map(spec)

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_dir = Path(tmp_dir)
            self._write_csv(artifact_dir / "tier1_ClusterA.csv", ["dummy"], expected["tier1"])
            self._write_csv(artifact_dir / "scaled_search_ClusterA.csv", ["dummy"], expected["tier2"] - 1)
            self._write_csv(artifact_dir / "predictor_latency_profile_ClusterA.csv", ["dummy"], expected["predictor_profile"])
            self._write_csv(artifact_dir / "synthesized_dispatch_latency_ClusterA.csv", ["dummy"], expected["tier4"])

            self._write_csv(artifact_dir / "tier1_ClusterB.csv", ["dummy"], expected["tier1"])
            self._write_csv(artifact_dir / "scaled_search_ClusterB.csv", ["dummy"], expected["tier2"])
            self._write_csv(artifact_dir / "predictor_latency_profile_ClusterB.csv", ["dummy"], expected["predictor_profile"])
            self._write_csv(artifact_dir / "synthesized_dispatch_latency_ClusterB.csv", ["dummy"], expected["tier4"])

            audit = build_artifact_audit(artifact_dir, spec)

            with self.assertRaises(ValueError):
                validate_artifact_audit(audit, mode="strict", partial_clusters=[])

            # ClusterA is the only permitted partial cluster.
            validate_artifact_audit(audit, mode="partial", partial_clusters=["ClusterA"])

            with self.assertRaises(ValueError):
                validate_artifact_audit(audit, mode="partial", partial_clusters=["ClusterB"])

    def test_default_spec_matches_three_algorithm_public_protocol(self) -> None:
        """Default spec matches the three-algorithm public protocol."""

        spec = _default_spec()

        self.assertEqual(spec.public_algorithms, ["EHA", "PTS", "BandPilot"])
        self.assertEqual(spec.representative_k_values, [16, 64])
        self.assertEqual(spec.tier2_public_repeat_num, 20)

    def test_expected_row_map_retains_boundary_k_equal_to_target_avail(self) -> None:
        """Tier 2 expected rows retain the `k == target_avail` boundary case."""

        spec = BenchmarkSpec(
            cluster_total_gpu=32,
            cluster_types=["ClusterA"],
            algorithms=["EHA", "BandPilot"],
            public_algorithms=["BandPilot"],
            representative_contention_mode="common",
            representative_avail_ratio=0.5,
            representative_inter_pod_factor=0.7,
            representative_k_values=[64],
            tier1_k_values=[8],
            tier1_contention_modes=["common"],
            tier1_repeat_num=2,
            tier2_gpu_counts=[128],
            tier2_k_values=[16, 32, 64],
            tier2_avail_ratios=[0.5],
            tier2_contention_modes=["common"],
            tier2_inter_pod_factors=[0.7],
            tier2_repeat_num=3,
            tier2_public_repeat_num=5,
            predictor_node_counts=[4],
            predictor_inference_repeats=7,
            tier4_target_gpu_counts=[128],
        )

        expected = _build_expected_row_map(spec)

        # `target_avail = max(64, round(128*0.5)=64) = 64`, so k=16/32/64 are feasible.
        # The representative group uses k=64 and five public repeats.
        # Tier 2 = 3(k) * 2 algorithms * 5 repeats = 30
        self.assertEqual(expected["tier2"], 30)
        self.assertEqual(expected["tier4"], 30)

    def test_representative_display_context_excludes_exact_fit_boundary(self) -> None:
        """Reviewer-facing display excludes exact-fit `total_gpu == k` rows."""

        spec = _default_spec()
        frame = pd.DataFrame(
            [
                {
                    "cluster_type": "ClusterA",
                    "algorithm": "BandPilot",
                    "total_gpu": 64,
                    "k": 64,
                    "contention_mode": "common",
                    "avail_ratio": 0.7,
                    "inter_pod_factor": 0.7,
                },
                {
                    "cluster_type": "ClusterA",
                    "algorithm": "BandPilot",
                    "total_gpu": 128,
                    "k": 64,
                    "contention_mode": "common",
                    "avail_ratio": 0.7,
                    "inter_pod_factor": 0.7,
                },
                {
                    "cluster_type": "ClusterA",
                    "algorithm": "BandPilot",
                    "total_gpu": 128,
                    "k": 16,
                    "contention_mode": "common",
                    "avail_ratio": 0.7,
                    "inter_pod_factor": 0.7,
                },
            ]
        )

        filtered = _filter_representative_display_context(frame, spec)

        self.assertEqual(sorted(filtered["total_gpu"].tolist()), [128, 128])
        self.assertEqual(sorted(filtered["k"].tolist()), [16, 64])
        self.assertFalse(((filtered["total_gpu"] == 64) & (filtered["k"] == 64)).any())

    def test_diagnostic_algorithm_prefers_bandpilot(self) -> None:
        """BandPilot diagnostics are selected for public summaries."""

        spec = _default_spec()
        self.assertEqual(_select_diagnostic_algorithm(spec), "BandPilot")

    def test_report_scope_distinguishes_full_and_partial(self) -> None:
        """Report scope distinguishes full and partial audit states."""

        full_scope_audit = [
            {"cluster_type": "ClusterA", "tier": "tier1", "status": "complete"},
            {"cluster_type": "ClusterA", "tier": "tier2", "status": "complete"},
        ]
        partial_scope_audit = [
            {"cluster_type": "ClusterA", "tier": "tier1", "status": "complete"},
            {"cluster_type": "ClusterA", "tier": "tier2", "status": "partial"},
        ]

        self.assertEqual(_determine_report_scope(pd.DataFrame(full_scope_audit)), "full")
        self.assertEqual(_determine_report_scope(pd.DataFrame(partial_scope_audit)), "partial")


if __name__ == "__main__":
    unittest.main()
