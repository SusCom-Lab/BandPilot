"""Unit tests for the scalability PTS-sidecar pipeline."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PTS_SIDECAR_DIR = REPO_ROOT / "evaluation" / "scalability" / "pts_sidecar"


def _load_module(module_name: str, relative_name: str):
    """Load a PTS-sidecar module by filename."""

    target_path = PTS_SIDECAR_DIR / relative_name
    spec = importlib.util.spec_from_file_location(module_name, target_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {target_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PTSSidecarPipelineTest(unittest.TestCase):
    """Validate the PTS-sidecar analysis and report path."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.analyze = _load_module("pts_sidecar_analyze", "analyze.py")
        cls.report_builder = _load_module("pts_sidecar_report_builder", "report_builder.py")

    def _build_raw_df(self) -> pd.DataFrame:
        """Create synthetic raw rows for a two-scale paired analysis."""

        rows = []
        for total_gpu, repeat_idx, pts_latency, hu_latency, pts_bw, hu_bw, same_combo in [
            (128, 0, 1.0, 0.5, 100.0, 100.0, True),
            (128, 1, 1.2, 0.6, 100.0, 99.0, False),
            (256, 0, 2.0, 1.0, 200.0, 198.0, True),
            (256, 1, 2.4, 1.2, 200.0, 200.0, True),
        ]:
            combo_signature = "0,1,2,3" if same_combo else "4,5,6,7"
            rows.append(
                {
                    "cluster_type": "H100_26H100_27H100_28H100_29",
                    "total_gpu": total_gpu,
                    "k": 64,
                    "avail_ratio": 0.7,
                    "contention_mode": "common",
                    "inter_pod_factor": 0.7,
                    "repeat_idx": repeat_idx,
                    "seed": 1000 + total_gpu + repeat_idx,
                    "algorithm": "legacy-PTS",
                    "measured_wall_time_s": pts_latency,
                    "predictor_time_s": pts_latency * 0.2,
                    "predictor_calls": 10,
                    "contention_time_s": 0.0,
                    "pts_phase_time_s": pts_latency * 0.8,
                    "non_predictor_search_time_s": pts_latency * 0.8,
                    "final_bw": pts_bw,
                    "combo_signature": "0,1,2,3",
                    "evidence_type": "simulated",
                    "latency_evidence_kind": "scaled_trace",
                    "bandwidth_evidence_kind": "scaled_estimated",
                }
            )
            rows.append(
                {
                    "cluster_type": "H100_26H100_27H100_28H100_29",
                    "total_gpu": total_gpu,
                    "k": 64,
                    "avail_ratio": 0.7,
                    "contention_mode": "common",
                    "inter_pod_factor": 0.7,
                    "repeat_idx": repeat_idx,
                    "seed": 1000 + total_gpu + repeat_idx,
                    "algorithm": "PTS",
                    "measured_wall_time_s": hu_latency,
                    "predictor_time_s": hu_latency * 0.1,
                    "predictor_calls": 5,
                    "contention_time_s": 0.0,
                    "pts_phase_time_s": hu_latency * 0.9,
                    "non_predictor_search_time_s": hu_latency * 0.9,
                    "final_bw": hu_bw,
                    "combo_signature": combo_signature,
                    "evidence_type": "simulated",
                    "latency_evidence_kind": "scaled_trace",
                    "bandwidth_evidence_kind": "scaled_estimated",
                }
            )
        return pd.DataFrame(rows)

    def test_build_summary_tables_computes_speedup_and_combo_rate(self) -> None:
        """Summary tables compute speedup, same-combo rate, and bandwidth drift."""

        raw_df = self._build_raw_df()
        paired_df, summary_df, breakdown_df = self.analyze.build_summary_tables(raw_df)

        self.assertEqual(len(paired_df), 4)
        self.assertEqual(len(summary_df), 2)
        self.assertEqual(len(breakdown_df), 4)

        row_128 = summary_df[summary_df["total_gpu"] == 128].iloc[0]
        self.assertAlmostEqual(float(row_128["speedup_mean"]), 2.0, places=6)
        self.assertAlmostEqual(float(row_128["same_combo_rate"]), 0.5, places=6)
        self.assertAlmostEqual(float(row_128["bw_delta_pct_pts_vs_legacy_pts_mean"]), -0.5, places=6)

    def test_report_builder_writes_required_sections(self) -> None:
        """Report builder writes required sections and evidence markers."""

        summary_df = pd.DataFrame(
            [
                {
                    "cluster_type": "H100_26H100_27H100_28H100_29",
                    "total_gpu": 128,
                    "k": 64,
                    "avail_ratio": 0.7,
                    "contention_mode": "common",
                    "inter_pod_factor": 0.7,
                    "repeat_count": 10,
                    "evidence_type": "simulated",
                    "latency_evidence_kind": "scaled_trace",
                    "bandwidth_evidence_kind": "scaled_estimated",
                    "speedup_mean": 2.1,
                    "speedup_std": 0.1,
                    "speedup_p50": 2.0,
                    "speedup_p95": 2.2,
                    "same_combo_rate": 0.9,
                    "bw_delta_pct_pts_vs_legacy_pts_mean": -0.3,
                    "bw_delta_pct_pts_vs_legacy_pts_std": 0.2,
                }
            ]
        )
        metadata = {
            "run_tag": "unit_test_run",
            "cluster_types": ["H100_26H100_27H100_28H100_29", "Het-4Mix"],
            "gpu_counts": [128, 256],
            "k_value": 64,
            "repeat_num": 10,
            "contention_mode": "common",
            "avail_ratio": 0.7,
            "inter_pod_factor": 0.7,
            "evidence_type": "simulated",
        }
        breakdown_df = pd.DataFrame(
            [
                {
                    "cluster_type": "H100_26H100_27H100_28H100_29",
                    "total_gpu": 128,
                    "algorithm": "legacy-PTS",
                    "repeat_count": 10,
                    "latency_mean_s": 1.0,
                    "predictor_time_mean_s": 0.6,
                    "non_predictor_time_mean_s": 0.4,
                    "contention_time_mean_s": 1.0,
                    "pts_phase_time_mean_s": 1.0,
                },
                {
                    "cluster_type": "H100_26H100_27H100_28H100_29",
                    "total_gpu": 128,
                    "algorithm": "PTS",
                    "repeat_count": 10,
                    "latency_mean_s": 0.5,
                    "predictor_time_mean_s": 0.2,
                    "non_predictor_time_mean_s": 0.3,
                    "contention_time_mean_s": 0.5,
                    "pts_phase_time_mean_s": 0.5,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            figure_dir = base_dir / "figures"
            report_dir = base_dir / "reports"
            figure_dir.mkdir(parents=True, exist_ok=True)
            # Minimal placeholder figures are enough for Markdown path generation.
            (figure_dir / "pts_speedup_vs_legacy_pts.png").write_text("", encoding="utf-8")
            (figure_dir / "latency_breakdown_legacy_pts_vs_pts.png").write_text("", encoding="utf-8")

            paths = self.report_builder.build_report(
                summary_df=summary_df,
                breakdown_df=breakdown_df,
                metadata=metadata,
                figure_dir=figure_dir,
                report_dir=report_dir,
            )
            report_text = Path(paths["report_md"]).read_text(encoding="utf-8")

        self.assertIn("## Summary", report_text)
        self.assertIn("## Protocol", report_text)
        self.assertIn("simulated", report_text)
        self.assertIn("## Figure 1: PTS Speedup Over legacy-PTS", report_text)


if __name__ == "__main__":
    unittest.main()
