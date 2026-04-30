"""Online mismatch-kNN support bank for BandPilot adaptive replay.

The bank stores labeled EHA-vs-PTS observations and estimates whether PTS is
likely to help for a new query. It is append-only within a replay stream and
uses fixed mismatch features so runtime adaptation, compare replay, and
scalability diagnostics share the same decision substrate.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from algorithms.adaptive_gate_support.online_mismatch_knn.features import (
    _cluster_group_key,
    _robust_scale_columns,
    _sample_to_mismatch_knn_vector,
)
from algorithms.adaptive_gate_support.utils import (
    _as_int,
    _write_csv,
)


def _sample_case_key(sample: Mapping[str, object]) -> Tuple[str, str, int, int]:
    """Return the case identity used to exclude self-neighbors."""
    return (
        str(sample.get("cluster_type", "")),
        str(sample.get("contention_mode", "")),
        _as_int(sample.get("test_num", 0)),
        _as_int(sample.get("repeat_idx", 0)),
    )


@dataclass(frozen=True)
class OnlineBankConfig:
    """Configuration for online mismatch-kNN risk estimation."""

    k_neighbors: int = 5
    same_contention_only: bool = True
    min_support: int | None = None
    low_trust_conflict_risk_threshold: float = 0.30


@dataclass(frozen=True)
class _SupportEntry:
    """One labeled support row stored in the online bank."""

    case_key: Tuple[str, str, int, int]
    sample: Dict[str, object]
    feature_row: Dict[str, object]
    vector: Tuple[float, ...]
    label: int


@dataclass(frozen=True)
class _NeighborVoteStats:
    """Weighted-neighbor vote statistics for one query.

    `helpful_weight_share` is the kNN risk score, `conflict_risk` captures
    disagreement between helpful and non-helpful neighbors, and `low_trust`
    marks insufficient local support.
    """

    helpful_weight_share: float
    support_count: int
    helpful_neighbor_count: int
    non_helpful_neighbor_count: int
    conflict_risk: float
    low_trust: bool


@dataclass
class OnlineMismatchBank:
    """Append-only bank of labeled support rows for online mismatch-kNN."""

    config: OnlineBankConfig
    labeled_samples: List[Dict[str, object]]
    labeled_feature_rows: List[Dict[str, object]]
    _entries_by_group: Dict[Tuple[str, str], List[_SupportEntry]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Build the grouped support-entry index after dataclass initialization."""

        self._entries_by_group = {}
        for sample, feature_row in zip(self.labeled_samples, self.labeled_feature_rows):
            self._append_support_entry(sample=dict(sample), feature_row=dict(feature_row))

    @classmethod
    def empty(cls, config: OnlineBankConfig) -> "OnlineMismatchBank":
        """Create an empty support bank."""
        return cls(config=config, labeled_samples=[], labeled_feature_rows=[])

    @classmethod
    def from_rows(
        cls,
        *,
        config: OnlineBankConfig,
        labeled_rows: Sequence[Mapping[str, object]],
    ) -> "OnlineMismatchBank":
        """Create a bank from labeled rows that already contain feature columns."""
        return cls(
            config=config,
            labeled_samples=[dict(row) for row in labeled_rows],
            labeled_feature_rows=[dict(row) for row in labeled_rows],
        )

    def _group_key(self, sample: Mapping[str, object]) -> Tuple[str, str]:
        """Return the `(cluster, contention)` support-group key for a sample."""

        cluster_type = _cluster_group_key(sample)
        contention_mode = str(sample.get("contention_mode", "")).strip().lower()
        if self.config.same_contention_only:
            return (cluster_type, contention_mode)
        return (cluster_type, "__all__")

    def _append_support_entry(
        self,
        *,
        sample: Dict[str, object],
        feature_row: Dict[str, object],
    ) -> None:
        """Append one labeled support row to the grouped bank index."""

        entry = _SupportEntry(
            case_key=_sample_case_key(sample),
            sample=sample,
            feature_row=feature_row,
            vector=tuple(_sample_to_mismatch_knn_vector(sample, feature_row)),
            label=1 if bool(feature_row.get("pts_helpful", False)) else 0,
        )
        self._entries_by_group.setdefault(self._group_key(sample), []).append(entry)

    @property
    def labeled_size(self) -> int:
        """Return the number of labeled rows stored in the bank."""
        return len(self.labeled_samples)

    def append_labeled_rows(self, rows: Sequence[Mapping[str, object]]) -> None:
        """Append newly admitted labeled rows to the bank."""
        recovered = OnlineMismatchBank.from_rows(config=self.config, labeled_rows=rows)
        self.labeled_samples.extend(recovered.labeled_samples)
        self.labeled_feature_rows.extend(recovered.labeled_feature_rows)
        for sample, feature_row in zip(recovered.labeled_samples, recovered.labeled_feature_rows):
            self._append_support_entry(sample=sample, feature_row=feature_row)

    def _effective_min_support(self) -> int:
        """Return the minimum neighbor count required to trust a query decision."""

        k_neighbors = max(1, int(self.config.k_neighbors))
        configured = self.config.min_support
        if configured in ("", None):
            return k_neighbors
        return max(1, min(k_neighbors, int(configured)))

    def _low_trust_conflict_risk_threshold(self) -> float:
        """Return the bounded conflict-risk threshold for low-trust detection."""

        return max(
            0.0,
            min(1.0, float(self.config.low_trust_conflict_risk_threshold)),
        )

    def _eligible_support_entries(
        self,
        *,
        query_sample: Mapping[str, object],
    ) -> List[_SupportEntry]:
        """Return support entries eligible for a query, excluding self-neighbors."""
        query_case_key = _sample_case_key(query_sample)
        eligible_entries: List[_SupportEntry] = []
        for entry in self._entries_by_group.get(self._group_key(query_sample), []):
            if entry.case_key == query_case_key:
                continue
            eligible_entries.append(entry)
        return eligible_entries

    def _evaluate_query_only_mismatch_knn_risk(
        self,
        *,
        query_sample: Mapping[str, object],
        query_feature_row: Mapping[str, object],
        support_entries: Sequence[_SupportEntry],
    ) -> Dict[str, object]:
        """Estimate mismatch-kNN risk for one query against the current support bank."""

        query_vector = list(_sample_to_mismatch_knn_vector(query_sample, query_feature_row))
        support_vectors = [list(entry.vector) for entry in support_entries]
        medians, scales = _robust_scale_columns([*support_vectors, query_vector])

        normalized_query = [
            (float(value) - medians[col_idx]) / scales[col_idx]
            for col_idx, value in enumerate(query_vector)
        ]
        neighbors: List[Tuple[float, _SupportEntry]] = []
        for entry, support_vector in zip(support_entries, support_vectors):
            normalized_support = [
                (float(value) - medians[col_idx]) / scales[col_idx]
                for col_idx, value in enumerate(support_vector)
            ]
            distance = math.sqrt(
                sum(
                    (normalized_query[col_idx] - normalized_support[col_idx]) ** 2
                    for col_idx in range(len(query_vector))
                )
            )
            neighbors.append((distance, entry))

        neighbors.sort(key=lambda item: item[0])
        selected = neighbors[: max(1, min(int(self.config.k_neighbors), len(neighbors)))]

        vote_stats = self._summarize_neighbor_votes(selected)

        group_key = self._group_key(query_sample)
        return {
            "online_mismatch_knn_risk": vote_stats.helpful_weight_share,
            "online_mismatch_support": vote_stats.support_count,
            "online_mismatch_scope": f"{group_key[0]}|{group_key[1]}",
            "online_mismatch_helpful_neighbor_count": vote_stats.helpful_neighbor_count,
            "online_mismatch_non_helpful_neighbor_count": vote_stats.non_helpful_neighbor_count,
            "online_mismatch_conflict_risk": vote_stats.conflict_risk,
            "online_mismatch_low_trust": vote_stats.low_trust,
            "online_mismatch_min_support": self._effective_min_support(),
            "bank_active": False,
            "support_insufficient": vote_stats.support_count < self._effective_min_support(),
        }

    def _summarize_neighbor_votes(
        self,
        neighbors: Sequence[Tuple[float, _SupportEntry]],
    ) -> _NeighborVoteStats:
        """Summarize weighted helpful/non-helpful votes from selected neighbors."""

        if not neighbors:
            return _NeighborVoteStats(
                helpful_weight_share=1.0,
                support_count=0,
                helpful_neighbor_count=0,
                non_helpful_neighbor_count=0,
                conflict_risk=0.0,
                low_trust=False,
            )

        weighted_sum = 0.0
        weight_total = 0.0
        helpful_neighbor_count = 0
        for distance, entry in neighbors:
            weight = 1.0 / max(1e-9, distance + 1e-6)
            weighted_sum += weight * float(entry.label)
            weight_total += weight
            helpful_neighbor_count += int(entry.label)

        support_count = len(neighbors)
        helpful_weight_share = weighted_sum / max(weight_total, 1e-9)
        non_helpful_neighbor_count = support_count - helpful_neighbor_count

        # Conflict risk is highest near a 50/50 weighted vote split.
        conflict_risk = max(0.0, 1.0 - 2.0 * abs(helpful_weight_share - 0.5))
        low_trust = (
            helpful_neighbor_count > 0
            and non_helpful_neighbor_count > 0
            and conflict_risk >= self._low_trust_conflict_risk_threshold()
        )
        return _NeighborVoteStats(
            helpful_weight_share=helpful_weight_share,
            support_count=support_count,
            helpful_neighbor_count=helpful_neighbor_count,
            non_helpful_neighbor_count=non_helpful_neighbor_count,
            conflict_risk=conflict_risk,
            low_trust=low_trust,
        )

    def evaluate_query(
        self,
        *,
        query_sample: Mapping[str, object],
        query_feature_row: Mapping[str, object],
    ) -> Dict[str, object]:
        """Return the online mismatch-risk row for one query."""
        eligible_entries = self._eligible_support_entries(query_sample=query_sample)
        if not eligible_entries:
            return {
                "online_mismatch_knn_risk": 1.0,
                "online_mismatch_support": 0,
                "online_mismatch_scope": "support_insufficient",
                "online_mismatch_helpful_neighbor_count": 0,
                "online_mismatch_non_helpful_neighbor_count": 0,
                "online_mismatch_conflict_risk": 0.0,
                "online_mismatch_low_trust": False,
                "online_mismatch_min_support": self._effective_min_support(),
                "bank_active": False,
                "support_insufficient": True,
            }
        return self._evaluate_query_only_mismatch_knn_risk(
            query_sample=query_sample,
            query_feature_row=query_feature_row,
            support_entries=eligible_entries,
        )


def persist_online_bank_state(
    *,
    output_dir: Path,
    query_pool_rows: Sequence[Mapping[str, object]],
    labeled_bank_rows: Sequence[Mapping[str, object]],
    snapshot_rows: Sequence[Mapping[str, object]],
) -> None:
    """Persist query-pool, labeled-bank, and bank-snapshot diagnostics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = output_dir / "bank_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale snapshots so the directory reflects the current replay only.
    for snapshot_path in snapshot_dir.glob("*.json"):
        snapshot_path.unlink()

    _write_csv(output_dir / "query_pool.csv", list(query_pool_rows))
    _write_csv(output_dir / "labeled_bank.csv", list(labeled_bank_rows))
    _write_csv(output_dir / "bank_snapshots.csv", list(snapshot_rows))

    for row in snapshot_rows:
        bank_version = _as_int(row.get("bank_version", 0))
        (snapshot_dir / f"bank_{bank_version:04d}.json").write_text(
            json.dumps(dict(row), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
