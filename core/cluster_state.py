"""Cluster state management.

Implements resource allocation, contention detection, and bandwidth prediction for multi-tenant scenarios.
"""
from __future__ import annotations

import numpy as np
import copy
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import List, Dict, Tuple, Optional, Set, Callable
from pathlib import Path

from core.bandwidth import prepare_model_inputs, calculate_bandwidth_values, SwitchBandwidthConfig
from training.evaluator import predict_with_model

logger = logging.getLogger(__name__)

# Contention mode whitelist to keep modes consistent across modules (avoid case/whitespace drift).
VALID_CONTENTION_MODES = ("common", "intensive", "idle")


def normalize_contention_mode(mode: str) -> str:
    """Normalize contention mode to lowercase and validate."""
    if mode is None:
        raise ValueError("contention_mode cannot be None")
    normalized = str(mode).strip().lower()
    if normalized not in VALID_CONTENTION_MODES:
        raise ValueError(
            f"Unsupported contention_mode '{mode}', must be one of {sorted(VALID_CONTENTION_MODES)}"
        )
    return normalized


class ContentionProfiler:
    """Track contention-related latency."""

    __slots__ = ("total_time",)

    def __init__(self) -> None:
        self.total_time: float = 0.0

    def add(self, duration: float) -> None:
        self.total_time += duration


_contention_profiler_ctx: ContextVar[Optional[ContentionProfiler]] = ContextVar(
    "contention_profiler_ctx", default=None
)


@contextmanager
def contention_profiling_session():
    """Start a contention timing session."""
    profiler = ContentionProfiler()
    token = _contention_profiler_ctx.set(profiler)
    try:
        yield profiler
    finally:
        _contention_profiler_ctx.reset(token)


def create_bandwidth_predictor(
    if_real_data: bool,
    total_gpu: int,
    gpu_bw_dict_list: List,
    switch_config: SwitchBandwidthConfig,
    training_data_path: str,
    evaluation_data_path: Optional[str] = None,
    model=None,
    device=None,
    artifact_dir: Optional[Path] = None,
    num_train_samples: Optional[int] = None,
) -> Callable[[np.ndarray], float]:
    """Factory that creates a bandwidth predictor callable.

    Chooses real-data lookup vs. model prediction based on if_real_data. The returned
    callable is injected into ClusterStateManager so it stays decoupled from mode.

    Args:
        if_real_data: True to use real data lookup; False to use model prediction.
        total_gpu: Total GPU count.
        gpu_bw_dict_list: List of GPU bandwidth dictionaries.
        switch_config: Switch configuration.
        training_data_path: Data path for training/model prediction.
        evaluation_data_path: Data path for real-data evaluation (defaults to training path).
        model: PyTorch model (for prediction mode).
        device: PyTorch device (for prediction mode).
        artifact_dir: Directory containing model/scalers (for prediction mode).

    Returns:
        Callable taking np.ndarray (GPU combo) and returning float bandwidth.

    Raises:
        ValueError if if_real_data=False but model/device/artifact_dir are missing.
    """
    real_data_path = evaluation_data_path or training_data_path

    if if_real_data:
        # Real-data mode: use calculate_bandwidth_values
        def predictor(combo: np.ndarray) -> float:
            if np.sum(combo) == 0:
                return 0.0
            bw, _, _ = calculate_bandwidth_values(
                combo, total_gpu, gpu_bw_dict_list, switch_config, real_data_path
            )
            return float(bw)
        
        return predictor
    else:
        # Model-prediction mode: use predict_with_model
        if model is None or device is None or artifact_dir is None:
            raise ValueError(
                "Model prediction mode requires model, device and artifact_dir."
            )
        
        def predictor(combo: np.ndarray) -> float:
            if np.sum(combo) == 0:
                return 0.0
            part_bws, node_counts, total_counts = prepare_model_inputs(
                np.array([combo]),
                total_gpu,
                gpu_bw_dict_list,
                switch_config,
                training_data_path,
            )
            bw = predict_with_model(
                model,
                part_bws,
                node_counts,
                total_counts,
                device,
                artifact_dir,
                num_train_samples,
            )
            return float(bw[0])
        
        return predictor

class ClusterStateManager:
    """Cluster state manager for multi-tenant GPU allocation and contention.

    Design:
    - Decoupled from evaluation mode; inject bandwidth_predictor to support real/model paths.
    - if_real_data is an algorithm-level choice and should not be hardcoded here.
    - Use create_bandwidth_predictor() to ensure consistent predictors.
    - contention_mode controls contention modeling:
        - "intensive" (default): cross-node tasks assumed at peak; split by bottleneck capacity.
        - "common": simulates moderate usage (25%–75% peak) for allocated jobs; new job modeled at peak.
        - "idle": no contention; always exclusive bandwidth.

    Example:
        predictor = create_bandwidth_predictor(
            if_real_data=False,
            total_gpu=32,
            gpu_bw_dict_list=gpu_bw_dict_list,
            switch_config=switch_config,
            training_data_path="train_data.csv",
            evaluation_data_path="eval_data.csv",
            model=model,
            device=device,
            artifact_dir=artifact_dir,
        )
        manager = ClusterStateManager(total_gpu=32, bandwidth_predictor=predictor)
    """
    
    def __init__(
        self, 
        total_gpu: int,
        bandwidth_predictor: Callable[[np.ndarray], float],
        contention_mode: str = "intensive",
        occupancy_seed: Optional[int] = None,
    ):
        """Initialize cluster state manager.

        Args:
            total_gpu: Total GPUs.
            bandwidth_predictor: Callable taking np.ndarray combo -> float bandwidth.
            contention_mode: Contention mode; "intensive" always considers cross-node contention,
                "idle" skips contention (exclusive bandwidth).
        """
        contention_mode = normalize_contention_mode(contention_mode)
        self.total_gpu = total_gpu
        self.bandwidth_predictor = bandwidth_predictor
        self.contention_mode = contention_mode

        # 0: idle, 1: busy
        self.allocated_gpu_mask = np.zeros(total_gpu, dtype=int)
        
        # Active job info
        # Format: {
        #   'job_id': int,
        #   'combo': np.array,
        #   'standalone_bw': float,  # peak bandwidth when exclusive
        #   'occupancy_bw': float,   # real-time occupancy in common mode (else same as standalone)
        #   'current_bw': float,     # bandwidth after contention
        #   'history': List[float]   # bandwidth history
        # }
        self.active_jobs: List[Dict] = []
        
        # Node size (assume 8 GPUs per node)
        self.node_size = 8

        # Occupancy ratio control for common mode
        self._occupancy_seed = occupancy_seed if occupancy_seed is not None else 0
        self._occupancy_ratio_cache: Dict[int, float] = {}
        self._current_job_context: Optional[int] = None

    def _should_apply_contention(self) -> bool:
        """Whether contention logic should apply."""
        return self.contention_mode in {"intensive", "common"}

    def _is_common_mode(self) -> bool:
        return self.contention_mode == "common"

    def _derive_effective_demand(self, standalone_bw: float, job_id: Optional[int]) -> float:
        """
        Derive the effective bandwidth demand for contention calculation.
        In common mode, randomly samples 25%~75% of peak bandwidth to simulate 
        real-time traffic of allocated tasks.
        Note: For new tasks not yet allocated, this contention phase always uses 
        standalone_bw as the demand; only allocate_job writes the random occupancy 
        into state for future tasks to read.
        """
        if self._is_common_mode():
            if job_id is None:
                raise ValueError(
                    "common mode requires job_id; caller must invoke set_job_context or explicitly pass job_id"
                )
            ratio = self._get_or_create_occupancy_ratio(job_id)
            return standalone_bw * ratio
        return standalone_bw

    def _get_job_demand(self, job: Dict) -> float:
        """Get a job’s demand bandwidth under the current mode."""
        if self._is_common_mode():
            return job.get("occupancy_bw", job["standalone_bw"])
        return job["standalone_bw"]

    def _validate_combo_request(
        self,
        combo: np.ndarray,
        enforce_availability: bool = True,
    ) -> np.ndarray:
        """Validate basic legality of a candidate combo.

        - Shape must match total_gpu
        - Elements must be {0,1}
        - Per-node GPU count cannot exceed node_size
        - When enforce_availability=True, no overlap with allocated GPUs
        """
        combo_arr = np.asarray(combo)
        if combo_arr.ndim != 1 or combo_arr.size != self.total_gpu:
            raise ValueError(f"Invalid combo length: expect {self.total_gpu}, got {combo_arr.size}")

        if not np.all((combo_arr == 0) | (combo_arr == 1)):
            raise ValueError("combo contains non {0,1} elements; cannot map to GPUs")

        combo_int = combo_arr.astype(int, copy=False)

        # Single-node capacity check: demand > node_size is invalid
        reshaped = combo_int.reshape(-1, self.node_size)
        node_usage = reshaped.sum(axis=1)
        if np.any(node_usage > self.node_size):
            raise ValueError(f"combo exceeds node capacity: usage {node_usage.tolist()}, capacity {self.node_size}")

        if enforce_availability:
            conflicted = (self.allocated_gpu_mask > 0) & (combo_int > 0)
            if np.any(conflicted):
                conflict_indices = np.where(conflicted)[0][:5]
                raise ValueError(f"combo overlaps allocated GPUs (sample indices: {conflict_indices.tolist()})")

        return combo_int

    def get_available_gpus(self) -> List[int]:
        """Return the list of currently available GPU indices."""
        return list(np.where(self.allocated_gpu_mask == 0)[0])

    def get_active_combos(self) -> List[np.ndarray]:
        """Return a snapshot of currently allocated GPU combinations (copies)."""
        return [job['combo'].copy() for job in self.active_jobs]

    def get_total_active_bandwidth(self) -> float:
        """Return the total bandwidth of all active tasks."""
        return float(sum(job['current_bw'] for job in self.active_jobs))

    def set_job_context(self, job_id: Optional[int]) -> None:
        """Set the current evaluation job_id for predict_with_contention."""
        self._current_job_context = job_id

    def clear_job_context(self) -> None:
        """Clear the current job context."""
        self._current_job_context = None

    def _resolve_job_id(self, job_id: Optional[int]) -> Optional[int]:
        if job_id is not None:
            return job_id
        return self._current_job_context

    def _get_or_create_occupancy_ratio(self, job_id: int) -> float:
        """Generate a stable occupancy ratio for a job_id, ensuring consistency between prediction and submission phases."""
        if job_id in self._occupancy_ratio_cache:
            return self._occupancy_ratio_cache[job_id]
        mask = (1 << 64) - 1
        base = (int(self._occupancy_seed) if self._occupancy_seed is not None else 0) & mask
        job_component = (int(job_id) * 0x9E3779B185EBCA87) & mask
        seed = base ^ job_component
        rng = np.random.default_rng(np.uint64(seed))
        ratio = float(rng.uniform(0.25, 0.75))
        self._occupancy_ratio_cache[job_id] = ratio
        return ratio

    def _predict_combo_bandwidth(self, combo: np.ndarray) -> float:
        """Predict bandwidth for a combo (ignores external contention).

        Uses injected bandwidth_predictor so ClusterStateManager stays mode-agnostic.

        Args:
            combo: GPU combo (0/1 vector)

        Returns:
            Predicted bandwidth.
        """
        return int(self.bandwidth_predictor(combo))

    def _is_cross_node_combo(self, combo: np.ndarray) -> bool:
        """Check whether a GPU combo spans multiple nodes.

        Only cross-node tasks contend with others; single-node tasks do not.
        """
        involved_nodes = set()
        for i in range(len(combo)):
            if combo[i] == 1:
                involved_nodes.add(i // self.node_size)
        
        # If multiple nodes are involved, it is a cross-node combo
        return len(involved_nodes) > 1

    def _get_nodes_for_combo(self, combo: np.ndarray) -> set:
        """Get node indices involved in a combo."""
        nodes = set()
        for i in range(len(combo)):
            if combo[i] == 1:
                nodes.add(i // self.node_size)
        return nodes

    def _project_combo_to_nodes(self, combo: np.ndarray, target_nodes: set) -> np.ndarray:
        """Project combo onto a target node set (keep GPUs on target nodes only)."""
        projected = np.zeros_like(combo)
        for i in range(len(combo)):
            if combo[i] == 1 and (i // self.node_size) in target_nodes:
                projected[i] = 1
        return projected

    def _canonicalize_combo(self, combo: np.ndarray) -> np.ndarray:
        """Canonicalize a combo so each node marks the first `count` GPUs as 1.

        Super combos may contain >1 or non-contiguous values per GPU slot, causing lookup failures.
        This normalizes by counting per node and marking the first `count` slots.
        """
        canonical = np.zeros_like(combo, dtype=int)
        num_nodes = len(combo) // self.node_size
        for node_idx in range(num_nodes):
            start = node_idx * self.node_size
            end = start + self.node_size
            node_slice = combo[start:end]
            count = int(np.sum(node_slice))
            if count <= 0:
                continue
            if count > self.node_size:

                count = self.node_size
            canonical[start : start + count] = 1
        return canonical

    def _is_super_combo_feasible(self, combo: np.ndarray) -> bool:
        """Check if the super combo demand on each node does not exceed the node capacity."""
        reshaped = combo.reshape(-1, self.node_size)
        node_usage = reshaped.sum(axis=1)
        return bool(np.all(node_usage <= self.node_size))

    def _calculate_dual_super_combo_capacity(
        self, 
        combo1: np.ndarray, 
        nodes1: set, 
        combo2: np.ndarray, 
        nodes2: set
    ) -> Optional[float]:
        """Calculate the bidirectional super combo capacity between two cross-node tasks and return the minimum.
        
        For two cross-node tasks, compute:
        1. From combo1's perspective: combo1 + project(combo2, nodes1)
        2. From combo2's perspective: combo2 + project(combo1, nodes2)
        Return the minimum of the bandwidth predictions for these two super combos.
        
        Example:
            Job X = [6,4,0,2] (nodes {0,1,3}), Job Y = [0,2,2,2] (nodes {1,2,3})
            Shared nodes: {1,3}
            
            From X's perspective:
                project(Y, {0,1,3}) = [0,2,0,2]
                super_X = [6,4,0,2] + [0,2,0,2] = [6,6,0,4]
                capacity1 = f([6,6,0,4])
            
            From Y's perspective:
                project(X, {1,2,3}) = [0,4,0,2]
                super_Y = [0,2,2,2] + [0,4,0,2] = [0,6,2,4]
                capacity2 = f([0,6,2,4])
            
            Return: min(capacity1, capacity2)
        
        Args:
            combo1: GPU combination of the first task
            nodes1: Set of node indices involved in the first task
            combo2: GPU combination of the second task
            nodes2: Set of node indices involved in the second task
        
        Returns:
            The minimum of the bidirectional super combo capacities; returns None if the super combo
            in either direction exceeds the per-node capacity on any node (indicating that the GPU
            combination is physically infeasible for parallel execution, and the caller should skip it)
        """
        # From combo1’s perspective: combo1 + project(combo2, nodes1)
        projection1 = self._project_combo_to_nodes(combo2, nodes1)
        super_combo1 = combo1 + projection1
        if not self._is_super_combo_feasible(super_combo1):
            return None
        canonical_super_combo1 = self._canonicalize_combo(super_combo1)
        capacity1 = self._predict_combo_bandwidth(canonical_super_combo1)
        
        # From combo2’s perspective: combo2 + project(combo1, nodes2)
        projection2 = self._project_combo_to_nodes(combo1, nodes2)
        super_combo2 = combo2 + projection2
        if not self._is_super_combo_feasible(super_combo2):
            return None
        canonical_super_combo2 = self._canonicalize_combo(super_combo2)
        capacity2 = self._predict_combo_bandwidth(canonical_super_combo2)
        
        # Use the minimum of the two directional capacities as the effective bottleneck.
        return min(capacity1, capacity2)

    def predict_with_contention(
        self,
        candidate_combo: np.ndarray,
        job_id: Optional[int] = None,
    ) -> float:
        """
        Probe interface: Predicts the bandwidth that the candidate combination would obtain if admitted.
        Implementation logic:
        1. Determine whether the candidate combination spans multiple nodes.
        2. If the candidate combination is single-node, directly return the standalone bandwidth (single-node tasks do not contend with other tasks).
        3. If the candidate combination is cross-node, only consider other cross-node active tasks for contention calculation.
        4. For each cross-node task that shares nodes with the candidate task, compute the bidirectional super combo capacity:
           - From the candidate task's perspective: candidate + project(job, candidate_nodes)
           - From the existing task's perspective: job + project(candidate, job_nodes)
           - Take the minimum of these two capacities
           - If the demand on any single node in either direction exceeds 8 GPUs, skip this GPU set (considered infeasible for concurrent execution)
        5. The final bottleneck capacity = the minimum of all bidirectional capacities
        6. Total demand = aggregated candidate demand + demand bandwidth of all related tasks
           - New tasks are always modeled with exclusive/peak demand (upper bound bandwidth)
           - Existing tasks use randomly sampled 25%~75% peak bandwidth in "common" mode,
             and exclusive bandwidth in other modes
        7. If total demand > bottleneck, allocate proportionally.
        When contention_mode="idle", this function directly returns the standalone bandwidth.
        """
        profiler = _contention_profiler_ctx.get()
        start_time = time.perf_counter() if profiler is not None else None
        try:
            candidate_combo = self._validate_combo_request(candidate_combo, enforce_availability=True)
            resolved_job_id = self._resolve_job_id(job_id)

            # Calculate the standalone bandwidth and demand bandwidth of the candidate
            candidate_standalone = self._predict_combo_bandwidth(candidate_combo)
            if self._is_common_mode():
                # In common mode, the candidate job must use the upper bound bandwidth as demand,
                # so that the output final_bw also has the upper bound semantics.
                # The random occupancy is only written into state in the allocate_job phase.
                candidate_demand = candidate_standalone
                if __debug__:
                    assert abs(candidate_demand - candidate_standalone) < 1e-9, (
                        "In common mode, the demand of the candidate job must be equal to the standalone bandwidth"
                    )
            else:
                candidate_demand = self._derive_effective_demand(
                    candidate_standalone, resolved_job_id
                )

            if not self._should_apply_contention():
                # In idle mode, tasks are assumed to be staggered, and no contention occurs
                return candidate_standalone
            
            # 1. Determine whether the candidate combination spans multiple nodes
            if not self._is_cross_node_combo(candidate_combo):
                # Single-node tasks do not contend with other tasks
                return candidate_standalone
            
            # 2. The candidate combination is cross-node, need to check contention with other cross-node tasks
            # Determine the nodes involved in the candidate task
            candidate_nodes = self._get_nodes_for_combo(candidate_combo)
            
            if not candidate_nodes:
                return 0.0

            # 3. Collect all cross-node tasks that share nodes with the candidate task, and compute the bidirectional super combo capacity
            # For each related task, compute the bidirectional capacity between the candidate task and the related task, and take the minimum
            dual_capacities = []
            existing_demands = 0.0
            
            # Only consider cross-node active tasks
            for job in self.active_jobs:
                job_combo = job['combo']
                # Only process cross-node tasks
                if not self._is_cross_node_combo(job_combo):
                    continue
                
                # check if there are shared nodes
                job_nodes = self._get_nodes_for_combo(job_combo)
                
                # If there are shared nodes, participate in contention calculation
                if candidate_nodes & job_nodes:  # intersection is not empty
                    # Compute the bidirectional super combo capacity between the candidate task and the related task
                    dual_cap = self._calculate_dual_super_combo_capacity(
                        candidate_combo, candidate_nodes,
                        job_combo, job_nodes
                    )
                    if dual_cap is None:
                        # The super combo is already over-provisioned on some node, so the GPU set does not produce effective contention
                        continue
                    dual_capacities.append(dual_cap)
                    # Use the demand bandwidth of the task in the current mode
                    existing_demands += self._get_job_demand(job)

            # 4. Calculate the bottleneck bandwidth: take the minimum of all bidirectional capacities
            if not dual_capacities:
                # No other related tasks
                return candidate_standalone
            
            max_capacity = min(dual_capacities)
            
            # 5. Contention determination
            total_demand = candidate_demand + existing_demands
            
            if total_demand <= max_capacity:
                # Not reached the bottleneck
                return candidate_standalone
            else:
                # Reached the bottleneck, allocate proportionally
                ratio = candidate_demand / total_demand
                allocated_bw = max_capacity * ratio
                return min(candidate_standalone, allocated_bw)
        finally:
            if profiler is not None and start_time is not None:
                profiler.add(time.perf_counter() - start_time)

    def allocate_job(self, job_id: int, combo: np.ndarray) -> float:
        """
        Commit interface: formally allocate GPUs to a job, update state, and adjust the bandwidth of affected historical jobs.
        
        According to the requirements, only cross-node jobs interfere with each other. Single-node jobs do not affect the bandwidth of other jobs.
        When contention_mode="common", each job randomly generates an occupancy bandwidth of 25%~75% upon joining,
        which is stored in the state as "background traffic of allocated jobs"; the new job still uses
        the upper-bound bandwidth (standalone_bw) as its demand in the current contention calculation. In "idle" mode, jobs always maintain exclusive bandwidth.
        Contention recalculation only constructs super combos pair-by-pair between the new job and other cross-node jobs;
        if either direction exceeds the 8-card capacity on a single node, that pair of jobs is skipped, maintaining consistency with the probe phase.
        """
        combo = self._validate_combo_request(combo, enforce_availability=True)

        # 1. Register the new task
        standalone_bw = self._predict_combo_bandwidth(combo)
        occupancy_bw = self._derive_effective_demand(standalone_bw, job_id)  # common mode once samples the occupancy bandwidth
        self.allocated_gpu_mask += combo
        
        new_job = {
            'job_id': job_id,
            'combo': combo,
            'standalone_bw': standalone_bw,
            'occupancy_bw': occupancy_bw,
            'current_bw': standalone_bw,
            'history': []
        }
        self.active_jobs.append(new_job)
        
        if not self._should_apply_contention():
            # In idle mode, jobs always maintain exclusive bandwidth
            for job in self.active_jobs:
                job['history'].append(job['current_bw'])
            return standalone_bw

        # 2. Determine whether the new task spans multiple nodes
        if not self._is_cross_node_combo(combo):
            # Single-node tasks do not contend with other tasks
            # Do not update other tasks, record history and return
            for job in self.active_jobs:
                job['history'].append(job['current_bw'])
            return standalone_bw
        
        # 3. The new task is cross-node, need to check contention with other cross-node tasks
        # Global recalculation (handle contention)
        # The logic here is that the addition of the new task may cause the bandwidth of other cross-node tasks with shared nodes to decrease (correction).
        # According to the requirements: correction value = min(original recorded value, new allocation value)
        
        # Determine the nodes involved in the new task
        new_job_nodes = self._get_nodes_for_combo(combo)
                
        if not new_job_nodes:
            # Record history and return
            for job in self.active_jobs:
                job['history'].append(job['current_bw'])
            return 0.0
            
        # Collect all cross-node tasks that share nodes with the new task (excluding the new task itself)
        # For each related task, compute the bidirectional super combo capacity between the new task and the related task
        clashing_jobs = []  # List of job_dict (excluding the new task itself)
        dual_capacities = []
        if self._is_common_mode():
            # The new task uses the peak demand in common mode, and the random occupancy is only written into new_job['occupancy_bw']
            new_job_demand = new_job['standalone_bw']
            if __debug__:
                assert abs(new_job_demand - new_job['standalone_bw']) < 1e-9, (
                    "In common mode, the demand of the new task must be equal to the standalone bandwidth"
                )
        else:
            new_job_demand = self._get_job_demand(new_job)
        
        for job in self.active_jobs:
            if job is new_job:
                continue
            job_combo = job['combo']
            # Only process cross-node tasks
            if not self._is_cross_node_combo(job_combo):
                continue
            
            # Check if there are shared nodes
            job_nodes = self._get_nodes_for_combo(job_combo)
            
            # If there are shared nodes, participate in contention calculation
            if new_job_nodes & job_nodes:  # intersection is not empty
                # Compute the bidirectional super combo capacity between the new task and the related task
                dual_cap = self._calculate_dual_super_combo_capacity(
                    combo, new_job_nodes,
                    job_combo, job_nodes
                )
                if dual_cap is None:
                    continue
                dual_capacities.append(dual_cap)
                clashing_jobs.append(job)
        
        # Calculate the bottleneck: take the minimum of all bidirectional capacities
        if not dual_capacities:
            # No other related tasks, the new task maintains exclusive bandwidth
            new_job['current_bw'] = standalone_bw
            for job in self.active_jobs:
                job['history'].append(job['current_bw'])
            return standalone_bw
        
        max_capacity = min(dual_capacities)
        total_demand = new_job_demand + sum(self._get_job_demand(job) for job in clashing_jobs)
        
        # Update the bandwidth
        if total_demand > max_capacity:
            for job in clashing_jobs:
                job_demand = self._get_job_demand(job)
                ratio = job_demand / total_demand
                allocated_part = max_capacity * ratio
                job['current_bw'] = min(job['standalone_bw'], allocated_part)
            new_ratio = new_job_demand / total_demand if total_demand > 0 else 0.0
            new_allocated = max_capacity * new_ratio
            new_job['current_bw'] = min(new_job['standalone_bw'], new_allocated)

        else:
            # Not reached the bottleneck, the related tasks maintain exclusive bandwidth
            for job in clashing_jobs:
                job['current_bw'] = job['standalone_bw']
            new_job['current_bw'] = new_job['standalone_bw']

        # Record history
        for job in self.active_jobs:
            job['history'].append(job['current_bw'])
            
        return self.active_jobs[-1]['current_bw']

    def get_state_summary(self):
        """Get the current cluster state summary"""
        return [
            {
                'job_id': j['job_id'], 
                'bw': j['current_bw'], 
                'gpu_count': int(np.sum(j['combo']))
            } 
            for j in self.active_jobs
        ]