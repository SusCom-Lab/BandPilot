#!/usr/bin/env bash
# Nested/cumulative sensitivity rerun pipeline.
#
# The script regenerates predictor-level and dispatch-level sensitivity
# artifacts, checks the dispatch acceptance gate, and escalates to a larger
# rerun protocol when required.
#
# Usage:
#   bash evaluation/sensitivity-analysis/run_nested_cumulative_rerun.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG_PATH="${CONFIG_PATH:-config/default_config.yaml}"
MASTER_SEED="${MASTER_SEED:-42}"
CASE_SEED="${CASE_SEED:-1111}"
SAMPLE_SIZES="${SAMPLE_SIZES:-100,250,500}"
STRATEGIES="${STRATEGIES:-Random,Stratified,Worst-Case}"
CONTENTION_MODES="${CONTENTION_MODES:-idle,common,intensive}"
K_VALUES="${K_VALUES:-4,8,12,16,20,24,28}"
REPEAT_INDICES="${REPEAT_INDICES:-0,1,2,3,4,5,6,7,8,9}"

PREDICTOR_FIRST_PASS_DIR="evaluation/sensitivity-analysis/artifacts/predictor-level_nested_ms${MASTER_SEED}_5seed_100-250-500"
DISPATCH_FIRST_PASS_DIR="evaluation/sensitivity-analysis/artifacts/dispatch_sidecar/1111RS_5seed_10RN_4-8-12-16-20-24-28K_idle-common-intensiveCM_nested_ms${MASTER_SEED}"
PREDICTOR_SECOND_PASS_DIR="evaluation/sensitivity-analysis/artifacts/predictor-level_nested_ms${MASTER_SEED}_10seed_100-250-500"
DISPATCH_SECOND_PASS_DIR="evaluation/sensitivity-analysis/artifacts/dispatch_sidecar/1111RS_10seed_10RN_4-8-12-16-20-24-28K_idle-common-intensiveCM_nested_ms${MASTER_SEED}"

echo "[NestedSensitivity] First pass: predictor-level (5 seeds)"
conda run -n gpu_dp_opt python -m training.sample_sensitivity_experiment \
  --config "${CONFIG_PATH}" \
  --sample-sizes "${SAMPLE_SIZES}" \
  --num-seeds 5 \
  --master-seed "${MASTER_SEED}" \
  --sampling-protocol nested \
  --output-dir "${PREDICTOR_FIRST_PASS_DIR}"

echo "[NestedSensitivity] First pass: dispatch-level (5 seeds, 10 repeats)"
conda run -n gpu_dp_opt python -m training.sample_sensitivity_dispatch_experiment \
  --config "${CONFIG_PATH}" \
  --sample-sizes "${SAMPLE_SIZES}" \
  --strategies "${STRATEGIES}" \
  --num-seeds 5 \
  --master-seed "${MASTER_SEED}" \
  --sampling-protocol nested \
  --contention-modes "${CONTENTION_MODES}" \
  --k-values "${K_VALUES}" \
  --repeat-indices "${REPEAT_INDICES}" \
  --case-seed "${CASE_SEED}" \
  --output-dir "${DISPATCH_FIRST_PASS_DIR}"

echo "[NestedSensitivity] Checking acceptance gate after first pass"
if conda run -n gpu_dp_opt python evaluation/sensitivity-analysis/check_nested_acceptance.py \
  --summary "${DISPATCH_FIRST_PASS_DIR}/summary.csv"; then
  echo "[NestedSensitivity] First pass already satisfies the acceptance gate."
  exit 0
fi

echo "[NestedSensitivity] Acceptance gate failed; escalating to 10 seeds + 10 repeats"
conda run -n gpu_dp_opt python -m training.sample_sensitivity_experiment \
  --config "${CONFIG_PATH}" \
  --sample-sizes "${SAMPLE_SIZES}" \
  --num-seeds 10 \
  --master-seed "${MASTER_SEED}" \
  --sampling-protocol nested \
  --output-dir "${PREDICTOR_SECOND_PASS_DIR}"

conda run -n gpu_dp_opt python -m training.sample_sensitivity_dispatch_experiment \
  --config "${CONFIG_PATH}" \
  --sample-sizes "${SAMPLE_SIZES}" \
  --strategies "${STRATEGIES}" \
  --num-seeds 10 \
  --master-seed "${MASTER_SEED}" \
  --sampling-protocol nested \
  --contention-modes "${CONTENTION_MODES}" \
  --k-values "${K_VALUES}" \
  --repeat-indices "${REPEAT_INDICES}" \
  --case-seed "${CASE_SEED}" \
  --output-dir "${DISPATCH_SECOND_PASS_DIR}"

echo "[NestedSensitivity] Final acceptance check (10 seeds)"
conda run -n gpu_dp_opt python evaluation/sensitivity-analysis/check_nested_acceptance.py \
  --summary "${DISPATCH_SECOND_PASS_DIR}/summary.csv"
