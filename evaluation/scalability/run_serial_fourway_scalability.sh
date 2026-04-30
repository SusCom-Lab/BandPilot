#!/usr/bin/env bash
# Purpose:
# - Run the reviewer-facing scalability benchmark in four serial stages:
#   1. H100 default
#   2. H100 aggressive
#   3. Het-4Mix default
#   4. Het-4Mix aggressive
# - Keep the launcher usable under `nohup`, so a dropped SSH session does not
#   stop the benchmark.
# - Persist per-stage state files so an interrupted sequence can be resumed.
# - Support manual restart from a later stage with `--start-from h100_aggressive`.
#
# Stage semantics:
# - each stage has its own run directory and logs;
# - pass `--foreground` to run the benchmark process in the current shell;
# - state files use:
#   - `running`: stage process has been launched;
#   - `done`: stage completed successfully;
#   - `failed`: stage exited with a non-zero code;
# - completed stages are skipped on resume;
# - a stale `runner.pid` with `running` state is treated as failed before retry.
#
# Usage:
# 1. Launch the full sequence in background-friendly mode:
#    `bash evaluation/scalability/run_serial_fourway_scalability.sh`
# 2. Resume from a specific stage:
#    `bash evaluation/scalability/run_serial_fourway_scalability.sh --start-from h100_aggressive`
# 3. Reset one stage before running:
#    `bash evaluation/scalability/run_serial_fourway_scalability.sh --reset-stage h100_aggressive`
# 4. Follow launcher progress:
#    `tail -f evaluation/scalability/artifacts/benchmark/current/serial_fourway_runner/launcher.log`
# 5. Follow a stage log directly:
#    `tail -f evaluation/scalability/artifacts/benchmark/current/serial_fourway_runner/h100_default/h100_default.log`
# 6. Inspect the launcher process:
#    `ps -fp "$(cat evaluation/scalability/artifacts/benchmark/current/serial_fourway_runner/runner.pid)"`
#
# Notes:
# - use this script for long-running reviewer-facing benchmarks, not smoke tests;
# - The benchmark runs `BandPilot` in Tier 1 / Tier 2 with a fresh runtime bank.
# - per-stage case-level cache files are intentionally preserved for resume.
# - use `--start-from <stage>` together with `--reset-stage <stage>` when a
#   single stage should be regenerated.
# - commands run through `conda run -n gpu_dp_opt`; update `PYTHON_CMD` below if
#   a different environment is required.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
RUN_ROOT="${REPO_ROOT}/evaluation/scalability/artifacts/benchmark/current/serial_fourway_runner"
LAUNCHER_LOG="${RUN_ROOT}/launcher.log"
RUNNER_PID_FILE="${RUN_ROOT}/runner.pid"

mkdir -p "${RUN_ROOT}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

CURRENT_STAGE=""
FOREGROUND_MODE=0
START_FROM_STAGE=""
RESET_STAGE=""

STAGE_ORDER=(
  "h100_default"
  "h100_aggressive"
  "het4mix_default"
  "het4mix_aggressive"
)

stage_config_path() {
  case "$1" in
    h100_default) echo "evaluation/scalability/configs/h100_default_refresh.yaml" ;;
    h100_aggressive) echo "evaluation/scalability/configs/h100_aggressive_sidecar.yaml" ;;
    het4mix_default) echo "evaluation/scalability/configs/het4mix_default_refresh.yaml" ;;
    het4mix_aggressive) echo "evaluation/scalability/configs/het4mix_aggressive_sidecar.yaml" ;;
    *) return 1 ;;
  esac
}

stage_index() {
  local target="$1"
  local idx=0
  for stage in "${STAGE_ORDER[@]}"; do
    if [[ "${stage}" == "${target}" ]]; then
      printf '%s\n' "${idx}"
      return 0
    fi
    idx=$((idx + 1))
  done
  return 1
}

validate_stage_name() {
  local stage="$1"
  if ! stage_index "${stage}" > /dev/null 2>&1; then
    printf 'Unknown stage: %s\n' "${stage}" >&2
    printf 'Valid stages: %s\n' "${STAGE_ORDER[*]}" >&2
    exit 2
  fi
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --foreground)
        FOREGROUND_MODE=1
        shift
        ;;
      --resume)
        shift
        ;;
      --start-from)
        START_FROM_STAGE="${2:-}"
        if [[ -z "${START_FROM_STAGE}" ]]; then
          printf 'Missing value for --start-from\n' >&2
          exit 2
        fi
        validate_stage_name "${START_FROM_STAGE}"
        shift 2
        ;;
      --reset-stage)
        RESET_STAGE="${2:-}"
        if [[ -z "${RESET_STAGE}" ]]; then
          printf 'Missing value for --reset-stage\n' >&2
          exit 2
        fi
        validate_stage_name "${RESET_STAGE}"
        shift 2
        ;;
      *)
        printf 'Unknown argument: %s\n' "$1" >&2
        exit 2
        ;;
    esac
  done
}

mark_failed() {
  local exit_code="$1"
  local stage_id="${CURRENT_STAGE:-unknown}"
  local stage_dir="${RUN_ROOT}/${stage_id}"

  mkdir -p "${stage_dir}"
  rm -f "${stage_dir}/running"
  printf '%s\n' "${exit_code}" > "${stage_dir}/failed"
  rm -f "${RUNNER_PID_FILE}"
  log "Runner aborted | stage=${stage_id} | exit_code=${exit_code}"
}

reset_stage_state() {
  local stage_id="$1"
  local stage_dir="${RUN_ROOT}/${stage_id}"

  mkdir -p "${stage_dir}"
  rm -f "${stage_dir}/done" "${stage_dir}/running" "${stage_dir}/failed"
  log "Reset stage state | stage=${stage_id}"
}

reconcile_stale_state() {
  local stale_pid=""
  if [[ -f "${RUNNER_PID_FILE}" ]]; then
    stale_pid="$(cat "${RUNNER_PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${stale_pid}" ]] && ! ps -p "${stale_pid}" > /dev/null 2>&1; then
      log "Detected stale runner pid | pid=${stale_pid}"
      rm -f "${RUNNER_PID_FILE}"
    fi
  fi

  for stage_id in "${STAGE_ORDER[@]}"; do
    local stage_dir="${RUN_ROOT}/${stage_id}"
    local running_flag="${stage_dir}/running"
    local done_flag="${stage_dir}/done"
    local failed_flag="${stage_dir}/failed"
    if [[ -f "${running_flag}" ]]; then
      rm -f "${running_flag}"
      if [[ ! -f "${done_flag}" ]]; then
        printf '%s\n' "stale_runner" > "${failed_flag}"
        log "Recovered stale running stage -> failed | stage=${stage_id}"
      fi
    fi
  done
}

run_stage() {
  local stage_id="$1"
  local config_path="$2"
  local force_run="${3:-0}"
  local stage_dir="${RUN_ROOT}/${stage_id}"
  local stage_log="${stage_dir}/${stage_id}.log"
  local running_flag="${stage_dir}/running"
  local done_flag="${stage_dir}/done"
  local failed_flag="${stage_dir}/failed"

  CURRENT_STAGE="${stage_id}"
  mkdir -p "${stage_dir}"

  if [[ "${force_run}" == "1" ]]; then
    rm -f "${done_flag}" "${running_flag}" "${failed_flag}"
    log "Force rerun stage | stage=${stage_id}"
  fi

  if [[ -f "${done_flag}" ]]; then
    log "Skip ${stage_id}: done flag already exists"
    return 0
  fi

  rm -f "${failed_flag}"
  printf '%s\n' "$$" > "${running_flag}"

  log "Start ${stage_id} | config=${config_path}"
  (
    cd "${REPO_ROOT}"
    conda run -n gpu_dp_opt python main.py --config "${config_path}"
  ) >> "${stage_log}" 2>&1

  rm -f "${running_flag}"
  touch "${done_flag}"
  log "Finish ${stage_id}"
}

run_all_stages() {
  cd "${REPO_ROOT}"
  printf '%s\n' "$$" > "${RUNNER_PID_FILE}"
  local start_idx=-1
  if [[ -n "${START_FROM_STAGE}" ]]; then
    start_idx="$(stage_index "${START_FROM_STAGE}")"
  fi

  log "Serial four-way runner started | pid=$$ | start_from=${START_FROM_STAGE:-auto}"

  local idx=0
  for stage_id in "${STAGE_ORDER[@]}"; do
    local config_path
    config_path="$(stage_config_path "${stage_id}")"
    if [[ "${start_idx}" -ge 0 && "${idx}" -lt "${start_idx}" ]]; then
      log "Skip ${stage_id}: before requested start stage ${START_FROM_STAGE}"
      idx=$((idx + 1))
      continue
    fi
    if [[ "${start_idx}" -ge 0 ]]; then
      run_stage "${stage_id}" "${config_path}" 1
    else
      run_stage "${stage_id}" "${config_path}" 0
    fi
    idx=$((idx + 1))
  done

  rm -f "${RUNNER_PID_FILE}"
  log "Serial four-way runner completed successfully"
}

main() {
  parse_args "$@"

  if [[ -n "${RESET_STAGE}" ]]; then
    reset_stage_state "${RESET_STAGE}"
    exit 0
  fi

  if [[ "${FOREGROUND_MODE}" != "1" ]]; then
    if [[ -f "${RUNNER_PID_FILE}" ]]; then
      local existing_pid
      existing_pid="$(cat "${RUNNER_PID_FILE}" 2>/dev/null || true)"
      if [[ -n "${existing_pid}" ]] && ps -p "${existing_pid}" > /dev/null 2>&1; then
        printf 'Serial runner already active: pid=%s\n' "${existing_pid}"
        printf 'Launcher log: %s\n' "${LAUNCHER_LOG}"
        exit 0
      fi
    fi

    reconcile_stale_state

    local relay_args=()
    if [[ -n "${START_FROM_STAGE}" ]]; then
      relay_args+=(--start-from "${START_FROM_STAGE}")
    else
      relay_args+=(--resume)
    fi

    nohup bash "${SCRIPT_PATH}" --foreground "${relay_args[@]}" >> "${LAUNCHER_LOG}" 2>&1 &
    local bg_pid=$!
    printf 'Serial four-way runner launched in background: pid=%s\n' "${bg_pid}"
    printf 'Launcher log: %s\n' "${LAUNCHER_LOG}"
    exit 0
  fi

  reconcile_stale_state

  trap 'exit_code=$?; mark_failed "${exit_code}"; exit "${exit_code}"' ERR
  run_all_stages
}

main "$@"
