#!/bin/bash
# =============================================================================
# Robust ablation wrapper: auto-restarts run_ablation.sh after kernel oops or
# any non-zero exit, until the sweep emits "=== Critic Ablation complete ===".
# Logs to ~/logs/ablation_<timestamp>.log (persistent, survives reboots).
#
# Completed runs are tracked via marker files under $ABLATION_DONE_DIR
# (default ~/logs/ablation_done), so on restart the inner script skips runs
# that already finished and resumes from where it left off.
#
# Usage (env vars override defaults):
#   NUM_ENVS=512 SEEDS="42 123" TOTAL_FRAMES=51200000 \
#     ./scripts/run_ablation_robust.sh
#
#   # to reset completed-run state and start fresh:
#   rm -rf ~/logs/ablation_done && ./scripts/run_ablation_robust.sh
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$SCRIPT_DIR/run_ablation.sh"

LOG_DIR="${ABLATION_LOG_DIR:-$HOME/logs}"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/ablation_$TS.log"

MAX_RESTARTS="${MAX_RESTARTS:-20}"
RESTART_DELAY="${RESTART_DELAY:-60}"

attempt=1
echo "[$(date +%H:%M:%S)] === Robust ablation wrapper starting, log=$LOG ===" | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] env: NUM_ENVS=${NUM_ENVS:-default} SEEDS=${SEEDS:-default} TOTAL_FRAMES=${TOTAL_FRAMES:-default}" | tee -a "$LOG"
echo "[$(date +%H:%M:%S)] completed-run markers: ${ABLATION_DONE_DIR:-$HOME/logs/ablation_done}" | tee -a "$LOG"

while (( attempt <= MAX_RESTARTS )); do
    echo "" | tee -a "$LOG"
    echo "[$(date +%H:%M:%S)] ===== ATTEMPT $attempt/$MAX_RESTARTS =====" | tee -a "$LOG"
    "$INNER" "$@" 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    if tail -10 "$LOG" | grep -q "=== Critic Ablation complete"; then
        echo "[$(date +%H:%M:%S)] === sweep finished successfully on attempt $attempt ===" | tee -a "$LOG"
        exit 0
    fi
    echo "[$(date +%H:%M:%S)] inner script exited rc=$rc without 'complete' marker" | tee -a "$LOG"
    echo "[$(date +%H:%M:%S)] sleeping ${RESTART_DELAY}s before restart..." | tee -a "$LOG"
    sleep "$RESTART_DELAY"
    attempt=$((attempt + 1))
done

echo "[$(date +%H:%M:%S)] === max restart attempts ($MAX_RESTARTS) reached, giving up ===" | tee -a "$LOG"
exit 1
