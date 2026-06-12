#!/bin/bash
# =============================================================================
# Critic Ablation — 5 critics × 2 arenas (Experiment 1)
#
# All actors: CNN+GRU with sensor="both" (depth + segmap), 1 past action.
#
# Critic variants:
#   1. Vs       — V(s):     MLP on privileged state
#   2. Vsz      — V(s,z):   MLP on state + detached actor GRU hidden z
#   3. Vsh      — V(s,h):   Critic-side CNN+GRU on (image, past_actions) + state
#   4. Vo       — V(o,a):   CNN critic on same observation as actor (symmetric)
#   5. Vsoa     — V(s,o,a): Dict critic on state + image + past_actions (Geles et al.)
#
# Arenas:      open, wall
# Total:       5 × 2 = 10 experiments per seed
#
# Usage:
#   ./scripts/run_ablation.sh                    # run all 10 configs
#   ./scripts/run_ablation.sh --dry-run          # print commands without running
#   ./scripts/run_ablation.sh --seeds "42 123"   # multiple seeds
#   ./scripts/run_ablation.sh --arena open       # open map only (5 runs/seed)
#   ./scripts/run_ablation.sh --arena wall       # wall map only (5 runs/seed)
#   NUM_ENVS=256 ./scripts/run_ablation.sh       # for 12 GB GPUs
# =============================================================================
set -euo pipefail

# Defaults
SEEDS="${SEEDS:-42}"
NUM_ENVS="${NUM_ENVS:-1024}"
TOTAL_FRAMES="${TOTAL_FRAMES:-102400000}"   # 400K timesteps × 256 envs
PER_RUN_TIMEOUT="${PER_RUN_TIMEOUT:-32400}" # 9 h per run; PXR/USD races hang Isaac Sim periodically
TASK="Ablation-vision-vs-trajectories"
WANDB_PROJECT="${WANDB_PROJECT:-critic_ablation}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
DRY_RUN=false
ARENA_FILTER=""  # "" = both, "open" or "wall"
CRITIC_FILTER=""  # "" = all 5, else space-separated subset of {Vs Vsz Vsh Vo Vsoa}

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --seeds=*) SEEDS="${1#*=}"; shift ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --arena=*) ARENA_FILTER="${1#*=}"; shift ;;
        --arena) ARENA_FILTER="$2"; shift 2 ;;
        --critics=*) CRITIC_FILTER="${1#*=}"; shift ;;
        --critics) CRITIC_FILTER="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Critic variants: (label, --agent entry point, extra_flags)
# All actors are CNN+GRU with sensor=both and 1 past action.
# Vsoa (Geles) requires --unbiased-critic for Dict critic state.
ALL_CRITICS=(
    "Vs|skrl_ppo_vision_rnn_cfg_entry_point|"
    "Vsz|skrl_ppo_vision_rnn_sz_cfg_entry_point|"
    "Vsh|skrl_ppo_vision_rnn_sh_cfg_entry_point|"
    "Vo|skrl_ppo_vision_rnn_symmetric_cfg_entry_point|"
    "Vsoa|skrl_ppo_vision_rnn_geles_cfg_entry_point|--unbiased-critic"
)

# Apply optional --critics filter (e.g. --critics "Vs Vsz Vsh")
if [ -n "$CRITIC_FILTER" ]; then
    CRITICS=()
    for spec in "${ALL_CRITICS[@]}"; do
        IFS='|' read -r label _ _ <<< "$spec"
        for keep in $CRITIC_FILTER; do
            if [ "$label" = "$keep" ]; then
                CRITICS+=("$spec")
                break
            fi
        done
    done
    if [ ${#CRITICS[@]} -eq 0 ]; then
        echo "ERROR: --critics='$CRITIC_FILTER' matched none of: Vs Vsz Vsh Vo Vsoa" >&2
        exit 1
    fi
else
    CRITICS=("${ALL_CRITICS[@]}")
fi

SENSOR="both"

# Arena configs: (label, extra_flags)
# Wall arena uses gamma=0.999 (longer credit-assignment horizon needed for
# obstacle-aware planning — captures take 0.6s but require route choice
# around the wall, which the agent cannot value-bootstrap under gamma=0.99).
# Open arena uses the YAML default gamma=0.99 (~50-step typical catch, tight
# horizon gives cleaner PPO value learning — gamma>=0.995 broke Vsz here).
if [ "$ARENA_FILTER" = "open" ]; then
    ARENAS=("open|")
elif [ "$ARENA_FILTER" = "wall" ]; then
    ARENAS=("wall|--enable-obstacles --discount-factor=0.999")
else
    ARENAS=("wall|--enable-obstacles --discount-factor=0.999" "open|")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/skrl/train.py"

count=0
total=0

for seed in $SEEDS; do
    for critic_spec in "${CRITICS[@]}"; do
        IFS='|' read -r label entry_point critic_flags <<< "$critic_spec"
        for arena_spec in "${ARENAS[@]}"; do
            total=$((total + 1))
        done
    done
done

echo "=== Critic Ablation: $total runs (seeds: $SEEDS) ==="
echo ""

for seed in $SEEDS; do
    for critic_spec in "${CRITICS[@]}"; do
        IFS='|' read -r label entry_point critic_flags <<< "$critic_spec"
        for arena_spec in "${ARENAS[@]}"; do
            IFS='|' read -r arena_label arena_flags <<< "$arena_spec"
            count=$((count + 1))
            NAME="${label}_${arena_label}_s${seed}"

            EXTRA_ARGS="${critic_flags:-} ${arena_flags:-}"

            WANDB_ARGS="--wandb-project=$WANDB_PROJECT --wandb-name=$NAME"
            if [ -n "$WANDB_ENTITY" ]; then
                WANDB_ARGS="$WANDB_ARGS --wandb-entity=$WANDB_ENTITY"
            fi

            CMD="${PYTHONEXE:-python} $TRAIN_SCRIPT \
    --task=$TASK \
    --agent=$entry_point \
    --sensor-mode=$SENSOR \
    --num-past-actions=1 \
    --seed=$seed \
    --num_envs=$NUM_ENVS \
    --total_frames=$TOTAL_FRAMES \
    --headless --enable_cameras \
    $WANDB_ARGS \
    $EXTRA_ARGS"

            echo "[$count/$total] $NAME"
            # Resume support: skip if a completion marker exists. Markers are written
            # under ABLATION_DONE_DIR after a successful run, so a sweep restarted
            # after a kernel oops resumes where it left off instead of redoing all runs.
            DONE_DIR="${ABLATION_DONE_DIR:-$HOME/logs/ablation_done}"
            if [ -f "$DONE_DIR/$NAME.done" ]; then
                echo "  [SKIP] $NAME already complete (marker at $DONE_DIR/$NAME.done)"
                echo ""
                continue
            fi
            if [ "$DRY_RUN" = true ]; then
                echo "  $CMD"
                echo ""
            else
                # Per-run watchdog: if the run hangs (PXR/USD race) `timeout` kills
                # it after PER_RUN_TIMEOUT seconds; `|| rc=$?` keeps the sweep going.
                # Fall back to plain eval if `timeout` isn't available (minimal
                # containers without coreutils).
                rc=0
                if command -v timeout >/dev/null 2>&1; then
                    timeout --kill-after=60 "$PER_RUN_TIMEOUT" bash -c "$CMD" || rc=$?
                else
                    echo "[WARN] 'timeout' not found in PATH; running without watchdog"
                    eval "$CMD" || rc=$?
                fi
                if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
                    echo "[WARN] $NAME timed out after ${PER_RUN_TIMEOUT}s — moving on"
                elif [ "$rc" -eq 127 ]; then
                    echo "[WARN] $NAME exited with rc=127 (command not found). Check that 'python' and 'timeout' resolve inside the venv/container. — moving on"
                elif [ "$rc" -ne 0 ]; then
                    echo "[WARN] $NAME exited with rc=$rc — moving on"
                else
                    mkdir -p "$DONE_DIR"
                    touch "$DONE_DIR/$NAME.done"
                    echo "  [DONE] marker written to $DONE_DIR/$NAME.done"
                fi
                echo ""
            fi
        done
    done
done

echo "=== Critic Ablation complete: $count/$total runs ==="
