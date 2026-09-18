#!/usr/bin/env bash
# =============================================================================
# Reproduce Experiment 1 of
#   "Memory-State Critic for Asymmetric Actor-Critic with Application to
#    Vision-Based Pursuit-Evasion" (EWRL 2026)
#
# Runs the exact critic x arena x seed grid behind Figures 3 and 4: 160 runs at
# 512 envs x 100K timesteps = 5.12e7 environment steps each.
#
#   critic  paper symbol   seeds (identical in both arenas)
#   ------  -------------  ---------------------------------------------------
#   Vs      V(s)           1 2 3 4 5 6 7 8 9 11 12 13 15 17 19 21 27 37 42 123
#   Vsh     V(s, z^c)      (same)
#   Vsz     V(s, z^a)      (same)
#   Vsoa    V(s, o, a)     (same)
#
# Every cell runs the same twenty seeds, so the comparison between critics is
# paired. The May 2026 submission used a smaller, unbalanced grid; the
# camera-ready replaces it entirely.
#
# Budget: ~5.5 h per run on an RTX 4090 at 512 envs, so ~37 GPU-days in total.
# Runs are sequential; split the grid across machines with --critics / --arena.
#
# Usage:
#   ./scripts/reproduce_paper.sh --dry-run          # print the 160 commands
#   ./scripts/reproduce_paper.sh                    # run everything
#   ./scripts/reproduce_paper.sh --arena open       # open arena only (80 runs)
#   ./scripts/reproduce_paper.sh --critics "Vsz Vsh"
#
# Environment:
#   WANDB_ENTITY    your wandb entity (required for logging; unset = anonymous)
#   WANDB_PROJECT   defaults to "critic_ablation"
#   NUM_ENVS        defaults to 512 (use 256 on 12 GB GPUs; results will differ)
#   ABLATION_DONE_DIR  marker dir for resuming an interrupted sweep
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ABLATION="$REPO_ROOT/scripts/run_ablation.sh"

DRY_RUN=""
ARENA_FILTER=""
CRITIC_FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        --arena=*) ARENA_FILTER="${1#*=}"; shift ;;
        --arena) ARENA_FILTER="$2"; shift 2 ;;
        --critics=*) CRITIC_FILTER="${1#*=}"; shift ;;
        --critics) CRITIC_FILTER="$2"; shift 2 ;;
        -h|--help) sed -n '2,37p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

export WANDB_PROJECT="${WANDB_PROJECT:-critic_ablation}"

# (critic, arena, seeds) — the grid exactly as it was run for the paper.
# Every cell runs the same twenty seeds: 4 critics x 2 arenas x 20 = 160 runs.
PAPER_SEEDS="1 2 3 4 5 6 7 8 9 11 12 13 15 17 19 21 27 37 42 123"
GRID=(
    "Vs|open|$PAPER_SEEDS"
    "Vsh|open|$PAPER_SEEDS"
    "Vsz|open|$PAPER_SEEDS"
    "Vsoa|open|$PAPER_SEEDS"
    "Vs|wall|$PAPER_SEEDS"
    "Vsh|wall|$PAPER_SEEDS"
    "Vsz|wall|$PAPER_SEEDS"
    "Vsoa|wall|$PAPER_SEEDS"
)

_wanted() {  # _wanted <value> <space-separated filter>; empty filter = keep all
    local value="$1" filter="$2" item
    [ -z "$filter" ] && return 0
    for item in $filter; do
        [ "$item" = "$value" ] && return 0
    done
    return 1
}

for entry in "${GRID[@]}"; do
    IFS='|' read -r critic arena seeds <<< "$entry"
    _wanted "$arena" "$ARENA_FILTER" || continue
    _wanted "$critic" "$CRITIC_FILTER" || continue

    echo "=== ${critic} / ${arena} / seeds: ${seeds} ==="
    "$ABLATION" --critics "$critic" --arena "$arena" --seeds "$seeds" ${DRY_RUN}
done

echo "=== reproduce_paper.sh complete ==="
echo "Next: regenerate the figures with"
echo "  python scripts/plot_ablation_results.py --paper \\"
echo "      --entity \"\$WANDB_ENTITY\" --project \"$WANDB_PROJECT\" \\"
echo "      --output-dir figures/reproduced"
