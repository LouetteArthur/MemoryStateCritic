#!/usr/bin/env bash
# =============================================================================
# Reproduce Experiment 1 of
#   "Memory-State Critic for Asymmetric Actor-Critic with Application to
#    Vision-Based Pursuit-Evasion" (EWRL 2026)
#
# Runs the exact critic x arena x seed grid behind Figures 3 and 4: 32 runs at
# 512 envs x 100K timesteps = 5.12e7 environment steps each.
#
#   critic  paper symbol  open-arena seeds     wall-arena seeds
#   ------  ------------  -------------------  -------------------
#   Vs      V(s)          1, 5, 7, 42, 123     1, 5, 15, 42, 123
#   Vsh     V(s, z^c)     1, 5, 7, 42, 123     1, 5, 15, 42, 123
#   Vsz     V(s, z^a)     1, 7, 15, 27, 42     1, 5, 15, 42, 123
#   Vsoa    V(s, o, a)    1, 42, 123           1, 42, 123
#
# The seed sets are recorded exactly as they were run. The open-arena Vsz set
# differs from its Vs / Vsh siblings; this is a fact about how the sweep was
# executed, not a deliberate design choice. V(s,o,a) has three seeds per arena
# rather than five, as stated in Section 4.3 of the paper.
#
# Budget: ~5.5 h per run on an RTX 4090 at 512 envs, so ~7 GPU-days in total.
# Runs are sequential; split the grid across machines with --critics / --arena.
#
# Usage:
#   ./scripts/reproduce_paper.sh --dry-run          # print the 32 commands
#   ./scripts/reproduce_paper.sh                    # run everything
#   ./scripts/reproduce_paper.sh --arena open       # open arena only (18 runs)
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
GRID=(
    "Vs|open|1 5 7 42 123"
    "Vsh|open|1 5 7 42 123"
    "Vsz|open|1 7 15 27 42"
    "Vsoa|open|1 42 123"
    "Vs|wall|1 5 15 42 123"
    "Vsh|wall|1 5 15 42 123"
    "Vsz|wall|1 5 15 42 123"
    "Vsoa|wall|1 42 123"
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
