# Reproducing the paper

This document covers **Experiment 1** of *"Memory-State Critic for Asymmetric
Actor-Critic with Application to Vision-Based Pursuit-Evasion"* (EWRL 2026,
[OpenReview](https://openreview.net/forum?id=iKiJUMvYT7)):
the critic ablation behind Figures 3 and 4 and Table 1.

The population-based self-play pipeline, the Elo tournament and the Crazyswarm
sim-to-real deployment belong to separate work and are not shipped here.

---

## 1. The critics in the code

Every critic runs on the same agent, `PPO_RNN_ASYM`
(`skrl_ext/agents/ppo_rnn_asym.py`). What separates them is the critic model,
selected by the `value.class` field of the agent YAML, and one flag:

| Paper (Table 1) | Symbol | Label | Critic model (`value.class`) | Agent YAML | Extra flag |
|---|---|---|---|---|---|
| State-only | `V(s)` | `state` | `DeterministicMixin` | `skrl_ppo_state_critic_cfg.yaml` | — |
| History-state | `V(s,z^c)` | `history-state` | `HistoryStateCriticMixin` | `skrl_ppo_history_state_critic_cfg.yaml` | — |
| **Memory-state (ours)** | `V(s,z^a)` | `memory-state` | `MemoryStateCriticMixin` | `skrl_ppo_memory_state_critic_cfg.yaml` | — |
| Observation-state | `V(s,o,a)` | `observation-state` | `DeterministicMixin` (Dict input) | `skrl_ppo_observation_state_critic_cfg.yaml` | `--unbiased-critic` |

The `--agent` entry point is the YAML name with `.yaml` replaced by `_entry_point`,
e.g. `skrl_ppo_memory_state_critic_cfg_entry_point`. The label is what
`run_ablation.sh --critics` takes and what wandb run names start with.

- `MemoryStateCriticMixin` (`skrl_ext/models/memory_state_critic.py`) is an MLP
  over `[privileged_state ‖ z^a]`, where the agent detaches `z^a` before passing
  it. That `.detach()` is the stop-gradient of Figure 1; there is no second
  recurrent encoder. The agent enables it with `memory_state_critic: True`.
- `HistoryStateCriticMixin` (`skrl_ext/models/history_state_critic.py`) owns a
  second CNN+GRU that encodes `(image, past_actions)` into `z^c` from the value
  loss alone. The agent enables it with `history_state_critic: True`.

[`minimal/memory_state_critic.py`](minimal/memory_state_critic.py) is the same
idea in one short file of plain PyTorch, with no simulator.

`skrl_ppo_symmetric_critic_cfg.yaml` (label `symmetric`, `V(o,a)`) is also
provided but is not in the paper; `--paper` mode excludes it.

**Older names.** The runs behind the paper were launched before the code adopted
the paper's notation, so their wandb names and `figures/paper/runs.pkl` use
`Vs`, `Vsh`, `Vsz`, `Vsoa` and `Vo` for the five labels above, in that order.
Those names, the old entry points (`skrl_ppo_vision_rnn_sz_cfg_entry_point`, ...),
the old agent classes (`PPO_RNN_SZ`, `PPO_RNN_SH`, `PPO_RNN_VSH`) and the old
config keys (`sz_critic`, `sh_critic`, ...) are all still accepted. Note that
`Vsz` is the memory-state critic (ours) and `Vsh` the history-state baseline.

---

## 2. Settings

Shared across all critic variants, matching Appendix Table 2:

| | |
|---|---|
| Task | `Ablation-vision-vs-trajectories` |
| Drone platform | `crazyflie` (brushed 2.x) — downloaded from NVIDIA Nucleus at runtime |
| Sensors | `--sensor-mode=both` (64×64 depth + binary opponent segmentation) |
| Past actions in observation | `--num-past-actions=1` |
| Parallel environments | 512 |
| Total environment steps | 51,200,000 (= 100,000 timesteps × 512 envs) |
| Open arena | γ = 0.99 (YAML default) |
| Wall arena | `--enable-obstacles --discount-factor=0.999` |

All remaining PPO hyperparameters (rollout 32, 8 learning epochs, 4 mini-batches,
lr 3e-4 with KL-adaptive scheduling at τ=0.008, clip 0.2, entropy 0.01, value
loss 2.0, GRU 256×1, BPTT sequence length 16) live in the agent YAMLs and are
identical across variants.

## 3. Seeds

The grid exactly as it was run:

| Critic | Paper symbol | Open arena | Wall arena |
|---|---|---|---|
| `state` | `V(s)` | 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 17, 19, 21, 27, 37, 42, 123 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 17, 19, 21, 27, 37, 42, 123 |
| `history-state` | `V(s,z^c)` | 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 17, 19, 21, 27, 37, 42, 123 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 17, 19, 21, 27, 37, 42, 123 |
| `memory-state` | `V(s,z^a)` | 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 17, 19, 21, 27, 37, 42, 123 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 17, 19, 21, 27, 37, 42, 123 |
| `observation-state` | `V(s,o,a)` | 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 17, 19, 21, 27, 37, 42, 123 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 17, 19, 21, 27, 37, 42, 123 |

**160 runs total**: 4 critics x 2 arenas x 20 seeds, the same seed set in every
cell, so the comparison between critics is paired.

(The May 2026 submission used a smaller, unbalanced grid — five seeds for most
cells, three for `V(s,o,a)`, and a different open-arena seed set for `V(s,z^a)`. The
camera-ready replaces it entirely; nothing from that grid is reused.)

Aggregation is the interquartile mean (Agarwal et al., 2022), recomputed
**independently at each evaluation point**: at every point the best and worst
quartile of seeds are discarded and the middle ten averaged. Shaded bands and
error bars are **95% bootstrap confidence intervals** over seeds
(20,000 resamples), which is the interval Agarwal et al. recommend — not the
spread of the retained seeds.

---

## 4. Running it

```bash
# Preview the full 160-run plan without launching anything
./scripts/reproduce_paper.sh --dry-run

# Run everything (sequential; ~5.5 h/run on an RTX 4090 → ~37 GPU-days)
WANDB_ENTITY=<your-entity> ./scripts/reproduce_paper.sh

# Split across machines
WANDB_ENTITY=<your-entity> ./scripts/reproduce_paper.sh --arena open
WANDB_ENTITY=<your-entity> ./scripts/reproduce_paper.sh --arena wall
WANDB_ENTITY=<your-entity> ./scripts/reproduce_paper.sh --critics "memory-state history-state"
```

A single run, if you only want to see the memory-state critic train:

```bash
python scripts/skrl/train.py \
    --task=Ablation-vision-vs-trajectories \
    --agent=skrl_ppo_memory_state_critic_cfg_entry_point \
    --sensor-mode=both --num-past-actions=1 \
    --seed=42 --num_envs=512 --total_frames=51200000 \
    --headless --enable_cameras
```

On a 12 GB GPU, `NUM_ENVS=256` fits but changes the batch composition, so
results will not match the paper exactly.

**Short runs produce no Figure-4 data.** The aggregate
`Info / TerminationRate/<reason>` series that Figure 4 is built from is emitted
once per rolling window of 10,240 completed episodes
(`termination_rate_window` in
`tasks/direct/pursuit_evasion/tools/stats_tracker.py`). A brief smoke run logs
only the per-evader `TerminationRate/<heuristic>/<reason>` breakdown and none of
the aggregates. At 512 envs the first aggregate point lands early in training,
so a full run is unaffected.

Interrupted sweeps resume from marker files under `$ABLATION_DONE_DIR`
(default `~/logs/ablation_done`). `--dry-run` always prints the full plan and
ignores those markers.

---

## 5. Regenerating the figures

The plotted series for all 160 runs ship in `figures/paper/runs.pkl`, so the
paper's figures regenerate **without a wandb account**:

```bash
python scripts/plot_ablation_results.py --paper \
    --cache figures/paper/runs.pkl \
    --output-dir figures/reproduced
```

This writes `reward_evolution.{pdf,png}` (**Figure 3**),
`termination_heatmap.{pdf,png}` (**Figure 4**), plus
`termination_summary.csv`, `win_rate_summary.csv` and the capture-rate curves.
`figures/reproduced/termination_summary.csv` should be byte-identical to
`figures/paper/termination_summary.csv`.

Two auxiliary plots — reward-component magnitudes and per-heuristic capture
rates — read per-run wandb *summary* fields that the history cache does not
carry, and are skipped in offline mode. Neither appears in the paper.

To plot from your own runs instead of the shipped cache:

```bash
python scripts/plot_ablation_results.py --paper \
    --entity <your-entity> --project critic_ablation \
    --cache figures/reproduced/runs.pkl \
    --output-dir figures/reproduced
```

`--paper` pins IQM aggregation, excludes `symmetric`, sets the 100K-timestep axis, and
filters to the seed grid in `PAPER_GRID` (`scripts/plot_ablation_results.py`).
It warns about any cell where runs are missing.

---

## 6. Environment

The experiments ran on **Isaac Sim 5.1.0** with Isaac Lab pinned at commit
`d2579ea`, CUDA 12.8, Python 3.11. `./install.sh` builds this environment;
`requirements-freeze.txt` records the exact package set that produced the
results.

**NumPy must stay at 1.26.x.** Isaac Sim 5.1's `omni.syntheticdata` camera
annotator fails under NumPy 2.x with `TypeError: Unable to write from unknown
dtype, kind=f, size=0` during `annotator.attach()`
([IsaacLab#3312](https://github.com/isaac-sim/IsaacLab/issues/3312)). If a
dependency upgrades NumPy, run `pip install isaacsim` to pull it back down.

Long sweeps occasionally hang in a PXR/USD thread reset — the process stays
alive while the log goes silent. `run_ablation.sh` wraps each run in a
`PER_RUN_TIMEOUT` watchdog (default 9 h) so the sweep moves on instead of
stalling overnight.

---

## 6b. Running in Docker

Docker is the recommended way to reproduce on a second machine: it pins the whole stack, and
the image is what removes the largest source of silent drift.

**The Isaac Lab pin is the part that matters.** The runs in this repository were produced
against `github.com/colson-louis/IsaacLab` at commit `d2579ea`, which is *not* an official
Isaac Lab release. That commit was submitted upstream as pull request #5725, and GitHub keeps
pull-request head refs fetchable indefinitely, so the Dockerfile fetches it from the official
repository via `refs/pull/5725/head`. Cloning the fork and checking out the SHA is **not**
reliable: the commit has since diverged from the fork's `main` (14 ahead, 4 behind).

Isaac Lab is the physics layer. Building against any other commit produces runs that cannot be
pooled with the existing ones, and nothing in the output would tell you. The build therefore
asserts the resulting SHA and fails if it does not match.

The rest of the stack was already pinned and matches the host that produced the runs:
Isaac Sim 5.1.0.0, torch 2.7.0+cu128, numpy 1.26.0.

```bash
cp docker/.env.example docker/.env      # fill WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT
docker compose -f docker/docker-compose.yaml build

# Verify the pin before committing days of GPU time
docker compose -f docker/docker-compose.yaml run --rm train git -C /IsaacLab rev-parse HEAD
# -> d2579eacf8eba0864e2328f3f383a0fdb411e00b

docker compose -f docker/docker-compose.yaml run --rm train bash
```

Inside the container:

```bash
export WANDB_PROJECT=<the project holding the rest of the grid>
# Resume markers must live inside the bind-mounted repo. The launcher defaults them to
# $HOME/ablation_done, which is /root inside the container and is lost when it exits --
# an interrupted sweep would then redo every finished cell.
export ABLATION_DONE_DIR=/workspace/MemoryStateCritic/logs/ablation_done
./scripts/run_ablation.sh --arena open --critics "state memory-state history-state" --seeds "<seeds>"
```

`PYTHONPATH` needs no attention: `docker/entrypoint.sh` installs the vendored skrl fork and the
project package into the interpreter the training scripts actually invoke (`python`, which in
these images is *not* `python3`). Logs, checkpoints and markers land on the host through the
bind mount.

**Splitting a grid across machines.** Keep whole cells -- and preferably a whole arena -- on one
machine. `NUM_ENVS` must be identical everywhere: 512 needs a 24 GB card, and the 256 fallback
for 12 GB cards produces runs that are not comparable with 512-env runs.

## 7. Verification performed on this tree

Checked on 2026-08-24, single RTX 4090, from a clean copy of the released file
set (no editable install pointing back at a development checkout):

- **Figures.** `--paper --cache figures/paper/runs.pkl` regenerates
  `termination_summary.csv` and `win_rate_summary.csv` byte-identically to the
  shipped versions, with no wandb account.
- **Plan.** `scripts/reproduce_paper.sh --dry-run` emits 160 commands, all at
  `--num_envs=512 --total_frames=51200000`, wall runs carrying
  `--enable-obstacles --discount-factor=0.999`. (Plain
  `scripts/run_ablation.sh --dry-run` emits 10 — five critics x two arenas at
  its single default seed; it is the per-cell launcher, not the paper grid.)
- **Training.** Memory-state and history-state critics, open arena, seed 42, 512 envs, run for 30,000
  of the paper's 100,000 environment timesteps (~1 h each).
- **Architecture, from the trained checkpoints.** The memory-state critic is an
  MLP with input width 320 = 64 privileged state + 256 actor GRU hidden, 123 K
  parameters, and **no** recurrent or convolutional weights. The history-state
  critic carries its own CNN+GRU, 628 K parameters — 5.1x larger. This is
  Figure 1 realised in weights. The actor in both is CNN(2x64x64) ->
  Linear(128) -> concat 4 past-action dims -> GRU(256) -> 4 actions, matching
  Appendix A.
- **Learning curve.** The memory-state run lies inside the envelope of the
  paper's five cached open-arena memory-state seeds at 90% of logged points.
- **Comparative claim.** At equal budget, the memory-state critic led the
  history-state critic at every checkpoint and crossed return 0 at 2,100
  timesteps versus 19,800.

These checks predate the renaming described in section 1, which also removed
code no experiment used. The renaming was checked without a GPU: the shipped
figures regenerate pixel-identically, and the unit tests pass. The training
check above has not been rerun on the renamed tree.

One seed is not the paper's evidence — Figures 3 and 4 are interquartile means
over twenty seeds, and single runs are noisy early in training. These checks
establish that the released tree runs and behaves as described, not that a
single re-run rederives the paper's aggregates.
