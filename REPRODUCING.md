# Reproducing the paper

This document covers **Experiment 1** of *"Memory-State Critic for Asymmetric
Actor-Critic with Application to Vision-Based Pursuit-Evasion"* (EWRL 2026):
the critic ablation behind Figures 3 and 4 and Table 1.

Everything else in this repository — the AMSPB population-based pipeline, the
Elo tournament, and the Crazyswarm sim-to-real deployment — belongs to separate
work and is **not** part of this paper.

---

## 1. Critic naming: paper ↔ code

The code labels predate the paper's notation and do not read the way you would
guess. **`Vsz` is the paper's contribution; `Vsh` is the baseline it is compared
against.** Getting these backwards silently runs the wrong experiment.

| Paper (Table 1) | Symbol | Code label | Critic model | Agent class | Agent YAML | Extra flag |
|---|---|---|---|---|---|---|
| Observation-state | `V(s,o,a)` | `Vsoa` | `DeterministicMixin` (Dict input) | `PPO_RNN_ASYM` | `skrl_ppo_vision_rnn_geles_cfg.yaml` | `--unbiased-critic` |
| State-only | `V(s)` | `Vs` | `DeterministicMixin` | `PPO_RNN_ASYM` | `skrl_ppo_vision_rnn_cfg.yaml` | — |
| History-state | `V(s,z^c)` | `Vsh` | `HistoryStateCriticMixin` | `PPO_RNN_SH` | `skrl_ppo_vision_rnn_sh_cfg.yaml` | — |
| **Memory-state (ours)** | `V(s,z^a)` | `Vsz` | `SzCriticMixin` | `PPO_RNN_SZ` | `skrl_ppo_vision_rnn_sz_cfg.yaml` | — |

**Where the contribution actually lives.** `PPO_RNN_SZ` and `PPO_RNN_SH` are
*aliases of the same agent*, `PPO_RNN_VSH`
(`skrl_ext/agents/ppo_rnn_vsh.py`). What separates the memory-state critic from
the history-state critic is the **critic model**, selected by the `value.class`
field of the YAML:

- `SzCriticMixin` (`skrl_ext/models/vsh_critic.py`) — an MLP over
  `[privileged_state ‖ detach(z^a)]`. The `detach` is the stop-gradient of
  Figure 1; there is no second recurrent encoder.
- `HistoryStateCriticMixin` (`skrl_ext/models/history_state_critic.py`) — owns a
  second CNN+GRU that encodes `(image, past_actions)` into `z^c` from the value
  loss alone.

Searching for a file named `ppo_rnn_sz.py` will not find anything; start from
`vsh_critic.py`.

The repository also contains `skrl_ppo_vision_rnn_symmetric_cfg.yaml` (label
`Vo`), `..._shh_cfg.yaml` and `..._szz_cfg.yaml`. None of these appear in the
paper; `--paper` mode excludes them.

The memory-state critic itself is `SzCriticMixin` plus `PPO_RNN_SZ` under
`source/isaac_pursuit_evasion/isaac_pursuit_evasion/skrl_ext/`. The stop-gradient
on `z^a` (Figure 1, right) is what distinguishes it from a jointly-trained
encoder.

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
| `Vs` | `V(s)` | 1, 5, 7, 42, 123 | 1, 5, 15, 42, 123 |
| `Vsh` | `V(s,z^c)` | 1, 5, 7, 42, 123 | 1, 5, 15, 42, 123 |
| `Vsz` | `V(s,z^a)` | 1, 7, 15, 27, 42 | 1, 5, 15, 42, 123 |
| `Vsoa` | `V(s,o,a)` | 1, 42, 123 | 1, 42, 123 |

36 runs total. Two things to note, stated plainly:

- The **open-arena `Vsz` seed set differs** from its `Vs` / `Vsh` siblings. This
  reflects how the sweep was actually executed across machines, not a deliberate
  design choice.
- **`V(s,o,a)` has three seeds per arena, not five**, as the paper states in
  Section 4.3. It never learns a useful policy in either arena, and the variance
  across those three seeds is small.

Aggregation is the interquartile mean (Agarwal et al., 2022): with five seeds,
the best- and worst-performing seed are dropped (ranked by the final 10% of
training on the plotted metric) and the middle three are averaged. Shaded bands
and error bars span the min and max of the three kept seeds.

---

## 4. Running it

```bash
# Preview the full 36-run plan without launching anything
./scripts/run_ablation.sh --dry-run

# Run everything (sequential; ~5.5 h/run on an RTX 4090 → ~7 GPU-days)
WANDB_ENTITY=<your-entity> ./scripts/run_ablation.sh

# Split across machines
WANDB_ENTITY=<your-entity> ./scripts/run_ablation.sh --arena open
WANDB_ENTITY=<your-entity> ./scripts/run_ablation.sh --arena wall
WANDB_ENTITY=<your-entity> ./scripts/run_ablation.sh --critics "Vsz Vsh"
```

A single run, if you only want to see the memory-state critic train:

```bash
python scripts/skrl/train.py \
    --task=Ablation-vision-vs-trajectories \
    --agent=skrl_ppo_vision_rnn_sz_cfg_entry_point \
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

The plotted series for all 36 runs ship in `figures/paper/runs.pkl`, so the
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

`--paper` pins IQM aggregation, excludes `Vo`, sets the 100K-timestep axis, and
filters to the seed grid in `PAPER_GRID` (`scripts/plot_ablation_results.py`).
It warns about any cell where runs are missing.

---

## 6. Environment

The experiments ran on **Isaac Sim 5.1.0** with Isaac Lab pinned at commit
`5497685`, CUDA 12.8, Python 3.11. `./install.sh` builds this environment;
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
export ABLATION_DONE_DIR=/workspace/IsaacPursuitEvasion/logs/ablation_done
./scripts/run_ablation.sh --arena open --critics "Vs Vsz Vsh" --seeds "<seeds>"
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
- **Plan.** `scripts/run_ablation.sh --dry-run` emits 36 commands, all at
  `--num_envs=512 --total_frames=51200000`, wall runs carrying
  `--enable-obstacles --discount-factor=0.999`.
- **Training.** `Vsz` and `Vsh`, open arena, seed 42, 512 envs, run for 30,000
  of the paper's 100,000 environment timesteps (~1 h each).
- **Architecture, from the trained checkpoints.** The memory-state critic is an
  MLP with input width 320 = 64 privileged state + 256 actor GRU hidden, 123 K
  parameters, and **no** recurrent or convolutional weights. The history-state
  critic carries its own CNN+GRU, 628 K parameters — 5.1x larger. This is
  Figure 1 realised in weights. The actor in both is CNN(2x64x64) ->
  Linear(128) -> concat 4 past-action dims -> GRU(256) -> 4 actions, matching
  Appendix A.
- **Learning curve.** The `Vsz` run lies inside the envelope of the paper's five
  cached open-arena `Vsz` seeds at 90% of logged points.
- **Comparative claim.** At equal budget, `Vsz` led `Vsh` at every checkpoint
  and crossed return 0 at 2,100 timesteps versus 19,800 for `Vsh`.

One seed is not the paper's evidence — Figures 3 and 4 are interquartile means
over five seeds, and single runs are noisy early in training. These checks
establish that the released tree runs and behaves as described, not that a
single re-run rederives the paper's aggregates.
