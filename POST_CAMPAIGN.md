# Deferred cleanup — apply only after the seed campaign finishes

Three changes are intentionally held back so that all 34 runs of the camera-ready
seed campaign execute byte-identical code. Applying any of them mid-campaign would
make the seeds non-comparable.

Run these once `scratchpad/seeds/campaign.out` reports `CAMPAGNE TERMINÉE`.

## 1. Repository-wide formatting

`pre-commit` reformats 218 files (black + isort). Semantics-preserving, but it
touches 67 files on the training import path.

```bash
pre-commit run --all-files    # run twice; the first pass rewrites, the second verifies
```

## 2. Third-party wandb entity

`kthxulg/ppo_baseline` is hard-coded in 7 places across
`tasks/direct/pursuit_evasion/pursuit_evasion_cfg.py` and `deployment/*_policy_loader.py`.
External users cannot access it, so every task that loads a pretrained opponent
(`Bench-*`, warmstart) fails for them with an opaque wandb error.

Replace the defaults with `None` and read the entity/project from
`PE_ARTIFACT_ENTITY` / `PE_ARTIFACT_PROJECT`. No experiment in the paper is affected.

## 3. Initialisation-order bug

`pursuit_evasion_env.py` assigns `self._wall_cfg_for_trajectories` *after*
`super().__init__()`, which calls `_setup_scene()` and consumes it — so any task using
the visual-ball evader dies with `AttributeError`. The paper's ablation task sets
`use_visual_ball_evader = False` and is unaffected, which is why this went unnoticed.

Fix: make `_build_wall_cfg_for_trajectories(cfg)` a `@staticmethod` taking `cfg`
explicitly, and move the assignment above `super().__init__()`.

## Verify afterwards

```bash
pytest tests/ -q                                   # expect 77 passed
python scripts/plot_ablation_results.py --paper \
    --cache figures/paper/runs.pkl --output-dir /tmp/figcheck
diff <(sort /tmp/figcheck/termination_summary.csv) \
     <(sort figures/paper/termination_summary.csv)  # must be empty
```
