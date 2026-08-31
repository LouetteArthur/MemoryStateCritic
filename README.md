# Memory-State Critic — reproduction code

Companion code for **"Memory-State Critic for Asymmetric Actor-Critic with Application to
Vision-Based Pursuit-Evasion"**. It reproduces Experiment 1: a single-agent, vision-based
pursuit-evasion POMDP (a Crazyflie pursuer learns to catch an evader drawn from a fixed
pool of heuristics) in which asymmetric critics are compared.

This is a curated single-agent slice of a larger research codebase: only the env, the
critic variants, and the ablation/plotting scripts needed to regenerate the paper's
figures are kept. The self-play league, the multi-agent environments, the Elo tournament
and the Crazyswarm flight scripts live in the parent codebase and are not shipped here.
Two subsystems do remain because the paper's code path imports them: `deployment/`, which
loads a policy from a checkpoint (used by `controllers/` to drive a frozen opponent), and
some staged-training (AMSPB) plumbing in the environment config. No experiment in the
paper uses the latter.

## What's here

```
source/isaac_pursuit_evasion/           # editable package
  isaac_pursuit_evasion/
    tasks/direct/pursuit_evasion/        # PursuitEvasionEnv + cfg + critic YAMLs
    skrl_ext/                            # asymmetric recurrent PPO extension of skrl (PPO_ASYM, PPO_RNN_VSH)
  assets/ dynamics/ controllers/         # Crazyflie assets, propeller dynamics, heuristic evaders
source/third_parties/skrl/               # vendored skrl fork
scripts/
  skrl/train.py  skrl/play.py            # train / visualize
  run_ablation.sh                        # the 5 critics x 2 arenas experiment
  run_exp1_extra_seeds.sh                # additional seeds
  plot_ablation_results.py               # regenerate the paper figures (mean/IQM over seeds)
```

## The critics (paper Table 1 / Fig 2-4)

| Label | Symbol   | Agent config (`--agent ...`)                    | Critic input |
|-------|----------|-------------------------------------------------|--------------|
| Vs    | V(s)     | `skrl_ppo_vision_rnn_cfg_entry_point`            | privileged state only |
| **Vsz** | **V(s,z)** | `skrl_ppo_vision_rnn_sz_cfg_entry_point`     | state + **detached actor GRU hidden** (ours) |
| Vsh   | V(s,h)   | `skrl_ppo_vision_rnn_sh_cfg_entry_point`         | state + critic-side CNN+GRU history |
| Vo    | V(o,a)   | `skrl_ppo_vision_rnn_symmetric_cfg_entry_point`  | actor obs only (symmetric) |
| Vsoa  | V(s,o,a) | `skrl_ppo_vision_rnn_geles_cfg_entry_point`      | state + image + past actions (`--unbiased-critic`) |

All variants share the same CNN+GRU actor (depth + segmentation, 1 past action); only the
critic head changes.

## Install

```bash
./install.sh          # .venv (Python 3.11), Isaac Sim 5.1, Isaac Lab, vendored skrl, this package
source .venv/bin/activate
```
Requirements: Ubuntu 22.04/24.04, NVIDIA GPU (driver >= 535, CUDA 12.x), Git LFS. If you
already have an Isaac Lab checkout, symlink it to `./IsaacLab` before `install.sh` to skip
the clone.

> **Use the dedicated `.venv`.** If you instead reuse an existing environment that already
> has another copy of `isaac_pursuit_evasion` installed in editable mode, `import
> isaac_pursuit_evasion` — which carries the environment and the task registration —
> silently resolves to *that* checkout, not to this one, and you will train against the
> wrong code. `install.sh` avoids this. To reuse an environment anyway, put this
> repository's package directory first on the path:
>
> ```bash
> export PYTHONPATH="$PWD/source/isaac_pursuit_evasion:$PWD:$PWD/source/third_parties/skrl"
> python -c "import isaac_pursuit_evasion as m; print(m.__file__)"   # must print this repo
> ```

## Reproduce the experiment

```bash
# all 5 critics x {open, wall}; the paper uses 5 seeds (open) / 3 seeds (wall)
OMNI_KIT_ACCEPT_EULA=YES NUM_ENVS=256 ./scripts/run_ablation.sh --seeds "42 123 7 9 21"

# subsets / inspection
./scripts/run_ablation.sh --arena wall --critics "Vs Vsz Vsh"
./scripts/run_ablation.sh --dry-run            # print the exact train commands

# figures (pulls runs named <Critic>_<arena>_s<seed> from wandb)
python scripts/plot_ablation_results.py --entity <you> --project critic_ablation --output-dir figures/ablation
```

A single training command (what the script emits) — memory-state critic, open arena:
```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/skrl/train.py \
    --task Ablation-vision-vs-trajectories \
    --agent skrl_ppo_vision_rnn_sz_cfg_entry_point \
    --sensor-mode both --num-past-actions 1 --seed 42 \
    --num_envs 256 --total_frames 102400000 --headless --enable_cameras
# wall arena: add  --enable-obstacles --discount-factor=0.999
```

## Notes / reproducibility

- The evader is part of the environment (fixed heuristic pool), so each run is a
  single-agent POMDP — no opponent learning.
- Compute: 5 critics x 2 arenas x 5 seeds ~ 50 runs of ~1e8 frames each; budget accordingly.
- Pin the stack: Isaac Sim 5.1.0.0, Isaac Lab @ the commit in `install.sh`, vendored skrl
  in `source/third_parties/skrl`. `requirements-freeze.txt` records the exact Python deps.

---

## Citation

```bibtex
@inproceedings{memorystatecritic2026,
  title     = {Memory-State Critic for Asymmetric Actor-Critic with
               Application to Vision-Based Pursuit-Evasion},
  author    = {Louette, Arthur and S{\'a}nchez Roncero, Alejandro and
               Lambrechts, Gaspard and Leroy, Pascal and Hansen, Julien and
               {\"O}gren, Petter and Ernst, Damien},
  booktitle = {European Workshop on Reinforcement Learning (EWRL)},
  year      = {2026}
}
```

Machine-readable metadata: [CITATION.cff](CITATION.cff).

## License

[BSD 3-Clause](LICENSE). Third-party components are listed in [NOTICE](NOTICE);
in particular this repository vendors a **modified** copy of
[skrl](https://github.com/Toni-SM/skrl) 1.4.3 (MIT) under `source/third_parties/`.

## Acknowledgements

Claude (Anthropic) assisted with parts of the code, under the authors' direction
and review. The research contribution and the reported results are the authors'.
