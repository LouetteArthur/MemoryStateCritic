"""Task registration for the memory-state-critic ablation (paper Experiment 1).

A single task — ``Ablation-vision-vs-trajectories`` — exposes the five critic variants
compared in the paper through ``--agent`` entry points. The actor is identical across
variants (CNN+GRU on depth+segmap + past action); only the critic head changes.

Critic variants (see scripts/run_ablation.sh):
  Vs    V(s)      skrl_ppo_vision_rnn_cfg          MLP on privileged state
  Vsz   V(s,z)    skrl_ppo_vision_rnn_sz_cfg       state + detached actor GRU hidden (ours)
  Vsh   V(s,h)    skrl_ppo_vision_rnn_sh_cfg       state + critic-side CNN+GRU history
  Vo    V(o,a)    skrl_ppo_vision_rnn_symmetric_cfg  symmetric critic (no privileged state)
  Vsoa  V(s,o,a)  skrl_ppo_vision_rnn_geles_cfg    dict critic, state+image+past (--unbiased-critic)

Open vs wall arena is selected at train time with ``--enable-obstacles``.
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Ablation-vision-vs-trajectories",
    entry_point=f"{__name__}.pursuit_evasion_env:PursuitEvasionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pursuit_evasion_cfg:ablation_vision_vs_trajectories_cfg",
        # default safe variant
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_vision_rnn_cfg.yaml",
        # the five critic variants
        "skrl_ppo_vision_rnn_cfg_entry_point": f"{agents.__name__}:skrl_ppo_vision_rnn_cfg.yaml",          # Vs
        "skrl_ppo_vision_rnn_sz_cfg_entry_point": f"{agents.__name__}:skrl_ppo_vision_rnn_sz_cfg.yaml",      # Vsz (ours)
        "skrl_ppo_vision_rnn_sh_cfg_entry_point": f"{agents.__name__}:skrl_ppo_vision_rnn_sh_cfg.yaml",      # Vsh
        "skrl_ppo_vision_rnn_symmetric_cfg_entry_point": f"{agents.__name__}:skrl_ppo_vision_rnn_symmetric_cfg.yaml",  # Vo
        "skrl_ppo_vision_rnn_geles_cfg_entry_point": f"{agents.__name__}:skrl_ppo_vision_rnn_geles_cfg.yaml",  # Vsoa
    },
)
