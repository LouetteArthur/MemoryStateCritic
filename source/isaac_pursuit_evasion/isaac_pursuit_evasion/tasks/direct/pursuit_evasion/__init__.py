# Copyright (c) 2026, the MemoryStateCritic authors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task registration for the memory-state-critic ablation.

A single task, ``Ablation-vision-vs-trajectories``, exposes the critic variants
through ``--agent`` entry points. The actor is identical across variants (CNN+GRU
on depth + segmentation + past action); only the critic changes.

  --agent skrl_ppo_<name>_cfg_entry_point     critic
  -------------------------------------------  -------------------------------------------
  state_critic                                 V(s)        MLP on the privileged state
  history_state_critic                         V(s, z^c)   state + critic-side CNN+GRU (baseline)
  memory_state_critic                          V(s, z^a)   state + detached actor GRU state (ours)
  observation_state_critic                     V(s, o, a)  state + image + past action (needs --unbiased-critic)
  symmetric_critic                             V(o, a)     actor observations only (not in the paper)

The open or wall arena is selected at train time with ``--enable-obstacles``.
"""

import gymnasium as gym

from . import agents

_CRITIC_CFGS = {
    "state_critic": "skrl_ppo_state_critic_cfg.yaml",
    "history_state_critic": "skrl_ppo_history_state_critic_cfg.yaml",
    "memory_state_critic": "skrl_ppo_memory_state_critic_cfg.yaml",
    "observation_state_critic": "skrl_ppo_observation_state_critic_cfg.yaml",
    "symmetric_critic": "skrl_ppo_symmetric_critic_cfg.yaml",
}

# Entry points from before the code adopted the paper's notation, kept so that
# older commands keep working: Vs, Vsz, Vsh, Vsoa, Vo.
_LEGACY_ENTRY_POINTS = {
    "skrl_ppo_vision_rnn_cfg_entry_point": "state_critic",
    "skrl_ppo_vision_rnn_sz_cfg_entry_point": "memory_state_critic",
    "skrl_ppo_vision_rnn_sh_cfg_entry_point": "history_state_critic",
    "skrl_ppo_vision_rnn_geles_cfg_entry_point": "observation_state_critic",
    "skrl_ppo_vision_rnn_symmetric_cfg_entry_point": "symmetric_critic",
}

_entry_points = {f"skrl_ppo_{name}_cfg_entry_point": f"{agents.__name__}:{yaml}" for name, yaml in _CRITIC_CFGS.items()}
_entry_points.update(
    {legacy: f"{agents.__name__}:{_CRITIC_CFGS[name]}" for legacy, name in _LEGACY_ENTRY_POINTS.items()}
)

gym.register(
    id="Ablation-vision-vs-trajectories",
    entry_point=f"{__name__}.pursuit_evasion_env:PursuitEvasionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pursuit_evasion_cfg:ablation_vision_vs_trajectories_cfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:{_CRITIC_CFGS['state_critic']}",
        **_entry_points,
    },
)
