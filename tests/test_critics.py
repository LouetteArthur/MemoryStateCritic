# Copyright (c) 2026, the MemoryStateCritic authors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the critics, their stop-gradient, and the names they answer to.

Pure-torch tests, no Isaac Sim needed. Covers:

- the standalone reference in ``minimal/memory_state_critic.py``: the value loss
  must not reach the actor's encoder, the policy loss must;
- the skrl memory-state and history-state critic models: shapes, and the guard
  that stops the memory-state critic from silently degrading to V(s);
- the names from before the code adopted the paper's notation, which must still
  resolve to the same classes and config keys.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import pytest
import torch
from isaac_pursuit_evasion.skrl_ext.models.history_state_critic import (
    HistoryStateCriticModel,
)
from isaac_pursuit_evasion.skrl_ext.models.memory_state_critic import (
    MemoryStateCriticModel,
    memory_state_critic_model,
)

from minimal.memory_state_critic import (
    HistoryStateCritic,
    MemoryStateCritic,
    RecurrentActor,
    ppo_losses,
)

B, T, OBS, ACT, STATE, MEM = 4, 5, 7, 2, 3, 16


def _batch() -> dict[str, torch.Tensor]:
    return {
        "obs": torch.randn(B, T, OBS),
        "prev_action": torch.randn(B, T, ACT),
        "state": torch.randn(B, T, STATE),
        "action": torch.randn(B, T, ACT),
        "log_prob": torch.randn(B, T),
        "advantage": torch.randn(B, T),
        "return": torch.randn(B, T),
    }


# ---------------------------------------------------------------------------
# minimal/memory_state_critic.py
# ---------------------------------------------------------------------------


def test_minimal_value_loss_does_not_reach_actor():
    torch.manual_seed(0)
    actor, critic = RecurrentActor(OBS, ACT, MEM), MemoryStateCritic(STATE, MEM)
    _, value_loss = ppo_losses(actor, critic, _batch())
    value_loss.backward()
    assert all(p.grad is None for p in actor.parameters())
    assert all(p.grad is not None for p in critic.parameters())


def test_minimal_policy_loss_trains_actor_encoder():
    torch.manual_seed(0)
    actor, critic = RecurrentActor(OBS, ACT, MEM), MemoryStateCritic(STATE, MEM)
    policy_loss, _ = ppo_losses(actor, critic, _batch())
    policy_loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.f_theta.parameters())


def test_minimal_history_state_critic_trains_its_own_encoder():
    torch.manual_seed(0)
    batch = _batch()
    critic = HistoryStateCritic(OBS, ACT, STATE, MEM)
    value = critic(batch["state"], batch["obs"], batch["prev_action"])
    assert value.shape == (B, T)
    value.sum().backward()
    assert all(p.grad is not None for p in critic.f_psi.parameters())


# ---------------------------------------------------------------------------
# skrl models
# ---------------------------------------------------------------------------


def _spaces(state_dim: int = STATE):
    return gym.spaces.Box(-1.0, 1.0, shape=(state_dim,)), gym.spaces.Box(-1.0, 1.0, shape=(ACT,))


def test_memory_state_critic_model_shapes():
    obs_space, act_space = _spaces()
    m = memory_state_critic_model(obs_space, act_space, device="cpu", actor_hidden_size=MEM, layers=[8])
    assert isinstance(m, MemoryStateCriticModel)
    v, _ = m.compute({"states": torch.randn(B, STATE), "z_a": torch.randn(B, MEM)})
    assert v.shape == (B, 1)
    v, _ = m.compute({"states": torch.randn(B, T, STATE), "z_a": torch.randn(B, T, MEM)})
    assert v.shape == (B, T, 1)


def test_memory_state_critic_zero_fallback_disarms_after_first_real_input():
    """Zeros stand in for z^a only before the first rollout; afterwards a missing z^a is a bug."""
    obs_space, act_space = _spaces()
    m = MemoryStateCriticModel(obs_space, act_space, device="cpu", actor_hidden_size=MEM, layers=[8])
    m.compute({"states": torch.randn(B, STATE)})  # init-time dummy forward: allowed
    m.compute({"states": torch.randn(B, STATE), "z_a": torch.randn(B, MEM)})
    with pytest.raises(RuntimeError, match="state-only critic"):
        m.compute({"states": torch.randn(B, STATE)})


def test_memory_state_critic_depends_on_memory():
    torch.manual_seed(0)
    obs_space, act_space = _spaces()
    m = MemoryStateCriticModel(obs_space, act_space, device="cpu", actor_hidden_size=MEM, layers=[8])
    s = torch.randn(B, STATE)
    v0, _ = m.compute({"states": s, "z_a": torch.zeros(B, MEM)})
    v1, _ = m.compute({"states": s, "z_a": torch.ones(B, MEM)})
    assert not torch.allclose(v0, v1)


def test_history_state_critic_model_has_one_recurrent_state():
    obs_space, _ = _spaces()
    m = HistoryStateCriticModel(
        obs_space,
        gym.spaces.Box(-1.0, 1.0, shape=(ACT,)),
        device="cpu",
        image_channels=1,
        image_height=48,
        image_width=48,
        past_actions_size=ACT,
        rnn={"hidden_size": MEM, "num_layers": 1, "sequence_length": T},
        num_envs=B,
        cnn_feature_size=8,
        layers=[8],
    )
    assert m.get_specification()["rnn"]["sizes"] == [(1, B, MEM)]
    v, out = m.compute({
        "states": torch.randn(B, STATE),
        "critic_image": torch.randn(B, 1, 48, 48),
        "critic_past_actions": torch.randn(B, ACT),
    })
    assert v.shape == (B, 1)
    assert len(out["rnn"]) == 1 and out["rnn"][0].shape == (1, B, MEM)


# ---------------------------------------------------------------------------
# Names from before the paper's notation
# ---------------------------------------------------------------------------


def test_legacy_agent_names_resolve_to_ppo_rnn_asym():
    from isaac_pursuit_evasion.skrl_ext import CustomRunner
    from isaac_pursuit_evasion.skrl_ext.agents import PPO_RNN_ASYM

    for name in ("PPO_RNN_ASYM", "PPO_RNN_VSH", "PPO_RNN_SZ", "PPO_RNN_SH"):
        assert CustomRunner._component(None, name) is PPO_RNN_ASYM


def test_legacy_critic_names_resolve_to_memory_state_critic():
    from isaac_pursuit_evasion.skrl_ext import CustomRunner

    for name in ("MemoryStateCriticMixin", "SzCriticMixin", "VshCriticMixin"):
        assert CustomRunner._component(None, name) is memory_state_critic_model


@pytest.mark.parametrize(
    ("legacy", "current"),
    [
        ({"sz_critic": True, "sz_z_dim": 32}, {"memory_state_critic": True, "memory_dim": 32}),
        ({"vsh_critic": True, "vsh_actor_hidden_size": 32}, {"memory_state_critic": True, "memory_dim": 32}),
        (
            {"sh_critic": True, "sh_image_shape": [1, 8, 8], "sh_past_actions_size": 3},
            {"history_state_critic": True, "history_image_shape": [1, 8, 8], "history_past_actions_size": 3},
        ),
    ],
)
def test_legacy_agent_config_keys_are_translated(legacy, current):
    from isaac_pursuit_evasion.skrl_ext.agents import PPO_RNN_ASYM

    obs_space, act_space = _spaces()
    agent = PPO_RNN_ASYM(models={}, observation_space=obs_space, action_space=act_space, device="cpu", cfg=legacy)
    for key, value in current.items():
        assert agent.cfg[key] == value
    assert not set(legacy) & set(agent.cfg)


def test_task_registers_paper_and_legacy_entry_points():
    import importlib
    import sys

    sys.modules.pop("isaac_pursuit_evasion.tasks.direct.pursuit_evasion", None)
    with pytest.MonkeyPatch.context() as mp:
        registered = {}
        mp.setattr(gym, "register", lambda **kw: registered.update(kw))
        importlib.import_module("isaac_pursuit_evasion.tasks.direct.pursuit_evasion")
    kwargs = registered["kwargs"]
    pairs = {
        "skrl_ppo_vision_rnn_cfg_entry_point": "skrl_ppo_state_critic_cfg_entry_point",
        "skrl_ppo_vision_rnn_sh_cfg_entry_point": "skrl_ppo_history_state_critic_cfg_entry_point",
        "skrl_ppo_vision_rnn_sz_cfg_entry_point": "skrl_ppo_memory_state_critic_cfg_entry_point",
        "skrl_ppo_vision_rnn_geles_cfg_entry_point": "skrl_ppo_observation_state_critic_cfg_entry_point",
        "skrl_ppo_vision_rnn_symmetric_cfg_entry_point": "skrl_ppo_symmetric_critic_cfg_entry_point",
    }
    for legacy, current in pairs.items():
        assert kwargs[legacy] == kwargs[current]
        module, yaml_name = kwargs[current].split(":")
        assert (Path(importlib.import_module(module).__file__).parent / yaml_name).exists()
