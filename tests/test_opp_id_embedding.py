# Copyright (c) 2026, the MemoryStateCritic authors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the learned opponent-identifier embedding e(k_t) wiring.

Covers the paper-faithful V(s, z, z_opp, e(k_t)) critic head added for
Experiment 2 (paper v22, Prop. 2). Pure-torch tests — no Isaac Sim needed.

What we check:

- The embedding is only instantiated when ``opp_id_dim > 0`` (so the
  V(s, z) and V(s, z, z_opp) configs used by Experiment 1 are bit-identical
  to before the change).
- ``opp_id_num > 0`` is required as soon as ``opp_id_dim > 0`` (loud failure
  beats a silent embedding-lookup crash at first forward pass).
- The forward path is shape-correct for both rollout (2D input) and BPTT
  (3D input) layouts.
- Different ``opp_id`` integers actually produce different value outputs.
  This is the real "is the embedding wired" check — if the kwarg were
  silently dropped somewhere, value outputs would be identical across
  identifiers.
- When ``opp_id`` is missing from inputs (initial-eval / bootstrap), the
  zero fallback runs without raising.
- ``opp_id_dim = 0`` ignores any ``opp_id`` kwarg passed in (no accidental
  read).
"""

from __future__ import annotations

import gymnasium as gym
import pytest
import torch

from source.isaac_pursuit_evasion.isaac_pursuit_evasion.skrl_ext.models.history_state_critic import (
    HistoryStateCriticModel,
    history_state_critic_model,
)
from source.isaac_pursuit_evasion.isaac_pursuit_evasion.skrl_ext.models.vsh_critic import (
    VshCriticModel,
    vsh_critic_model,
)


def _make_model(
    state_dim: int = 8,
    actor_hidden_size: int = 16,
    z_opp_dim: int = 0,
    opp_id_dim: int = 0,
    opp_id_num: int = 0,
) -> VshCriticModel:
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(state_dim,))
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
    return VshCriticModel(
        observation_space=obs_space,
        action_space=act_space,
        device="cpu",
        actor_hidden_size=actor_hidden_size,
        z_opp_dim=z_opp_dim,
        opp_id_dim=opp_id_dim,
        opp_id_num=opp_id_num,
        layers=(32, 16),
    )


def test_embedding_disabled_by_default():
    """V(s, z) / V(s, z, z_opp) configs do not instantiate the embedding —
    Experiment 1 critic graph is unchanged."""
    m = _make_model(opp_id_dim=0)
    assert m.opp_id_embedding is None
    m_zopp = _make_model(z_opp_dim=4, opp_id_dim=0)
    assert m_zopp.opp_id_embedding is None


def test_embedding_dim_requires_num():
    """opp_id_dim > 0 with opp_id_num == 0 is misconfiguration — fail loudly."""
    with pytest.raises(ValueError, match="opp_id_num"):
        _make_model(opp_id_dim=8, opp_id_num=0)


def test_embedding_instantiated_when_enabled():
    m = _make_model(opp_id_dim=8, opp_id_num=16)
    assert isinstance(m.opp_id_embedding, torch.nn.Embedding)
    assert m.opp_id_embedding.num_embeddings == 16
    assert m.opp_id_embedding.embedding_dim == 8


def test_forward_rollout_shape():
    """2D rollout shape: states (N, state_dim), z (N, h), opp_id (N, 1)."""
    state_dim, h, z_opp, k_dim = 8, 16, 4, 8
    m = _make_model(
        state_dim=state_dim,
        actor_hidden_size=h,
        z_opp_dim=z_opp,
        opp_id_dim=k_dim,
        opp_id_num=16,
    )
    n = 5
    inputs = {
        "states": torch.zeros(n, state_dim),
        "z_theta": torch.zeros(n, h),
        "z_opp": torch.zeros(n, z_opp),
        "opp_id": torch.tensor([[0], [1], [2], [3], [0]], dtype=torch.long),
    }
    v, _ = m.compute(inputs)
    assert v.shape == (n, 1)


def test_forward_bptt_shape():
    """3D BPTT shape: states (num_seq, seq_len, state_dim) — embedding lookup
    must reshape back to (num_seq, seq_len, k_dim)."""
    state_dim, h, k_dim = 8, 16, 8
    m = _make_model(state_dim=state_dim, actor_hidden_size=h, opp_id_dim=k_dim, opp_id_num=16)
    num_seq, seq_len = 3, 4
    inputs = {
        "states": torch.zeros(num_seq, seq_len, state_dim),
        "z_theta": torch.zeros(num_seq, seq_len, h),
        "opp_id": torch.randint(0, 16, (num_seq, seq_len, 1)),
    }
    v, _ = m.compute(inputs)
    assert v.shape == (num_seq, seq_len, 1)


def test_different_ids_produce_different_values():
    """The whole point of the embedding: V(s, z, e(k=0)) != V(s, z, e(k=1)).

    With identical s and z, two batches differing only in opp_id must yield
    different value outputs — otherwise the kwarg is being silently dropped.
    """
    torch.manual_seed(0)
    state_dim, h, k_dim = 8, 16, 8
    m = _make_model(state_dim=state_dim, actor_hidden_size=h, opp_id_dim=k_dim, opp_id_num=16)
    n = 4
    states = torch.randn(n, state_dim)
    z = torch.randn(n, h)
    v0, _ = m.compute({"states": states, "z_theta": z, "opp_id": torch.zeros(n, 1, dtype=torch.long)})
    v1, _ = m.compute({"states": states, "z_theta": z, "opp_id": torch.ones(n, 1, dtype=torch.long)})
    # Probabilistic guarantee: with random init, the chance that the
    # embedding's k=0 row equals k=1 row exactly is zero.
    assert not torch.allclose(v0, v1), "opp_id is not affecting the value head — embedding kwarg dropped somewhere."


def test_missing_opp_id_uses_zero_fallback():
    """Initial-eval / bootstrap path: no ``opp_id`` in inputs must not crash;
    the critic produces a deterministic output using the zero fallback."""
    state_dim, h, k_dim = 8, 16, 8
    m = _make_model(state_dim=state_dim, actor_hidden_size=h, opp_id_dim=k_dim, opp_id_num=16)
    n = 4
    states = torch.zeros(n, state_dim)
    z = torch.zeros(n, h)
    v, _ = m.compute({"states": states, "z_theta": z})  # no opp_id
    assert v.shape == (n, 1)
    assert torch.isfinite(v).all()


def test_opp_id_ignored_when_disabled():
    """When opp_id_dim == 0, even if ``opp_id`` is in inputs it is not read.
    This guards Vsz / Vsh configs against an accidental read if the env
    starts emitting extras["opp_id"] in a future change."""
    state_dim, h = 8, 16
    m = _make_model(state_dim=state_dim, actor_hidden_size=h, opp_id_dim=0)
    n = 4
    common = {
        "states": torch.zeros(n, state_dim),
        "z_theta": torch.zeros(n, h),
    }
    v0, _ = m.compute(common)
    v1, _ = m.compute({**common, "opp_id": torch.arange(n).reshape(n, 1)})
    assert torch.allclose(v0, v1), "opp_id is being read by a critic that has opp_id_dim=0 — would taint Vsz/Vsh."


# --------------------------------------------------------------------------
# History-state critic — SHH joint history-state with e(k) embedding
# --------------------------------------------------------------------------


def _make_shh_model(
    state_dim: int = 8,
    opp_id_dim: int = 0,
    opp_id_num: int = 0,
    opp_branch: bool = True,
) -> HistoryStateCriticModel:
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(state_dim,))
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
    return HistoryStateCriticModel(
        observation_space=obs_space,
        action_space=act_space,
        device="cpu",
        image_channels=2,
        image_height=64,
        image_width=64,
        past_actions_size=4,
        rnn={"hidden_size": 32, "num_layers": 1, "sequence_length": 4},
        num_envs=1,
        cnn_feature_size=16,
        layers=(32, 16),
        opp_branch=opp_branch,
        opp_image_channels=2,
        opp_past_actions_size=4,
        opp_id_dim=opp_id_dim,
        opp_id_num=opp_id_num,
    )


def test_shh_embedding_disabled_by_default():
    """SHH with opp_branch but no opp_id_dim should not instantiate the
    embedding — preserves backward compatibility."""
    m = _make_shh_model(opp_id_dim=0)
    assert m.opp_id_embedding is None


def test_shh_embedding_instantiated_when_enabled():
    m = _make_shh_model(opp_id_dim=16, opp_id_num=64)
    assert isinstance(m.opp_id_embedding, torch.nn.Embedding)
    assert m.opp_id_embedding.num_embeddings == 64
    assert m.opp_id_embedding.embedding_dim == 16


def test_shh_different_ids_produce_different_values():
    """SHH must use e(k_t) just like SZZ — different k_t with identical
    everything else must yield different value outputs."""
    torch.manual_seed(0)
    state_dim, k_dim, k_num = 8, 16, 32
    m = _make_shh_model(state_dim=state_dim, opp_id_dim=k_dim, opp_id_num=k_num)
    n = 4
    img = torch.zeros(n, 2, 64, 64)
    pa = torch.zeros(n, 4)
    opp_img = torch.zeros(n, 2, 64, 64)
    opp_pa = torch.zeros(n, 4)
    states = torch.randn(n, state_dim)
    base = {
        "states": states,
        "critic_image": img,
        "critic_past_actions": pa,
        "opp_image": opp_img,
        "opp_prev_action": opp_pa,
        "rnn": [torch.zeros(1, n, 32), torch.zeros(1, n, 32)],
    }
    v0, _ = m.compute({**base, "opp_id": torch.zeros(n, 1, dtype=torch.long)})
    v1, _ = m.compute({**base, "opp_id": torch.ones(n, 1, dtype=torch.long)})
    assert not torch.allclose(v0, v1), "SHH critic ignores opp_id — embedding wiring is broken."


def test_shh_factory_threads_kwargs():
    """history_state_critic_model factory must thread opp_id_* to the model."""
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,))
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
    m = history_state_critic_model(
        observation_space=obs_space,
        action_space=act_space,
        device="cpu",
        image_channels=2,
        image_height=64,
        image_width=64,
        past_actions_size=4,
        rnn={"hidden_size": 32, "num_layers": 1, "sequence_length": 4},
        cnn_feature_size=16,
        layers=(32, 16),
        opp_branch=True,
        opp_id_dim=16,
        opp_id_num=64,
    )
    assert isinstance(m, HistoryStateCriticModel)
    assert isinstance(m.opp_id_embedding, torch.nn.Embedding)
    assert m.opp_id_embedding.num_embeddings == 64


def test_factory_threads_kwargs():
    """The ``vsh_critic_model`` factory used by the skrl Runner must thread
    opp_id_* through to the model — otherwise YAML settings get silently
    lost."""
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,))
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
    m = vsh_critic_model(
        observation_space=obs_space,
        action_space=act_space,
        device="cpu",
        actor_hidden_size=16,
        opp_id_dim=8,
        opp_id_num=16,
        layers=(32, 16),
    )
    assert isinstance(m, VshCriticModel)
    assert isinstance(m.opp_id_embedding, torch.nn.Embedding)
    assert m.opp_id_embedding.num_embeddings == 16
    assert m.opp_id_embedding.embedding_dim == 8
