# Copyright (c) 2026, the MemoryStateCritic authors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sanity checks for the CNN+GRU actor and its stored-state recurrent pipeline.

These tests do **not** require Isaac Sim — they import
``isaac_pursuit_evasion.skrl_ext`` directly via a package stub so that the
project's ``__init__.py`` (which pulls in Isaac Lab) is bypassed.

They answer three narrow questions that must be true before running any
expensive training, independent of whether skrl does true BPTT or stored-state
recurrence:

1. **past_actions actually reaches the policy output.** The R2D2/IMPALA
   fix only matters if swapping past actions on an otherwise-identical
   observation changes the action. If this fails, the concat is dead code.

2. **past_actions are in the gradient path.** Training must be able to
   update weights that consume past actions; otherwise the past-action
   branch of the network is frozen even if the forward pass uses it.

3. **Stored-state alignment is correct.** At rollout time the GRU is run
   with ``h_{t-1}`` and produces ``(action_t, h_t)``; the hidden state
   ``h_{t-1}`` is what gets stored in memory for transition ``t``. At
   training time we re-run with the stored hidden state and the stored
   observation, and must get the same action back. If this alignment is
   off by one step, recurrent credit assignment silently corrupts.

These three checks gate the 3x2 ablation: if any fails, the pipeline is
wrong and benchmark numbers would be meaningless.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Bypass isaac_pursuit_evasion/__init__.py (which imports Isaac Lab) by
# installing a thin package stub pointing at the real source directory.
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SKRL_EXT_PARENT = _PROJECT_ROOT / "source" / "isaac_pursuit_evasion"
sys.path.insert(0, str(_SKRL_EXT_PARENT))

if "isaac_pursuit_evasion" not in sys.modules or not hasattr(sys.modules["isaac_pursuit_evasion"], "skrl_ext"):
    _pkg = types.ModuleType("isaac_pursuit_evasion")
    _pkg.__path__ = [str(_SKRL_EXT_PARENT / "isaac_pursuit_evasion")]
    sys.modules["isaac_pursuit_evasion"] = _pkg

from isaac_pursuit_evasion.skrl_ext.models.gaussian_cnn_rnn import (  # noqa: E402
    GaussianCNNGRUModel,
)

# ---------------------------------------------------------------------------
# Shared fixture: a small CNN+GRU model with a Dict observation space whose
# image is 1x16x16 (tiny, so tests run fast) and past_actions = 3*4 = 12.
# ---------------------------------------------------------------------------

N_PAST_ACTIONS = 3
ACTION_DIM = 4
IMG_CHANNELS = 1
# Must be >= 48 because the CNN uses kernels 8/4/3 with strides 4/2/1 and no
# padding — matches the production config (64x64).
IMG_H, IMG_W = 64, 64
PAST_ACTIONS_SIZE = N_PAST_ACTIONS * ACTION_DIM
FLAT_OBS_SIZE = IMG_CHANNELS * IMG_H * IMG_W + PAST_ACTIONS_SIZE


def _build_model(seed: int = 0, hidden_size: int = 32) -> GaussianCNNGRUModel:
    torch.manual_seed(seed)
    observation_space = spaces.Dict({
        "image": spaces.Box(low=0.0, high=1.0, shape=(IMG_CHANNELS, IMG_H, IMG_W), dtype=np.float32),
        "past_actions": spaces.Box(low=-1.0, high=1.0, shape=(PAST_ACTIONS_SIZE,), dtype=np.float32),
    })
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)
    model = GaussianCNNGRUModel(
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
        rnn={"type": "gru", "hidden_size": hidden_size, "num_layers": 1, "sequence_length": 4},
        num_envs=2,
    )
    # Force lazy modules to materialize by running a dummy forward pass.
    dummy = torch.zeros(1, FLAT_OBS_SIZE)
    with torch.no_grad():
        model.compute({"states": dummy, "rnn": []})
    return model


def _flat_obs(image: torch.Tensor, past_actions: torch.Tensor) -> torch.Tensor:
    """Flatten a Dict observation into the layout skrl stores in memory.

    ``skrl.utils.spaces.torch.flatten_tensorized_space`` concatenates Dict
    sub-spaces in ``sorted(space.keys())`` order, which for us is
    ``image, past_actions``.
    """
    batch = image.shape[0]
    return torch.cat([image.reshape(batch, -1), past_actions.reshape(batch, -1)], dim=-1)


# ---------------------------------------------------------------------------
# Test 1 — past_actions actually influence the policy output
# ---------------------------------------------------------------------------


def test_past_actions_change_output():
    """Same image, different past_actions → different policy mean.

    If this fails, the ``concat(cnn_features, past_actions)`` in
    ``GaussianCNNGRUModel.compute()`` is bypassed and the past_actions
    branch of the network is dead.
    """
    model = _build_model(seed=0)
    model.eval()

    torch.manual_seed(123)
    image = torch.rand(1, IMG_CHANNELS, IMG_H, IMG_W)

    past_a = torch.zeros(1, PAST_ACTIONS_SIZE)
    past_b = torch.full((1, PAST_ACTIONS_SIZE), 0.7)

    with torch.no_grad():
        out_a, _, _ = model.compute({"states": _flat_obs(image, past_a), "rnn": []})
        out_b, _, _ = model.compute({"states": _flat_obs(image, past_b), "rnn": []})

    assert out_a.shape == (1, ACTION_DIM)
    # Any meaningful difference is fine; use a conservative threshold so we
    # catch "identical" (broken) without being noise-sensitive.
    assert not torch.allclose(
        out_a, out_b, atol=1e-6
    ), f"past_actions had no effect on policy output: max diff {(out_a - out_b).abs().max().item():.2e}"


# ---------------------------------------------------------------------------
# Test 2 — past_actions are in the gradient path during training
# ---------------------------------------------------------------------------


def test_past_actions_have_gradient():
    """Loss on the policy output yields non-zero gradient w.r.t. past_actions.

    This checks that during training, gradients flow from the policy head
    back through the concat into the past-action features. A zero gradient
    here would mean the past-action branch cannot learn.
    """
    model = _build_model(seed=1)
    model.train()

    torch.manual_seed(456)
    # Single "sampled transition" (what skrl's sample_all returns for stored
    # state recurrent training: 2D flat tensor).
    batch = 4
    image = torch.rand(batch, IMG_CHANNELS, IMG_H, IMG_W)
    past = torch.rand(batch, PAST_ACTIONS_SIZE) * 2 - 1
    flat = _flat_obs(image, past).clone().detach().requires_grad_(True)

    # Stored initial hidden state, as skrl would pass from memory.
    h0 = torch.zeros(1, batch, model._rnn_hidden_size)

    out, _, _ = model.compute({"states": flat, "rnn": [h0]})
    loss = out.pow(2).mean()
    loss.backward()

    assert flat.grad is not None
    # past_actions slice starts at index IMG_CHANNELS*IMG_H*IMG_W in the flat
    # layout (sorted Dict key order: 'image' before 'past_actions').
    image_end = IMG_CHANNELS * IMG_H * IMG_W
    past_grad = flat.grad[:, image_end : image_end + PAST_ACTIONS_SIZE]
    image_grad = flat.grad[:, :image_end]

    assert past_grad.abs().sum() > 0, "no gradient flowing into past_actions slice"
    # Sanity: image side also receives gradient (otherwise the test is
    # checking nothing about the concat specifically).
    assert image_grad.abs().sum() > 0, "no gradient flowing into image slice"


# ---------------------------------------------------------------------------
# Test 3 — rollout/train hidden-state alignment (no off-by-one)
# ---------------------------------------------------------------------------


def test_stored_state_alignment_rollout_vs_training():
    """The stored hidden state for transition t reproduces action_t exactly.

    Simulates skrl's stored-state recipe:
        rollout:   h_0 -> (obs_0) -> action_0, h_1
                   h_1 -> (obs_1) -> action_1, h_2
                   ...
        memory:    stores (obs_t, h_t_before_step, action_t) per t
        training:  feeds (obs_t, h_t_before_step) back into the SAME model
                   and must recover action_t bit-for-bit.

    The state that must be stored at index t is ``h_t_before_step`` — i.e.,
    the GRU hidden state BEFORE processing obs_t. If the implementation
    mistakenly stores ``h_{t+1}`` (the state AFTER the step), credit
    assignment is off by one and training silently corrupts.
    """
    model = _build_model(seed=2)
    model.eval()

    torch.manual_seed(789)
    num_steps = 5
    num_envs = 2
    images = torch.rand(num_steps, num_envs, IMG_CHANNELS, IMG_H, IMG_W)
    past_actions = torch.rand(num_steps, num_envs, PAST_ACTIONS_SIZE) * 2 - 1

    # Rollout: step through one transition at a time, recording h_before.
    h = torch.zeros(1, num_envs, model._rnn_hidden_size)
    rollout_actions = []
    stored_h_before = []

    with torch.no_grad():
        for t in range(num_steps):
            stored_h_before.append(h.clone())
            flat_t = _flat_obs(images[t], past_actions[t])
            out_t, _, rnn_out = model.compute({"states": flat_t, "rnn": [h]})
            rollout_actions.append(out_t.clone())
            h = rnn_out["rnn"][0]  # updated hidden state for the next step

    # "Training" replay: take each stored transition independently and
    # reconstruct the rollout output from (obs_t, h_before_t).
    with torch.no_grad():
        for t in range(num_steps):
            flat_t = _flat_obs(images[t], past_actions[t])
            replay_out, _, _ = model.compute({"states": flat_t, "rnn": [stored_h_before[t]]})
            assert torch.allclose(replay_out, rollout_actions[t], atol=1e-6), (
                f"stored-state misalignment at step {t}: "
                f"max diff {(replay_out - rollout_actions[t]).abs().max().item():.2e}"
            )


# ---------------------------------------------------------------------------
# Negative control for Test 3: if we use the WRONG hidden state (off-by-one),
# the check should fail. Catching this gives confidence that Test 3 would
# actually detect a regression rather than passing trivially because the
# hidden state does nothing.
# ---------------------------------------------------------------------------


def test_stored_state_alignment_detects_off_by_one():
    """Negative control: using h_{t+1} instead of h_t must disagree.

    Guarantees Test 3 is not vacuous — i.e. the GRU hidden state actually
    carries information that changes the policy output, so a misaligned
    state would produce a detectable error.
    """
    model = _build_model(seed=3)
    model.eval()

    torch.manual_seed(999)
    num_steps = 4
    num_envs = 1
    images = torch.rand(num_steps, num_envs, IMG_CHANNELS, IMG_H, IMG_W)
    past_actions = torch.rand(num_steps, num_envs, PAST_ACTIONS_SIZE) * 2 - 1

    h = torch.zeros(1, num_envs, model._rnn_hidden_size)
    rollout_actions = []
    h_sequence = [h.clone()]
    with torch.no_grad():
        for t in range(num_steps):
            flat_t = _flat_obs(images[t], past_actions[t])
            out_t, _, rnn_out = model.compute({"states": flat_t, "rnn": [h_sequence[-1]]})
            rollout_actions.append(out_t.clone())
            h_sequence.append(rnn_out["rnn"][0])

    # Replay obs_t with h_{t+1} (wrong) — at least one step must disagree.
    any_disagreement = False
    with torch.no_grad():
        for t in range(num_steps):
            flat_t = _flat_obs(images[t], past_actions[t])
            wrong_out, _, _ = model.compute({"states": flat_t, "rnn": [h_sequence[t + 1]]})
            if not torch.allclose(wrong_out, rollout_actions[t], atol=1e-5):
                any_disagreement = True
                break

    assert any_disagreement, (
        "using the wrong hidden state produced identical outputs — "
        "Test 3 would be vacuous because the GRU hidden state is ignored"
    )
