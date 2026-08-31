# Copyright (c) 2026, the MemoryStateCritic authors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for AMSPB stage resolution and env-var plumbing.

Covers the new mechanism where _amspb_vision_cfg(stage=None) reads the
stage from the AMSPB_STAGE env var and the checkpoint map from
AMSPB_CHECKPOINTS.  These are the seams between the staged-training driver
and the env config factory.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@pytest.fixture(scope="module")
def cfg_mod():
    return importlib.import_module(
        "source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.pursuit_evasion.pursuit_evasion_cfg"
    )


@pytest.fixture
def restore_env(monkeypatch):
    """Convenience fixture: monkeypatch.setenv auto-reverts in teardown."""
    return monkeypatch


# ---------------------------------------------------------------------------
# _load_amspb_checkpoint_map
# ---------------------------------------------------------------------------


def test_load_checkpoint_map_empty_when_unset(cfg_mod, restore_env):
    restore_env.delenv("AMSPB_CHECKPOINTS", raising=False)
    assert cfg_mod._load_amspb_checkpoint_map() == {}


def test_load_checkpoint_map_parses_json(cfg_mod, restore_env):
    payload = {"pursuer_rl_bodyrates_pretrain": "/tmp/p.pt", "evader_rl_bodyrates_stage1": "/tmp/e.pt"}
    restore_env.setenv("AMSPB_CHECKPOINTS", json.dumps(payload))
    assert cfg_mod._load_amspb_checkpoint_map() == payload


def test_load_checkpoint_map_invalid_json_returns_empty(cfg_mod, restore_env):
    """Bad JSON should not crash — the env factory will fall back to FRPN-only pools."""
    restore_env.setenv("AMSPB_CHECKPOINTS", "not-valid-json")
    assert cfg_mod._load_amspb_checkpoint_map() == {}


def test_load_checkpoint_map_non_dict_returns_empty(cfg_mod, restore_env):
    restore_env.setenv("AMSPB_CHECKPOINTS", json.dumps([1, 2, 3]))
    assert cfg_mod._load_amspb_checkpoint_map() == {}


# ---------------------------------------------------------------------------
# _resolve_amspb_prob
# ---------------------------------------------------------------------------


def test_resolve_prob_uses_explicit_value_when_passed(cfg_mod, restore_env):
    restore_env.setenv("AMSPB_BASELINE_PROB", "0.9")
    assert cfg_mod._resolve_amspb_prob(0.3) == pytest.approx(0.3)  # CLI arg wins


def test_resolve_prob_falls_back_to_env(cfg_mod, restore_env):
    restore_env.setenv("AMSPB_BASELINE_PROB", "0.7")
    assert cfg_mod._resolve_amspb_prob(None) == pytest.approx(0.7)


def test_resolve_prob_default_when_no_signal(cfg_mod, restore_env):
    restore_env.delenv("AMSPB_BASELINE_PROB", raising=False)
    assert cfg_mod._resolve_amspb_prob(None) == pytest.approx(0.5)


def test_resolve_prob_clamps_to_unit_interval(cfg_mod, restore_env):
    assert cfg_mod._resolve_amspb_prob(2.5) == 1.0
    assert cfg_mod._resolve_amspb_prob(-0.3) == 0.0


# ---------------------------------------------------------------------------
# _previous_stage_key
# ---------------------------------------------------------------------------


def test_previous_stage_key_pretrain_for_stage_1(cfg_mod):
    """Stage 1 warm-starts from the pretrain checkpoint, not from a numbered stage."""
    assert cfg_mod._previous_stage_key("pursuer", "rl_bodyrates", 1) == "pursuer_rl_bodyrates_pretrain"
    assert cfg_mod._previous_stage_key("evader", "rl_bodyrates", 1) == "evader_rl_bodyrates_pretrain"


def test_previous_stage_key_decrements(cfg_mod):
    assert cfg_mod._previous_stage_key("pursuer", "rl_bodyrates", 5) == "pursuer_rl_bodyrates_stage4"


def test_previous_stage_key_zero_or_negative_returns_none(cfg_mod):
    assert cfg_mod._previous_stage_key("pursuer", "rl_bodyrates", 0) is None
    assert cfg_mod._previous_stage_key("pursuer", "rl_bodyrates", -1) is None


# ---------------------------------------------------------------------------
# _amspb_vision_cfg env-var stage resolution
# ---------------------------------------------------------------------------


def test_amspb_vision_cfg_reads_stage_from_env_var(cfg_mod, restore_env):
    """When stage=None, the factory must use AMSPB_STAGE."""
    restore_env.setenv("AMSPB_STAGE", "3")
    restore_env.setenv("AMSPB_BASELINE_PROB", "0.5")
    restore_env.setenv("AMSPB_ACTION_MODE", "rl_bodyrates")
    # Provide all checkpoint keys the stage-3 pursuer pool will demand.
    ckpts = {
        "pursuer_rl_bodyrates_pretrain": "/fake/p_pretrain.pt",
        "pursuer_rl_bodyrates_stage1": "/fake/p_s1.pt",
        "pursuer_rl_bodyrates_stage2": "/fake/p_s2.pt",
        "evader_rl_bodyrates_stage3": "/fake/e_s3.pt",
    }
    restore_env.setenv("AMSPB_CHECKPOINTS", json.dumps(ckpts))

    cfg = cfg_mod._amspb_vision_cfg(stage=None, training_agent="pursuer", num_envs=64)
    # The wandb_run_name encodes the stage — quick way to verify it was resolved to 3.
    assert "stage3" in cfg.wandb_run_name
    # Evader pool at stage 3 must reference evader_stage3
    evader_names = [s.name for s in cfg.evader_controllers]
    assert "rl_bodyrates_evader_stage3" in evader_names


def test_amspb_vision_cfg_explicit_stage_overrides_env(cfg_mod, restore_env):
    """An explicit stage argument must take precedence over the env var."""
    restore_env.setenv("AMSPB_STAGE", "9")
    restore_env.setenv("AMSPB_ACTION_MODE", "rl_bodyrates")
    ckpts = {
        "pursuer_rl_bodyrates_pretrain": "/fake/p_pretrain.pt",
        "evader_rl_bodyrates_stage1": "/fake/e_s1.pt",
    }
    restore_env.setenv("AMSPB_CHECKPOINTS", json.dumps(ckpts))

    cfg = cfg_mod._amspb_vision_cfg(stage=1, training_agent="pursuer", num_envs=64)
    assert "stage1" in cfg.wandb_run_name
