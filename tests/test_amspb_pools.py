# Copyright (c) 2026, the MemoryStateCritic authors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the AMSPB pool builders.

The pool builders generate ControllerSpec lists that the env consumes to
sample opponents per environment.  The previous implementation hardcoded
stage 1 / stage 2 branches; these tests pin down the generalized N-stage
behavior introduced for the vision+RNN AMSPB pipeline.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@pytest.fixture(scope="module")
def cfg_mod():
    """Import the cfg module once with Isaac stubs in place (provided by conftest)."""
    return importlib.import_module(
        "source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.pursuit_evasion.pursuit_evasion_cfg"
    )


def _make_checkpoint_map(extra_keys: list[str] | None = None) -> dict[str, str]:
    """Build a map of all keys the pool builders may need at stages 1..3."""
    keys = [
        "pursuer_rl_bodyrates_pretrain",
        "evader_rl_bodyrates_stage1",
        "pursuer_rl_bodyrates_stage1",
        "evader_rl_bodyrates_stage2",
        "pursuer_rl_bodyrates_stage2",
        "evader_rl_bodyrates_stage3",
    ]
    if extra_keys:
        keys.extend(extra_keys)
    return {k: f"/fake/path/{k}.pt" for k in keys}


# ---------------------------------------------------------------------------
# _amspb_pursuer_pool
# ---------------------------------------------------------------------------


def test_pursuer_pool_stage1_has_frpn_and_pretrain_only(cfg_mod):
    """Stage 1: opponents are FRPN baseline + pretrained pursuer (no checkpoint chain yet)."""
    ckpts = _make_checkpoint_map()
    specs = cfg_mod._amspb_pursuer_pool(
        stage_index=1,
        rl_kind="rl_bodyrates",
        num_envs=256,
        baseline_prob=0.5,
        checkpoints=ckpts,
    )
    names = [s.name for s in specs]
    assert "frpn_pursuer" in names
    assert "rl_bodyrates_pursuer_pretrain" in names
    # Stage 1 should NOT yet reference any pursuer_stage{N} checkpoint
    assert not any("pursuer_stage" in n for n in names)
    # Probabilities sum to 1.0
    assert pytest.approx(sum(s.probability for s in specs), abs=1e-6) == 1.0


def test_pursuer_pool_stage2_includes_stage1_checkpoint(cfg_mod):
    """Stage 2: FRPN (bp/2) + pretrain (bp/2) + stage1 latest (1 - bp)."""
    ckpts = _make_checkpoint_map()
    specs = cfg_mod._amspb_pursuer_pool(
        stage_index=2,
        rl_kind="rl_bodyrates",
        num_envs=256,
        baseline_prob=0.6,
        checkpoints=ckpts,
    )
    names = [s.name for s in specs]
    assert "rl_bodyrates_pursuer_stage1" in names
    # Latest checkpoint should get probability 1 - baseline_prob = 0.4
    latest = next(s for s in specs if s.name == "rl_bodyrates_pursuer_stage1")
    assert pytest.approx(latest.probability) == 0.4
    # FRPN and pretrain split the baseline mass equally
    frpn = next(s for s in specs if s.name == "frpn_pursuer")
    pretrain = next(s for s in specs if s.name == "rl_bodyrates_pursuer_pretrain")
    assert pytest.approx(frpn.probability) == 0.3
    assert pytest.approx(pretrain.probability) == 0.3
    assert pytest.approx(sum(s.probability for s in specs), abs=1e-6) == 1.0


def test_pursuer_pool_stage3_uses_stage2_latest(cfg_mod):
    """Stage 3 must point at the stage-2 checkpoint (not stage-1)."""
    ckpts = _make_checkpoint_map()
    specs = cfg_mod._amspb_pursuer_pool(
        stage_index=3,
        rl_kind="rl_bodyrates",
        num_envs=256,
        baseline_prob=0.5,
        checkpoints=ckpts,
    )
    names = [s.name for s in specs]
    assert "rl_bodyrates_pursuer_stage2" in names
    assert "rl_bodyrates_pursuer_stage1" not in names  # do not chain old stages


def test_pursuer_pool_stage_n_missing_checkpoint_raises(cfg_mod):
    """If the previous-stage checkpoint is missing, the builder must raise."""
    incomplete = _make_checkpoint_map()
    incomplete.pop("pursuer_rl_bodyrates_stage1")
    with pytest.raises(ValueError, match="Missing AMSPB checkpoint"):
        cfg_mod._amspb_pursuer_pool(
            stage_index=2,
            rl_kind="rl_bodyrates",
            num_envs=256,
            baseline_prob=0.5,
            checkpoints=incomplete,
        )


def test_pursuer_pool_kind_propagates_to_specs(cfg_mod):
    """RL-spec entries should carry kind=rl_kind so the env knows which controller to instantiate."""
    ckpts = _make_checkpoint_map()
    specs = cfg_mod._amspb_pursuer_pool(
        stage_index=2,
        rl_kind="rl_velocity",
        num_envs=128,
        baseline_prob=0.4,
        checkpoints={k.replace("rl_bodyrates", "rl_velocity"): v for k, v in ckpts.items()},
    )
    rl_specs = [s for s in specs if s.name.startswith("rl_velocity")]
    assert rl_specs, "expected at least one RL-kind spec in the pool"
    for s in rl_specs:
        assert s.kind == "rl_velocity"


# ---------------------------------------------------------------------------
# _amspb_evader_pool
# ---------------------------------------------------------------------------


def test_evader_pool_stage1_uses_stage1_checkpoint(cfg_mod):
    """When training pursuer at stage 1, the evader pool needs evader_{rl}_stage1."""
    ckpts = _make_checkpoint_map()
    specs = cfg_mod._amspb_evader_pool(
        stage_index=1,
        rl_kind="rl_bodyrates",
        num_envs=256,
        baseline_prob=0.6,
        checkpoints=ckpts,
    )
    names = [s.name for s in specs]
    assert "rl_bodyrates_evader_stage1" in names
    # Three trajectory baselines (hover, circular, lemniscate) should appear
    assert {"hover", "circular", "lemniscate"}.issubset(names)
    assert pytest.approx(sum(s.probability for s in specs), abs=1e-6) == 1.0


def test_evader_pool_stage_n_uses_stage_n_checkpoint(cfg_mod):
    """The evader pool keys on stage_index (not stage_index - 1) per the AMSPB convention."""
    ckpts = _make_checkpoint_map()
    for stage in (1, 2, 3):
        specs = cfg_mod._amspb_evader_pool(
            stage_index=stage,
            rl_kind="rl_bodyrates",
            num_envs=256,
            baseline_prob=0.5,
            checkpoints=ckpts,
        )
        expected = f"rl_bodyrates_evader_stage{stage}"
        names = [s.name for s in specs]
        assert expected in names, f"stage={stage}: expected '{expected}' in {names}"


def test_evader_pool_baseline_prob_zero_skips_trajectories(cfg_mod):
    """With baseline_prob=0 there are no trajectory controllers, only the latest RL evader."""
    ckpts = _make_checkpoint_map()
    specs = cfg_mod._amspb_evader_pool(
        stage_index=1,
        rl_kind="rl_bodyrates",
        num_envs=256,
        baseline_prob=0.0,
        checkpoints=ckpts,
    )
    names = [s.name for s in specs]
    assert "hover" not in names and "circular" not in names and "lemniscate" not in names
    assert names == ["rl_bodyrates_evader_stage1"]
