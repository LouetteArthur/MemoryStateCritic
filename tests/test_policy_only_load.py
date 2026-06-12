"""Unit tests for the policy-only checkpoint loader (`--checkpoint-policy-only`).

The loader exists to warm-start AMSPB agents from Experiment-1 checkpoints
whose critic architecture mismatches the AMSPB agent's critic. It must:

- Load only the policy state_dict (not value, not optimizer).
- Reject corrupt checkpoints (NaN actor weights).
- Tolerate architecture differences in the actor (e.g. extra heads, extra
  log_std parameters) via strict=False.
- Optionally load the actor's state_preprocessor if both checkpoint and
  agent expose one.

Pure-torch tests — no Isaac Sim needed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Make `scripts/skrl/train.py` importable without invoking its argparse/Isaac
# Lab side-effects at import time. The helper we want is a pure function that
# only depends on torch + collections.abc.Mapping, so we import it by
# extracting and evaluating its source. This avoids needing to refactor
# train.py.

_TRAIN_PATH = Path(__file__).resolve().parent.parent / "scripts" / "skrl" / "train.py"


def _extract_function_source(source_path: Path, fn_name: str) -> str:
    """Return the source code of a top-level function from a script."""
    import ast

    tree = ast.parse(source_path.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            start_line = node.lineno
            end_line = node.end_lineno
            lines = source_path.read_text().splitlines()
            return "\n".join(lines[start_line - 1 : end_line])
    raise RuntimeError(f"function {fn_name} not found in {source_path}")


@pytest.fixture(scope="module")
def load_policy_only():
    """Compile the helper out of train.py without triggering the script's
    side-effects (argparse / Isaac Lab AppLauncher / etc.)."""
    src = _extract_function_source(_TRAIN_PATH, "_load_policy_only_from_checkpoint")
    namespace: dict = {"torch": torch}
    from collections.abc import Mapping
    namespace["Mapping"] = Mapping
    namespace["Any"] = object  # only used in type hints
    exec(src, namespace)
    return namespace["_load_policy_only_from_checkpoint"]


class _Actor(nn.Module):
    """Tiny stand-in for an skrl actor: linear + log_std parameter."""

    def __init__(self, in_dim: int = 8, out_dim: int = 4) -> None:
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)
        self.log_std_parameter = nn.Parameter(torch.zeros(out_dim))


class _Agent:
    """Tiny stand-in for an skrl Agent: exposes ``policy`` (and optionally
    ``_state_preprocessor``) just like the real agent does."""

    def __init__(self, policy: nn.Module, state_preprocessor=None) -> None:
        self.policy = policy
        self._state_preprocessor = state_preprocessor


def _build_payload(policy_state, value_state=None, state_preprocessor=None):
    """Match the layout `torch.save(agent.checkpoint())` writes in skrl: a
    dict with 'policy', 'value', and optionally 'state_preprocessor'."""
    payload = {"policy": policy_state}
    if value_state is not None:
        payload["value"] = value_state
    if state_preprocessor is not None:
        payload["state_preprocessor"] = state_preprocessor
    return payload


def test_load_policy_only_happy_path(tmp_path, load_policy_only):
    """A well-formed checkpoint loads cleanly; missing/unexpected = empty."""
    source_actor = _Actor()
    # Mutate source weights so we can check the load actually transferred them
    with torch.no_grad():
        source_actor.net.weight.fill_(0.42)
        source_actor.log_std_parameter.fill_(0.17)
    payload = _build_payload(source_actor.state_dict())
    path = tmp_path / "src.pt"
    torch.save(payload, path)

    target_actor = _Actor()
    agent = _Agent(target_actor)
    load_policy_only(agent, str(path))

    assert torch.allclose(target_actor.net.weight, torch.full_like(target_actor.net.weight, 0.42))
    assert torch.allclose(target_actor.log_std_parameter, torch.full_like(target_actor.log_std_parameter, 0.17))


def test_load_policy_only_rejects_nan(tmp_path, load_policy_only):
    """A checkpoint with any NaN in the actor weights must be refused."""
    source_actor = _Actor()
    with torch.no_grad():
        source_actor.net.weight[0, 0] = float("nan")
    payload = _build_payload(source_actor.state_dict())
    path = tmp_path / "nan.pt"
    torch.save(payload, path)

    target_actor = _Actor()
    agent = _Agent(target_actor)
    with pytest.raises(ValueError, match="NaN"):
        load_policy_only(agent, str(path))


def test_load_policy_only_rejects_bad_payload(tmp_path, load_policy_only):
    """A non-skrl-shaped payload (no 'policy' key) is refused with a
    diagnostic that lists what was actually inside."""
    path = tmp_path / "bogus.pt"
    torch.save({"only_value": 1.0}, path)
    agent = _Agent(_Actor())
    with pytest.raises(ValueError, match="not a skrl checkpoint"):
        load_policy_only(agent, str(path))


def test_load_policy_only_tolerates_architecture_diff(tmp_path, load_policy_only):
    """When the source actor has an extra parameter the target doesn't
    have (or vice versa), strict=False load logs missing / unexpected and
    proceeds — does NOT raise."""
    class _SourceActorWithExtra(_Actor):
        def __init__(self, in_dim=8, out_dim=4):
            super().__init__(in_dim, out_dim)
            self.extra_head = nn.Linear(out_dim, 2)
    source_actor = _SourceActorWithExtra()
    payload = _build_payload(source_actor.state_dict())
    path = tmp_path / "extra.pt"
    torch.save(payload, path)

    target_actor = _Actor()  # lacks extra_head
    agent = _Agent(target_actor)
    # Should not raise; load_state_dict with strict=False handles it.
    load_policy_only(agent, str(path))


def test_load_policy_only_does_not_touch_value(tmp_path, load_policy_only):
    """The value (critic) head — even if present in the checkpoint — must
    NOT be loaded. The agent's value head stays at random init."""
    source_actor = _Actor()
    bogus_value = {"fake_layer.weight": torch.full((4, 4), 999.0)}
    payload = _build_payload(source_actor.state_dict(), value_state=bogus_value)
    path = tmp_path / "with_value.pt"
    torch.save(payload, path)

    target_actor = _Actor()
    # No value attribute on the agent — if the loader tried to touch it,
    # it would AttributeError. Passing this test confirms the loader skips
    # the value section entirely.
    agent = _Agent(target_actor)
    load_policy_only(agent, str(path))


def test_load_policy_only_loads_state_preprocessor_if_both_sides_have_one(
    tmp_path, load_policy_only,
):
    """Optional: if the checkpoint AND the agent both expose a
    state_preprocessor, the loader transfers its state too (so the actor
    sees the same observation normalisation as during pretrain)."""
    source_actor = _Actor()
    # Stand-in preprocessor — anything with load_state_dict / state_dict
    class _Preproc(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(8))
    src_pre = _Preproc()
    with torch.no_grad():
        src_pre.scale.fill_(2.5)
    payload = _build_payload(
        source_actor.state_dict(),
        state_preprocessor=src_pre.state_dict(),
    )
    path = tmp_path / "with_pre.pt"
    torch.save(payload, path)

    target_pre = _Preproc()
    agent = _Agent(_Actor(), state_preprocessor=target_pre)
    load_policy_only(agent, str(path))

    assert torch.allclose(target_pre.scale, torch.full_like(target_pre.scale, 2.5))
