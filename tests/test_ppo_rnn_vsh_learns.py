"""End-to-end smoke test: PPO_RNN_VSH + CustomRunner must learn a toy POMDP.

This test is a *gate* for the full pipeline. It constructs a minimal
partially-observed environment that is only solvable by a policy with
working recurrence, instantiates ``PPO_RNN_VSH`` via ``CustomRunner``, runs
a short training budget on CPU, and asserts that the average episode
return improves meaningfully from the first few iterations to the last.

If this test fails, something in the stack is broken:
- the recurrent forward/backward (caught earlier by ``test_gru_actor.py``),
- the BPTT sequence training in PPO_RNN_VSH,
- the asymmetric critic wiring (critic reads a 2-dim privileged state),
- the V(s, h) augmentation (actor_hidden injected into the critic),
- the ``CustomRunner`` registration of our model/agent classes.

The test is intentionally *slow by unit-test standards* (tens of seconds
on CPU) because no faster check can show convergence. It runs on CPU with
tiny networks and does not require Isaac Sim. If you don't want it in
the fast-lane pytest run, mark it with ``pytest.mark.slow`` — but don't
delete it.

The POMDP
---------
- ``num_envs`` parallel 1-D point agents in ``[-1, 1]``.
- Each episode a random ``target`` is drawn in ``[-1, 1]``.
- On ``t == 0``, the observation image is a 48x48 frame with the target's
  x-coordinate encoded as a bright vertical stripe. On ``t > 0``, the image
  is all zeros (no information).
- ``past_actions``: last 3 actions, flat (3 dims here, 1-D action).
- Privileged state (critic only): ``[agent_x, target_x]``.
- Action: 1-D velocity in ``[-1, 1]``, applied with step size 0.25.
- Reward: ``-|agent_x - target_x|`` at every step.
- Episode length: 8 steps.

Because the target is only visible at ``t == 0``, a memoryless policy
cannot distinguish episodes with different targets after the first step
and is stuck at the expected-value baseline. A recurrent policy that
stores the target in its GRU hidden state can do strictly better, which
is exactly what we want to verify.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Package stub: avoid importing the heavy isaac_pursuit_evasion/__init__.py
# which pulls in Isaac Lab.
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SKRL_EXT_PARENT = _PROJECT_ROOT / "source" / "isaac_pursuit_evasion"
sys.path.insert(0, str(_SKRL_EXT_PARENT))

if "isaac_pursuit_evasion" not in sys.modules or not hasattr(
    sys.modules["isaac_pursuit_evasion"], "skrl_ext"
):
    _pkg = types.ModuleType("isaac_pursuit_evasion")
    _pkg.__path__ = [str(_SKRL_EXT_PARENT / "isaac_pursuit_evasion")]
    sys.modules["isaac_pursuit_evasion"] = _pkg

# Disable wandb and tensorboard writers during the test.
os.environ.setdefault("WANDB_MODE", "disabled")

from isaac_pursuit_evasion.skrl_ext import CustomRunner  # noqa: E402
from skrl.envs.wrappers.torch.base import Wrapper  # noqa: E402
from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal batched POMDP — directly implements skrl's Wrapper interface so
# we don't need a real gymnasium env + wrap_env indirection.
# ---------------------------------------------------------------------------


IMG_H, IMG_W = 48, 48  # CNN (k=8/4/3, s=4/2/1) needs >= 48
N_PAST_ACTIONS = 3
ACTION_DIM = 1
EPISODE_LEN = 8
STATE_DIM = 2  # [agent_x, target_x]


class RememberTargetPOMDP(Wrapper):
    """Skrl-wrapper-compatible tiny POMDP.

    The target is only visible in the first-step image; after that the
    agent must rely on its GRU hidden state to keep moving toward it.
    """

    def __init__(self, num_envs: int, device: str = "cpu", seed: int = 0) -> None:
        self._num_envs = num_envs
        self._device = torch.device(device)
        self._rng = torch.Generator(device=self._device).manual_seed(seed)

        self._unwrapped = self  # satisfy Wrapper.__getattr__ / .device
        self._env = self
        self._agent = torch.zeros(num_envs, device=self._device)
        self._target = torch.zeros(num_envs, device=self._device)
        self._past_actions = torch.zeros(num_envs, N_PAST_ACTIONS, device=self._device)
        self._step_idx = torch.zeros(num_envs, dtype=torch.long, device=self._device)

        # Episode-return bookkeeping for the learning check.
        self._running_return = torch.zeros(num_envs, device=self._device)
        self.completed_returns: list[float] = []

    # ---- skrl Wrapper interface -------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def num_agents(self) -> int:
        # Override Wrapper.num_agents so it doesn't recurse via ``self._unwrapped``.
        return 1

    @property
    def observation_space(self) -> spaces.Space:
        return spaces.Dict(
            {
                "image": spaces.Box(low=0.0, high=1.0, shape=(1, IMG_H, IMG_W), dtype=np.float32),
                "past_actions": spaces.Box(low=-1.0, high=1.0, shape=(N_PAST_ACTIONS,), dtype=np.float32),
            }
        )

    @property
    def action_space(self) -> spaces.Space:
        return spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)

    @property
    def state_space(self) -> spaces.Space:
        return spaces.Box(low=-1.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32)

    # ---- internal ---------------------------------------------------------------

    def _build_observation(self) -> torch.Tensor:
        """Produce a flat observation tensor matching the Dict layout.

        ``skrl.utils.spaces.torch.flatten_tensorized_space`` concatenates the
        Dict sub-spaces in ``sorted(space.keys())`` order, which here is
        ``image`` then ``past_actions``.
        """
        img = torch.zeros(self._num_envs, 1, IMG_H, IMG_W, device=self._device)
        # Step 0: encode target as a vertical bright stripe.
        visible = self._step_idx == 0
        if visible.any():
            cols = ((self._target[visible] + 1.0) * 0.5 * (IMG_W - 1)).long().clamp(0, IMG_W - 1)
            for env_idx, col in zip(visible.nonzero(as_tuple=True)[0].tolist(), cols.tolist()):
                img[env_idx, 0, :, col] = 1.0

        obs_dict = {"image": img, "past_actions": self._past_actions}
        tensorized = tensorize_space(self.observation_space, obs_dict, device=self._device)
        return flatten_tensorized_space(tensorized)

    def _privileged_state(self) -> torch.Tensor:
        return torch.stack([self._agent, self._target], dim=-1)

    # ---- reset/step -------------------------------------------------------------

    def reset(self):
        self._agent.zero_()
        self._target.uniform_(-1.0, 1.0, generator=self._rng)
        self._past_actions.zero_()
        self._step_idx.zero_()
        self._running_return.zero_()

        obs = self._build_observation()
        info = {"critic_states": self._privileged_state()}
        return obs, info

    def step(self, actions: torch.Tensor):
        # Clip actions to the box and apply dynamics.
        a = actions.clamp(-1.0, 1.0).reshape(self._num_envs, ACTION_DIM)
        self._agent = (self._agent + 0.25 * a.squeeze(-1)).clamp(-1.0, 1.0)

        # Slide the past-actions buffer: [a_{t-2}, a_{t-1}, a_t].
        self._past_actions = torch.cat(
            [self._past_actions[:, 1:], a], dim=-1
        ).reshape(self._num_envs, N_PAST_ACTIONS)

        reward = -(self._agent - self._target).abs().reshape(self._num_envs, 1)
        self._running_return += reward.reshape(self._num_envs)
        self._step_idx += 1

        terminated = torch.zeros(self._num_envs, 1, dtype=torch.bool, device=self._device)
        truncated = (self._step_idx >= EPISODE_LEN).reshape(self._num_envs, 1)

        obs = self._build_observation()
        # For PPO_RNN bootstrapping, critic_states must track the *current*
        # (next) state; next_critic_states is used at the end of rollout.
        info = {
            "critic_states": self._privileged_state(),
            "next_critic_states": self._privileged_state(),
        }

        # Auto-reset terminated/truncated envs (like an Isaac Lab env would).
        done = truncated.reshape(self._num_envs)
        if done.any():
            # Record the returns of completed episodes.
            self.completed_returns.extend(self._running_return[done].tolist())
            self._agent[done] = 0.0
            self._target[done] = (torch.rand(done.sum(), generator=self._rng, device=self._device) * 2 - 1)
            self._past_actions[done] = 0.0
            self._step_idx[done] = 0
            self._running_return[done] = 0.0
            obs = self._build_observation()

        return obs, reward, terminated, truncated, info

    def render(self, *args, **kwargs):  # noqa: D401 — skrl interface
        return None

    def close(self) -> None:  # noqa: D401 — skrl interface
        return None

    def state(self) -> torch.Tensor:
        return self._privileged_state()


# ---------------------------------------------------------------------------
# Runner config — minimal PPO_RNN_VSH setup scaled for a CPU smoke test.
# ---------------------------------------------------------------------------


def _make_cfg(timesteps: int) -> dict:
    return {
        "seed": 42,
        "models": {
            "separate": True,
            "policy": {
                "class": "GaussianCNNRNNMixin",
                "clip_actions": False,
                "clip_log_std": True,
                "min_log_std": -20.0,
                "max_log_std": 2.0,
                "initial_log_std": 0.0,
                "rnn": {
                    "type": "gru",
                    "hidden_size": 32,
                    "num_layers": 1,
                    "sequence_length": 8,  # BPTT across full episode
                },
            },
            "value": {
                "class": "VshCriticMixin",
                "clip_actions": False,
                "actor_hidden_size": 32,
                "layers": [32, 32],
            },
        },
        "memory": {"class": "RandomMemory", "memory_size": -1},
        "agent": {
            "class": "PPO_RNN_VSH",
            "rollouts": 32,
            "learning_epochs": 2,
            "mini_batches": 2,
            "discount_factor": 0.99,
            "lambda": 0.95,
            "learning_rate": 1.0e-04,
            "state_preprocessor": None,
            "state_preprocessor_kwargs": None,
            "critic_state_preprocessor": "RunningStandardScaler",
            "critic_state_preprocessor_kwargs": None,
            "value_preprocessor": "RunningStandardScaler",
            "value_preprocessor_kwargs": None,
            "random_timesteps": 0,
            "learning_starts": 0,
            "grad_norm_clip": 1.0,
            "ratio_clip": 0.2,
            "value_clip": 0.2,
            "clip_predicted_values": True,
            "entropy_loss_scale": 0.01,
            "value_loss_scale": 1.0,
            "kl_threshold": 0.0,
            "rewards_shaper_scale": 1.0,
            "time_limit_bootstrap": False,
            "vsh_critic": True,
            "vsh_actor_hidden_size": 32,
            "mixed_precision": False,
            "experiment": {
                "directory": "",
                "experiment_name": "",
                "write_interval": 0,
                "checkpoint_interval": 0,
                "wandb": False,
            },
        },
        "trainer": {
            "class": "SequentialTrainer",
            "timesteps": timesteps,
            "environment_info": "log",
            "close_environment_at_exit": False,
        },
    }


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_ppo_rnn_vsh_learns_remember_target():
    """A short training run must noticeably improve episode return.

    Runs PPO_RNN_VSH with BPTT on the ``RememberTargetPOMDP``. The
    task is fully solvable by a recurrent policy: with a random policy the
    expected return is around ``-4`` (mean distance ~0.5 over 8 steps); a
    perfect policy gets close to ``-0.5``. We only assert that the last
    window of episode returns is meaningfully better than the first
    window, which is robust to the exact budget and initialization.
    """
    torch.manual_seed(0)
    num_envs = 16
    timesteps = 30000  # ~59 PPO iterations (rollouts=32 per env)

    env = RememberTargetPOMDP(num_envs=num_envs, device="cpu", seed=0)

    cfg = _make_cfg(timesteps=timesteps)
    runner = CustomRunner(env, cfg)
    runner.run(mode="train")

    returns = np.array(env.completed_returns, dtype=np.float64)
    assert len(returns) >= 40, (
        f"too few completed episodes to judge learning: {len(returns)}. "
        f"Increase timesteps or num_envs."
    )

    # Fixed window sizes: capture the *untrained* baseline in "early" and the
    # *trained* policy in "late". A fractional window would let early creep
    # into training — with ~40k episodes per run, even 20% covers thousands of
    # PPO updates and stops being a baseline.
    window = 200
    early = returns[:window]
    late = returns[-window:]
    improvement = float(late.mean() - early.mean())

    # Per-episode return variance on this POMDP is dominated by the random
    # target distance (path-integral cost depends on |target|), not by policy
    # noise — so we use reward-range thresholds instead of a within-window
    # std check. Random policy ≈ -4, optimal policy ≈ -1, so the learning
    # headroom is ~3 reward units.
    print(
        f"\n[ppo_rnn_vsh smoke] episodes={len(returns)} "
        f"early_mean={early.mean():.3f}+-{early.std():.3f} "
        f"late_mean={late.mean():.3f} improvement={improvement:.3f}"
    )

    # Random policy ≈ -4 to -4.7, optimal policy ≈ -1. An improvement
    # of 0.8 is ~25% of the 3-unit headroom and well above the noise
    # floor of a non-learning run (~0). This threshold is conservative
    # enough to pass reliably on CPU with the tiny training budget while
    # catching pipeline regressions (broken BPTT, dead gradients, etc.).
    assert improvement > 0.8, (
        f"PPO_RNN_VSH did not learn the remember-target POMDP: "
        f"early mean return {early.mean():.3f}, late mean return {late.mean():.3f}, "
        f"improvement {improvement:.3f}. "
        f"Either the pipeline is broken or the training budget is too small."
    )


# ---------------------------------------------------------------------------
# V(s, h) history-state critic config
# ---------------------------------------------------------------------------


def _make_sh_cfg(timesteps: int) -> dict:
    """Config for PPO_RNN_SH with HistoryStateCriticMixin."""
    return {
        "seed": 42,
        "models": {
            "separate": True,
            "policy": {
                "class": "GaussianCNNRNNMixin",
                "clip_actions": False,
                "clip_log_std": True,
                "min_log_std": -20.0,
                "max_log_std": 2.0,
                "initial_log_std": 0.0,
                "rnn": {
                    "type": "gru",
                    "hidden_size": 32,
                    "num_layers": 1,
                    "sequence_length": 8,
                },
            },
            "value": {
                "class": "HistoryStateCriticMixin",
                "clip_actions": False,
                "image_channels": 1,
                "image_height": IMG_H,
                "image_width": IMG_W,
                "past_actions_size": N_PAST_ACTIONS,
                "cnn_feature_size": 32,
                "layers": [32, 32],
                "rnn": {
                    "hidden_size": 32,
                    "num_layers": 1,
                    "sequence_length": 8,
                },
            },
        },
        "memory": {"class": "RandomMemory", "memory_size": -1},
        "agent": {
            "class": "PPO_RNN_SH",
            "rollouts": 32,
            "learning_epochs": 2,
            "mini_batches": 2,
            "discount_factor": 0.99,
            "lambda": 0.95,
            "learning_rate": 1.0e-04,
            "state_preprocessor": None,
            "state_preprocessor_kwargs": None,
            "critic_state_preprocessor": "RunningStandardScaler",
            "critic_state_preprocessor_kwargs": None,
            "value_preprocessor": "RunningStandardScaler",
            "value_preprocessor_kwargs": None,
            "random_timesteps": 0,
            "learning_starts": 0,
            "grad_norm_clip": 1.0,
            "ratio_clip": 0.2,
            "value_clip": 0.2,
            "clip_predicted_values": True,
            "entropy_loss_scale": 0.01,
            "value_loss_scale": 1.0,
            "kl_threshold": 0.0,
            "rewards_shaper_scale": 1.0,
            "time_limit_bootstrap": False,
            "sh_critic": True,
            "sh_image_shape": [1, IMG_H, IMG_W],
            "sh_past_actions_size": N_PAST_ACTIONS,
            "mixed_precision": False,
            "experiment": {
                "directory": "",
                "experiment_name": "",
                "write_interval": 0,
                "checkpoint_interval": 0,
                "wandb": False,
            },
        },
        "trainer": {
            "class": "SequentialTrainer",
            "timesteps": timesteps,
            "environment_info": "log",
            "close_environment_at_exit": False,
        },
    }


@pytest.mark.slow
def test_ppo_rnn_sh_learns_remember_target():
    """V(s,h) history-state critic must also learn the remember-target POMDP.

    Same test as V(s,z) but using HistoryStateCriticMixin with its own
    CNN+GRU. The critic has separate visual processing and recurrence from
    the actor, so BPTT flows through both RNNs independently.
    """
    torch.manual_seed(0)
    num_envs = 16
    timesteps = 30000

    env = RememberTargetPOMDP(num_envs=num_envs, device="cpu", seed=0)

    cfg = _make_sh_cfg(timesteps=timesteps)
    runner = CustomRunner(env, cfg)
    runner.run(mode="train")

    returns = np.array(env.completed_returns, dtype=np.float64)
    assert len(returns) >= 40, (
        f"too few completed episodes to judge learning: {len(returns)}. "
        f"Increase timesteps or num_envs."
    )

    window = 200
    early = returns[:window]
    late = returns[-window:]
    improvement = float(late.mean() - early.mean())

    print(
        f"\n[ppo_rnn_sh smoke] episodes={len(returns)} "
        f"early_mean={early.mean():.3f}+-{early.std():.3f} "
        f"late_mean={late.mean():.3f} improvement={improvement:.3f}"
    )

    assert improvement > 0.8, (
        f"PPO_RNN_SH did not learn the remember-target POMDP: "
        f"early mean return {early.mean():.3f}, late mean return {late.mean():.3f}, "
        f"improvement {improvement:.3f}. "
        f"Either the pipeline is broken or the training budget is too small."
    )
