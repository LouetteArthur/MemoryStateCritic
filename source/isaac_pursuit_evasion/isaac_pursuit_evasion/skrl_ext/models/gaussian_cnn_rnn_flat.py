# Copyright (c) 2026, the MemoryStateCritic authors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CNN+GRU model for flat (non-Dict) image observations.

Designed for benchmarking on standard IsaacLab camera envs (e.g.
CartPole-Camera) where the observation space is a simple Box [H, W, C]
rather than a Dict with "image" and "past_actions" keys.

Architecture:
    permute(NHWC → NCHW) → CNN → Linear(128) → LayerNorm → GRU → Linear → actions

Uses the same design choices as the pursuit-evasion GaussianCNNGRUModel:
    - LayerNorm before GRU input
    - Orthogonal init for GRU weights
    - Done-state hidden resets during BPTT
"""

from collections.abc import Mapping
from typing import Any

import gymnasium
import torch
import torch.nn as nn
from skrl.models.torch import GaussianMixin, Model


class GaussianCNNGRUFlatModel(GaussianMixin, Model):
    """CNN+GRU for flat NHWC image observations (no Dict space)."""

    def __init__(
        self,
        observation_space: int | tuple[int] | gymnasium.Space | None = None,
        action_space: int | tuple[int] | gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20,
        max_log_std: float = 2,
        reduction: str = "sum",
        initial_log_std: float = 0.0,
        fixed_log_std: bool = False,
        rnn: Mapping[str, Any] | None = None,
        num_envs: int = 1,
        **kwargs,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        # Infer image shape from observation space [H, W, C]
        obs_shape = observation_space.shape
        if len(obs_shape) == 3:
            self._img_h, self._img_w, self._img_c = obs_shape
        else:
            raise ValueError(f"Expected [H, W, C] observation space, got shape {obs_shape}")

        # CNN encoder (same architecture as skrl CartPole-Camera baseline)
        self.cnn = nn.Sequential(
            nn.Conv2d(self._img_c, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # RNN configuration
        rnn_cfg = rnn or {}
        self._rnn_hidden_size = int(rnn_cfg.get("hidden_size", 128))
        self._rnn_num_layers = int(rnn_cfg.get("num_layers", 1))
        self._rnn_sequence_length = int(rnn_cfg.get("sequence_length", 16))
        self._rnn_num_envs = max(int(num_envs), 1)
        rnn_dropout = float(rnn_cfg.get("dropout", 0.0)) if self._rnn_num_layers > 1 else 0.0

        cnn_feature_size = 128
        self.cnn_linear = nn.Sequential(
            nn.LazyLinear(cnn_feature_size),
            nn.ELU(),
        )

        # LayerNorm before GRU
        self.input_ln = nn.LayerNorm(cnn_feature_size)

        self.rnn = nn.GRU(
            input_size=cnn_feature_size,
            hidden_size=self._rnn_hidden_size,
            num_layers=self._rnn_num_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )
        # Orthogonal init for GRU weights
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        self.output_layer = nn.Linear(self._rnn_hidden_size, self.num_actions)
        self.log_std_parameter = nn.Parameter(
            torch.full((self.num_actions,), float(initial_log_std)),
            requires_grad=not fixed_log_std,
        )

    def get_specification(self) -> Mapping[str, Any]:
        return {
            "rnn": {
                "sizes": [(self._rnn_num_layers, self._rnn_num_envs, self._rnn_hidden_size)],
                "sequence_length": self._rnn_sequence_length,
            }
        }

    def compute(self, inputs: Mapping[str, Any], role: str = ""):
        raw_states = inputs.get("states")

        # Handle sequence dimension for BPTT training
        has_seq = raw_states.dim() == 3
        if has_seq:
            batch_size, seq_len, flat_dim = raw_states.shape
            states_2d = raw_states.reshape(batch_size * seq_len, flat_dim)
        else:
            batch_size = raw_states.shape[0]
            seq_len = 1
            states_2d = raw_states

        # Reshape flat observation to [N, H, W, C] then permute to [N, C, H, W]
        images = states_2d.reshape(-1, self._img_h, self._img_w, self._img_c)
        images = images.permute(0, 3, 1, 2)  # NHWC → NCHW

        # CNN forward
        cnn_out = self.cnn(images)
        features = self.cnn_linear(cnn_out)  # (batch*seq, 128)

        # LayerNorm before GRU
        features = self.input_ln(features)

        # Reshape for GRU: (batch, seq, feature_size)
        x = features.reshape(batch_size, seq_len, -1)

        # --- RNN hidden state handling ---
        rnn_states = inputs.get("rnn")
        if isinstance(rnn_states, torch.Tensor):
            rnn_states = [rnn_states]
        if rnn_states and len(rnn_states) > 0 and x.dim() == 3:
            h_batch = rnn_states[0].shape[1]
            if x.shape[0] != h_batch and x.shape[1] == h_batch:
                x = x.transpose(0, 1)
        if not rnn_states or len(rnn_states) == 0:
            h0 = torch.zeros(
                self._rnn_num_layers,
                x.shape[0],
                self._rnn_hidden_size,
                device=x.device,
                dtype=x.dtype,
            )
        else:
            h0 = rnn_states[0]
            if h0.dim() == 2:
                h0 = h0.unsqueeze(0)
            h0 = h0.contiguous()

        if has_seq and seq_len > 1:
            # BPTT with mid-sequence done resets
            terminated = inputs.get("terminated")
            outputs = []
            h = h0
            for t in range(seq_len):
                if terminated is not None and t > 0:
                    done_prev = terminated[:, t - 1]
                    if done_prev.dim() > 1:
                        done_prev = done_prev[..., 0]
                    done_prev = done_prev.bool()
                    if done_prev.any():
                        h = h.clone()
                        h[:, done_prev] = 0.0
                out_t, h = self.rnn(x[:, t : t + 1, :], h)
                outputs.append(out_t)
            rnn_out = torch.cat(outputs, dim=1)
            features = rnn_out.reshape(batch_size * seq_len, -1)
        else:
            # Single step (rollout)
            terminated = inputs.get("terminated")
            if terminated is not None:
                done = terminated
                if done.dim() > 1:
                    done = done[..., 0]
                if done.dim() > 1:
                    done = done[:, -1]
                done = done.reshape(-1).bool()
                if done.any():
                    h0 = h0.clone()
                    h0[:, done] = 0.0
            rnn_out, h = self.rnn(x, h0)
            features = rnn_out[:, -1, :]

        output = self.output_layer(features)
        return output, self.log_std_parameter, {"rnn": [h]}


def gaussian_cnn_rnn_flat_model(
    observation_space: int | tuple[int] | gymnasium.Space | None = None,
    action_space: int | tuple[int] | gymnasium.Space | None = None,
    device: str | torch.device | None = None,
    clip_actions: bool = False,
    clip_log_std: bool = True,
    min_log_std: float = -20,
    max_log_std: float = 2,
    reduction: str = "sum",
    initial_log_std: float = 0.0,
    fixed_log_std: bool = False,
    rnn: Mapping[str, Any] | None = None,
    return_source: bool = False,
    num_envs: int = 1,
    *args,
    **kwargs,
) -> Model | str:
    """Factory function for the flat-image CNN+GRU Gaussian model."""
    rnn_cfg = rnn or {}
    if return_source:
        return (
            "GaussianCNNGRUFlatModel(\n"
            "  CNN: Conv2d(in→32, k=8,s=4) → Conv2d(32→64, k=4,s=2) → Conv2d(64→64, k=3,s=1) → Flatten → Linear(128)\n"
            f"  LayerNorm(128) → GRU(hidden={rnn_cfg.get('hidden_size', 128)}, "
            f"layers={rnn_cfg.get('num_layers', 1)}, seq={rnn_cfg.get('sequence_length', 16)}) [orthogonal init]\n"
            f"  Output: Linear({rnn_cfg.get('hidden_size', 128)}, num_actions)\n"
            ")"
        )

    return GaussianCNNGRUFlatModel(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        clip_actions=clip_actions,
        clip_log_std=clip_log_std,
        min_log_std=min_log_std,
        max_log_std=max_log_std,
        reduction=reduction,
        initial_log_std=initial_log_std,
        fixed_log_std=fixed_log_std,
        rnn=rnn,
        num_envs=num_envs,
    )
