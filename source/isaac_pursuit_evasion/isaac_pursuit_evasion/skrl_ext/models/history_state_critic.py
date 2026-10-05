# Copyright (c) 2026, the MemoryStateCritic authors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""History-state critic V(s, z^c): the paper's baseline (Baisero & Amato, 2022).

The critic has its **own** CNN encoder and GRU ``f_psi``, separate from the
actor's. It reads the same observation-action stream as the actor, with
independent weights, and its GRU output ``z^c = f_psi(h)`` is concatenated with
the privileged state and fed through an MLP. Unlike the memory-state critic
(``memory_state_critic.py``), it is trained end to end by the value loss.

The model exposes RNN specification metadata so that ``PPO_RNN_ASYM`` can
manage the critic's hidden state (reset on episode boundaries, BPTT over
sequences, etc.).
"""

from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium
import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, Model


class HistoryStateCriticModel(DeterministicMixin, Model):
    """V(s, z^c) critic: value = MLP(concat(state, GRU(CNN(o), a))).

    The critic maintains its own CNN+GRU pipeline over the learner's
    observation-action stream. The GRU hidden state ``z^c`` is the critic's
    own encoding of the history h = (o_0, a_0, ..., o_t).

    Parameters
    ----------
    observation_space : gymnasium.Space
        The privileged state space (e.g. 35-dim flat Box).
    action_space : gymnasium.Space
        The action space (used to determine action_dim for GRU input).
    image_space : gymnasium.Space or tuple
        Shape of the image observation (C, H, W). Must be provided so the
        critic CNN can be sized correctly.
    past_actions_size : int
        Number of past action dimensions concatenated with CNN features
        before the GRU (same as actor's obs_num_past_actions * action_dim).
    rnn : dict
        RNN config: hidden_size, num_layers, sequence_length.
    num_envs : int
        Number of parallel environments (for hidden state sizing).
    cnn_feature_size : int
        Output size of the CNN linear projection.
    layers : tuple of int
        MLP head layer sizes after concat(state, gru_output).
    """

    def __init__(
        self,
        observation_space: int | tuple[int] | gymnasium.Space | None = None,
        action_space: int | tuple[int] | gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        clip_actions: bool = False,
        image_channels: int = 2,
        image_height: int = 64,
        image_width: int = 64,
        past_actions_size: int = 4,
        rnn: Mapping[str, Any] | None = None,
        num_envs: int = 1,
        cnn_feature_size: int = 128,
        layers: Sequence[int] = (256, 128, 64),
        **kwargs,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self._past_actions_size = past_actions_size
        self._cnn_feature_size = cnn_feature_size
        # RNN configuration
        rnn_cfg = rnn or {}
        self._rnn_hidden_size = int(rnn_cfg.get("hidden_size", 256))
        self._rnn_num_layers = int(rnn_cfg.get("num_layers", 1))
        self._rnn_sequence_length = int(rnn_cfg.get("sequence_length", 16))
        self._rnn_num_envs = max(int(num_envs), 1)

        # --- Learner branch: CNN + GRU ---
        self.cnn = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.cnn_linear = nn.Sequential(
            nn.LazyLinear(cnn_feature_size),
            nn.ELU(),
        )

        gru_input_size = cnn_feature_size + past_actions_size
        self.input_ln = nn.LayerNorm(gru_input_size)

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=self._rnn_hidden_size,
            num_layers=self._rnn_num_layers,
            batch_first=True,
        )
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        # MLP head: concat(state, z^c) -> value
        mlp_input_size = self.num_observations + self._rnn_hidden_size

        modules: list[nn.Module] = []
        in_dim = mlp_input_size
        for out_dim in layers:
            modules.extend([nn.Linear(in_dim, out_dim), nn.ELU()])
            in_dim = out_dim
        modules.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*modules)

        # Store image shape for unflattening
        self._image_shape = (image_channels, image_height, image_width)
        self._image_size = image_channels * image_height * image_width

    def get_specification(self) -> Mapping[str, Any]:
        return {
            "rnn": {
                "sizes": [(self._rnn_num_layers, self._rnn_num_envs, self._rnn_hidden_size)],
                "sequence_length": self._rnn_sequence_length,
            }
        }

    def _run_gru_branch(
        self,
        gru: nn.GRU,
        x: torch.Tensor,
        h0: torch.Tensor,
        has_seq: bool,
        seq_len: int,
        terminated: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a GRU branch with BPTT and mid-sequence done resets.

        Returns (features, h_final) where features is (batch*seq, hidden) or (batch, hidden).
        """
        batch_size = x.shape[0]
        if has_seq and seq_len > 1:
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
                out_t, h = gru(x[:, t : t + 1, :], h)
                outputs.append(out_t)
            rnn_out = torch.cat(outputs, dim=1)
            features = rnn_out.reshape(batch_size * seq_len, -1)
        else:
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
            rnn_out, h = gru(x, h0)
            features = rnn_out[:, -1, :]
        return features, h

    def compute(self, inputs: Mapping[str, Any], role: str = ""):
        """Forward pass.

        Expected inputs:
        - "states": preprocessed privileged state (batch, state_dim) or (batch, seq, state_dim)
        - "critic_image": image tensor (batch, C, H, W) or (batch, seq, C, H, W)
        - "critic_past_actions": past actions (batch, A) or (batch, seq, A)
        - "rnn": [h] hidden state of the critic GRU
        - "terminated": done flags for mid-sequence hidden state resets
        """
        states = inputs.get("states")
        critic_image = inputs.get("critic_image")
        critic_past_actions = inputs.get("critic_past_actions")

        # Handle sequence dimension for BPTT training
        has_seq = states.dim() == 3
        if has_seq:
            batch_size, seq_len, state_dim = states.shape
        else:
            batch_size = states.shape[0]
            seq_len = 1
            state_dim = states.shape[-1]

        # If no image provided, use zeros (fallback for initial eval)
        if critic_image is None:
            flat_batch = batch_size * seq_len if has_seq else batch_size
            critic_image = torch.zeros(
                flat_batch,
                *self._image_shape,
                device=states.device,
                dtype=states.dtype,
            )
        if critic_past_actions is None:
            flat_batch = batch_size * seq_len if has_seq else batch_size
            critic_past_actions = torch.zeros(
                flat_batch,
                self._past_actions_size,
                device=states.device,
                dtype=states.dtype,
            )

        # Flatten seq dim for CNN
        if has_seq and critic_image.dim() == 5:
            critic_image = critic_image.reshape(batch_size * seq_len, *self._image_shape)
        if has_seq and critic_past_actions.dim() == 3:
            critic_past_actions = critic_past_actions.reshape(batch_size * seq_len, -1)

        # CNN forward
        cnn_out = self.cnn(critic_image)
        cnn_features = self.cnn_linear(cnn_out)

        if self._past_actions_size > 0:
            features = torch.cat([cnn_features, critic_past_actions], dim=-1)
        else:
            features = cnn_features
        features = self.input_ln(features)

        # Reshape for GRU: (batch, seq, gru_input_size)
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

        terminated = inputs.get("terminated")
        gru_features, h = self._run_gru_branch(self.gru, x, h0, has_seq, seq_len, terminated)

        # Flatten states for MLP if needed
        if has_seq:
            states_flat = states.reshape(batch_size * seq_len, state_dim)
        else:
            states_flat = states

        value = self.net(torch.cat([states_flat, gru_features], dim=-1))
        return value, {"rnn": [h]}


def history_state_critic_model(
    observation_space: int | tuple[int] | gymnasium.Space | None = None,
    action_space: int | tuple[int] | gymnasium.Space | None = None,
    device: str | torch.device | None = None,
    clip_actions: bool = False,
    image_channels: int = 2,
    image_height: int = 64,
    image_width: int = 64,
    past_actions_size: int = 4,
    rnn: Mapping[str, Any] | None = None,
    num_envs: int = 1,
    cnn_feature_size: int = 128,
    layers: Sequence[int] = (256, 128, 64),
    return_source: bool = False,
    *args,
    **kwargs,
) -> Model | str:
    """Factory function for the V(s, z^c) history-state critic.

    Called by the skrl Runner when the YAML config specifies
    ``class: HistoryStateCriticMixin``.
    """
    rnn_cfg = rnn or {}
    if return_source:
        return (
            "HistoryStateCriticModel(\n"
            f"  CNN: Conv2d({image_channels}→32, k=8,s=4) → Conv2d(32→64, k=4,s=2) → "
            f"Conv2d(64→64, k=3,s=1) → Linear({cnn_feature_size})\n"
            f"  GRU: input={cnn_feature_size}+{past_actions_size}, "
            f"hidden={rnn_cfg.get('hidden_size', 256)}, "
            f"layers={rnn_cfg.get('num_layers', 1)}, "
            f"seq_len={rnn_cfg.get('sequence_length', 16)}\n"
            f"  MLP: state({observation_space}) + gru_out → {list(layers)} → 1\n"
            ")"
        )

    return HistoryStateCriticModel(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        clip_actions=clip_actions,
        image_channels=image_channels,
        image_height=image_height,
        image_width=image_width,
        past_actions_size=past_actions_size,
        rnn=rnn,
        num_envs=num_envs,
        cnn_feature_size=cnn_feature_size,
        layers=layers,
    )
