from typing import Any, Mapping, Optional, Tuple, Union

import gymnasium
import numpy as np
from gymnasium import spaces

import torch
import torch.nn as nn

from skrl.models.torch import GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space


class GaussianCNNGRUModel(GaussianMixin, Model):
    """CNN encoder + GRU for vision-based RL with temporal memory.

    Architecture:
        z_t = CNN(image_t) -> Linear(128)
        x_t = concat(z_t, a_{t-1})               # previous action appended
        h_t = GRU(x_t, h_{t-1})
        a_t = Linear(h_t)

    This follows the canonical recurrent RL recipe (R2D2, IMPALA, Ni et al.
    2022): the RNN receives both the visual encoding and the previous
    action at each step.  Feeding ``a_{t-1}`` is essential in a POMDP with
    partial visual observability — without it the agent cannot disambiguate
    self-motion from target motion or close the proprioceptive loop from
    images alone.

    Observation space must be a gymnasium Dict with:
      - ``"image"``: ``(C, H, W)`` visual input
      - ``"past_actions"``: flat vector of the last N actions (any N >= 1;
        all of them are concatenated with the CNN features before the GRU)
    """

    def __init__(
        self,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20,
        max_log_std: float = 2,
        reduction: str = "sum",
        initial_log_std: float = 0.0,
        fixed_log_std: bool = False,
        rnn: Optional[Mapping[str, Any]] = None,
        num_envs: int = 1,
        **kwargs,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        # Determine image channels and past-actions size from observation space
        if isinstance(observation_space, spaces.Dict):
            image_shape = observation_space["image"].shape  # (C, H, W)
            in_channels = image_shape[0]
            if "past_actions" in observation_space.spaces:
                self._past_actions_size = int(np.prod(observation_space["past_actions"].shape))
            else:
                self._past_actions_size = 0
        else:
            raise ValueError(
                f"GaussianCNNGRUModel requires Dict observation space with 'image' key, "
                f"got {type(observation_space)}"
            )

        # CNN encoder — same architecture as skrl_ppo_vision_*_cfg.yaml
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # RNN configuration
        rnn_cfg = rnn or {}
        self._rnn_hidden_size = int(rnn_cfg.get("hidden_size", 256))
        self._rnn_num_layers = int(rnn_cfg.get("num_layers", 1))
        self._rnn_sequence_length = int(rnn_cfg.get("sequence_length", 16))
        self._rnn_num_envs = max(int(num_envs), 1)
        rnn_dropout = float(rnn_cfg.get("dropout", 0.0)) if self._rnn_num_layers > 1 else 0.0

        cnn_feature_size = 128
        self.cnn_linear = nn.Sequential(
            nn.LazyLinear(cnn_feature_size),
            nn.ELU(),
        )

        # LayerNorm on the full GRU input stabilizes scale across CNN features
        # and past_actions, preventing either from dominating at init or during
        # training.  Standard practice in recurrent RL (R2D2, IMPALA, DreamerV3).
        gru_input_size = cnn_feature_size + self._past_actions_size
        self.input_ln = nn.LayerNorm(gru_input_size)

        self.rnn = nn.GRU(
            input_size=gru_input_size,
            hidden_size=self._rnn_hidden_size,
            num_layers=self._rnn_num_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )
        # Orthogonal init for GRU weights — prevents vanishing/exploding
        # gradients through long sequences (standard in recurrent RL).
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

        # Unflatten Dict observation space (image + past_actions)
        states = unflatten_tensorized_space(self.observation_space, states_2d)

        # CNN forward pass on (batch*seq, C, H, W)
        image = states["image"]
        cnn_out = self.cnn(image)
        cnn_features = self.cnn_linear(cnn_out)  # (batch*seq, 128)

        # Concatenate previous action(s) with CNN features
        # (R2D2/IMPALA/Ni et al. recipe: RNN input = [visual_encoding, a_{t-1}])
        if self._past_actions_size > 0:
            past_actions = states["past_actions"].reshape(cnn_features.shape[0], -1)
            features = torch.cat([cnn_features, past_actions], dim=-1)
        else:
            features = cnn_features

        # LayerNorm before GRU — stabilizes scale of CNN features + past_actions
        features = self.input_ln(features)

        # Reshape for GRU: (batch, seq, gru_input_size)
        x = features.reshape(batch_size, seq_len, -1)

        # --- RNN hidden state handling (same logic as GaussianRNNModel) ---
        rnn_states = inputs.get("rnn", None)
        if isinstance(rnn_states, torch.Tensor):
            rnn_states = [rnn_states]
        if rnn_states and len(rnn_states) > 0 and x.dim() == 3:
            h_batch = rnn_states[0].shape[1]
            if x.shape[0] != h_batch and x.shape[1] == h_batch:
                x = x.transpose(0, 1)
        if not rnn_states or len(rnn_states) == 0:
            h0 = torch.zeros(
                self._rnn_num_layers, x.shape[0], self._rnn_hidden_size,
                device=x.device, dtype=x.dtype,
            )
        else:
            h0 = rnn_states[0]
            if h0.dim() == 2:
                h0 = h0.unsqueeze(0)
            h0 = h0.contiguous()

        if has_seq and seq_len > 1:
            # BPTT: step through sequence with mid-sequence done resets.
            # At each step, if done_{t} is True the env auto-reset, so we
            # zero the hidden state before processing obs_{t+1}.
            terminated = inputs.get("terminated", None)
            outputs = []
            h = h0
            for t in range(seq_len):
                # Reset h for envs that finished at the PREVIOUS step
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
            rnn_out = torch.cat(outputs, dim=1)  # (batch, seq_len, hidden)
            # Return ALL timestep outputs, flattened to 2D for GaussianMixin
            features = rnn_out.reshape(batch_size * seq_len, -1)
        else:
            # Single step (rollout / stored-state fallback)
            # Reset h0 for done envs
            terminated = inputs.get("terminated", None)
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
            features = rnn_out[:, -1, :]  # (batch, hidden)

        output = self.output_layer(features)
        return output, self.log_std_parameter, {"rnn": [h]}


def gaussian_cnn_rnn_model(
    observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
    action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
    device: Optional[Union[str, torch.device]] = None,
    clip_actions: bool = False,
    clip_log_std: bool = True,
    min_log_std: float = -20,
    max_log_std: float = 2,
    reduction: str = "sum",
    initial_log_std: float = 0.0,
    fixed_log_std: bool = False,
    rnn: Optional[Mapping[str, Any]] = None,
    return_source: bool = False,
    num_envs: int = 1,
    *args,
    **kwargs,
) -> Union[Model, str]:
    """Factory function for the CNN+GRU Gaussian model.

    Called by the skrl Runner when the YAML config specifies
    ``class: GaussianCNNRNNMixin``.
    """
    rnn_cfg = rnn or {}
    if return_source:
        return (
            f"GaussianCNNGRUModel(\n"
            f"  CNN: Conv2d(in→32, k=8,s=4) → Conv2d(32→64, k=4,s=2) → Conv2d(64→64, k=3,s=1) → Flatten → Linear(128)\n"
            f"  GRU input: LayerNorm(concat(CNN_features=128, past_actions))\n"
            f"  GRU: hidden_size={rnn_cfg.get('hidden_size', 256)}, "
            f"num_layers={rnn_cfg.get('num_layers', 1)}, "
            f"sequence_length={rnn_cfg.get('sequence_length', 16)} (orthogonal init)\n"
            f"  Output: Linear({rnn_cfg.get('hidden_size', 256)}, num_actions)\n"
            f")"
        )

    return GaussianCNNGRUModel(
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
