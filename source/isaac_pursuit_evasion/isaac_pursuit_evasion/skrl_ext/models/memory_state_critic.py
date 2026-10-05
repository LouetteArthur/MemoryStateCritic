# Copyright (c) 2026, the MemoryStateCritic authors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Memory-state critic V(s, z^a): the paper's contribution.

The critic is an MLP over the privileged state ``s`` and the actor's own GRU
hidden state ``z^a = f_theta(h)``. ``z^a`` arrives already detached (the agent
calls ``.detach()`` before storing or passing it), which is the stop-gradient of
Figure 1: the value loss never reaches the actor's recurrent encoder, and there
is no second recurrent encoder on the critic side.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium
import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, Model


class MemoryStateCriticModel(DeterministicMixin, Model):
    """V(s, z^a) critic: value = MLP(concat(state, z^a)).

    GRU hidden states are bounded by tanh (values in [-1, 1]) so ``z^a`` needs
    no preprocessing.
    """

    def __init__(
        self,
        observation_space: int | tuple[int] | gymnasium.Space | None = None,
        action_space: int | tuple[int] | gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        clip_actions: bool = False,
        actor_hidden_size: int = 128,
        layers: Sequence[int] = (256, 128, 64),
        **kwargs,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self._z_dim = actor_hidden_size

        # observation_space is the privileged state_space
        modules: list[nn.Module] = []
        in_dim = self.num_observations + actor_hidden_size
        for out_dim in layers:
            modules.extend([nn.Linear(in_dim, out_dim), nn.ELU()])
            in_dim = out_dim
        modules.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*modules)

    def compute(self, inputs: Mapping[str, Any], role: str = ""):
        states = inputs.get("states")  # preprocessed critic states
        z_a = inputs.get("z_a")

        # Fallback: if no z^a is provided, use zeros. This exists ONLY for the
        # dummy forward pass skrl runs inside Model.init_state_dict() before any
        # rollout has happened. Leaving it armed during training is dangerous:
        # the critic silently degrades to V(s) with self._z_dim dead inputs, and
        # nothing downstream -- not the run name, not the logs -- would say so.
        # _allow_zero_fallback is cleared by the first real forward (see below).
        if z_a is None:
            if not getattr(self, "_allow_zero_fallback", True):
                raise RuntimeError(
                    f"{type(self).__name__} received no 'z_a' after initialisation. The "
                    "memory-state critic would silently become a state-only critic with "
                    "zeroed memory inputs. Check that the agent config sets "
                    "memory_state_critic: True and that the rollout stores the actor hidden state."
                )
            z_a = torch.zeros(*states.shape[:-1], self._z_dim, device=states.device, dtype=states.dtype)
        else:
            # A real memory vector arrived: from here on, a missing z^a is a bug.
            self._allow_zero_fallback = False

        return self.net(torch.cat([states, z_a], dim=-1)), {}


def memory_state_critic_model(
    observation_space: int | tuple[int] | gymnasium.Space | None = None,
    action_space: int | tuple[int] | gymnasium.Space | None = None,
    device: str | torch.device | None = None,
    clip_actions: bool = False,
    actor_hidden_size: int = 128,
    layers: Sequence[int] = (256, 128, 64),
    return_source: bool = False,
    *args,
    **kwargs,
) -> Model | str:
    """Factory function for the V(s, z^a) memory-state critic.

    Called by the skrl Runner when the YAML config specifies
    ``class: MemoryStateCriticMixin``.
    """
    if return_source:
        return (
            "MemoryStateCriticModel(\n"
            f"  Input: state({observation_space}) + z_a({actor_hidden_size})\n"
            f"  MLP: {list(layers)} → 1\n"
            ")"
        )

    return MemoryStateCriticModel(
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        clip_actions=clip_actions,
        actor_hidden_size=actor_hidden_size,
        layers=layers,
    )
