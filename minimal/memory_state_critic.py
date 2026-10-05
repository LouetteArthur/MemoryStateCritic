# Copyright (c) 2026, the MemoryStateCritic authors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The memory-state critic in plain PyTorch, next to the history-state baseline.

This file is self-contained: it needs only ``torch``, not Isaac Sim, skrl or the
rest of this repository. It is the method of the paper with everything else
stripped away; the full training code lives in
``source/isaac_pursuit_evasion/isaac_pursuit_evasion/skrl_ext``.

Notation (as in the paper):
    h_t   history of observations and past actions
    s_t   privileged state, available to the critic at training time only
    z^a   the actor's memory, z^a_t = f_theta(h_t), computed by its GRU
    z^c   the critic's own encoding of the history, z^c_t = f_psi(h_t)

    History-state critic (baseline)   V_psi(s_t, z^c_t)   two recurrent encoders
    Memory-state critic (ours)        V_psi(s_t, z^a_t)   one recurrent encoder

The memory-state critic reuses the z^a the actor already computes and blocks the
value loss from reaching f_theta with a stop-gradient, so f_theta is trained by
the policy gradient alone.

Run ``python minimal/memory_state_critic.py`` to check, on random data, that the
value loss leaves the actor's encoder untouched.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim: int, out_dim: int, hidden: int = 64) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, out_dim)
    )


class RecurrentActor(nn.Module):
    """pi_theta(a | h) = g_theta(a | f_theta(h)), with f_theta a GRU."""

    def __init__(self, obs_dim: int, act_dim: int, memory_dim: int = 64) -> None:
        super().__init__()
        self.f_theta = nn.GRU(obs_dim + act_dim, memory_dim, batch_first=True)
        self.g_theta = mlp(memory_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(
        self, obs: torch.Tensor, prev_action: torch.Tensor, z0: torch.Tensor | None = None
    ) -> tuple[torch.distributions.Normal, torch.Tensor]:
        """obs: (B, T, obs_dim), prev_action: (B, T, act_dim) -> (pi, z^a of shape (B, T, memory_dim))."""
        z_a, _ = self.f_theta(torch.cat([obs, prev_action], dim=-1), z0)
        pi = torch.distributions.Normal(self.g_theta(z_a), self.log_std.exp())
        return pi, z_a


class MemoryStateCritic(nn.Module):
    """V_psi(s, z^a): an MLP on the state and the actor's memory. No recurrent encoder of its own."""

    def __init__(self, state_dim: int, memory_dim: int = 64) -> None:
        super().__init__()
        self.v_psi = mlp(state_dim + memory_dim, 1)

    def forward(self, state: torch.Tensor, z_a: torch.Tensor) -> torch.Tensor:
        # The stop-gradient: the value loss trains v_psi only, never f_theta.
        return self.v_psi(torch.cat([state, z_a.detach()], dim=-1)).squeeze(-1)


class HistoryStateCritic(nn.Module):
    """V_psi(s, z^c): the baseline, with a second GRU f_psi trained by the value loss."""

    def __init__(self, obs_dim: int, act_dim: int, state_dim: int, memory_dim: int = 64) -> None:
        super().__init__()
        self.f_psi = nn.GRU(obs_dim + act_dim, memory_dim, batch_first=True)
        self.v_psi = mlp(state_dim + memory_dim, 1)

    def forward(self, state: torch.Tensor, obs: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
        z_c, _ = self.f_psi(torch.cat([obs, prev_action], dim=-1))
        return self.v_psi(torch.cat([state, z_c], dim=-1)).squeeze(-1)


def ppo_losses(
    actor: RecurrentActor,
    critic: MemoryStateCritic,
    batch: dict[str, torch.Tensor],
    clip: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clipped PPO policy loss and value loss for the memory-state critic.

    ``batch`` holds (B, T, ...) sequences: obs, prev_action, state, action,
    log_prob, advantage, return. The critic reads the z^a the actor computes in
    this same forward pass: nothing is stored for it and nothing is re-encoded.
    """
    pi, z_a = actor(batch["obs"], batch["prev_action"])
    ratio = (pi.log_prob(batch["action"]).sum(-1) - batch["log_prob"]).exp()
    adv = batch["advantage"]
    policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
    value_loss = F.mse_loss(critic(batch["state"], z_a), batch["return"])
    return policy_loss, value_loss


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, obs_dim, act_dim, state_dim = 8, 16, 10, 2, 6
    batch = {
        "obs": torch.randn(B, T, obs_dim),
        "prev_action": torch.randn(B, T, act_dim),
        "state": torch.randn(B, T, state_dim),
        "action": torch.randn(B, T, act_dim),
        "log_prob": torch.randn(B, T),
        "advantage": torch.randn(B, T),
        "return": torch.randn(B, T),
    }
    actor, critic = RecurrentActor(obs_dim, act_dim), MemoryStateCritic(state_dim)
    policy_loss, value_loss = ppo_losses(actor, critic, batch)

    value_loss.backward(retain_graph=True)
    assert all(p.grad is None for p in actor.parameters()), "the value loss reached the actor"
    print("value loss  -> actor encoder f_theta: no gradient (stop-gradient holds)")

    policy_loss.backward()
    grad = sum(p.grad.norm() ** 2 for p in actor.f_theta.parameters()) ** 0.5
    print(f"policy loss -> actor encoder f_theta: gradient norm {grad:.3f}")

    n_ours = sum(p.numel() for p in critic.parameters())
    n_base = sum(p.numel() for p in HistoryStateCritic(obs_dim, act_dim, state_dim).parameters())
    print(f"critic parameters: memory-state {n_ours}, history-state {n_base}")
