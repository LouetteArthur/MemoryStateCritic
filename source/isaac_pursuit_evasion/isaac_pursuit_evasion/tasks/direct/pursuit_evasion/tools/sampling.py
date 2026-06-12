# Copyright (c) 2023 Botian Xu
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.distributions as D


def get_seeds_list(n_seeds: int) -> List[int]:
    return np.arange(0, n_seeds, dtype=int).tolist()


def rectangular_ring_sampling(
    big_low: torch.Tensor,
    big_high: torch.Tensor,
    small_low: torch.Tensor,
    small_high: torch.Tensor,
    num_points: int,
) -> torch.Tensor:
    """Sample uniformly in a rectangular ring defined by two axis-aligned boxes."""
    big_dist = D.Uniform(big_low, big_high)
    points = torch.empty((num_points, 3), device=big_low.device, dtype=big_low.dtype)

    collected = 0
    batch = max(100, num_points)
    while collected < num_points:
        outer_points = big_dist.sample((batch,))
        mask = (
            (outer_points[:, 0] < small_low[0])
            | (outer_points[:, 0] > small_high[0])
            | (outer_points[:, 1] < small_low[1])
            | (outer_points[:, 1] > small_high[1])
            | (outer_points[:, 2] < small_low[2])
            | (outer_points[:, 2] > small_high[2])
        )
        valid = outer_points[mask]
        if not valid.numel():
            continue
        take = min(num_points - collected, valid.shape[0])
        points[collected : collected + take] = valid[:take]
        collected += take
    return points


def min_separation_sampling(
    low: torch.Tensor,
    high: torch.Tensor,
    ref_points: torch.Tensor,
    min_separation: float,
    attempts: int = 10,
) -> torch.Tensor:
    """Sample points that maintain a minimum distance from reference points."""
    points_dist = D.Uniform(low, high)
    num_points = ref_points.shape[0]
    points = torch.empty((num_points, 3), device=low.device, dtype=low.dtype)
    remaining_mask = torch.ones(num_points, device=low.device, dtype=torch.bool)

    while remaining_mask.any():
        remain = remaining_mask.sum().item()
        sampled = points_dist.sample((remain, attempts))
        distances = torch.norm(sampled - ref_points[remaining_mask].unsqueeze(1), dim=-1)
        feasible = distances > min_separation
        has_feasible = feasible.any(dim=1)
        if not has_feasible.any():
            continue
        chosen_idx = torch.argmax(feasible.int(), dim=1)
        feasible_points = sampled[has_feasible]
        assign_mask = torch.zeros_like(remaining_mask)
        assign_mask[torch.where(remaining_mask)[0][has_feasible]] = True
        points[assign_mask] = feasible_points[has_feasible, chosen_idx[has_feasible],:]
        remaining_mask = remaining_mask & ~assign_mask

    if not torch.all(torch.norm(points - ref_points, dim=-1) > min_separation):
        raise RuntimeError("Sampling failed to respect minimum separation constraint.")
    return points


def policy_sampling(policy_pool: Dict[str, float], num_sampled: int) -> Dict[str, List[int]]:
    """Stratified sampling over discrete policy names according to a probability pool."""
    policies = list(policy_pool.keys())
    probs = np.array([policy_pool[p] for p in policies], dtype=float)
    total = probs.sum()
    if not np.isclose(total, 1.0):
        probs /= total

    cdf = np.cumsum(probs)
    assignments: Dict[str, List[int]] = {p: [] for p in policies}
    for i in range(num_sampled):
        threshold = (i + 0.5) / num_sampled
        for policy, cum_prob in zip(policies, cdf):
            if threshold <= cum_prob:
                assignments[policy].append(i)
                break
    return assignments
