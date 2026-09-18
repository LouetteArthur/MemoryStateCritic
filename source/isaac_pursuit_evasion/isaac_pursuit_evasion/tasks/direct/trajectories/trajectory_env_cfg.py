# Copyright (c) 2026, the MemoryStateCritic authors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration helpers for the trajectory tracking environment."""

from __future__ import annotations

from .tracking_env import TrajectorySpecConfig, TrajectoryTrackingEnvCfg


def _split_counts(num_envs: int) -> tuple[int, int, int]:
    """Split envs across hover/circular/lemniscate using a 1:2:1 ratio."""
    circular = int(num_envs // 2)
    hover = int((num_envs - circular) // 2)
    lemniscate = int(num_envs - hover - circular)
    return hover, circular, lemniscate


def trajectory_tracking_cfg(
    num_envs: int | None = None,
    hover_count: int | None = 8,
    circular_count: int | None = 8,
    lemniscate_count: int | None = 8,
    enable_yaw_tracking: bool | None = None,
) -> TrajectoryTrackingEnvCfg:
    """Return a TrajectoryTrackingEnvCfg with balanced trajectory specs."""
    cfg = TrajectoryTrackingEnvCfg()
    cfg.controller_type = "crazyflie_pid"
    cfg.drone_name = "crazyflie_brushless"

    if hover_count is not None or circular_count is not None or lemniscate_count is not None:
        hover = int(hover_count or 0)
        circular = int(circular_count or 0)
        lemniscate = int(lemniscate_count or 0)
        total = hover + circular + lemniscate
        if total <= 0:
            raise ValueError("Trajectory counts must sum to a positive value.")
        cfg.trajectory_specs = (
            TrajectorySpecConfig(name="hovertrajectory", count=hover),
            TrajectorySpecConfig(name="circulartrajectory", count=circular),
            TrajectorySpecConfig(name="lemniscatetrajectory", count=lemniscate),
        )
        cfg.scene.num_envs = total
    elif num_envs is not None:
        hover, circular, lemniscate = _split_counts(int(num_envs))
        cfg.trajectory_specs = (
            TrajectorySpecConfig(name="hovertrajectory", count=hover),
            TrajectorySpecConfig(name="circulartrajectory", count=circular),
            TrajectorySpecConfig(name="lemniscatetrajectory", count=lemniscate),
        )
        cfg.scene.num_envs = int(num_envs)

    if enable_yaw_tracking is not None:
        cfg.enable_yaw_tracking = bool(enable_yaw_tracking)

    return cfg
