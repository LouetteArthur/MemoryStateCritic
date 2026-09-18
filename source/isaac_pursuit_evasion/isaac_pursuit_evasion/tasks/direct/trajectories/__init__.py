# Copyright (c) 2026, the MemoryStateCritic authors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

gym.register(
    id="Trajectory-Tracking-v0",
    entry_point=f"{__name__}.tracking_env:TrajectoryTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trajectory_env_cfg:trajectory_tracking_cfg",
    },
)
