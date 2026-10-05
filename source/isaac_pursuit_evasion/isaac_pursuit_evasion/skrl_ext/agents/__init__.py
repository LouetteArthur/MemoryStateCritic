# Copyright (c) 2026, the MemoryStateCritic authors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom skrl agent: recurrent PPO with an asymmetric critic."""

from isaac_pursuit_evasion.skrl_ext.agents.ppo_rnn_asym import (
    PPO_RNN_ASYM,
    PPO_RNN_ASYM_DEFAULT_CONFIG,
    PPO_RNN_SH,
    PPO_RNN_SH_DEFAULT_CONFIG,
    PPO_RNN_SZ,
    PPO_RNN_SZ_DEFAULT_CONFIG,
    PPO_RNN_VSH,
    PPO_RNN_VSH_DEFAULT_CONFIG,
)

__all__ = [
    "PPO_RNN_ASYM",
    "PPO_RNN_ASYM_DEFAULT_CONFIG",
    # deprecated aliases of PPO_RNN_ASYM
    "PPO_RNN_SH",
    "PPO_RNN_SH_DEFAULT_CONFIG",
    "PPO_RNN_SZ",
    "PPO_RNN_SZ_DEFAULT_CONFIG",
    "PPO_RNN_VSH",
    "PPO_RNN_VSH_DEFAULT_CONFIG",
]
