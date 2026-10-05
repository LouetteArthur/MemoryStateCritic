# Copyright (c) 2026, the MemoryStateCritic authors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom skrl model instantiators for vision-based recurrent RL."""

from isaac_pursuit_evasion.skrl_ext.models.gaussian_cnn_rnn import (
    GaussianCNNGRUModel,
    gaussian_cnn_rnn_model,
)
from isaac_pursuit_evasion.skrl_ext.models.history_state_critic import (
    HistoryStateCriticModel,
    history_state_critic_model,
)
from isaac_pursuit_evasion.skrl_ext.models.memory_state_critic import (
    MemoryStateCriticModel,
    memory_state_critic_model,
)

__all__ = [
    "GaussianCNNGRUModel",
    "gaussian_cnn_rnn_model",
    "HistoryStateCriticModel",
    "history_state_critic_model",
    "MemoryStateCriticModel",
    "memory_state_critic_model",
]
