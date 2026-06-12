"""Custom skrl model instantiators for vision-based recurrent RL."""

from isaac_pursuit_evasion.skrl_ext.models.gaussian_cnn_rnn import (
    GaussianCNNGRUModel,
    gaussian_cnn_rnn_model,
)
from isaac_pursuit_evasion.skrl_ext.models.gaussian_cnn_rnn_flat import (
    GaussianCNNGRUFlatModel,
    gaussian_cnn_rnn_flat_model,
)
from isaac_pursuit_evasion.skrl_ext.models.history_state_critic import (
    HistoryStateCriticModel,
    history_state_critic_model,
)
from isaac_pursuit_evasion.skrl_ext.models.vsh_critic import (
    SzCriticModel,
    VshCriticModel,
    sz_critic_model,
    vsh_critic_model,
)

__all__ = [
    "GaussianCNNGRUModel",
    "gaussian_cnn_rnn_model",
    "GaussianCNNGRUFlatModel",
    "gaussian_cnn_rnn_flat_model",
    "HistoryStateCriticModel",
    "history_state_critic_model",
    "SzCriticModel",
    "sz_critic_model",
    "VshCriticModel",
    "vsh_critic_model",
]
