"""Custom skrl agents with asymmetric actor-critic, V(s,z), and V(s,h) support."""

from isaac_pursuit_evasion.skrl_ext.agents.ppo_asym import (
    PPO_ASYM,
    PPO_ASYM_DEFAULT_CONFIG,
)
from isaac_pursuit_evasion.skrl_ext.agents.ppo_rnn_vsh import (
    PPO_RNN_SH,
    PPO_RNN_SH_DEFAULT_CONFIG,
    PPO_RNN_SZ,
    PPO_RNN_SZ_DEFAULT_CONFIG,
    PPO_RNN_VSH,
    PPO_RNN_VSH_DEFAULT_CONFIG,
)

# PPO_RNN_ASYM: alias for PPO_RNN_VSH with sz_critic=False (the default).
# Use PPO_RNN_ASYM in YAMLs for asymmetric RNN PPO without V(s,z).
PPO_RNN_ASYM = PPO_RNN_VSH
PPO_RNN_ASYM_DEFAULT_CONFIG = PPO_RNN_VSH_DEFAULT_CONFIG

__all__ = [
    "PPO_ASYM",
    "PPO_ASYM_DEFAULT_CONFIG",
    "PPO_RNN_ASYM",
    "PPO_RNN_ASYM_DEFAULT_CONFIG",
    "PPO_RNN_SH",
    "PPO_RNN_SH_DEFAULT_CONFIG",
    "PPO_RNN_SZ",
    "PPO_RNN_SZ_DEFAULT_CONFIG",
    "PPO_RNN_VSH",
    "PPO_RNN_VSH_DEFAULT_CONFIG",
]
