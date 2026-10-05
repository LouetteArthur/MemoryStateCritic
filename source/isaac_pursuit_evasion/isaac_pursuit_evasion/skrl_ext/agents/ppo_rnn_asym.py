# Copyright (c) 2026, the MemoryStateCritic authors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO_RNN_ASYM: recurrent PPO with an asymmetric critic.

This is a modified copy of ``skrl.agents.torch.ppo.PPO_RNN``. One agent runs
every critic in the paper; the YAML picks the critic model and these flags:

1. State-only V(s) (default): the critic reads the privileged state, taken
   from ``infos["critic_states"]`` and stored in a ``critic_states`` memory
   tensor with its own preprocessor.

2. Memory-state V(s, z^a) (ours, ``memory_state_critic: True``): the actor's
   GRU hidden state ``z^a`` is detached, stored in memory and passed to the
   critic at every call site. The critic model is ``MemoryStateCriticModel``.
   The ``.detach()`` is the stop-gradient: the value loss never reaches the
   actor's recurrent encoder.

3. History-state V(s, z^c) (baseline, ``history_state_critic: True``): the
   critic has its own CNN+GRU over the observation-action stream. The critic
   model is ``HistoryStateCriticModel``, which exposes an RNN specification so
   its hidden state is managed by the same value-RNN plumbing as upstream.
   The agent slices image and past actions out of the flat observation and
   passes them to the critic.

4. Observation-state V(s, o, a): a Dict-shaped critic state whose ``state``
   component is normalised by an internal ``RunningStandardScaler``.

Differences from upstream ``skrl.PPO_RNN``:
- ``critic_state_preprocessor`` / ``critic_state_preprocessor_kwargs`` fields
- ``memory_state_critic`` / ``memory_dim`` fields (default: disabled)
- ``history_state_critic`` / ``history_image_shape`` / ``history_past_actions_size`` fields
- Memory tensor ``z_a`` (only when ``memory_state_critic`` is True)
- ``z_a`` or image inputs threaded through ``self.value.act`` at 3 call sites
"""

import copy
import itertools
from collections.abc import Mapping
from typing import Any

import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.utils.spaces.torch import (
    compute_space_size,
    flatten_tensorized_space,
    unflatten_tensorized_space,
)

# fmt: off
# [start-config-dict-torch]
PPO_RNN_ASYM_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 1e-3,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "critic_state_preprocessor": None,      # separate preprocessor for critic states (asymmetric actor-critic)
    "critic_state_preprocessor_kwargs": {},  # critic state preprocessor's kwargs (e.g. {"size": env.state_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.5,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.0,      # entropy loss scaling factor
    "value_loss_scale": 1.0,        # value loss scaling factor

    "kl_threshold": 0,              # KL divergence threshold for early stopping

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)

    "memory_state_critic": False,   # V(s, z^a): pass the detached actor GRU hidden state z^a to the critic
    "memory_dim": 128,              # dimension of z^a (must match the actor RNN hidden_size)
    "history_state_critic": False,  # V(s, z^c): the critic has its own CNN+GRU over the (o, a) stream
    "history_image_shape": [2, 64, 64],  # (C, H, W) of the image in the flat observation
    "history_past_actions_size": 4,      # number of past-action dims in the flat observation

    "mixed_precision": False,       # enable automatic mixed precision for higher performance

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-torch]
# fmt: on

# Config keys from before the code adopted the paper's names. Still accepted so
# that older YAMLs keep working; translated in PPO_RNN_ASYM.__init__.
_LEGACY_CFG_KEYS = {
    "sz_critic": "memory_state_critic",
    "vsh_critic": "memory_state_critic",
    "sz_z_dim": "memory_dim",
    "vsh_actor_hidden_size": "memory_dim",
    "sh_critic": "history_state_critic",
    "sh_image_shape": "history_image_shape",
    "sh_past_actions_size": "history_past_actions_size",
}


class PPO_RNN_ASYM(Agent):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Memory | tuple[Memory] | None = None,
        observation_space: int | tuple[int] | gymnasium.Space | None = None,
        action_space: int | tuple[int] | gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg: dict | None = None,
    ) -> None:
        """Proximal Policy Optimization (PPO) with support for Recurrent Neural Networks (RNN, GRU, LSTM, etc.)

        https://arxiv.org/abs/1707.06347

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = copy.deepcopy(PPO_RNN_ASYM_DEFAULT_CONFIG)
        for key, value in (cfg if cfg is not None else {}).items():
            if key in _LEGACY_CFG_KEYS:
                if value is None:
                    continue
                logger.warning(f"Config key '{key}' is deprecated, use '{_LEGACY_CFG_KEYS[key]}'")
                key = _LEGACY_CFG_KEYS[key]
            _cfg[key] = value
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        # models
        self.policy = self.models.get("policy")
        self.value = self.models.get("value")

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info("Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
                if self.value is not None and self.policy is not self.value:
                    self.value.broadcast_parameters()

        # configuration
        self._learning_epochs = self.cfg["learning_epochs"]
        self._mini_batches = self.cfg["mini_batches"]
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._grad_norm_clip = self.cfg["grad_norm_clip"]
        self._ratio_clip = self.cfg["ratio_clip"]
        self._value_clip = self.cfg["value_clip"]
        self._clip_predicted_values = self.cfg["clip_predicted_values"]

        self._value_loss_scale = self.cfg["value_loss_scale"]
        self._entropy_loss_scale = self.cfg["entropy_loss_scale"]

        self._kl_threshold = self.cfg["kl_threshold"]

        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]
        self._critic_state_preprocessor = self.cfg["critic_state_preprocessor"]
        self._value_preprocessor = self.cfg["value_preprocessor"]

        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._rewards_shaper = self.cfg["rewards_shaper"]
        self._time_limit_bootstrap = self.cfg["time_limit_bootstrap"]

        self._mixed_precision = self.cfg["mixed_precision"]

        # V(s, z^a) memory-state critic
        self._memory_state_critic = bool(self.cfg["memory_state_critic"])
        self._memory_dim = int(self.cfg["memory_dim"])

        # V(s, z^c) history-state critic: critic has own CNN+GRU
        self._history_state_critic = bool(self.cfg["history_state_critic"])
        if self._history_state_critic:
            self._history_image_shape = tuple(self.cfg["history_image_shape"])  # (C, H, W)
            self._history_image_size = int(np.prod(self._history_image_shape))
            self._history_past_actions_size = self.cfg["history_past_actions_size"]

        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # set up optimizer and learning rate scheduler
        if self.policy is not None and self.value is not None:
            if self.policy is self.value:
                self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate)
            else:
                self.optimizer = torch.optim.Adam(
                    itertools.chain(self.policy.parameters(), self.value.parameters()), lr=self._learning_rate
                )
            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(
                    self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )

            self.checkpoint_modules["optimizer"] = self.optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

        # set up separate critic state preprocessor for asymmetric actor-critic
        if self._critic_state_preprocessor:
            self._critic_state_preprocessor = self._critic_state_preprocessor(
                **self.cfg["critic_state_preprocessor_kwargs"]
            )
            self.checkpoint_modules["critic_state_preprocessor"] = self._critic_state_preprocessor
        else:
            # Fall back to state_preprocessor if critic_state_preprocessor not specified
            self._critic_state_preprocessor = None

        if self._value_preprocessor:
            self._value_preprocessor = self._value_preprocessor(**self.cfg["value_preprocessor_kwargs"])
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor

        # Dict critic support: normalize the "state" component of Dict critic states
        # (used by Geles-style unbiased critic where critic receives image + past_actions + state)
        self._state_component_scaler = None
        if self.value is not None:
            val_obs_space = getattr(self.value, "observation_space", None)
            if isinstance(val_obs_space, gymnasium.spaces.Dict) and "state" in val_obs_space.spaces:
                from skrl.resources.preprocessors.torch import RunningStandardScaler

                self._state_component_scaler = RunningStandardScaler(size=val_obs_space["state"], device=device)
                self.checkpoint_modules["state_component_scaler"] = self._state_component_scaler

    def init(self, trainer_cfg: Mapping[str, Any] | None = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # Determine critic state size from config or fall back to observation space
        critic_pp_kwargs = self.cfg.get("critic_state_preprocessor_kwargs") or {}
        critic_state_size = critic_pp_kwargs.get("size", self.observation_space)
        # Dict critic states (Geles-style): compute flat size for memory storage
        if isinstance(critic_state_size, gymnasium.spaces.Dict):
            critic_state_size = compute_space_size(critic_state_size)

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="critic_states", size=critic_state_size, dtype=torch.float32)

            # V(s, z^a): store the actor GRU hidden state z^a
            if self._memory_state_critic:
                self.memory.create_tensor(name="z_a", size=self._memory_dim, dtype=torch.float32)

            # tensors sampled during training
            self._tensors_names = [
                "states",
                "actions",
                "terminated",
                "truncated",
                "log_prob",
                "values",
                "returns",
                "advantages",
                "critic_states",
            ]
            if self._memory_state_critic:
                self._tensors_names.append("z_a")

        # RNN specifications
        self._rnn = False  # flag to indicate whether RNN is available
        self._rnn_tensors_names = []  # used for sampling during training
        self._rnn_final_states = {"policy": [], "value": []}
        self._rnn_initial_states = {"policy": [], "value": []}
        self._rnn_sequence_length = self.policy.get_specification().get("rnn", {}).get("sequence_length", 1)

        # policy
        for i, size in enumerate(self.policy.get_specification().get("rnn", {}).get("sizes", [])):
            self._rnn = True
            # create tensors in memory
            if self.memory is not None:
                self.memory.create_tensor(
                    name=f"rnn_policy_{i}", size=(size[0], size[2]), dtype=torch.float32, keep_dimensions=True
                )
                self._rnn_tensors_names.append(f"rnn_policy_{i}")
            # default RNN states
            self._rnn_initial_states["policy"].append(torch.zeros(size, dtype=torch.float32, device=self.device))

        # value
        if self.value is not None:
            if self.policy is self.value:
                self._rnn_initial_states["value"] = self._rnn_initial_states["policy"]
            else:
                for i, size in enumerate(self.value.get_specification().get("rnn", {}).get("sizes", [])):
                    self._rnn = True
                    # create tensors in memory
                    if self.memory is not None:
                        self.memory.create_tensor(
                            name=f"rnn_value_{i}", size=(size[0], size[2]), dtype=torch.float32, keep_dimensions=True
                        )
                        self._rnn_tensors_names.append(f"rnn_value_{i}")
                    # default RNN states
                    self._rnn_initial_states["value"].append(torch.zeros(size, dtype=torch.float32, device=self.device))

        # create temporary variables needed for storage and computation
        self._current_log_prob = None
        self._current_next_states = None
        self._current_critic_states = None
        self._current_next_critic_states = None

        # Log critic architecture for experiment verification
        critic_mode = "V(s)"
        critic_input_dim = critic_state_size if isinstance(critic_state_size, int) else "dict"
        if self._memory_state_critic:
            critic_mode = f"V(s, z^a) — memory_dim={self._memory_dim}"
            if isinstance(critic_state_size, int):
                critic_input_dim = f"{critic_state_size} + {self._memory_dim} = {critic_state_size + self._memory_dim}"
        if self._history_state_critic:
            critic_mode = "V(s, z^c) — critic-side CNN+GRU"
        logger.info(f"[PPO_RNN_ASYM] Critic mode: {critic_mode} | critic_state_dim: {critic_input_dim}")

    def _history_critic_inputs(self, flat_obs: torch.Tensor) -> dict:
        """Extract image and past_actions from the flat observation for the V(s, z^c) critic.

        The flat observation layout (from gymnasium Dict sorted keys) is:
        [image_flat, past_actions]. This matches the flatten order used by
        skrl's ``flatten_tensorized_space`` which sorts Dict keys alphabetically.
        """
        if not self._history_state_critic:
            return {}
        # Handle 2D (batch, flat) and 3D (batch, seq, flat)
        orig_shape = flat_obs.shape
        if flat_obs.dim() == 3:
            batch, seq, flat_dim = orig_shape
            flat_obs_2d = flat_obs.reshape(batch * seq, flat_dim)
        else:
            flat_obs_2d = flat_obs

        image = flat_obs_2d[:, : self._history_image_size].reshape(-1, *self._history_image_shape)
        past_actions = flat_obs_2d[
            :, self._history_image_size : self._history_image_size + self._history_past_actions_size
        ]

        if len(orig_shape) == 3:
            # Reshape back to (batch, seq, ...) for BPTT
            image = image.reshape(batch, seq, *self._history_image_shape)
            past_actions = past_actions.reshape(batch, seq, -1)

        return {"critic_image": image, "critic_past_actions": past_actions}

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        rnn = {"rnn": self._rnn_initial_states["policy"]} if self._rnn else {}

        # sample random actions (random_timesteps=0 in all configs, so this is never hit)
        if timestep < self._random_timesteps:
            return self.policy.random_act({"states": self._state_preprocessor(states), **rnn}, role="policy")

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            actions, log_prob, outputs = self.policy.act(
                {"states": self._state_preprocessor(states), **rnn}, role="policy"
            )
            self._current_log_prob = log_prob

        if self._rnn:
            self._rnn_final_states["policy"] = outputs.get("rnn", [])

        return actions, log_prob, outputs

    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

        if self.memory is not None:
            self._current_next_states = next_states

            # Extract critic states from infos for asymmetric actor-critic
            critic_states = infos.get("critic_states") if isinstance(infos, dict) else None
            next_critic_states = infos.get("next_critic_states") if isinstance(infos, dict) else None
            # Current critic states should align with current observations
            self._current_critic_states = critic_states if critic_states is not None else states
            # Next critic states are used for bootstrapping
            self._current_next_critic_states = next_critic_states if next_critic_states is not None else next_states

            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # V(s, z^a): detached actor hidden state for the critic and for storage
            if self._memory_state_critic and self._rnn_final_states["policy"]:
                # shape: (num_layers, batch, hidden_size) — take the last layer
                self._current_z_a = self._rnn_final_states["policy"][0][-1].detach()
            else:
                self._current_z_a = None

            # compute values using critic states (asymmetric actor-critic)
            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                rnn = {"rnn": self._rnn_initial_states["value"]} if self._rnn else {}
                if self._critic_state_preprocessor is not None:
                    value_input = self._critic_state_preprocessor(self._current_critic_states)
                elif isinstance(self._current_critic_states, dict):
                    # Dict critic states (Geles-style): normalize state component
                    value_input = self._current_critic_states
                    if self._state_component_scaler is not None:
                        value_input = {**value_input, "state": self._state_component_scaler(value_input["state"])}
                else:
                    value_input = self._state_preprocessor(states)
                critic_kwargs = {"states": value_input, **rnn}
                if self._current_z_a is not None:
                    critic_kwargs["z_a"] = self._current_z_a
                critic_kwargs.update(self._history_critic_inputs(states))
                values, _, outputs = self.value.act(critic_kwargs, role="value")
                values = self._value_preprocessor(values, inverse=True)

            # time-limit (truncation) bootstrapping
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated

            # package RNN states
            rnn_states = {}
            if self._rnn:
                rnn_states.update(
                    {f"rnn_policy_{i}": s.transpose(0, 1) for i, s in enumerate(self._rnn_initial_states["policy"])}
                )
                if self.policy is not self.value:
                    rnn_states.update(
                        {f"rnn_value_{i}": s.transpose(0, 1) for i, s in enumerate(self._rnn_initial_states["value"])}
                    )

            # V(s, z^a): include the actor hidden state in stored samples
            z_a_kwargs = {"z_a": self._current_z_a} if self._current_z_a is not None else {}

            # Flatten Dict critic states for memory storage
            critic_states_for_memory = (
                flatten_tensorized_space(self._current_critic_states)
                if isinstance(self._current_critic_states, dict)
                else self._current_critic_states
            )

            # storage transition in memory
            self.memory.add_samples(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                log_prob=self._current_log_prob,
                values=values,
                critic_states=critic_states_for_memory,
                **rnn_states,
                **z_a_kwargs,
            )
            for memory in self.secondary_memories:
                memory.add_samples(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    log_prob=self._current_log_prob,
                    values=values,
                    critic_states=critic_states_for_memory,
                    **rnn_states,
                    **z_a_kwargs,
                )

        # update RNN states
        if self._rnn:
            self._rnn_final_states["value"] = (
                self._rnn_final_states["policy"] if self.policy is self.value else outputs.get("rnn", [])
            )

            # reset states if the episodes have ended
            finished_episodes = (terminated | truncated).nonzero(as_tuple=False)
            if finished_episodes.numel():
                for rnn_state in self._rnn_final_states["policy"]:
                    rnn_state[:, finished_episodes[:, 0]] = 0
                if self.policy is not self.value:
                    for rnn_state in self._rnn_final_states["value"]:
                        rnn_state[:, finished_episodes[:, 0]] = 0

            self._rnn_initial_states = self._rnn_final_states

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(  # noqa: C901  (PPO update loop; mirrors upstream skrl structure)
        self, timestep: int, timesteps: int
    ) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
        ) -> torch.Tensor:
            """Compute the Generalized Advantage Estimator (GAE)

            :param rewards: Rewards obtained by the agent
            :type rewards: torch.Tensor
            :param dones: Signals to indicate that episodes have ended
            :type dones: torch.Tensor
            :param values: Values obtained by the agent
            :type values: torch.Tensor
            :param next_values: Next values obtained by the agent
            :type next_values: torch.Tensor
            :param discount_factor: Discount factor
            :type discount_factor: float
            :param lambda_coefficient: Lambda coefficient
            :type lambda_coefficient: float

            :return: Generalized Advantage Estimator
            :rtype: torch.Tensor
            """
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages

        # compute returns and advantages
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            self.value.train(False)
            rnn = {"rnn": self._rnn_initial_states["value"]} if self._rnn else {}
            if self._critic_state_preprocessor is not None and self._current_next_critic_states is not None:
                value_input = self._critic_state_preprocessor(self._current_next_critic_states.float())
            elif isinstance(self._current_next_critic_states, dict):
                value_input = self._current_next_critic_states
                if self._state_component_scaler is not None:
                    value_input = {**value_input, "state": self._state_component_scaler(value_input["state"])}
            else:
                value_input = self._state_preprocessor(self._current_next_states.float())
            bootstrap_kwargs = {"states": value_input, **rnn}
            # V(s, z^a): bootstrap with the latest actor hidden state
            if self._memory_state_critic and self._rnn_final_states["policy"]:
                bootstrap_kwargs["z_a"] = self._rnn_final_states["policy"][0][-1].detach()
            # V(s, z^c): extract image+past_actions from the latest next_states
            bootstrap_kwargs.update(self._history_critic_inputs(self._current_next_states))
            last_values, _, _ = self.value.act(bootstrap_kwargs, role="value")
            self.value.train(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(
            names=self._tensors_names, mini_batches=self._mini_batches, sequence_length=self._rnn_sequence_length
        )

        rnn_policy, rnn_value = {}, {}
        if self._rnn:
            sampled_rnn_batches = self.memory.sample_all(
                names=self._rnn_tensors_names,
                mini_batches=self._mini_batches,
                sequence_length=self._rnn_sequence_length,
            )

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        # learning epochs
        for epoch in range(self._learning_epochs):
            kl_divergences = []

            # mini-batches loop
            for i, sampled in enumerate(sampled_batches):
                sampled_states = sampled[0]
                sampled_actions = sampled[1]
                sampled_terminated = sampled[2]
                sampled_truncated = sampled[3]
                sampled_log_prob = sampled[4]
                sampled_values = sampled[5]
                sampled_returns = sampled[6]
                sampled_advantages = sampled[7]
                sampled_critic_states = sampled[8]
                sampled_z_a = sampled[9] if self._memory_state_critic else None

                # ---- BPTT: reshape data into sequences ----
                seq_len = self._rnn_sequence_length
                use_bptt = self._rnn and seq_len > 1

                if use_bptt:
                    N = sampled_states.shape[0]
                    num_seq = N // seq_len

                    # States → 3D for BPTT through GRU
                    sampled_states_3d = sampled_states.reshape(num_seq, seq_len, -1)
                    # Terminated → 3D for mid-sequence done resets
                    done_3d = (sampled_terminated | sampled_truncated).reshape(num_seq, seq_len, -1)

                    # RNN initial states: take only the first h per sequence
                    if self.policy is self.value:
                        rnn_policy = {
                            "rnn": [s[::seq_len].transpose(0, 1) for s in sampled_rnn_batches[i]],
                            "terminated": done_3d,
                        }
                        rnn_value = rnn_policy
                    else:
                        rnn_policy = {
                            "rnn": [
                                s[::seq_len].transpose(0, 1)
                                for s, n in zip(sampled_rnn_batches[i], self._rnn_tensors_names)
                                if "policy" in n
                            ],
                            "terminated": done_3d,
                        }
                        rnn_value = {
                            "rnn": [
                                s[::seq_len].transpose(0, 1)
                                for s, n in zip(sampled_rnn_batches[i], self._rnn_tensors_names)
                                if "value" in n
                            ],
                            "terminated": done_3d,
                        }
                elif self._rnn:
                    # Stored-state (seq_len=1): pass all h, terminated as 2D
                    if self.policy is self.value:
                        rnn_policy = {
                            "rnn": [s.transpose(0, 1) for s in sampled_rnn_batches[i]],
                            "terminated": sampled_terminated | sampled_truncated,
                        }
                        rnn_value = rnn_policy
                    else:
                        rnn_policy = {
                            "rnn": [
                                s.transpose(0, 1)
                                for s, n in zip(sampled_rnn_batches[i], self._rnn_tensors_names)
                                if "policy" in n
                            ],
                            "terminated": sampled_terminated | sampled_truncated,
                        }
                        rnn_value = {
                            "rnn": [
                                s.transpose(0, 1)
                                for s, n in zip(sampled_rnn_batches[i], self._rnn_tensors_names)
                                if "value" in n
                            ],
                            "terminated": sampled_terminated | sampled_truncated,
                        }

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):

                    policy_input = sampled_states_3d if use_bptt else sampled_states
                    policy_input = self._state_preprocessor(policy_input, train=not epoch)

                    _, next_log_prob, _ = self.policy.act(
                        {"states": policy_input, "taken_actions": sampled_actions, **rnn_policy}, role="policy"
                    )

                    # compute approximate KL divergence
                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    # early stopping with KL divergence
                    if self._kl_threshold and kl_divergence > self._kl_threshold:
                        break

                    # compute entropy loss
                    if self._entropy_loss_scale:
                        entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    # compute policy loss
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )

                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # compute value loss
                    if self._state_component_scaler is not None:
                        # Dict critic: unflatten from memory, normalize state component
                        critic_dict = unflatten_tensorized_space(self.value.observation_space, sampled_critic_states)
                        critic_dict["state"] = self._state_component_scaler(critic_dict["state"], train=not epoch)
                        critic_input = critic_dict
                    elif self._critic_state_preprocessor is not None:
                        critic_input = self._critic_state_preprocessor(sampled_critic_states, train=not epoch)
                    else:
                        critic_input = sampled_states
                    # V(s, z^c): reshape critic states to 3D during BPTT so the
                    # critic's RNN can process sequences properly.
                    if self._history_state_critic and use_bptt:
                        critic_input = critic_input.reshape(num_seq, seq_len, -1)
                    value_kwargs = {"states": critic_input, **rnn_value}
                    if sampled_z_a is not None:
                        value_kwargs["z_a"] = sampled_z_a
                    # V(s, z^c): extract image+past_actions from sampled states for BPTT
                    history_source = sampled_states_3d if use_bptt else sampled_states
                    value_kwargs.update(self._history_critic_inputs(history_source))
                    predicted_values, _, _ = self.value.act(value_kwargs, role="value")

                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
                        )
                    value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                # optimization step
                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(self.policy.parameters(), self.value.parameters()), self._grad_norm_clip
                        )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            # update learning rate
            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    # reduce (collect from all workers/processes) KL in distributed runs
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        # record data
        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batches))
        if self._entropy_loss_scale:
            self.track_data(
                "Loss / Entropy loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches)
            )

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])


# Names from before the code adopted the paper's notation, kept so that older
# YAMLs and imports keep working. All of them are the same agent.
PPO_RNN_VSH = PPO_RNN_SZ = PPO_RNN_SH = PPO_RNN_ASYM
PPO_RNN_VSH_DEFAULT_CONFIG = PPO_RNN_SZ_DEFAULT_CONFIG = PPO_RNN_SH_DEFAULT_CONFIG = PPO_RNN_ASYM_DEFAULT_CONFIG
