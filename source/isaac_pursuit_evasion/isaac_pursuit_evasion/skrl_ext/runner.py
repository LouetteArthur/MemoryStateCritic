"""Custom skrl Runner for isaac_pursuit_evasion.

Subclasses the upstream ``skrl.utils.runner.torch.Runner`` and extends it with:

* **Custom model factories**: ``gaussian_cnn_rnn_model`` (CNN+GRU actor with past
  action concat) and ``vsh_critic_model`` / ``sz_critic_model`` (V(s, z)
  memory-state critic that consumes the detached actor GRU hidden state).
  These replace the built-in ``gaussiancnnrnnmixin`` / ``vshcriticmixin`` /
  ``szcriticmixin`` registrations that would otherwise require patching skrl.
* **Custom agent classes**: ``PPO_ASYM`` and ``PPO_RNN_VSH`` (alias
  ``PPO_RNN_SZ``), which implement asymmetric actor-critic and V(s, z)
  training respectively (see ``skrl_ext.agents``).
* **Asymmetric actor-critic plumbing**: when a ``value`` role is declared and
  the environment exposes a separate ``state_space``, the value network is
  built on that privileged state space rather than the actor's observation
  space, and ``critic_state_preprocessor`` is wired up automatically.

Keeping this subclass project-local means the vendored ``third_parties/skrl``
copy can stay pristine and be upgraded without merge conflicts.
"""

import copy
from typing import Any
from collections.abc import Mapping

from skrl import logger
from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import MultiAgentEnvWrapper, Wrapper
from skrl.models.torch import Model
from skrl.resources.noises.torch import (  # noqa: F401  (used by eval)
    GaussianNoise,
    OrnsteinUhlenbeckNoise,
)
from skrl.resources.preprocessors.torch import (  # noqa: F401  (used by eval)
    RunningStandardScaler,
)
from skrl.resources.schedulers.torch import KLAdaptiveLR  # noqa: F401  (used by eval)
from skrl.utils.runner.torch import Runner


class CustomRunner(Runner):
    """Project-local skrl Runner with asymmetric AC and V(s, h) support."""

    def _component(self, name: str) -> type:
        """Resolve component name → class/factory.

        Extends the base resolver with project-local model instantiators and
        agent classes. Unknown names fall back to the upstream lookup.
        """
        lname = name.lower()
        # project-local model instantiators
        if lname == "gaussiancnnrnnmixin":
            from isaac_pursuit_evasion.skrl_ext.models import gaussian_cnn_rnn_model

            return gaussian_cnn_rnn_model
        if lname == "gaussiancnngruflatmixin":
            from isaac_pursuit_evasion.skrl_ext.models import gaussian_cnn_rnn_flat_model

            return gaussian_cnn_rnn_flat_model
        if lname in ("vshcriticmixin", "szcriticmixin"):
            from isaac_pursuit_evasion.skrl_ext.models import vsh_critic_model

            return vsh_critic_model
        if lname == "historystatecriticmixin":
            from isaac_pursuit_evasion.skrl_ext.models import history_state_critic_model

            return history_state_critic_model
        # project-local agents
        if lname in ["ppo_asym", "ppo_asym_default_config"]:
            from isaac_pursuit_evasion.skrl_ext.agents import (
                PPO_ASYM,
                PPO_ASYM_DEFAULT_CONFIG,
            )

            return PPO_ASYM_DEFAULT_CONFIG if "default_config" in lname else PPO_ASYM
        if lname in [
            "ppo_rnn_vsh", "ppo_rnn_vsh_default_config",
            "ppo_rnn_sz", "ppo_rnn_sz_default_config",
            "ppo_rnn_sh", "ppo_rnn_sh_default_config",
            "ppo_rnn_asym", "ppo_rnn_asym_default_config",
        ]:
            from isaac_pursuit_evasion.skrl_ext.agents import (
                PPO_RNN_VSH,
                PPO_RNN_VSH_DEFAULT_CONFIG,
            )

            return PPO_RNN_VSH_DEFAULT_CONFIG if "default_config" in lname else PPO_RNN_VSH
        # fall back to upstream resolver
        return super()._component(name)

    def _process_cfg(self, cfg: dict) -> dict:
        """Add ``critic_state_preprocessor`` to the direct-eval allow-list.

        The upstream implementation converts a handful of preprocessor string
        names (e.g. ``"RunningStandardScaler"``) into their class objects via
        ``eval``; we need the same treatment for the asymmetric critic
        preprocessor.
        """
        _direct_eval = {
            "learning_rate_scheduler",
            "shared_state_preprocessor",
            "state_preprocessor",
            "critic_state_preprocessor",
            "value_preprocessor",
            "amp_state_preprocessor",
            "noise",
            "smooth_regularization_noise",
        }

        def reward_shaper_function(scale):
            def reward_shaper(rewards, *args, **kwargs):
                return rewards * scale

            return reward_shaper

        def update_dict(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    update_dict(value)
                else:
                    if key in _direct_eval:
                        if isinstance(value, str):
                            d[key] = eval(value)
                    elif key.endswith("_kwargs"):
                        d[key] = value if value is not None else {}
                    elif key in ["rewards_shaper_scale"]:
                        d["rewards_shaper"] = reward_shaper_function(value)
            return d

        return update_dict(copy.deepcopy(cfg))

    def _generate_models(
        self, env: Wrapper | MultiAgentEnvWrapper, cfg: Mapping[str, Any]
    ) -> Mapping[str, Mapping[str, Model]]:
        """Instantiate models, routing the ``value`` role to the privileged state space.

        This is a reimplementation of the upstream method with two project-local
        tweaks: (1) the value network observation space is swapped for the
        environment's ``state_space`` when asymmetric training is active, and
        (2) RNN model configs receive a ``num_envs`` default so our recurrent
        instantiator can size its hidden buffers.
        """
        multi_agent = isinstance(env, MultiAgentEnvWrapper)
        device = env.device
        possible_agents = env.possible_agents if multi_agent else ["agent"]
        num_envs = env.num_envs if hasattr(env, "num_envs") else 1
        state_spaces = env.state_spaces if multi_agent else {"agent": env.state_space}
        observation_spaces = env.observation_spaces if multi_agent else {"agent": env.observation_space}
        action_spaces = env.action_spaces if multi_agent else {"agent": env.action_space}

        agent_class = cfg.get("agent", {}).get("class", "").lower()
        # Agent classes that use asymmetric actor-critic (value network sees
        # privileged state_space instead of actor observation_space).
        _asym_agents = {"ppo_asym", "ppo_rnn_vsh", "ppo_rnn_sz", "ppo_rnn_sh", "ppo_rnn_asym"}

        models = {}
        for agent_id in possible_agents:
            _cfg = copy.deepcopy(cfg)
            models[agent_id] = {}
            models_cfg = _cfg.get("models")
            if not models_cfg:
                raise ValueError("No 'models' are defined in cfg")
            try:
                separate = models_cfg["separate"]
                del models_cfg["separate"]
            except KeyError:
                separate = True
                logger.warning("No 'separate' field defined in 'models' cfg. Defining it as True by default")
            if separate:
                for role in models_cfg:
                    model_class = models_cfg[role].get("class")
                    if not model_class:
                        raise ValueError(f"No 'class' field defined in 'models:{role}' cfg")
                    del models_cfg[role]["class"]
                    model_class = self._component(model_class)
                    observation_space = observation_spaces[agent_id]
                    # asymmetric actor-critic: value network sees privileged state.
                    # Only swap for asymmetric agent classes.  Symmetric PPO uses
                    # the same observation_space for both actor and critic.
                    if role == "value" and state_spaces[agent_id] is not None and agent_class in _asym_agents:
                        observation_space = state_spaces[agent_id]
                    if agent_class == "mappo" and role == "value":
                        observation_space = state_spaces[agent_id]
                    if agent_class == "amp" and role == "discriminator":
                        try:
                            observation_space = env.amp_observation_space
                        except Exception:
                            logger.warning(
                                "Unable to get AMP space via 'env.amp_observation_space'."
                                " Using 'env.observation_space' instead"
                            )
                    model_kwargs = self._process_cfg(models_cfg[role])
                    if "rnn" in model_kwargs:
                        model_kwargs.setdefault("num_envs", num_envs)
                    source = model_class(
                        observation_space=observation_space,
                        action_space=action_spaces[agent_id],
                        device=device,
                        **model_kwargs,
                        return_source=True,
                    )
                    logger.info(f"Model (role): {role}\n{source}")
                    models[agent_id][role] = model_class(
                        observation_space=observation_space,
                        action_space=action_spaces[agent_id],
                        device=device,
                        **model_kwargs,
                    )
            else:
                roles = list(models_cfg.keys())
                if len(roles) != 2:
                    raise ValueError(
                        "Runner currently only supports shared models, made up of exactly two models. "
                        "Set 'separate' field to True to create non-shared models for the given cfg"
                    )
                structure = []
                parameters = []
                for role in roles:
                    model_structure = models_cfg[role].get("class")
                    if not model_structure:
                        raise ValueError(f"No 'class' field defined in 'models:{role}' cfg")
                    del models_cfg[role]["class"]
                    structure.append(model_structure)
                    parameters.append(self._process_cfg(models_cfg[role]))
                model_class = self._component("Shared")
                source = model_class(
                    observation_space=observation_spaces[agent_id],
                    action_space=action_spaces[agent_id],
                    device=device,
                    structure=structure,
                    roles=roles,
                    parameters=parameters,
                    return_source=True,
                )
                logger.info(f"Shared model (roles): {roles}\n{source}")
                models[agent_id][roles[0]] = model_class(
                    observation_space=observation_spaces[agent_id],
                    action_space=action_spaces[agent_id],
                    device=device,
                    structure=structure,
                    roles=roles,
                    parameters=parameters,
                )
                models[agent_id][roles[1]] = models[agent_id][roles[0]]

        for agent_id in possible_agents:
            for role, model in models[agent_id].items():
                model.init_state_dict(role)

        return models

    def _generate_agent(
        self,
        env: Wrapper | MultiAgentEnvWrapper,
        cfg: Mapping[str, Any],
        models: Mapping[str, Mapping[str, Model]],
    ) -> Agent:
        """Instantiate the agent, wiring up the asymmetric critic preprocessor.

        This mirrors the upstream method but also accepts our project-local
        agent classes (``ppo_asym``, ``ppo_rnn_vsh``) and configures
        ``critic_state_preprocessor_kwargs`` from the environment's
        ``state_space`` when asymmetric training is active. When the env does
        not expose a state space (e.g. symmetric vision training), the
        preprocessor is silently disabled.
        """
        multi_agent = isinstance(env, MultiAgentEnvWrapper)
        device = env.device
        num_envs = env.num_envs
        possible_agents = env.possible_agents if multi_agent else ["agent"]
        state_spaces = env.state_spaces if multi_agent else {"agent": env.state_space}
        observation_spaces = env.observation_spaces if multi_agent else {"agent": env.observation_space}
        action_spaces = env.action_spaces if multi_agent else {"agent": env.action_space}

        agent_class = cfg.get("agent", {}).get("class", "").lower()
        if not agent_class:
            raise ValueError("No 'class' field defined in 'agent' cfg")

        # delegate standard / multi-agent classes (ppo, amp, ippo, mappo, ...)
        # to the upstream implementation — only PPO_ASYM and PPO_RNN_VSH need
        # the asymmetric state_space wiring below.
        if agent_class not in ["ppo_asym", "ppo_rnn_vsh", "ppo_rnn_sz", "ppo_rnn_sh", "ppo_rnn_asym"]:
            return super()._generate_agent(env, cfg, models)

        # project-local: PPO_ASYM / PPO_RNN_VSH — create memories + wire
        # critic_state_preprocessor from the privileged state_space
        if "memory" not in cfg:
            logger.warning(
                "Deprecation warning: No 'memory' field defined in cfg. Using the default generated configuration"
            )
            cfg["memory"] = {"class": "RandomMemory", "memory_size": -1}
        try:
            memory_class = self._component(cfg["memory"]["class"])
            del cfg["memory"]["class"]
        except KeyError:
            memory_class = self._component("RandomMemory")
            logger.warning("No 'class' field defined in 'memory' cfg. 'RandomMemory' will be used as default")
        memories = {}
        if cfg["memory"]["memory_size"] < 0:
            cfg["memory"]["memory_size"] = cfg["agent"]["rollouts"]
        for agent_id in possible_agents:
            memories[agent_id] = memory_class(num_envs=num_envs, device=device, **self._process_cfg(cfg["memory"]))

        agent_id = possible_agents[0]
        agent_cfg = self._component(f"{agent_class}_DEFAULT_CONFIG").copy()
        agent_cfg.update(self._process_cfg(cfg["agent"]))
        agent_cfg.get("state_preprocessor_kwargs", {}).update({"size": observation_spaces[agent_id], "device": device})
        # asymmetric actor-critic: always size critic_state_preprocessor_kwargs from state_space.
        # PPO_RNN_VSH.init reads `critic_state_preprocessor_kwargs.size` to allocate the
        # memory.critic_states slot, even when `critic_state_preprocessor` is null
        # (e.g. Geles V(s,o,a) does its own state-component scaling internally). Without this,
        # the slot defaults to self.observation_space (actor obs 8196) and add_samples crashes
        # on critic states of size 8260.
        state_space = state_spaces[agent_id]
        if agent_cfg.get("critic_state_preprocessor_kwargs", None) is None:
            agent_cfg["critic_state_preprocessor_kwargs"] = {}
        invalid_state_space = state_space is None
        if not invalid_state_space and isinstance(state_space, (int, float)):
            invalid_state_space = state_space <= 0
        if invalid_state_space:
            agent_cfg["critic_state_preprocessor"] = None
            agent_cfg["critic_state_preprocessor_kwargs"] = {}
        else:
            agent_cfg["critic_state_preprocessor_kwargs"].update({"size": state_space, "device": device})
        agent_cfg.get("value_preprocessor_kwargs", {}).update({"size": 1, "device": device})
        agent_kwargs = {
            "models": models[agent_id],
            "memory": memories[agent_id],
            "observation_space": observation_spaces[agent_id],
            "action_space": action_spaces[agent_id],
        }
        return self._component(agent_class)(cfg=agent_cfg, device=device, **agent_kwargs)
