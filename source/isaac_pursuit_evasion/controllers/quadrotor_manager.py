from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from tensordict import TensorDictBase

from isaaclab.assets import Articulation, ArticulationData

from source.isaac_pursuit_evasion.controllers.evader import CrazyflieAPFEvaderWrapper
from source.isaac_pursuit_evasion.controllers.pursuer import CrazyflieFRPNPursuerWrapper
from source.isaac_pursuit_evasion.controllers.rl_controllers import (
    CrazyflieRLBodyRatesWrapper,
    CrazyflieRLVelocityWrapper,
)
from source.isaac_pursuit_evasion.deployment.actor_policy_loader import (
    ActorPolicyCallable,
    ActorPolicyConfig,
    RecurrentActorConfig,
    RecurrentActorPolicyCallable,
    load_actor_from_checkpoint,
    load_actor_policy_config,
    load_recurrent_actor_from_checkpoint,
)
from source.isaac_pursuit_evasion.deployment.critic_policy_loader import (
    CriticPolicyConfig,
    load_critic_from_checkpoint,
    load_critic_policy_config,
)
from source.isaac_pursuit_evasion.controllers.config import load_controller_config
from source.isaac_pursuit_evasion.dynamics.propellers import Drone_cfg
from source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.trajectories.trajectory import (
    CrazyflieTrajectoryWrapper,
    TrajectoryBatchManager,
    TrajectorySpec,
    WallConfig,
)

TRAJECTORY_TYPES = {"hover", "circular", "lemniscate"}
TRAJECTORY_CLASS_MAP = {
    "hover": "hovertrajectory",
    "circular": "circulartrajectory",
    "lemniscate": "lemniscatetrajectory",
}
RL_KINDS = {"rl_velocity", "rl_bodyrates", "rl_policy"}


class QuadrotorManager:
    """Manages heterogeneous controllers (geometric, RL, trajectories) for a quadrotor."""

    def __init__(
        self,
        robot: Articulation,
        drone_cfg: Drone_cfg,
        dt: float,
        num_envs: int,
        total_timesteps: int,
        controller_assignment: Dict[str, Dict[str, Any]],
        device: str,
        arena_bounds: torch.Tensor,
        env_origins: torch.Tensor,
        obs_dim: int,
        action_dim: int,
        pid_params: Optional[Mapping[str, Any]] = None,
        pid_dt: Optional[float] = None,
        role: Literal["pursuer", "evader"] = "pursuer",
        wall_cfg: Optional[WallConfig] = None,
    ) -> None:
        self.robot = robot
        self.drone_cfg = drone_cfg
        self.dt = dt
        self.num_envs = num_envs
        self.total_timesteps = total_timesteps
        self.device = torch.device(device)
        self.role = role
        self.arena_bounds = arena_bounds
        self.env_origins = env_origins
        self.wall_cfg = wall_cfg
        self._obs_dim = int(obs_dim)
        self._action_dim = int(action_dim)
        self._pid_params = copy.deepcopy(pid_params) if pid_params is not None else None
        self._pid_dt = float(pid_dt) if pid_dt is not None else float(dt)

        self._trajectory: Optional[dict[str, Any]] = None
        self.prev_action = torch.zeros((num_envs, self._action_dim), device=self.device)
        self._controllers: list[Dict[str, Any]] = []
        self._critic_policies: dict[str, nn.Module] = {}
        # Opponent z tracking: maps global env_id → GRU hidden state (last layer)
        self._opp_z_dim: int = 0  # set when a recurrent policy is loaded
        self._opp_z: Optional[torch.Tensor] = None
        self._recurrent_policies: dict[str, RecurrentActorPolicyCallable] = {}
        self._register_controllers(controller_assignment or {})
        self._global_frame = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def reset(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.prev_action[env_ids] = 0.0

        # Reset opponent z for reset envs
        if self._opp_z is not None:
            self._opp_z[env_ids] = 0.0

        for entry in self._controllers:
            controller = entry["controller"]
            if not hasattr(controller, "reset"):
                continue
            local_ids = _global_to_local(entry["env_ids"], env_ids)
            if local_ids.numel() == 0:
                continue
            controller.reset(local_ids)
            # Reset recurrent policy hidden state
            if entry.get("recurrent") and entry["name"] in self._recurrent_policies:
                self._recurrent_policies[entry["name"]].reset(local_ids)

        if self._trajectory is not None:
            traj = self._trajectory
            local_ids = traj["global_to_traj"][env_ids]
            mask = local_ids >= 0
            if mask.any():
                local_ids = torch.unique(local_ids[mask])
                traj["manager"].reset(local_ids)
                traj["controller"].reset(local_ids)
                self._regenerate_trajectory_series(local_ids)

    def set_curriculum_frame(self, frame: int) -> None:
        self._global_frame = int(frame)
        for entry in self._controllers:
            controller = entry["controller"]
            if hasattr(controller, "update_curriculum"):
                controller.update_curriculum(self._global_frame)

    def set_rate_gains(
        self,
        env_ids: torch.Tensor,
        rate_kp: torch.Tensor,
        rate_ki: torch.Tensor,
        rate_kd: torch.Tensor,
    ) -> None:
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        for entry in self._controllers:
            controller = entry["controller"]
            pid = getattr(controller, "pid", None)
            if pid is None or not hasattr(pid, "set_rate_gains"):
                continue
            local_ids = _global_to_local(entry["env_ids"], env_ids)
            if local_ids.numel() == 0:
                continue
            global_ids = entry["env_ids"][local_ids]
            pid.set_rate_gains(
                rate_kp=rate_kp[global_ids],
                rate_ki=rate_ki[global_ids],
                rate_kd=rate_kd[global_ids],
                env_ids=local_ids,
            )

        if self._trajectory is not None:
            traj_controller = self._trajectory["controller"]
            pid = getattr(traj_controller, "pid", None)
            if pid is not None and hasattr(pid, "set_rate_gains"):
                local_ids = _global_to_local(self._trajectory["global_ids"], env_ids)
                if local_ids.numel() > 0:
                    global_ids = self._trajectory["global_ids"][local_ids]
                    pid.set_rate_gains(
                        rate_kp=rate_kp[global_ids],
                        rate_ki=rate_ki[global_ids],
                        rate_kd=rate_kd[global_ids],
                        env_ids=local_ids,
                    )

    def compute_action(
        self,
        adversary_data: Optional[ArticulationData],
        rl_observations: Optional[Dict[str, TensorDictBase]] = None,
    ) -> torch.Tensor:
        commands = torch.zeros((self.num_envs, 4), device=self.device)
        robot_state = self.robot.data.root_state_w.clone()
        robot_state[..., :3] -= self.env_origins

        if self._trajectory is not None:
            traj_cmd = self._compute_trajectory_actions(robot_state)
            commands[self._trajectory["global_ids"]] = traj_cmd

        for entry in self._controllers:
            env_ids = entry["env_ids"]
            subset_state = robot_state[env_ids]
            kind = entry["kind"]
            controller = entry["controller"]

            if kind == "frpn_pursuer":
                if adversary_data is None:
                    raise ValueError("FRPN pursuer controllers require adversary data.")
                pursuer_state = subset_state
                target_state = adversary_data.root_state_w[env_ids].clone()
                target_state[..., :3] -= self.env_origins[env_ids]
                cmd = controller.command(pursuer_state, target_state)

            elif kind == "apf_evader":
                if adversary_data is None:
                    raise ValueError("APF evader controllers require adversary data.")
                evader_state = subset_state
                pursuer_state = adversary_data.root_state_w[env_ids].clone()
                pursuer_state[..., :3] -= self.env_origins[env_ids]
                cmd = controller.command(pursuer_state, evader_state)
                
            elif kind in RL_KINDS:
                if rl_observations is None or entry["name"] not in rl_observations:
                    raise KeyError(f"Missing observations for RL controller '{entry['name']}'.")
                cmd = controller.command(rl_observations[entry["name"]])
                # Capture z_opp from recurrent policies after forward pass
                if entry.get("recurrent") and entry["name"] in self._recurrent_policies:
                    rp = self._recurrent_policies[entry["name"]]
                    if self._opp_z is not None:
                        self._opp_z[env_ids] = rp.get_z()[:env_ids.numel()]
            else:
                raise KeyError(f"Unsupported controller kind '{kind}'.")

            commands[env_ids] = cmd

        self.prev_action = commands.clone()
        return commands

    def compute_wrench(self, commands: torch.Tensor) -> torch.Tensor:
        wrenches = torch.zeros((self.num_envs, 4), device=self.device)
        if commands.numel() == 0:
            return wrenches

        robot_state = self.robot.data.root_state_w.clone()
        robot_state[..., :3] -= self.env_origins

        if self._trajectory is not None:
            traj = self._trajectory
            global_ids = traj["global_ids"]
            if global_ids.numel() > 0:
                root_state = robot_state[global_ids]
                cmd = commands[global_ids]
                wrenches[global_ids] = traj["controller"].wrench_from_command(root_state, cmd)

        for entry in self._controllers:
            env_ids = entry["env_ids"]
            if env_ids.numel() == 0:
                continue
            root_state = robot_state[env_ids]
            cmd = commands[env_ids]
            wrenches[env_ids] = entry["controller"].wrench_from_command(root_state, cmd)

        return wrenches

    def _compute_trajectory_actions(self, robot_state: torch.Tensor) -> torch.Tensor:
        traj = self._trajectory
        if traj is None:
            return torch.zeros((0, 4), device=self.device)

        series = traj["series"]
        total_envs = traj["manager"].total_envs
        env_indices = torch.arange(total_envs, device=self.device)

        step = traj["step"]
        idx = torch.remainder(step, traj["horizon"])

        vel = series["vel"][idx, env_indices]
        speed_xy = torch.norm(vel[:, :2], dim=-1, keepdim=True)
        yaw = torch.atan2(vel[:, 1], vel[:, 0]).view(-1, 1)
        if "yaw" not in traj:
            traj["yaw"] = torch.zeros_like(yaw)
        yaw = torch.where(speed_xy > 1e-5, yaw, traj["yaw"])
        traj["yaw"] = yaw

        traj["step"] = step + 1
        return torch.cat((vel, yaw), dim=-1)

    def get_trajectory_setpoint(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the evader's current trajectory setpoint (pos, vel).

        Returns:
            pos:  (num_envs, 3)  target position
            vel:  (num_envs, 3)  target velocity

        For non-trajectory envs the values are zero.

        The step counter was already incremented in ``_compute_trajectory_actions``
        so we look up ``step - 1`` (the setpoint that was *just* commanded).
        """
        pos = torch.zeros(self.num_envs, 3, device=self.device)
        vel = torch.zeros(self.num_envs, 3, device=self.device)
        if self._trajectory is not None:
            traj = self._trajectory
            series = traj["series"]
            total_envs = traj["manager"].total_envs
            env_indices = torch.arange(total_envs, device=self.device)
            # step was already incremented, so current setpoint is at step-1
            idx = torch.remainder(traj["step"] - 1, traj["horizon"])
            global_ids = traj["global_ids"]
            pos[global_ids] = series["pos"][idx, env_indices]
            vel[global_ids] = series["vel"][idx, env_indices]
        return pos, vel

    def get_pid_integrals(self) -> torch.Tensor:
        """Return PID integrals for all envs as (num_envs, 9).

        Order per env: vel_pid(3) + att_pid(3) + rate_pid(3).
        Controllers without a ``.pid`` attribute contribute zeros.
        Trajectory controllers (hover/circular/lemniscate) are handled via
        ``self._trajectory``, not ``self._controllers``.
        """
        result = torch.zeros(self.num_envs, 9, device=self.device)
        for entry in self._controllers:
            ctrl = entry["controller"]
            env_ids = entry["env_ids"]
            if hasattr(ctrl, "pid"):
                integrals = ctrl.pid.get_pid_integrals(num_envs=env_ids.numel())
                result[env_ids] = integrals
        # Also include trajectory controller PID if present
        if self._trajectory is not None:
            traj = self._trajectory
            ctrl = traj["controller"]
            if hasattr(ctrl, "pid"):
                global_ids = traj["global_ids"]
                integrals = ctrl.pid.get_pid_integrals(num_envs=global_ids.numel())
                result[global_ids] = integrals
        return result

    def get_previous_action(self) -> torch.Tensor:
        return self.prev_action

    def get_opp_z(self) -> torch.Tensor:
        """Return opponent's RNN hidden state z^opp.

        Shape: (num_envs, z_dim). Returns zeros for envs with non-recurrent
        (scripted/MLP) opponents.
        """
        if self._opp_z is not None:
            return self._opp_z
        # No recurrent policies loaded — return zeros with default dim
        z_dim = self._opp_z_dim if self._opp_z_dim > 0 else 256
        return torch.zeros(self.num_envs, z_dim, device=self.device)

    def get_opp_prev_action(self) -> torch.Tensor:
        """Return opponent's previous action (all envs)."""
        return self.prev_action

    def get_critic(self, name: str) -> Optional[nn.Module]:
        return self._critic_policies.get(name)

    def rl_env_assignments(self) -> Dict[str, Dict[str, Any]]:
        assignments: Dict[str, Dict[str, Any]] = {}
        for entry in self._controllers:
            if entry["kind"] not in RL_KINDS:
                continue
            assignments[entry["name"]] = {
                "env_ids": entry["env_ids"].clone(),
                "kind": entry["kind"],
                "config": entry["config"],
                "recurrent": entry.get("recurrent", False),
            }
        return assignments

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _register_controllers(self, controller_assignment: Dict[str, Dict[str, Any]]) -> None:
        occupancy = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        trajectory_entries: list[Dict[str, Any]] = []

        for name, cfg in controller_assignment.items():
            cfg = cfg or {}
            env_ids = _normalize_env_ids(cfg.get("env_ids"), self.device, self.num_envs)
            if env_ids.numel() == 0:
                continue
            if occupancy[env_ids].any():
                raise ValueError(f"Environment IDs overlap for controller '{name}'.")
            occupancy[env_ids] = True

            kind = cfg.get("kind") or _infer_kind(name)
            if kind in TRAJECTORY_TYPES:
                trajectory_entries.append(
                    {"name": name, "kind": kind, "env_ids": env_ids, "config": cfg or {} }
                )
                continue

            controller, is_recurrent = self._instantiate_controller(name, kind, env_ids, cfg or {})
            entry = {
                "name": name,
                "kind": kind,
                "env_ids": env_ids,
                "config": cfg,
                "controller": controller,
                "recurrent": is_recurrent,
            }
            self._controllers.append(entry)

        self._init_trajectory_controllers(trajectory_entries)

    def _resolve_controller_cfg(self, kind: str, cfg: Dict[str, Any]) -> Dict[str, Any] | None:
        """Return the controller config dict, auto-loading from YAML if not explicitly provided.

        If ``cfg["config"]`` is already set it is returned as-is (backward compatible).
        Otherwise the YAML is loaded for the current drone and any overrides stored in
        ``cfg["config_overrides"]`` are merged on top.
        """
        controller_cfg = cfg.get("config")
        if controller_cfg is not None:
            return controller_cfg
        # Auto-load from YAML using the drone name
        drone_name = getattr(self.drone_cfg, "name", None) or getattr(self.drone_cfg, "model", "crazyflie_brushless")
        try:
            base_cfg = load_controller_config(kind, drone_name)
        except FileNotFoundError:
            return None
        # Merge any overrides stored alongside the spec (e.g. curriculum)
        overrides = cfg.get("config_overrides")
        if overrides:
            base_cfg.update(overrides)
        return base_cfg

    def _instantiate_controller(self, name: str, kind: str, env_ids: torch.Tensor, cfg: Dict[str, Any]) -> tuple[Any, bool]:
        """Instantiate a controller. Returns (controller, is_recurrent)."""
        env_count = env_ids.numel()

        if kind == "frpn_pursuer":
            controller_cfg = self._resolve_controller_cfg(kind, cfg)
            curriculum_cfg = None
            if controller_cfg:
                curriculum_cfg = controller_cfg.get("curriculum")
            total_frames = self.total_timesteps
            return CrazyflieFRPNPursuerWrapper(
                num_envs=env_count,
                drone_cfg=self.drone_cfg,
                dt=self.dt,
                pid_dt=self._pid_dt,
                total_frames=total_frames,
                device=str(self.device),
                command_heading=bool(cfg.get("command_heading", True)),
                controller_cfg=controller_cfg,
                curriculum_cfg=curriculum_cfg,
                pid_params=self._pid_params,
                wall_cfg=self.wall_cfg,
            ), False

        if kind == "apf_evader":
            arena_min = self.arena_bounds[:, 0] if self.arena_bounds is not None else None
            arena_max = self.arena_bounds[:, 1] if self.arena_bounds is not None else None
            return CrazyflieAPFEvaderWrapper(
                num_envs=env_count,
                drone_cfg=self.drone_cfg,
                dt=self.dt,
                pid_dt=self._pid_dt,
                device=str(self.device),
                command_heading=bool(cfg.get("command_heading", True)),
                controller_cfg=self._resolve_controller_cfg(kind, cfg),
                arena_min=arena_min,
                arena_max=arena_max,
                pid_params=self._pid_params,
                wall_cfg=self.wall_cfg,
            ), False

        if kind in {"rl_velocity", "rl_policy"}:
            policy, critic, is_recurrent = self._resolve_policy(cfg, label=name, env_count=env_count)
            if critic is not None:
                self._critic_policies[name] = critic
            if is_recurrent:
                self._register_recurrent_policy(name, policy)
            controller = CrazyflieRLVelocityWrapper(
                num_envs=env_count,
                drone_cfg=self.drone_cfg,
                policy=policy,
                dt=cfg.get("dt", self.dt),
                pid_dt=self._pid_dt,
                device=str(self.device),
                action_key=cfg.get("action_key", "action"),
                root_state_key=cfg.get("root_state_key", "root_state"),
                vel_scale=cfg.get("vel_scale"),
                yaw_rate_scale=cfg.get("yaw_rate_scale"),
                pid_params=self._pid_params,
            )
            return controller, is_recurrent

        if kind == "rl_bodyrates":
            policy, critic, is_recurrent = self._resolve_policy(cfg, label=name, env_count=env_count)
            if critic is not None:
                self._critic_policies[name] = critic
            if is_recurrent:
                self._register_recurrent_policy(name, policy)
            controller = CrazyflieRLBodyRatesWrapper(
                num_envs=env_count,
                drone_cfg=self.drone_cfg,
                policy=policy,
                dt=cfg.get("dt", self.dt),
                pid_dt=self._pid_dt,
                device=str(self.device),
                action_key=cfg.get("action_key", "action"),
                root_state_key=cfg.get("root_state_key", "root_state"),
                body_rate_key=cfg.get("body_rate_key", "body_rate"),
                thrust_scale=cfg.get("thrust_scale"),
                pid_params=self._pid_params,
            )
            return controller, is_recurrent

        raise ValueError(f"Unknown controller kind '{kind}'.")

    def _init_trajectory_controllers(self, entries: list[Dict[str, Any]]) -> None:
        if not entries:
            self._trajectory = None
            return
        if self.arena_bounds is None:
            raise ValueError("Trajectory controllers require arena bounds.")

        specs: list[TrajectorySpec] = []
        global_ids = []
        horizon = None
        for entry in entries:
            cfg = entry["config"] or {}
            specs.append(
                TrajectorySpec(
                    name=TRAJECTORY_CLASS_MAP[entry["kind"]],
                    count=entry["env_ids"].numel(),
                )
            )
            global_ids.append(entry["env_ids"])
            cfg_horizon = int(cfg.get("trajectory_horizon", 0))
            if cfg_horizon > 0:
                horizon = cfg_horizon if horizon is None else max(horizon, cfg_horizon)

        if horizon is None:
            horizon = 256

        manager = TrajectoryBatchManager(
            specs=specs,
            device=self.device,
            arena_min=self.arena_bounds[:, 0],
            arena_max=self.arena_bounds[:, 1],
            wall_cfg=self.wall_cfg,
        )

        global_ids_cat = torch.cat(global_ids, dim=0)
        global_to_traj = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        global_to_traj[global_ids_cat] = torch.arange(manager.total_envs, device=self.device)

        self._trajectory = {
            "manager": manager,
            "controller": CrazyflieTrajectoryWrapper(
                num_envs=manager.total_envs,
                drone_cfg=self.drone_cfg,
                dt=self.dt,
                pid_dt=self._pid_dt,
                device=self.device,
                pid_params=self._pid_params,
            ),
            "global_ids": global_ids_cat,
            "global_to_traj": global_to_traj,
            "horizon": horizon,
            "step": torch.zeros(manager.total_envs, dtype=torch.long, device=self.device),
        }
        self._regenerate_trajectory_series()

    def _regenerate_trajectory_series(self, local_ids: torch.Tensor | None = None) -> None:
        if self._trajectory is None:
            return
        traj = self._trajectory
        pos, vel, acc = traj["manager"].generate_series(traj["horizon"], self.dt)
        if local_ids is None:
            traj["series"] = {"pos": pos, "vel": vel, "acc": acc}
            traj["step"].zero_()
            traj["yaw"] = torch.zeros((traj["manager"].total_envs, 1), device=self.device)
            return

        if "series" not in traj:
            traj["series"] = {"pos": pos.clone(), "vel": vel.clone(), "acc": acc.clone()}
            traj["step"].zero_()
            traj["yaw"] = torch.zeros((traj["manager"].total_envs, 1), device=self.device)
            return

        local_ids = local_ids.to(dtype=torch.long, device=self.device)
        traj["series"]["pos"][:, local_ids] = pos[:, local_ids]
        traj["series"]["vel"][:, local_ids] = vel[:, local_ids]
        traj["series"]["acc"][:, local_ids] = acc[:, local_ids]
        traj["step"][local_ids] = 0
        if "yaw" not in traj:
            traj["yaw"] = torch.zeros((traj["manager"].total_envs, 1), device=self.device)
        traj["yaw"][local_ids] = 0.0

    def _resolve_policy(
        self, cfg: Dict[str, Any], label: str | None = None, env_count: int = 0
    ) -> tuple[Any, Optional[nn.Module], bool]:
        """Resolve policy from config, returning (policy, critic, is_recurrent)."""
        cfg = cfg or {}
        if "policy" in cfg:
            policy = cfg["policy"]
            is_recurrent = getattr(policy, "is_recurrent", False)
            return policy, None, is_recurrent

        checkpoint = self._resolve_checkpoint(cfg)
        if checkpoint is None:
            raise ValueError("RL controllers require either a policy object or a checkpoint path.")

        # Check if this is a recurrent (CNN+GRU) policy
        recurrent_cfg_raw = cfg.get("recurrent_actor_cfg")
        if recurrent_cfg_raw is not None:
            recurrent_cfg = self._resolve_recurrent_actor_cfg(recurrent_cfg_raw)
            actor = load_recurrent_actor_from_checkpoint(checkpoint, recurrent_cfg, device=self.device)
            policy = RecurrentActorPolicyCallable(actor, num_envs=env_count, device=self.device)
            tag = label or "policy"
            print(f"[INFO] Loaded recurrent actor ({tag}): hidden_size={recurrent_cfg.rnn_hidden_size}")
            return policy, None, True

        actor_cfg = self._resolve_actor_cfg(cfg.get("actor_cfg"))
        actor = load_actor_from_checkpoint(checkpoint, actor_cfg, device=self.device)
        policy = ActorPolicyCallable(actor, device=self.device)

        critic_cfg_raw = cfg.get("critic_cfg")
        critic = None
        if critic_cfg_raw is not None and critic_cfg_raw is not False:
            critic_cfg = self._resolve_critic_cfg(critic_cfg_raw)
            critic = load_critic_from_checkpoint(checkpoint, critic_cfg, device=self.device)
        tag = label or "policy"
        print(f"[INFO] Policy preprocessors ({tag}) actor obs scaler: not found")
        print(f"[INFO] Policy preprocessors ({tag}) critic/value scaler: not found")
        return policy, critic, False

    def _register_recurrent_policy(self, name: str, policy: RecurrentActorPolicyCallable) -> None:
        """Register a recurrent policy and allocate the global opp_z buffer."""
        self._recurrent_policies[name] = policy
        z_dim = policy.rnn_hidden_size
        if self._opp_z_dim == 0:
            self._opp_z_dim = z_dim
            self._opp_z = torch.zeros(self.num_envs, z_dim, device=self.device)
        elif z_dim != self._opp_z_dim:
            raise ValueError(
                f"Mismatched recurrent policy hidden sizes: {z_dim} vs {self._opp_z_dim}. "
                "All recurrent opponent policies must share the same GRU hidden size."
            )

    def _resolve_recurrent_actor_cfg(self, cfg: Any) -> RecurrentActorConfig:
        if isinstance(cfg, RecurrentActorConfig):
            return cfg
        if isinstance(cfg, Mapping):
            return RecurrentActorConfig.from_dict(cfg)
        raise TypeError(f"Unsupported recurrent_actor_cfg type: {type(cfg)}")

    def _resolve_actor_cfg(self, actor_cfg: Any) -> ActorPolicyConfig:
        if isinstance(actor_cfg, ActorPolicyConfig):
            return actor_cfg
        if isinstance(actor_cfg, Mapping):
            return ActorPolicyConfig.from_dict(actor_cfg)
        if isinstance(actor_cfg, (str, Path)):
            return load_actor_policy_config(actor_cfg)
        if actor_cfg is None:
            return load_actor_policy_config()
        raise TypeError(f"Unsupported actor_cfg type: {type(actor_cfg)}")

    def _resolve_critic_cfg(self, critic_cfg: Any) -> CriticPolicyConfig:
        if isinstance(critic_cfg, CriticPolicyConfig):
            return critic_cfg
        if isinstance(critic_cfg, Mapping):
            return CriticPolicyConfig.from_dict(critic_cfg)
        if isinstance(critic_cfg, (str, Path)):
            return load_critic_policy_config(critic_cfg)
        if isinstance(critic_cfg, bool):
            if critic_cfg:
                return load_critic_policy_config()
            raise ValueError("critic_cfg set to False; refusing to resolve.")
        if critic_cfg is None:
            return load_critic_policy_config()
        raise TypeError(f"Unsupported critic_cfg type: {type(critic_cfg)}")

    def _resolve_checkpoint(self, cfg: Dict[str, Any]) -> str | None:
        if "wandb_artifact" in cfg:
            return _download_wandb_artifact(cfg["wandb_artifact"])
        checkpoint = cfg.get("checkpoint") or cfg.get("path")
        return str(checkpoint) if checkpoint is not None else None


def _download_wandb_artifact(cfg: Dict[str, Any]) -> str:
    """Download a checkpoint from Weights & Biases and return the local file path."""
    try:
        import wandb  # type: ignore
    except Exception as exc:
        raise ImportError("wandb is required to download artifacts.") from exc

    artifact_path = cfg.get("artifact") or cfg.get("path")
    artifact_file = cfg.get("file") or cfg.get("artifact_file")
    local_dir = cfg.get("local_dir")
    api = wandb.Api()
    artifact = api.artifact(artifact_path)
    download_dir = Path(artifact.download(root=str(local_dir)) if local_dir else artifact.download())
    if artifact_file:
        candidate = download_dir / artifact_file
        if candidate.exists():
            return str(candidate)
    pt_files = sorted(download_dir.rglob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt checkpoints found in artifact {artifact_path}")
    return str(pt_files[-1])


# ---------------------------------------------------------------------- #
# Utility helpers
# ---------------------------------------------------------------------- #
def _normalize_env_ids(env_ids: Any, device: torch.device, num_envs: int) -> torch.Tensor:
    if env_ids is None:
        raise ValueError("Controller assignment requires 'env_ids'.")
    tensor = torch.as_tensor(env_ids, device=device, dtype=torch.long)
    tensor = tensor.flatten()
    tensor = tensor.clamp(min=0, max=num_envs - 1)
    tensor, _ = torch.sort(torch.unique(tensor))
    return tensor

def _infer_kind(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("rl_velocity"):
        return "rl_velocity"
    if lowered.startswith("rl_bodyrates"):
        return "rl_bodyrates"
    return lowered


def _global_to_local(controller_envs: torch.Tensor, reset_ids: torch.Tensor) -> torch.Tensor:
    if reset_ids.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=controller_envs.device)
    mask = (controller_envs.unsqueeze(0) == reset_ids.unsqueeze(1)).any(dim=0)
    return torch.nonzero(mask, as_tuple=False).squeeze(-1)
