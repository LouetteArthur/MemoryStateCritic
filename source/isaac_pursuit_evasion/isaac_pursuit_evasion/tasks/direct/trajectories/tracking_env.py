# Copyright (c) 2026, the MemoryStateCritic authors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import gymnasium as gym
import torch
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from source.isaac_pursuit_evasion.assets.crazyflie_brushless import (
    CrazyflieBrushlessPursuer,
)
from source.isaac_pursuit_evasion.controllers.crazy_controller import (
    CrazyfliePIDController,
)
from source.isaac_pursuit_evasion.dynamics.propellers import Drone_cfg, Propellers
from source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.trajectories.trajectory import (
    TrajectoryBatchManager,
    TrajectorySpec,
)


@dataclass
class TrajectorySpecConfig:
    name: str
    count: int


@configclass
class TrajectoryTrackingEnvCfg(DirectRLEnvCfg):
    """Configuration for the trajectory tracking demo environment."""

    episode_length_s = 10.0
    sim_rate_hz = 500
    policy_rate_hz = 50
    pid_loop_rate_hz = 500
    pid_posvel_loop_rate_hz = 100
    decimation = sim_rate_hz // policy_rate_hz
    action_space = 4
    observation_space = 0
    state_space = 0
    debug_vis = True

    sim: SimulationCfg = SimulationCfg(
        dt=1 / sim_rate_hz,
        render_interval=decimation,
        create_stage_in_memory=False,
    )
    terrain: TerrainImporterCfg = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", debug_vis=False)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=0.0, replicate_physics=True)
    robot = None  # lazily populated once IsaacLab assets are available
    controller_type: str = "crazyflie_pid"
    drone_name: str = "crazyflie_brushless"

    trajectory_specs: Sequence[TrajectorySpecConfig] = (
        TrajectorySpecConfig(name="hovertrajectory", count=2048),
        TrajectorySpecConfig(name="circulartrajectory", count=4096),
        TrajectorySpecConfig(name="lemniscatetrajectory", count=2048),
    )
    arena_min = (-2.0, -2.0, 0.0)
    arena_max = (2.0, 2.0, 2.0)

    position_error_weight: float = 1.0
    collision_penalty: float = 50.0
    collision_altitude: float = 0.1
    enable_yaw_tracking: bool = False


class TrajectoryTrackingEnv(DirectRLEnv):
    """Standalone environment that tracks analytic trajectories with a cascaded Crazyflie PID controller."""

    cfg: TrajectoryTrackingEnvCfg

    def __init__(
        self,
        cfg: TrajectoryTrackingEnvCfg,
        trajectory_specs: Sequence[TrajectorySpecConfig] | None = None,
        controller_overrides: dict | None = None,
        enable_yaw_tracking: bool | None = None,
        controller_type: str | None = None,
        drone_name: str | None = None,
        **kwargs,
    ) -> None:
        if trajectory_specs is not None:
            cfg.trajectory_specs = trajectory_specs
        if controller_type is not None:
            cfg.controller_type = controller_type
        if drone_name is not None:
            cfg.drone_name = drone_name

        total_envs = sum(spec.count for spec in cfg.trajectory_specs)
        cfg.scene.num_envs = total_envs

        drone_name = cfg.drone_name.lower()
        if drone_name not in ("crazyflie_brushless", "cf_brushless"):
            raise ValueError(f"Unsupported drone_name '{cfg.drone_name}'. Only crazyflie_brushless is supported.")

        if cfg.robot is None:
            cfg.robot = CrazyflieBrushlessPursuer.replace(prim_path="/World/envs/env_.*/Robot")

        self._enable_yaw_tracking = cfg.enable_yaw_tracking if enable_yaw_tracking is None else enable_yaw_tracking

        super().__init__(cfg, **kwargs)

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros_like(self._thrust)

        self._body_id = self._robot.find_bodies("body")[0]

        self._controller_type = cfg.controller_type.lower()
        if self._controller_type != "crazyflie_pid":
            raise ValueError(f"Unsupported controller_type '{cfg.controller_type}'. Use 'crazyflie_pid'.")
        self._drone_cfg = Drone_cfg(cfg.drone_name, device=self.device)
        self._propellers = Propellers(self.num_envs, self._drone_cfg, self.sim.cfg.dt, use=True, device=self.device)
        masses = self._robot.root_physx_view.get_masses()[0].to(self.device)
        mass_total = masses.sum()
        inertia_body = self._robot.root_physx_view.get_inertias()[0, self._body_id, :].view(3, 3).to(self.device)
        self._drone_cfg.set_physical_params(mass_total, inertia_body)

        arena_min = torch.tensor(cfg.arena_min, device=self.device, dtype=torch.float32)
        arena_max = torch.tensor(cfg.arena_max, device=self.device, dtype=torch.float32)
        self._arena_min_tensor = arena_min
        self._arena_max_tensor = arena_max
        specs = [TrajectorySpec(**asdict(spec)) for spec in cfg.trajectory_specs]
        self.trajectory_manager = TrajectoryBatchManager(specs, self.device, arena_min, arena_max)
        if self.trajectory_manager.total_envs != self.num_envs:
            raise ValueError("Sum of trajectory counts must equal number of environments.")

        self._dt = self.sim.cfg.dt * self.cfg.decimation
        self._reference_pos = torch.zeros(self.max_episode_length, self.num_envs, 3, device=self.device)
        self._reference_vel = torch.zeros_like(self._reference_pos)
        self._reference_acc = torch.zeros_like(self._reference_pos)

        controller_params = {
            "sim_rate_hz": cfg.sim_rate_hz,
            "pid_loop_rate_hz": cfg.pid_loop_rate_hz,
            "pid_posvel_loop_rate_hz": cfg.pid_posvel_loop_rate_hz,
        }
        self.crazy_controller = CrazyfliePIDController(
            dt=self.sim.cfg.dt,
            drone_cfg=self._drone_cfg,
            num_envs=self.num_envs,
            device=self.device,
            params=controller_params,
        )
        inertia_tensor = inertia_body.view(1, 3, 3).tile(self.num_envs, 1, 1)
        self.crazy_controller.set_physical_params(mass_total, inertia_tensor)

        self._refresh_reference_series()
        self._vel_markers: VisualizationMarkers | None = None
        self._ref_vel_markers: VisualizationMarkers | None = None
        self._target_markers: VisualizationMarkers | None = None
        self._velocity_scale = 0.5
        self._velocity_radius = 0.16
        self._setup_visualizers()

        self._current_target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._current_target_vel = torch.zeros_like(self._current_target_pos)
        self._current_target_acc = torch.zeros_like(self._current_target_pos)
        self._latest_position_error = torch.zeros(self.num_envs, device=self.device)
        self._latest_collision_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._collision_counts = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._current_target_yaw = torch.zeros(self.num_envs, 1, device=self.device)
        self._prev_target_yaw = torch.zeros_like(self._current_target_yaw)
        self._current_target_yaw_rate = torch.zeros_like(self._current_target_yaw)
        self._prop_joint_ids = self._find_prop_joints(self._robot)

    # -------------------------------------------------------------------------
    # IsaacLab interface implementation
    # -------------------------------------------------------------------------

    def _setup_scene(self) -> None:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation

        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _apply_action(self) -> None:
        root_state = self._robot.data.root_state_w
        thrust, moment = self.crazy_controller(
            root_state,
            target_pos=self._current_target_pos,
            target_vel=self._current_target_vel,
            target_yaw=self._current_target_yaw if self._enable_yaw_tracking else None,
            target_yaw_rate=self._current_target_yaw_rate if self._enable_yaw_tracking else None,
            command_level="position",
        )
        wrench = torch.cat((thrust, moment), dim=-1)
        omega_ref = self._propellers.compute_motor_speeds_from_wrench(wrench)
        self._propellers.compute_omega(omega_ref)
        vel_body = self._robot.data.root_lin_vel_b
        state_stub = torch.zeros(self.num_envs, 6, device=self.device)
        state_stub[:, 3:6] = vel_body
        self._thrust, self._moment = self._propellers.compute_force_and_torque(state_stub)
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)
        self._update_prop_visuals()

    def _get_observations(self) -> dict:
        obs_policy = torch.cat(
            [
                self._robot.data.root_pos_w,
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_link_quat_w,
                self._robot.data.root_ang_vel_b,
                self._propellers.omega / self._drone_cfg.omega_max,
            ],
            dim=-1,
        )
        return {"policy": obs_policy}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        timeout = self.episode_length_buf >= self.max_episode_length
        terminated = self._latest_collision_mask.clone()
        truncated = timeout & (~terminated)
        return terminated, truncated

    def _get_rewards(self):
        pos_error_vec = self._robot.data.root_pos_w - self._current_target_pos
        position_error = torch.norm(pos_error_vec, dim=-1)
        rewards = -self.cfg.position_error_weight * position_error

        collisions = self._detect_collisions()
        rewards[collisions] -= self.cfg.collision_penalty

        self._latest_position_error = position_error.detach()
        self._latest_collision_mask = collisions
        self._collision_counts += collisions.to(torch.int64)
        return rewards

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._propellers.reset(env_ids)
        self.trajectory_manager.reset(env_ids)
        self._refresh_reference_series()

    # -------------------------------------------------------------------------
    # Trajectory tracking logic
    # -------------------------------------------------------------------------

    def apply_controller_overrides(self, overrides: dict) -> None:
        for key, values in overrides.items():
            tensor = torch.as_tensor(values, device=self.device, dtype=torch.float32)
            if key == "k_pos":
                self.crazy_controller.pos_pid.kp = tensor
            elif key == "k_vel":
                self.crazy_controller.vel_pid.kp = tensor
            elif key == "k_att":
                self.crazy_controller.att_pid.kp = tensor
            elif key == "k_rate":
                self.crazy_controller.rate_kp = tensor
            elif hasattr(self.crazy_controller, key):
                setattr(self.crazy_controller, key, tensor)

    def _refresh_reference_series(self) -> None:
        pos, vel, acc = self.trajectory_manager.generate_series(self.max_episode_length, self._dt)
        self._reference_pos.copy_(pos)
        self._reference_vel.copy_(vel)
        self._reference_acc.copy_(acc)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        _ = actions  # actions ignored
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        step_idx = torch.remainder(self.episode_length_buf, self.max_episode_length)
        target_pos = self._reference_pos[step_idx, env_ids]
        env_origins = self._terrain.env_origins[env_ids]
        target_pos = target_pos + env_origins
        target_vel = self._reference_vel[step_idx, env_ids]
        target_acc = self._reference_acc[step_idx, env_ids]

        self._current_target_pos.copy_(target_pos)
        self._current_target_vel.copy_(target_vel)
        self._current_target_acc.copy_(target_acc)

        target_yaw = None
        target_yaw_rate = None
        if self._enable_yaw_tracking:
            planar_vel = target_vel[:, :2]
            new_yaw = torch.atan2(planar_vel[:, 1:2], planar_vel[:, 0:1])
            speed = torch.norm(planar_vel, dim=-1, keepdim=True)
            mask = speed > 1e-3
            self._prev_target_yaw.copy_(self._current_target_yaw)
            self._current_target_yaw = torch.where(mask, new_yaw, self._current_target_yaw)
            self._current_target_yaw_rate = (self._current_target_yaw - self._prev_target_yaw) / self._dt
            target_yaw = self._current_target_yaw
            target_yaw_rate = self._current_target_yaw_rate

        if self._enable_yaw_tracking:
            self._current_target_yaw.copy_(target_yaw)
            self._current_target_yaw_rate.copy_(target_yaw_rate)
        self._update_visualizers(step_idx)

    def _find_prop_joints(self, drone) -> list[int]:
        import re

        joint_ids, joint_names = drone.find_joints(["revolute_prop_.*"], preserve_order=True)
        if not joint_ids:
            joint_ids, joint_names = drone.find_joints(".*prop.*", preserve_order=True)
        if not joint_ids:
            return []
        indexed = []
        for joint_id, joint_name in zip(joint_ids, joint_names):
            match = re.search(r"(\d+)$", joint_name)
            if match:
                indexed.append((int(match.group(1)), joint_id))
        if indexed:
            indexed.sort(key=lambda item: item[0])
            joint_ids = [item[1] for item in indexed]
        return joint_ids

    def _update_prop_visuals(self) -> None:
        if not self._prop_joint_ids:
            return
        omega = self._propellers.omega
        count = min(len(self._prop_joint_ids), omega.shape[1])
        vis = omega[:, :count].clone()
        if count > 1:
            vis[:, 0::2] *= -1.0
        self._robot.write_joint_velocity_to_sim(vis, joint_ids=self._prop_joint_ids[:count])

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def _setup_visualizers(self) -> None:
        self._vel_markers = None
        self._ref_vel_markers = None
        self._target_markers = None
        if not self.cfg.debug_vis:
            return

        import isaaclab.sim as sim_utils
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

        vel_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/TrajectoryTracking/Velocity",
            markers={
                "arrow": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.8, 0.2, 0.2),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.5, 1.0)),
                )
            },
        )
        self._vel_markers = VisualizationMarkers(vel_cfg)

        ref_vel_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/TrajectoryTracking/RefVelocity",
            markers={
                "arrow": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.8, 0.2, 0.2),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 1.0, 0.2)),
                )
            },
        )
        self._ref_vel_markers = VisualizationMarkers(ref_vel_cfg)

        target_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/TrajectoryTracking/Targets",
            markers={
                "goal": sim_utils.SphereCfg(
                    radius=0.06,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.9, 0.1), emissive_color=(0.8, 0.7, 0.1)
                    ),
                )
            },
        )
        self._target_markers = VisualizationMarkers(target_cfg)

    def _update_visualizers(self, step_idx: torch.Tensor) -> None:
        if not self.cfg.debug_vis:
            return
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        positions = self._robot.data.root_pos_w

        if self._vel_markers is not None:
            orientations, scales = self._velocity_marker_data(self._robot.data.root_lin_vel_w)
            self._vel_markers.visualize(translations=positions, orientations=orientations, scales=scales)

        if self._ref_vel_markers is not None:
            target = self._reference_vel[step_idx, env_ids]
            orientations, scales = self._velocity_marker_data(target)
            self._ref_vel_markers.visualize(translations=positions, orientations=orientations, scales=scales)

        if self._target_markers is not None:
            self._target_markers.visualize(translations=self._current_target_pos)

    def _velocity_marker_data(self, vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        orientations = self._vectors_to_quat(vectors)
        magnitudes = torch.norm(vectors, dim=-1, keepdim=True)
        scales = torch.ones_like(vectors)
        scales[:, 0] = magnitudes.squeeze(-1) * self._velocity_scale + 0.05
        scales[:, 1] = self._velocity_radius
        scales[:, 2] = self._velocity_radius
        return orientations, scales

    def _vectors_to_quat(self, vectors: torch.Tensor) -> torch.Tensor:
        from isaaclab.utils import math as math_utils

        x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        base = x_axis.view(1, 3).expand_as(vectors)
        norms = torch.norm(vectors, dim=-1, keepdim=True)
        mask = norms <= 1e-6
        direction = torch.where(mask, base, vectors / norms.clamp_min(1e-6))
        dot = (base * direction).sum(dim=-1, keepdim=True)
        axis = torch.cross(base, direction, dim=-1)
        quat = torch.cat((axis, 1.0 + dot), dim=-1)
        quat = math_utils.normalize(quat, eps=1e-6)
        quat[mask.squeeze(-1)] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        return quat

    def _detect_collisions(self) -> torch.Tensor:
        pos = self._robot.data.root_pos_w - self._terrain.env_origins
        min_b = self._arena_min_tensor
        max_b = self._arena_max_tensor
        floor_contact = pos[:, 2] <= self.cfg.collision_altitude
        out_of_bounds = (
            (pos[:, 0] < min_b[0])
            | (pos[:, 0] > max_b[0])
            | (pos[:, 1] < min_b[1])
            | (pos[:, 1] > max_b[1])
            | (pos[:, 2] < min_b[2])
            | (pos[:, 2] > max_b[2])
        )

        return floor_contact | out_of_bounds

    def get_tracking_metrics(self) -> dict[str, torch.Tensor]:
        return {
            "position_error": self._latest_position_error.clone(),
            "collision_mask": self._latest_collision_mask.clone(),
            "collision_counts": self._collision_counts.clone(),
        }
