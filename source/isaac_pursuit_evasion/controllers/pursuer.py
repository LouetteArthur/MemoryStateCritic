# Copyright (c) 2026, the MemoryStateCritic authors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from isaaclab.assets import ArticulationData

from source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.trajectories.trajectory import (
    WallConfig,
)

from ..dynamics.propellers import Drone_cfg
from .config import load_controller_config
from .crazy_controller import build_crazyflie_pid
from .lee_controller import drone_cfg_name


class PDPursuerController:
    """Velocity-based PID pursuit controller that outputs velocity and yaw commands."""

    def __init__(
        self,
        num_envs: int,
        drone_cfg: Drone_cfg,
        dt: float,
        total_frames: int = 1,
        command_heading: bool = False,
        device: str = "cuda",
        controller_cfg: dict | None = None,
        lee_controller_cfg: dict | None = None,
    ) -> None:
        self.device = device
        self.num_envs = num_envs
        self.dt = dt
        self.total_frames = max(1, total_frames)

        self.command_heading = command_heading

        if controller_cfg is None:
            controller_cfg = load_controller_config("pd_pursuer", drone_cfg_name(drone_cfg))

        def to_tensor(values):
            return torch.as_tensor(values, device=device, dtype=torch.float32).flatten()

        self.kp = to_tensor(controller_cfg["kp"])
        self.kd = to_tensor(controller_cfg["kd"])
        self.derivative_limit = to_tensor(controller_cfg["derivative_limit"])
        self.filter_alpha = float(controller_cfg.get("filter_alpha", 0.0))
        self.max_speed = to_tensor(controller_cfg["max_speed"])

        self.curriculum_enabled = False
        self.start_speed = self.max_speed.clone()

        self.e_p = torch.zeros(num_envs, 3, device=device)
        self.e_d = torch.zeros_like(self.e_p)
        self.speed_limit = self.max_speed.unsqueeze(0).repeat(num_envs, 1)

    def to(self, device: str) -> PDPursuerController:
        attrs = ("kp", "kd", "derivative_limit", "max_speed", "start_speed")
        for attr in attrs:
            setattr(self, attr, getattr(self, attr).to(device))
        self.e_p = self.e_p.to(device)
        self.e_d = self.e_d.to(device)
        self.speed_limit = self.speed_limit.to(device)
        self.device = device
        return self

    def set_curriculum(self, enabled: bool, start_fraction: float = 0.1):
        self.curriculum_enabled = enabled
        if enabled:
            self.start_speed = self.max_speed * start_fraction
        else:
            self.start_speed = self.max_speed.clone()

    def reset(self, env_ids: torch.Tensor, frame: int = 0):
        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        self.e_p[env_ids] = 0.0
        self.e_d[env_ids] = 0.0
        current_speed = self._compute_speed_limit(frame)
        self.speed_limit[env_ids] = current_speed[env_ids]

    def _compute_speed_limit(self, frame: int) -> torch.Tensor:
        if not self.curriculum_enabled:
            return self.max_speed.unsqueeze(0).expand(self.num_envs, -1)
        progress = min(max(frame, 0), self.total_frames) / self.total_frames
        speed = self.start_speed + (self.max_speed - self.start_speed) * progress
        return speed.unsqueeze(0).expand(self.num_envs, -1)

    def _pd(self, state_pursuer: torch.Tensor, state_evader: torch.Tensor) -> torch.Tensor:
        pos_p_w = state_pursuer[..., :3]
        pos_e_w = state_evader[..., :3]

        prev_error = self.e_p.clone()
        self.e_p = pos_e_w - pos_p_w

        derivative = (self.e_p - prev_error) / self.dt
        derivative = torch.clamp(derivative, -self.derivative_limit, self.derivative_limit)
        self.e_d = torch.lerp(self.e_d, derivative, self.filter_alpha)

        u_cmd = self.kp * self.e_p + self.kd * self.e_d
        u_cmd = torch.clamp(u_cmd, -self.speed_limit, self.speed_limit)
        return u_cmd

    def forward(
        self,
        state_pursuer: torch.Tensor,
        state_evader: torch.Tensor,
    ):
        vel_cmd = self._pd(state_pursuer, state_evader)
        if self.command_heading:
            yaw_cmd = torch.atan2(vel_cmd[..., 1], vel_cmd[..., 0]).unsqueeze(-1)
        else:
            yaw_cmd = None
        if yaw_cmd is None:
            yaw_cmd = torch.zeros((vel_cmd.shape[0], 1), device=vel_cmd.device, dtype=vel_cmd.dtype)
        return torch.cat((vel_cmd, yaw_cmd), dim=-1)


class PDPursuerWrapper:
    def __init__(
        self,
        pd_controller: PDPursuerController,
    ):
        self.pd_controller = pd_controller

    def forward(
        self,
        data_pursuer: ArticulationData,
        data_evader: ArticulationData,
    ):
        return self.pd_controller(data_pursuer.root_state_w, data_evader.root_state_w)


class FRPNPursuerController:
    """Fast-response proportional navigation controller that outputs velocity and yaw commands."""

    def __init__(
        self,
        num_envs: int,
        drone_cfg: Drone_cfg,
        dt: float,
        total_frames: int = 1,
        device: str = "cuda",
        command_heading: bool = False,
        controller_cfg: dict | None = None,
        curriculum_cfg: dict | None = None,
        wall_cfg: WallConfig | None = None,
    ) -> None:
        self.device = device
        self.num_envs = num_envs
        self.dt = dt
        self.total_frames = max(1, total_frames)
        self.command_heading = command_heading

        if controller_cfg is None:
            controller_cfg = load_controller_config("frpn_pursuer", drone_cfg_name(drone_cfg))

        self.G = float(controller_cfg["G"])
        self.W = float(controller_cfg["W"])
        self.max_speed = torch.as_tensor(controller_cfg["max_speed"], device=device, dtype=torch.float32).flatten()

        # Wall avoidance parameters (reactive potential field overlay)
        self.wall_cfg = wall_cfg
        self.k_wall = float(controller_cfg.get("k_wall", 2.0))
        self.wall_order = float(controller_cfg.get("wall_order", 3.0))
        self.min_wall_dist = float(controller_cfg.get("min_wall_dist", 0.8))

        self.curriculum_enabled = False
        self.curriculum_start_fraction = 0.1
        self.curriculum_ramp_fraction = 0.75
        self.curriculum_end_frame = self.total_frames
        self.start_speed = self.max_speed.clone()
        if curriculum_cfg:
            self.curriculum_enabled = bool(curriculum_cfg.get("enabled", False))
            self.curriculum_start_fraction = float(curriculum_cfg.get("start_fraction", 0.1))
            self.curriculum_ramp_fraction = float(curriculum_cfg.get("ramp_fraction", 0.75))
            total_frames_cfg = int(curriculum_cfg.get("total_frames", self.total_frames))
            self.curriculum_end_frame = max(1, int(total_frames_cfg * self.curriculum_ramp_fraction))
            if self.curriculum_enabled:
                self.start_speed = self.max_speed * self.curriculum_start_fraction
        self.speed_limit = self.max_speed.unsqueeze(0).repeat(num_envs, 1)
        self.update_curriculum(0)

    def to(self, device: str) -> FRPNPursuerController:
        self.max_speed = self.max_speed.to(device)
        self.start_speed = self.start_speed.to(device)
        self.speed_limit = self.speed_limit.to(device)
        self.device = device
        return self

    def set_curriculum(self, enabled: bool, start_fraction: float = 0.1):
        self.curriculum_enabled = enabled
        if enabled:
            self.start_speed = self.max_speed * start_fraction
        else:
            self.start_speed = self.max_speed.clone()

    def reset(self, env_ids: torch.Tensor, frame: int = 0):
        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        self.update_curriculum(frame)

    def update_curriculum(self, frame: int) -> None:
        if not self.curriculum_enabled:
            self.speed_limit = self.max_speed.unsqueeze(0).repeat(self.num_envs, 1)
            return
        end_frame = max(1, self.curriculum_end_frame)
        clamped = max(0, min(int(frame), end_frame))
        progress = clamped / end_frame
        speed = self.start_speed + (self.max_speed - self.start_speed) * progress
        self.speed_limit = speed.unsqueeze(0).repeat(self.num_envs, 1)

    def forward(
        self,
        pursuer_state: torch.Tensor,
        evader_state: torch.Tensor,
    ):
        pos_p = pursuer_state[..., :3]
        vel_p = pursuer_state[..., 7:10]
        pos_e = evader_state[..., :3]
        vel_e = evader_state[..., 7:10]

        dp = pos_e - pos_p
        dv = vel_e - vel_p

        dp_norm = dp.norm(dim=-1, keepdim=True)
        dv_norm = dv.norm(dim=-1, keepdim=True)

        eps = 1e-6
        t_go = dp_norm / (dv_norm + eps)
        inv_t2 = 1.0 / (t_go.square() + eps)

        term_pn = (dp + dv * t_go) * inv_t2
        term_p = dp

        vel_cmd = self.G * (self.W * term_p + (1.0 - self.W) * term_pn)

        # Reactive wall avoidance overlay
        if self.wall_cfg is not None:
            x = pos_p[..., 0]
            y = pos_p[..., 1]
            ht = self.wall_cfg.half_thickness + self.wall_cfg.clearance
            y_lo, y_hi = self.wall_cfg.y_range
            in_wall_y = (y >= y_lo) & (y <= y_hi)
            dist_to_wall = x.abs() - ht
            dist_to_wall = dist_to_wall.clamp_min(1e-6)
            near_wall = in_wall_y & (dist_to_wall < self.min_wall_dist)
            gain = self.k_wall / (dist_to_wall**self.wall_order)
            vel_cmd[..., 0] = vel_cmd[..., 0] + torch.where(near_wall, gain * x.sign(), torch.zeros_like(x))

        vel_cmd = torch.clamp(vel_cmd, -self.speed_limit, self.speed_limit)

        if self.command_heading:
            yaw_cmd = torch.atan2(vel_cmd[..., 1], vel_cmd[..., 0]).unsqueeze(-1)
        else:
            yaw_cmd = None
        if yaw_cmd is None:
            yaw_cmd = torch.zeros((vel_cmd.shape[0], 1), device=vel_cmd.device, dtype=vel_cmd.dtype)
        return torch.cat((vel_cmd, yaw_cmd), dim=-1)


class CrazyflieFRPNPursuerWrapper:
    """Crazyflie wrapper that maps FRPN velocity commands to thrust/moment."""

    def __init__(
        self,
        num_envs: int,
        drone_cfg: Drone_cfg,
        dt: float,
        pid_dt: float | None = None,
        total_frames: int = 1,
        device: str = "cuda",
        command_heading: bool = False,
        controller_cfg: dict | None = None,
        curriculum_cfg: dict | None = None,
        pid_params: dict | None = None,
        wall_cfg: WallConfig | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.controller = FRPNPursuerController(
            num_envs=num_envs,
            drone_cfg=drone_cfg,
            dt=dt,
            total_frames=total_frames,
            device=device,
            command_heading=command_heading,
            controller_cfg=controller_cfg,
            curriculum_cfg=curriculum_cfg,
            wall_cfg=wall_cfg,
        )
        pid_dt = dt if pid_dt is None else pid_dt
        self.pid = build_crazyflie_pid(num_envs, drone_cfg, pid_dt, self.device, pid_params)

    def __call__(self, pursuer_state: torch.Tensor, evader_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cmd = self.command(pursuer_state, evader_state)
        wrench = self.wrench_from_command(pursuer_state, cmd)
        return cmd, wrench

    def command(self, pursuer_state: torch.Tensor, evader_state: torch.Tensor) -> torch.Tensor:
        return self.controller.forward(pursuer_state, evader_state)

    def wrench_from_command(self, root_state: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
        thrust, moment = self.pid(
            root_state=root_state,
            target_vel=cmd[:, :3],
            target_yaw=cmd[:, 3:4],
            command_level="velocity",
        )
        return torch.cat((thrust, moment), dim=-1)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.controller.reset(env_ids)
        self.pid.reset(env_ids)

    def update_curriculum(self, frame: int) -> None:
        self.controller.update_curriculum(frame)
