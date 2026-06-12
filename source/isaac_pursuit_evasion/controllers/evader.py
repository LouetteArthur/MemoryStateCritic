from __future__ import annotations

from typing import Optional

import torch

from isaaclab.utils import math as math_utils

from .config import load_controller_config
from .lee_controller import drone_cfg_name
from .crazy_controller import build_crazyflie_pid
from ..dynamics.propellers import Drone_cfg
from source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.trajectories.trajectory import WallConfig


class APFEvaderController:
    """Artificial potential field evader controller that outputs velocity and yaw commands."""

    def __init__(
        self,
        num_envs: int,
        drone_cfg: Drone_cfg,
        dt: float,
        device: str = "cuda",
        command_heading: bool = False,
        controller_cfg: Optional[dict] = None,
        lee_controller_cfg: Optional[dict] = None,
        arena_min: Optional[torch.Tensor] = None,
        arena_max: Optional[torch.Tensor] = None,
        wall_cfg: Optional[WallConfig] = None,
    ) -> None:
        self.device = device
        self.num_envs = num_envs
        self.dt = dt
        self.command_heading = command_heading

        if controller_cfg is None:
            controller_cfg = load_controller_config("apf_evader", drone_cfg_name(drone_cfg))

        self.k_pursuer = float(controller_cfg["k_pursuer"])
        self.k_wall = float(controller_cfg["k_wall"])
        self.order_den_pursuer = float(controller_cfg["order_den_pursuer"])
        self.order_den_wall = float(controller_cfg["order_den_wall"])
        self.min_wall_dist = float(controller_cfg["min_wall_dist"])
        self.min_pursuer_dist = float(controller_cfg["min_pursuer_dist"])
        self.max_speed = float(controller_cfg["max_speed"])

        if arena_min is None:
            arena_min = torch.as_tensor(controller_cfg.get("arena_min"), dtype=torch.float32)
        if arena_max is None:
            arena_max = torch.as_tensor(controller_cfg.get("arena_max"), dtype=torch.float32)

        self.arena_min = arena_min.to(device=device, dtype=torch.float32)
        self.arena_max = arena_max.to(device=device, dtype=torch.float32)

        self.wall_normals = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            device=device,
            dtype=torch.float32,
        )
        self.wall_offsets = torch.tensor(
            [
                self.arena_min[0].item(),
                -self.arena_max[0].item(),
                self.arena_min[1].item(),
                -self.arena_max[1].item(),
                self.arena_min[2].item(),
                -self.arena_max[2].item(),
            ],
            device=device,
            dtype=torch.float32,
        )

        # Central obstacle wall (optional)
        self.wall_cfg = wall_cfg

    def to(self, device: str) -> "APFEvaderController":
        self.device = device
        self.arena_min = self.arena_min.to(device)
        self.arena_max = self.arena_max.to(device)
        self.wall_normals = self.wall_normals.to(device)
        self.wall_offsets = self.wall_offsets.to(device)
        return self

    def reset(self, env_ids: torch.Tensor):
        # Stateless controller; nothing to reset.
        return

    def forward(
        self,
        pursuer_state: torch.Tensor,
        evader_state: torch.Tensor,
    ):
        pos_p = pursuer_state[..., :3]
        pos_e = evader_state[..., :3]

        e_p = pos_e - pos_p
        dist_pe = e_p.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        mask_near = dist_pe <= self.min_pursuer_dist
        f_pursuer = torch.zeros_like(e_p)
        repulse = self.k_pursuer * e_p / (dist_pe**self.order_den_pursuer)
        f_pursuer = torch.where(mask_near, repulse, f_pursuer)

        dist_w = (pos_e.unsqueeze(-2) * self.wall_normals).sum(dim=-1) - self.wall_offsets
        dist_w = dist_w.clamp_min(1e-6)
        mask_walls = dist_w < self.min_wall_dist

        wall_gain = self.k_wall / (dist_w**self.order_den_wall)
        wall_forces = (wall_gain.unsqueeze(-1) * self.wall_normals) * mask_walls.unsqueeze(-1)
        f_wall = wall_forces.sum(dim=-2)

        # Central obstacle wall repulsion (same APF formulation as boundary walls)
        f_obstacle = torch.zeros_like(e_p)
        if self.wall_cfg is not None:
            x = pos_e[..., 0]
            y = pos_e[..., 1]
            ht = self.wall_cfg.half_thickness + self.wall_cfg.clearance
            y_lo, y_hi = self.wall_cfg.y_range
            in_wall_y = (y >= y_lo) & (y <= y_hi)
            dist_to_wall = x.abs() - ht
            dist_to_wall = dist_to_wall.clamp_min(1e-6)
            near_wall = in_wall_y & (dist_to_wall < self.min_wall_dist)
            gain = self.k_wall / (dist_to_wall**self.order_den_wall)
            # Push in +x or -x depending on which side the drone is
            force_x = gain * x.sign()
            f_obstacle[..., 0] = torch.where(near_wall, force_x, f_obstacle[..., 0])

        total_force = f_pursuer + f_wall + f_obstacle
        direction = math_utils.normalize(total_force, eps=1e-6)
        vel_cmd = direction * self.max_speed

        if self.command_heading:
            yaw_cmd = torch.atan2(vel_cmd[..., 1], vel_cmd[..., 0]).unsqueeze(-1)
        else:
            yaw_cmd = None
        if yaw_cmd is None:
            yaw_cmd = torch.zeros((vel_cmd.shape[0], 1), device=vel_cmd.device, dtype=vel_cmd.dtype)
        return torch.cat((vel_cmd, yaw_cmd), dim=-1)


class CrazyflieAPFEvaderWrapper:
    """Crazyflie wrapper that maps APF velocity commands to thrust/moment."""

    def __init__(
        self,
        num_envs: int,
        drone_cfg: Drone_cfg,
        dt: float,
        pid_dt: float | None = None,
        device: str = "cuda",
        command_heading: bool = False,
        controller_cfg: Optional[dict] = None,
        arena_min: Optional[torch.Tensor] = None,
        arena_max: Optional[torch.Tensor] = None,
        pid_params: Optional[dict] = None,
        wall_cfg: Optional[WallConfig] = None,
    ) -> None:
        self.device = torch.device(device)
        self.controller = APFEvaderController(
            num_envs=num_envs,
            drone_cfg=drone_cfg,
            dt=dt,
            device=device,
            command_heading=command_heading,
            controller_cfg=controller_cfg,
            arena_min=arena_min,
            arena_max=arena_max,
            wall_cfg=wall_cfg,
        )
        pid_dt = dt if pid_dt is None else pid_dt
        self.pid = build_crazyflie_pid(num_envs, drone_cfg, pid_dt, self.device, pid_params)

    def __call__(self, pursuer_state: torch.Tensor, evader_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cmd = self.command(pursuer_state, evader_state)
        wrench = self.wrench_from_command(evader_state, cmd)
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
