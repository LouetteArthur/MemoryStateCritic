from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Type

import torch
import torch.distributions as D

from isaaclab.utils import math as math_utils

from source.isaac_pursuit_evasion.controllers.crazy_controller import build_crazyflie_pid
from source.isaac_pursuit_evasion.dynamics.propellers import Drone_cfg


@dataclass
class WallConfig:
    """Central-wall geometry for trajectory generators.

    The wall runs along the Y axis at x=0 with gaps at |y| > y_range bounds.
    Coordinates are arena-local (matching the per-env frame used by trajectories).
    """

    half_thickness: float
    y_range: Tuple[float, float]
    clearance: float
    cross_prob: float = 0.4
    max_resample_iters: int = 10
    check_samples: int = 128
    check_duration: float = 10.0


def scale_time(t: torch.Tensor, a: float = 1.0) -> torch.Tensor:
    return t / (1 + 1 / (a * torch.abs(t) + 1e-6))


def compute_derivatives(positions: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Central finite differences for velocity and acceleration."""
    vel = torch.gradient(positions, spacing=(dt,), dim=0)[0]
    acc = torch.gradient(vel, spacing=(dt,), dim=0)[0]
    return vel, acc


class TrajectoryRegistry:
    _registry: Dict[str, Type["Trajectory"]] = {}

    @classmethod
    def register(cls, traj_cls: Type["Trajectory"]) -> None:
        cls._registry[traj_cls.__name__.lower()] = traj_cls
        cls._registry[traj_cls.__name__] = traj_cls

    @classmethod
    def get(cls, name: str) -> Type["Trajectory"]:
        return cls._registry[name]

    @classmethod
    def names(cls) -> Iterable[str]:
        return cls._registry.keys()


class Trajectory:
    """Base trajectory with registration support."""

    # Whether this trajectory can meaningfully "cross" the wall through a gap.
    # Hover cannot (stationary); circular and lemniscate can.
    supports_cross_mode: bool = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        TrajectoryRegistry.register(cls)

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        arena_min: torch.Tensor,
        arena_max: torch.Tensor,
        wall_cfg: Optional[WallConfig] = None,
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self.arena_min = arena_min.to(device=device, dtype=torch.float32)
        self.arena_max = arena_max.to(device=device, dtype=torch.float32)
        self.wall_cfg = wall_cfg
        self.cross_mode = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def sample(self, time: torch.Tensor, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        time = time.unsqueeze(0).expand(env_ids.shape[0], -1)
        return self._positions(time, env_ids)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        raise NotImplementedError

    def _positions(self, time: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Wall-aware sampling helpers (used by circular/lemniscate subclasses)
    # ------------------------------------------------------------------ #
    def _sample_cross_mode(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Sample per-env cross/one-sided mode. Returns bool tensor [len(env_ids)]."""
        if self.wall_cfg is None or not self.supports_cross_mode:
            return torch.zeros(env_ids.shape[0], dtype=torch.bool, device=self.device)
        return torch.rand(env_ids.shape[0], device=self.device) < self.wall_cfg.cross_prob

    def _wall_constraint_mask(
        self, positions: torch.Tensor, cross_mode: torch.Tensor
    ) -> torch.Tensor:
        """Check if each env's curve satisfies its mode constraint AND stays within arena bounds.

        Args:
            positions: [N, K, 3] dense samples of the trajectory per env.
            cross_mode: [N] bool, True = mode B (cross through gap).

        Returns:
            [N] bool mask, True where the curve is acceptable.
        """
        wall = self.wall_cfg
        if wall is None:
            return torch.ones(positions.shape[0], dtype=torch.bool, device=self.device)

        margin = wall.half_thickness + wall.clearance
        x = positions[..., 0]
        y = positions[..., 1]
        z = positions[..., 2]

        # Arena bounds check (with small buffer for PID tracking error)
        pid_margin = 0.15
        in_bounds = (
            (x >= self.arena_min[0] + pid_margin)
            & (x <= self.arena_max[0] - pid_margin)
            & (y >= self.arena_min[1] + pid_margin)
            & (y <= self.arena_max[1] - pid_margin)
            & (z >= self.arena_min[2] + 0.3)
            & (z <= self.arena_max[2] - 0.1)
        )
        all_in_bounds = in_bounds.all(dim=-1)

        in_wall = (x.abs() < margin) & (y >= wall.y_range[0]) & (y <= wall.y_range[1])
        hits_wall = in_wall.any(dim=-1)

        # Mode A: entirely on one side (or entirely outside wall y-range).
        all_positive_x = (x > margin).all(dim=-1)
        all_negative_x = (x < -margin).all(dim=-1)
        all_outside_y = ((y < wall.y_range[0]) | (y > wall.y_range[1])).all(dim=-1)
        one_sided = all_positive_x | all_negative_x | all_outside_y

        # Mode B: spans both sides of x=0 with at least `margin` extent on each side.
        crosses = (x.max(dim=-1).values > margin) & (x.min(dim=-1).values < -margin)

        mode_a_ok = (~hits_wall) & one_sided & all_in_bounds
        mode_b_ok = (~hits_wall) & crosses & all_in_bounds
        return torch.where(cross_mode, mode_b_ok, mode_a_ok)

    def _resample_with_wall(
        self,
        env_ids: torch.Tensor,
        sample_fn,
    ) -> None:
        """Rejection loop: call sample_fn(ids, cross_mode) until all curves pass envelope check.

        Args:
            env_ids: Local env ids to (re)sample.
            sample_fn: Callable(env_ids_subset, cross_mode_subset) that writes params in place.
        """
        wall = self.wall_cfg
        cross_mode = self._sample_cross_mode(env_ids)
        self.cross_mode[env_ids] = cross_mode

        if wall is None:
            sample_fn(env_ids, cross_mode)
            return

        time_grid = torch.linspace(
            0.0, wall.check_duration, wall.check_samples, device=self.device
        )
        pending = env_ids.clone()
        pending_cross = cross_mode.clone()

        for _ in range(wall.max_resample_iters):
            if pending.numel() == 0:
                break
            sample_fn(pending, pending_cross)
            time_matrix = time_grid.unsqueeze(0).expand(pending.shape[0], -1)
            positions = self._positions(time_matrix, pending)
            ok = self._wall_constraint_mask(positions, pending_cross)
            rejected = ~ok
            pending = pending[rejected]
            pending_cross = pending_cross[rejected]

        # Fallback: any still-rejected envs get forced into Mode A (tight one-sided).
        if pending.numel() > 0:
            forced = torch.zeros_like(pending, dtype=torch.bool)
            self.cross_mode[pending] = False
            sample_fn(pending, forced)


class CrazyflieTrajectoryWrapper:
    """Crazyflie wrapper that maps trajectory velocity commands to thrust/moment."""

    def __init__(
        self,
        num_envs: int,
        drone_cfg: Drone_cfg,
        dt: float,
        device: torch.device,
        pid_dt: float | None = None,
        pid_params: dict | None = None,
    ) -> None:
        self.device = device
        pid_dt = dt if pid_dt is None else pid_dt
        self.pid = build_crazyflie_pid(num_envs, drone_cfg, pid_dt, device, pid_params)

    def __call__(self, root_state: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
        thrust, moment = self.pid(
            root_state=root_state,
            target_vel=cmd[:, :3],
            target_yaw=cmd[:, 3:4],
            command_level="velocity",
        )
        return torch.cat((thrust, moment), dim=-1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.pid.reset(env_ids)

    def wrench_from_command(self, root_state: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
        return self(root_state, cmd)


class HoverTrajectory(Trajectory):
    # Stationary — cannot cross the wall.
    supports_cross_mode: bool = False

    def __init__(self, num_envs, device, arena_min, arena_max, wall_cfg=None):
        super().__init__(num_envs, device, arena_min, arena_max, wall_cfg)
        self._dist = D.Uniform(self.arena_min + 1.0, self.arena_max - 0.5)
        self.positions = self._dist.sample((num_envs,))
        if self.wall_cfg is not None:
            self._reposition_for_wall(torch.arange(num_envs, device=device, dtype=torch.long))

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self.positions[env_ids] = self._dist.sample((env_ids.shape[0],))
        if self.wall_cfg is not None:
            self._reposition_for_wall(env_ids)

    def _reposition_for_wall(self, env_ids: torch.Tensor) -> None:
        """Move any hover position out of the wall region onto the nearest safe side."""
        wall = self.wall_cfg
        margin = wall.half_thickness + wall.clearance
        pos = self.positions[env_ids]
        x = pos[:, 0]
        y = pos[:, 1]
        in_wall = (x.abs() < margin) & (y >= wall.y_range[0]) & (y <= wall.y_range[1])
        if in_wall.any():
            # Push to whichever side is closest, preserving sign; default to +x if exactly 0.
            sign = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
            target_x = sign * (margin + 0.1)
            pos[in_wall, 0] = target_x[in_wall]
            self.positions[env_ids] = pos

    def _positions(self, time: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        return self.positions[env_ids].unsqueeze(1).expand(-1, time.shape[1], -1)


class CircularTrajectory(Trajectory):
    supports_cross_mode: bool = True

    def __init__(self, num_envs, device, arena_min, arena_max, wall_cfg=None):
        super().__init__(num_envs, device, arena_min, arena_max, wall_cfg)

        xy_extent = arena_max[:2] - arena_min[:2]
        xy_half = 0.5 * xy_extent
        # 0.3 m buffer keeps the commanded circle well inside the OOB boundary
        # to absorb PID tracking error.  Without this, r_max == y half-extent
        # and any overshoot triggers evader_out_of_bounds.
        xy_margin = 0.3
        r_max = torch.min(xy_half) - xy_margin
        r_max = torch.clamp(r_max, min=0.2)
        # Cap r_max so one-sided (Mode A) trajectories fit between wall and arena edge.
        # Available one-side width = arena_max_x - wall_margin; need 2*r + margins for center.
        if wall_cfg is not None:
            wall_margin = wall_cfg.half_thickness + wall_cfg.clearance
            one_side_width = float(arena_max[0]) - wall_margin
            r_max_wall = (one_side_width - 0.3) / 2.0
            r_max = torch.clamp(r_max, max=r_max_wall, min=0.2)
        r_min = r_max * 0.6

        speed_min = 1.0
        speed_max = 2.0
        omega_min = speed_min / r_max
        omega_max = speed_max / r_max

        z_offset_min = arena_min[2] + 0.5
        z_offset_max = arena_max[2] - 0.3

        self.radius_dist = D.Uniform(
            torch.as_tensor(r_min, device=device, dtype=torch.float32),
            torch.as_tensor(r_max, device=device, dtype=torch.float32),
        )
        self.omega_dist = D.Uniform(
            torch.as_tensor(omega_min, device=device, dtype=torch.float32),
            torch.as_tensor(omega_max, device=device, dtype=torch.float32),
        )
        self.phase_dist = D.Uniform(
            torch.as_tensor(0.0, device=device, dtype=torch.float32),
            torch.as_tensor(2 * torch.pi, device=device, dtype=torch.float32),
        )
        self.scale_dist = D.Uniform(
            torch.tensor([0.85, 0.85, 0.7], device=device),
            torch.tensor([1.0, 1.0, 1.0], device=device),
        )
        self.rpy_dist = D.Uniform(
            torch.tensor([0.0, 0.0, 0.0], device=device),
            torch.tensor([0.05, 0.05, 2.0], device=device),
        )
        self.z_offset_dist = D.Uniform(
            torch.as_tensor(z_offset_min, device=device, dtype=torch.float32),
            torch.as_tensor(z_offset_max, device=device, dtype=torch.float32),
        )

        self.radius = self.radius_dist.sample((num_envs, 1))
        self.omega = self.omega_dist.sample((num_envs, 1))
        self.phase = self.phase_dist.sample((num_envs, 1))
        self.scale = self.scale_dist.sample((num_envs, 1))
        rpy = self.rpy_dist.sample((num_envs, 1)) * torch.pi
        self.rot = math_utils.quat_from_euler_xyz(rpy[...,0], rpy[...,1], rpy[...,2])
        self.z_offset = self.z_offset_dist.sample((num_envs, 1))
        # Per-env center offset in the x-y plane (arena-local).
        self.cx = torch.zeros((num_envs, 1), device=device)
        self.cy = torch.zeros((num_envs, 1), device=device)

        if self.wall_cfg is not None:
            all_ids = torch.arange(num_envs, device=device, dtype=torch.long)
            self._resample_with_wall(all_ids, self._sample_params)

    def _sample_params(self, env_ids: torch.Tensor, cross_mode: torch.Tensor) -> None:
        """Sample trajectory params for env_ids with per-env mode (cross vs one-sided)."""
        count = env_ids.shape[0]
        self.radius[env_ids] = self.radius_dist.sample((count, 1))
        self.omega[env_ids] = self.omega_dist.sample((count, 1))
        self.phase[env_ids] = self.phase_dist.sample((count, 1))
        self.scale[env_ids] = self.scale_dist.sample((count, 1))
        rpy = self.rpy_dist.sample((count, 1)) * torch.pi
        self.rot[env_ids] = math_utils.quat_from_euler_xyz(rpy[..., 0], rpy[..., 1], rpy[..., 2])
        self.z_offset[env_ids] = self.z_offset_dist.sample((count, 1))
        self._sample_centers(env_ids, cross_mode)

    def _sample_centers(self, env_ids: torch.Tensor, cross_mode: torch.Tensor) -> None:
        """Sample (cx, cy) offsets. Mode A pushes to one side; Mode B places near the gap."""
        count = env_ids.shape[0]
        if self.wall_cfg is None or count == 0:
            self.cx[env_ids] = 0.0
            self.cy[env_ids] = 0.0
            return

        wall = self.wall_cfg
        margin = wall.half_thickness + wall.clearance
        r = self.radius[env_ids].squeeze(-1)
        scale_xy = self.scale[env_ids].squeeze(-2)[:, :2].max(dim=-1).values
        envelope = r * scale_xy

        # Mode A: one-sided. Shift cx so the circle stays on side ±1.
        sign = torch.where(
            torch.rand(count, device=self.device) < 0.5,
            torch.ones(count, device=self.device),
            -torch.ones(count, device=self.device),
        )
        a_lo = envelope + margin + 0.05
        a_hi = (self.arena_max[0] - 0.2 - envelope).clamp_min(a_lo + 0.1)
        cx_a = sign * (a_lo + torch.rand(count, device=self.device) * (a_hi - a_lo))
        cy_a = self.arena_min[1] + 0.3 + torch.rand(count, device=self.device) * (
            (self.arena_max[1] - self.arena_min[1] - 0.6).clamp_min(0.0)
        )

        # Mode B: crossing through a gap. cx near 0; cy placed in top or bottom gap.
        gap_sign = torch.where(
            torch.rand(count, device=self.device) < 0.5,
            torch.ones(count, device=self.device),
            -torch.ones(count, device=self.device),
        )
        cx_b = (torch.rand(count, device=self.device) - 0.5) * 0.4  # [-0.2, 0.2]
        gap_lo = torch.where(gap_sign > 0, wall.y_range[1] + 0.15, self.arena_min[1] + 0.2)
        gap_hi = torch.where(gap_sign > 0, self.arena_max[1] - 0.2, wall.y_range[0] - 0.15)
        gap_hi = torch.maximum(gap_hi, gap_lo + 0.05)
        cy_b = gap_lo + torch.rand(count, device=self.device) * (gap_hi - gap_lo)

        cx = torch.where(cross_mode, cx_b, cx_a).unsqueeze(-1)
        cy = torch.where(cross_mode, cy_b, cy_a).unsqueeze(-1)
        self.cx[env_ids] = cx
        self.cy[env_ids] = cy

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if self.wall_cfg is None:
            # Preserve legacy no-wall behavior (no rejection loop, cx=cy=0).
            self._sample_params(
                env_ids, torch.zeros(env_ids.shape[0], dtype=torch.bool, device=self.device)
            )
            return
        self._resample_with_wall(env_ids, self._sample_params)

    def _positions(self, time: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        phase = self.phase[env_ids]
        omega = self.omega[env_ids]
        radius = self.radius[env_ids]
        z_offset = self.z_offset[env_ids]

        t_scaled = phase + scale_time(time * omega)
        pos = torch.stack(
            (
                torch.cos(t_scaled) * radius,
                torch.sin(t_scaled) * radius,
                torch.ones_like(t_scaled, device=self.device) * z_offset,
            ),
            dim=-1,
        )
        pos = math_utils.quat_apply(self.rot[env_ids].expand(-1, time.shape[1], -1), pos)
        pos = pos * self.scale[env_ids]
        # Apply per-env center offset in the x-y plane (z already encoded in z_offset).
        pos[..., 0] = pos[..., 0] + self.cx[env_ids]
        pos[..., 1] = pos[..., 1] + self.cy[env_ids]
        return pos


class LemniscateTrajectory(Trajectory):
    supports_cross_mode: bool = True

    def __init__(self, num_envs, device, arena_min, arena_max, wall_cfg=None):
        super().__init__(num_envs, device, arena_min, arena_max, wall_cfg)

        xy_extent = arena_max[:2] - arena_min[:2]
        xy_half = 0.5 * xy_extent
        # See CircularTrajectory: 0.3 m margin absorbs PID tracking error so the
        # lobe tips don't graze the OOB boundary.
        xy_margin = 0.3
        r_max = torch.min(xy_half) - xy_margin
        r_max = torch.clamp(r_max, min=0.2)
        if wall_cfg is not None:
            wall_margin = wall_cfg.half_thickness + wall_cfg.clearance
            one_side_width = float(arena_max[0]) - wall_margin
            r_max_wall = (one_side_width - 0.3) / 2.0
            r_max = torch.clamp(r_max, max=r_max_wall, min=0.2)
        r_min = r_max * 0.5

        speed_min = 1.0
        speed_max = 2.0
        omega_min = speed_min / r_max
        omega_max = speed_max / r_max

        z_offset_min = arena_min[2] + 0.5
        z_offset_max = arena_max[2] - 0.3

        self.radius_dist = D.Uniform(
            torch.as_tensor(r_min, device=device, dtype=torch.float32),
            torch.as_tensor(r_max, device=device, dtype=torch.float32),
        )
        self.omega_dist = D.Uniform(
            torch.as_tensor(omega_min, device=device, dtype=torch.float32),
            torch.as_tensor(omega_max, device=device, dtype=torch.float32),
        )
        self.phase_dist = D.Uniform(
            torch.as_tensor(0.0, device=device, dtype=torch.float32),
            torch.as_tensor(2 * torch.pi, device=device, dtype=torch.float32),
        )
        self.scale_dist = D.Uniform(
            torch.tensor([0.5, 0.5, 0.7], device=device),
            torch.tensor([1.0, 1.0, 1.0], device=device),
        )
        self.rpy_dist = D.Uniform(
            torch.tensor([0.0, 0.0, 0.0], device=device),
            torch.tensor([0.05, 0.05, 2.0], device=device),
        )
        self.z_offset_dist = D.Uniform(
            torch.as_tensor(z_offset_min, device=device, dtype=torch.float32),
            torch.as_tensor(z_offset_max, device=device, dtype=torch.float32),
        )

        self.radius = self.radius_dist.sample((num_envs, 1))
        self.omega = self.omega_dist.sample((num_envs, 1))
        self.phase = self.phase_dist.sample((num_envs, 1))
        self.scale = self.scale_dist.sample((num_envs, 1))

        rpy = self.rpy_dist.sample((num_envs, 1)) * torch.pi
        self.rot = math_utils.quat_from_euler_xyz(rpy[...,0], rpy[...,1], rpy[...,2])
        self.z_offset = self.z_offset_dist.sample((num_envs, 1))
        self.cx = torch.zeros((num_envs, 1), device=device)
        self.cy = torch.zeros((num_envs, 1), device=device)

        if self.wall_cfg is not None:
            all_ids = torch.arange(num_envs, device=device, dtype=torch.long)
            self._resample_with_wall(all_ids, self._sample_params)

    def _sample_params(self, env_ids: torch.Tensor, cross_mode: torch.Tensor) -> None:
        count = env_ids.shape[0]
        self.radius[env_ids] = self.radius_dist.sample((count, 1))
        self.omega[env_ids] = self.omega_dist.sample((count, 1))
        self.phase[env_ids] = self.phase_dist.sample((count, 1))
        self.scale[env_ids] = self.scale_dist.sample((count, 1))
        rpy = self.rpy_dist.sample((count, 1)) * torch.pi
        self.rot[env_ids] = math_utils.quat_from_euler_xyz(rpy[..., 0], rpy[..., 1], rpy[..., 2])
        self.z_offset[env_ids] = self.z_offset_dist.sample((count, 1))
        self._sample_centers(env_ids, cross_mode)

    def _sample_centers(self, env_ids: torch.Tensor, cross_mode: torch.Tensor) -> None:
        count = env_ids.shape[0]
        if self.wall_cfg is None or count == 0:
            self.cx[env_ids] = 0.0
            self.cy[env_ids] = 0.0
            return

        wall = self.wall_cfg
        margin = wall.half_thickness + wall.clearance
        r = self.radius[env_ids].squeeze(-1)
        scale_xy = self.scale[env_ids].squeeze(-2)[:, :2].max(dim=-1).values
        envelope = r * scale_xy

        # Mode A: entire figure-8 on one side. Center offset ≥ envelope + margin.
        sign = torch.where(
            torch.rand(count, device=self.device) < 0.5,
            torch.ones(count, device=self.device),
            -torch.ones(count, device=self.device),
        )
        a_lo = envelope + margin + 0.05
        a_hi = (self.arena_max[0] - 0.2 - envelope).clamp_min(a_lo + 0.1)
        cx_a = sign * (a_lo + torch.rand(count, device=self.device) * (a_hi - a_lo))
        cy_a = self.arena_min[1] + 0.3 + torch.rand(count, device=self.device) * (
            (self.arena_max[1] - self.arena_min[1] - 0.6).clamp_min(0.0)
        )

        # Mode B: figure-8 centered near x=0 in the gap region — crossings happen through the gap.
        gap_sign = torch.where(
            torch.rand(count, device=self.device) < 0.5,
            torch.ones(count, device=self.device),
            -torch.ones(count, device=self.device),
        )
        cx_b = (torch.rand(count, device=self.device) - 0.5) * 0.4
        gap_lo = torch.where(gap_sign > 0, wall.y_range[1] + 0.15, self.arena_min[1] + 0.2)
        gap_hi = torch.where(gap_sign > 0, self.arena_max[1] - 0.2, wall.y_range[0] - 0.15)
        gap_hi = torch.maximum(gap_hi, gap_lo + 0.05)
        cy_b = gap_lo + torch.rand(count, device=self.device) * (gap_hi - gap_lo)

        cx = torch.where(cross_mode, cx_b, cx_a).unsqueeze(-1)
        cy = torch.where(cross_mode, cy_b, cy_a).unsqueeze(-1)
        self.cx[env_ids] = cx
        self.cy[env_ids] = cy

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if self.wall_cfg is None:
            self._sample_params(
                env_ids, torch.zeros(env_ids.shape[0], dtype=torch.bool, device=self.device)
            )
            return
        self._resample_with_wall(env_ids, self._sample_params)

    def _positions(self, time: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        phase = self.phase[env_ids]
        omega = self.omega[env_ids]
        radius = self.radius[env_ids]
        z_offset = self.z_offset[env_ids]

        t_scaled = phase + scale_time(time * omega)
        sin_t = torch.sin(t_scaled)
        cos_t = torch.cos(t_scaled)
        denom = sin_t.square() + 1.0

        pos = torch.stack(
            (
                radius * cos_t / denom,
                radius * sin_t * cos_t / denom,
                torch.ones_like(t_scaled, device=self.device) * z_offset,
            ),
            dim=-1,
        )
        pos = math_utils.quat_apply(self.rot[env_ids].expand(-1, time.shape[1], -1), pos)
        pos = pos * self.scale[env_ids]
        pos[..., 0] = pos[..., 0] + self.cx[env_ids]
        pos[..., 1] = pos[..., 1] + self.cy[env_ids]
        return pos


@dataclass
class TrajectorySpec:
    name: str
    count: int


def build_trajectories(
    specs: Sequence[TrajectorySpec],
    device: torch.device,
    arena_min: torch.Tensor,
    arena_max: torch.Tensor,
    wall_cfg: Optional[WallConfig] = None,
) -> Tuple[List[Trajectory], List[torch.Tensor]]:
    trajectories: List[Trajectory] = []
    env_groups: List[torch.Tensor] = []
    offset = 0
    for spec in specs:
        if spec.count <= 0:
            continue
        cls = TrajectoryRegistry.get(spec.name.lower())
        traj = cls(spec.count, device, arena_min, arena_max, wall_cfg=wall_cfg)
        trajectories.append(traj)
        env_ids = torch.arange(offset, offset + spec.count, device=device, dtype=torch.long)
        env_groups.append(env_ids)
        offset += spec.count
    return trajectories, env_groups


class TrajectoryBatchManager:
    """Utility that manages a set of trajectories for a vectorized environment."""

    def __init__(
        self,
        specs: Sequence[TrajectorySpec],
        device: torch.device,
        arena_min: torch.Tensor,
        arena_max: torch.Tensor,
        wall_cfg: Optional[WallConfig] = None,
    ) -> None:
        self.device = device
        self.wall_cfg = wall_cfg
        self.trajectories, self.env_groups = build_trajectories(
            specs, device, arena_min, arena_max, wall_cfg=wall_cfg
        )
        self.total_envs = sum(group.numel() for group in self.env_groups)

        self.group_index = torch.empty(self.total_envs, dtype=torch.long, device=device)
        self.local_index = torch.empty(self.total_envs, dtype=torch.long, device=device)
        for group_idx, group in enumerate(self.env_groups):
            self.group_index[group] = group_idx
            self.local_index[group] = torch.arange(group.numel(), device=device, dtype=torch.long)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            for traj in self.trajectories:
                traj.reset(None)
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        group_ids = torch.unique(self.group_index[env_ids])
        for g_idx in group_ids.tolist():
            local_ids = self.local_index[env_ids[self.group_index[env_ids] == g_idx]]
            self.trajectories[g_idx].reset(local_ids)

    def generate_series(self, horizon: int, dt: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time = torch.arange(horizon, device=self.device, dtype=torch.float32) * dt
        positions = torch.zeros(horizon, self.total_envs, 3, device=self.device)
        for traj, group in zip(self.trajectories, self.env_groups):
            local_envs = torch.arange(group.numel(), device=self.device, dtype=torch.long)
            time_matrix = time.unsqueeze(0).repeat(local_envs.shape[0], 1)
            pos_group = traj._positions(time_matrix, local_envs).permute(1, 0, 2)
            positions[:, group] = pos_group
        velocities, accelerations = compute_derivatives(positions, dt)
        return positions, velocities, accelerations
