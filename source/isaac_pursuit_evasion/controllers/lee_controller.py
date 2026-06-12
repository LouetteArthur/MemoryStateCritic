from typing import Optional, Tuple

import torch

from isaaclab.utils import math as math_utils

from .flight_controller import QuadMixer
from .config import load_controller_config
from ..dynamics.propellers import Drone_cfg


def _expand_to(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if tensor.dim() < reference.dim():
        for _ in range(reference.dim() - tensor.dim()):
            tensor = tensor.unsqueeze(0)
    return tensor.expand(reference.shape)


class LeePositionController:
    """Geometric position controller following Lee et al. (2010)."""

    def __init__(
        self,
        num_envs: int,
        drone_cfg: Drone_cfg,
        device: str = "cuda",
        controller_cfg: Optional[dict] = None,
    ) -> None:
        self.device = device
        self.num_envs = num_envs
        self.drone_cfg = drone_cfg

        if controller_cfg is None:
            drone_name = str(drone_cfg.name).lower()
            controller_cfg = load_controller_config("lee_controller", drone_name)

        self.mass = drone_cfg.mass.to(device)
        self.gravity = torch.tensor(9.81, device=device)
        self.g_vec = torch.tensor([0.0, 0.0, 1.0], device=device)
        self.inertia = drone_cfg.inertia.to(device)

        self.k_pos = torch.tensor(controller_cfg["position_gain"], device=device, dtype=torch.float32)
        self.k_vel = torch.tensor(controller_cfg["velocity_gain"], device=device, dtype=torch.float32)
        self.k_att = torch.tensor(controller_cfg["attitude_gain"], device=device, dtype=torch.float32)
        self.k_rate = torch.tensor(controller_cfg["angular_rate_gain"], device=device, dtype=torch.float32)
        self.max_acc = torch.tensor(controller_cfg.get("max_acceleration", float("inf")), device=device, dtype=torch.float32)

        self.mixer = QuadMixer(num_envs, drone_cfg, device=device)

    def to(self, device: str) -> "LeePositionController":
        for attr in ("mass", "gravity", "g_vec", "inertia", "k_pos", "k_vel", "k_att", "k_rate"):
            setattr(self, attr, getattr(self, attr).to(device))
        self.mixer = QuadMixer(self.num_envs, self.drone_cfg, device=device)
        self.device = device
        return self

    def __call__(
        self,
        root_state: torch.Tensor,
        target_pos: Optional[torch.Tensor] = None,
        target_vel: Optional[torch.Tensor] = None,
        target_acc: Optional[torch.Tensor] = None,
        target_yaw: Optional[torch.Tensor] = None,
        target_yaw_rate: Optional[torch.Tensor] = None,
        body_rate_input: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pos, quat, lin_vel, ang_vel = torch.split(root_state, [3, 4, 3, 3], dim=-1)
        if not body_rate_input:
            ang_vel = math_utils.quat_apply_inverse(quat, ang_vel)

        if target_pos is None:
            target_pos = pos
        if target_vel is None:
            target_vel = torch.zeros_like(lin_vel)
        if target_acc is None:
            target_acc = torch.zeros_like(lin_vel)
        if target_yaw is None:
            target_yaw = math_utils.euler_xyz_from_quat(quat)[2].unsqueeze(-1)
        else:
            target_yaw = target_yaw.to(device=self.device, dtype=pos.dtype)
            if target_yaw.dim() == pos.dim() - 1:
                target_yaw = target_yaw.unsqueeze(-1)
            if target_yaw.shape[-1] != 1:
                target_yaw = target_yaw.unsqueeze(-1)
            target_yaw = _expand_to(target_yaw, target_pos[..., :1])
        if target_yaw_rate is None:
            target_yaw_rate = torch.zeros_like(target_yaw)
        else:
            target_yaw_rate = target_yaw_rate.to(device=self.device, dtype=pos.dtype)
            if target_yaw_rate.dim() == pos.dim() - 1:
                target_yaw_rate = target_yaw_rate.unsqueeze(-1)
            if target_yaw_rate.shape[-1] != 1:
                target_yaw_rate = target_yaw_rate.unsqueeze(-1)
            target_yaw_rate = _expand_to(target_yaw_rate, target_yaw)

        target_pos = target_pos.to(device=self.device, dtype=pos.dtype)
        target_vel = target_vel.to(device=self.device, dtype=pos.dtype)
        target_acc = target_acc.to(device=self.device, dtype=pos.dtype)
        target_pos = _expand_to(target_pos, pos)
        target_vel = _expand_to(target_vel, lin_vel)
        target_acc = _expand_to(target_acc, lin_vel)

        pos_error = pos - target_pos
        vel_error = lin_vel - target_vel

        force_vector = (
            + self.k_pos * pos_error
            + self.k_vel * vel_error
            - self.mass * self.gravity * self.g_vec
            - self.mass * target_acc
        )
        if torch.isfinite(self.max_acc):
            acc_cmd = force_vector / self.mass
            norm = torch.norm(acc_cmd, dim=-1, keepdim=True).clamp_min(1e-9)
            clipped = torch.minimum(norm, self.max_acc)
            acc_cmd = acc_cmd * (clipped / norm)
            force_vector = acc_cmd * self.mass

        R = math_utils.matrix_from_quat(quat)
        b3_des = -math_utils.normalize(force_vector, eps=1e-6)
        b1_des = torch.cat(
            (
                torch.cos(target_yaw),
                torch.sin(target_yaw),
                torch.zeros_like(target_yaw),
            ),
            dim=-1,
        )
        b2_des = math_utils.normalize(torch.cross(b3_des, b1_des, dim=-1), eps=1e-6)
        b1_des = torch.cross(b2_des, b3_des, dim=-1)
        R_des = torch.stack((b1_des, b2_des, b3_des), dim=-1)
        # For cases where yaw aligns with thrust, use the more robust projection:
        # yaw_vec = torch.cat((torch.cos(target_yaw), torch.sin(target_yaw), torch.zeros_like(target_yaw)), dim=-1)
        # b1_proj = yaw_vec - (yaw_vec * b3_des).sum(dim=-1, keepdim=True) * b3_des
        # ... (normalize b1_proj, cross with b3_des) ...

        att_error_matrix = 0.5 * (R_des.transpose(-1, -2) @ R - R.transpose(-1, -2) @ R_des)
        e_R = torch.stack(
            (
                att_error_matrix[..., 2, 1],
                att_error_matrix[..., 0, 2],
                att_error_matrix[..., 1, 0],
            ),
            dim=-1,
        )

        omega_des = torch.zeros_like(ang_vel)
        omega_des[..., 2:3] = target_yaw_rate
        # omega_des_body = (R.transpose(-1, -2) @ (R_des @ omega_des.unsqueeze(-1))).squeeze(-1)

        e_Omega = ang_vel - omega_des
        coriolis = torch.cross(ang_vel, self.inertia * ang_vel, dim=-1)

        moment = -self.k_att * e_R - self.k_rate * e_Omega + coriolis

        thrust = -(force_vector * R[..., 2]).sum(dim=-1, keepdim=True)
        omega = self.mixer(thrust, moment)
        return omega, thrust, moment


def drone_cfg_name(cfg: Drone_cfg) -> str:
    return str(getattr(cfg, "name", "vaporx5")).lower()
