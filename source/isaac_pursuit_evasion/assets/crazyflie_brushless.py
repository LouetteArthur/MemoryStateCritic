# Copyright (c) 2026, the MemoryStateCritic authors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Crazyflie Brushless pursuer/evader."""

import os.path as osp
from dataclasses import dataclass

import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_mul

ASSET_DIR = osp.join(osp.dirname(__file__), "Crazyflie")


@dataclass
class CrazyflieCameraConfig:
    """Configurable camera parameters for the Crazyflie FPV camera."""

    width: int = 96
    height: int = 96
    frequency: float = 30.0
    fx: float = 27.7  # 96/2 / tan(60°) ≈ 27.7 → ~120° HFOV
    fy: float = 27.7
    cx: float = 48.0  # width / 2
    cy: float = 48.0  # height / 2
    tilt_deg: float = 0.0  # Camera tilt angle in degrees (positive = look down)
    data_types: list = None  # Default: ["rgb", "depth", "semantic_segmentation"]

    def __post_init__(self):
        if self.data_types is None:
            self.data_types = ["rgb", "depth", "semantic_segmentation"]
        # Update principal point if using non-default resolution
        if self.cx == 48.0 and self.width != 96:
            self.cx = self.width / 2.0
        if self.cy == 48.0 and self.height != 96:
            self.cy = self.height / 2.0


# Default camera configuration instance
DEFAULT_CAMERA_CONFIG = CrazyflieCameraConfig()


def _make_cfg(usd_name: str, prim_path: str) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=osp.join(ASSET_DIR, usd_name),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
            copy_from_source=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            joint_pos={
                ".*": 0.0,
            },
            joint_vel={
                ".*": 0.0,
            },
        ),
        actuators={
            "dummy": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )


CrazyflieBrushlessPursuer = _make_cfg("crazyflie_brushless_pursuer.usd", "/World/Pursuer")
CrazyflieBrushlessEvader = _make_cfg("crazyflie_brushless_evader.usd", "/World/Evader")


def fpv_camera_cfg(
    config: CrazyflieCameraConfig | None = None,
    width: int | None = None,
    height: int | None = None,
    frequency: float | None = None,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
    tilt_deg: float | None = None,
    data_types: list | None = None,
    prim_path: str = "/World/Pursuer",
    camera_link_path: str = "body/camera_link",
    tiled: bool = False,
) -> CameraCfg:
    """Return a reusable FPV camera configuration for the Crazyflie Brushless pursuer.

    Args:
        config: Optional CrazyflieCameraConfig to use as base. If None, uses DEFAULT_CAMERA_CONFIG.
        width: Override width from config
        height: Override height from config
        frequency: Override frequency from config
        fx: Override focal length x from config
        fy: Override focal length y from config
        cx: Override principal point x from config
        cy: Override principal point y from config
        tilt_deg: Camera tilt angle in degrees (positive = look down)
        data_types: Override data types from config
        prim_path: Base prim path for the camera
        camera_link_path: Path to camera link relative to robot
        tiled: Whether to use TiledCameraCfg

    Returns:
        Camera configuration ready for use in the environment
    """
    import math

    # Use provided config or default
    cfg = config or DEFAULT_CAMERA_CONFIG

    # Apply overrides
    _width = width if width is not None else cfg.width
    _height = height if height is not None else cfg.height
    _frequency = frequency if frequency is not None else cfg.frequency
    _fx = fx if fx is not None else cfg.fx
    _fy = fy if fy is not None else cfg.fy
    _cx = cx if cx is not None else cfg.cx
    _cy = cy if cy is not None else cfg.cy
    _tilt_deg = tilt_deg if tilt_deg is not None else cfg.tilt_deg
    _data_types = data_types if data_types is not None else cfg.data_types

    tilt_rad = _tilt_deg * 3.14159265 / 180.0
    offset_cfg = CameraCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),  # overwritten below
        convention="world",
    )
    half = tilt_rad * 0.5
    offset_cfg.rot = (math.cos(half), 0.0, math.sin(half), 0.0)

    intrinsic_matrix = [_fx, 0.0, _cx, 0.0, _fy, _cy, 0.0, 0.0, 1.0]
    spawn = PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=intrinsic_matrix,
        width=_width,
        height=_height,
        clipping_range=(0.02, 20.0),
        projection_type="pinhole",
        lock_camera=True,
    )

    camera_link_path = camera_link_path.strip("/")
    cfg_type = TiledCameraCfg if tiled else CameraCfg
    return cfg_type(
        prim_path=f"{prim_path}/{camera_link_path}/fpv_camera",
        update_period=1.0 / _frequency,
        height=_height,
        width=_width,
        data_types=_data_types,
        spawn=spawn,
        offset=offset_cfg,
        colorize_semantic_segmentation=False,
    )


def fpv_camera_center_line(length: float = 5.0, device: str = "cuda"):
    """Camera-frame center line endpoints for the FPV camera (+X forward, ROS convention)."""
    origin = torch.zeros(3, device=device, dtype=torch.float32)
    line_end = torch.tensor([length, 0.0, 0.0], device=device, dtype=torch.float32)
    return origin, line_end


def transform_camera_line(
    origin: torch.Tensor,
    line_end: torch.Tensor,
    link_pos: torch.Tensor,
    link_quat: torch.Tensor,
    cam_cfg: CameraCfg | None = None,
):
    """Transform camera-frame line into world coordinates using the camera_link pose."""
    cam_cfg = cam_cfg if cam_cfg is not None else fpv_camera_cfg()
    link_pos = link_pos.view(-1, 3)
    link_quat = link_quat.view(-1, 4)
    batch = link_pos.shape[0]

    offset_pos = (
        torch.tensor(cam_cfg.offset.pos, device=link_pos.device, dtype=link_pos.dtype).view(1, 3).expand(batch, -1)
    )
    offset_quat = (
        torch.tensor(cam_cfg.offset.rot, device=link_pos.device, dtype=link_pos.dtype).view(1, 4).expand(batch, -1)
    )
    cam_pos_w = link_pos + quat_apply(link_quat, offset_pos)
    cam_quat_w = quat_mul(link_quat, offset_quat)
    rot = matrix_from_quat(cam_quat_w).view(-1, 3, 3)

    origin_cam = origin.view(-1, 3)
    end_cam = line_end.view(-1, 3)
    if origin_cam.shape[0] == 1:
        origin_cam = origin_cam.expand(batch, -1)
    if end_cam.shape[0] == 1:
        end_cam = end_cam.expand(batch, -1)
    start_w = torch.bmm(rot, origin_cam.unsqueeze(-1)).squeeze(-1) + cam_pos_w
    end_w = torch.bmm(rot, end_cam.unsqueeze(-1)).squeeze(-1) + cam_pos_w
    return start_w, end_w, cam_pos_w, cam_quat_w
