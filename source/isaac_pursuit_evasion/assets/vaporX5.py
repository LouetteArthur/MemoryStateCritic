"""Configuration for the VaporX5 robot."""
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import FisheyeCameraCfg, PinholeCameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_mul
import os.path as osp
from isaaclab.sim.converters import UrdfConverterCfg
import torch


ASSET_PATH = osp.join(osp.dirname(__file__), "VaporX5")

# -------------- VaporX5 CONFIGURATION --------------
VaporX5 = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=osp.join(ASSET_PATH, "urdf/vaporX5.urdf"),
        usd_dir=osp.join(ASSET_PATH, "urdf/vaporX5"),
        usd_file_name="vaporX5.usd",
        merge_fixed_joints=False,
        fix_base=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
                    gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        )
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            "m1_joint": 200.0,
            "m2_joint": -200.0,
            "m3_joint": 200.0,
            "m4_joint": -200.0,
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


def fpv_camera_cfg(
    model: str = "fisheye",
    width: int = 320, # 640
    height: int = 240, # 480
    frequency: float = 50,
    fov_deg: float = 150.0,
    tilt_deg: float = -20.0,
    x_offset: float = 0.08,
    z_offset: float = 0.015,
    tiled: bool = False,
) -> CameraCfg:
    """Return a reusable FPV camera configuration for the VaporX5.

    Set ``tiled`` to True to get a ``TiledCameraCfg`` for tiled rendering.
    """
    import math

    tilt_rad = tilt_deg * math.pi / 180.0
    offset_cfg = CameraCfg.OffsetCfg(
        pos=(x_offset, 0.0, z_offset),
        rot=(1.0, 0.0, 0.0, 0.0),  # overwritten below
        convention="world",
    )
    # compute tilt quaternion explicitly to avoid reliance on float methods
    half = tilt_rad * 0.5
    offset_cfg.rot = (math.cos(half), 0.0, math.sin(half), 0.0)

    if model == "pinhole":
        spawn = PinholeCameraCfg(
            focal_length=8.0,
            horizontal_aperture=20.955,
            vertical_aperture=20.955 * height / width,
            clipping_range=(0.05, 200.0),
            lock_camera=True,
        )
    else:
        spawn = FisheyeCameraCfg(
            projection_type="fisheyePolynomial",
            fisheye_max_fov=fov_deg,
            fisheye_polynomial_b=0.0018,
            clipping_range=(0.05, 200.0),
            horizontal_aperture=20.0,
            vertical_aperture=20.0 * height / width,
            lock_camera=True,
        )

    cfg_type = TiledCameraCfg if tiled else CameraCfg
    return cfg_type(
        prim_path="/World/Drone/body/fpv_camera",
        update_period=1.0 / frequency,
        height=height,
        width=width,
        data_types=["rgb"],
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
    body_pos: torch.Tensor,
    body_quat: torch.Tensor,
    cam_cfg: CameraCfg | None = None,
):
    """Transform camera-frame line into world coordinates using body pose and built-in FPV offset/tilt."""
    cam_cfg = cam_cfg if cam_cfg is not None else fpv_camera_cfg()
    body_pos = body_pos.view(-1, 3)
    body_quat = body_quat.view(-1, 4)
    batch = body_pos.shape[0]

    offset_pos = torch.tensor(cam_cfg.offset.pos, device=body_pos.device, dtype=body_pos.dtype).view(1, 3).expand(batch, -1)
    offset_quat = torch.tensor(cam_cfg.offset.rot, device=body_pos.device, dtype=body_pos.dtype).view(1, 4).expand(batch, -1)
    cam_pos_w = body_pos + quat_apply(body_quat, offset_pos)
    cam_quat_w = quat_mul(body_quat, offset_quat)
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
