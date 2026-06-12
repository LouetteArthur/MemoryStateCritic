"""Drone configuration registry.

Every supported drone registers a ``DroneConfig`` so the environment can look
up articulation assets, body names, propeller joint patterns, camera helpers,
and dynamics config names by a single ``drone_name`` string.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any, Sequence

from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import CameraCfg


@dataclass
class DroneConfig:
    """Everything the environment needs to know about a drone platform."""

    # Canonical name used in config (e.g. "crazyflie_brushless").
    name: str

    # Articulation configs (pursuer / evader variants).
    pursuer_cfg: ArticulationCfg
    evader_cfg: ArticulationCfg

    # Name of the main rigid body in the USD hierarchy.
    body_name: str = "body"

    # Ordered list of regex patterns to try when locating propeller joints.
    prop_joint_patterns: list[list[str]] = field(default_factory=lambda: [
        ["revolute_prop_.*"],
        [".*prop.*"],
        ["m[1-4]_joint"],
    ])

    # Callable that returns a CameraCfg (signature must accept **kwargs).
    fpv_camera_cfg_fn: Callable[..., CameraCfg] | None = None
    fpv_camera_center_line_fn: Callable[..., Any] | None = None
    transform_camera_line_fn: Callable[..., Any] | None = None

    # Name used by dynamics/propellers.py to load the YAML config.
    dynamics_name: str | None = None

    # Alternative names that resolve to this config.
    aliases: Sequence[str] = ()


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, DroneConfig] = {}


def register_drone(config: DroneConfig) -> None:
    """Register a drone config under its ``name`` and all ``aliases``."""
    _REGISTRY[config.name] = config
    for alias in config.aliases:
        _REGISTRY[alias] = config


def get_drone_config(name: str) -> DroneConfig:
    """Look up a registered drone by name or alias (case-insensitive)."""
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown drone '{name}'. Available: {sorted(available_drones())}"
        )
    return _REGISTRY[key]


def available_drones() -> list[str]:
    """Return all registered drone names (canonical + aliases)."""
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Built-in registrations
# ---------------------------------------------------------------------------

def _register_builtins() -> None:
    # --- Crazyflie Brushless ---
    from source.isaac_pursuit_evasion.assets.crazyflie_brushless import (
        CrazyflieBrushlessPursuer,
        CrazyflieBrushlessEvader,
        fpv_camera_cfg as bl_fpv_camera_cfg,
        fpv_camera_center_line as bl_fpv_camera_center_line,
        transform_camera_line as bl_transform_camera_line,
    )

    register_drone(DroneConfig(
        name="crazyflie_brushless",
        pursuer_cfg=CrazyflieBrushlessPursuer,
        evader_cfg=CrazyflieBrushlessEvader,
        body_name="body",
        prop_joint_patterns=[
            ["revolute_prop_.*"],
            [".*prop.*"],
        ],
        fpv_camera_cfg_fn=bl_fpv_camera_cfg,
        fpv_camera_center_line_fn=bl_fpv_camera_center_line,
        transform_camera_line_fn=bl_transform_camera_line,
        dynamics_name="crazyflie_brushless",
        aliases=("cf_brushless",),
    ))

    # --- Standard Crazyflie (brushed, Nucleus asset) ---
    from source.isaac_pursuit_evasion.assets.crazyflie import (
        CrazyfliePursuer,
        CrazyflieEvader,
        fpv_camera_cfg as cf_fpv_camera_cfg,
        fpv_camera_center_line as cf_fpv_camera_center_line,
        transform_camera_line as cf_transform_camera_line,
    )

    register_drone(DroneConfig(
        name="crazyflie",
        pursuer_cfg=CrazyfliePursuer,
        evader_cfg=CrazyflieEvader,
        body_name="body",
        prop_joint_patterns=[
            ["m[1-4]_joint"],
            [".*prop.*"],
        ],
        fpv_camera_cfg_fn=cf_fpv_camera_cfg,
        fpv_camera_center_line_fn=cf_fpv_camera_center_line,
        transform_camera_line_fn=cf_transform_camera_line,
        dynamics_name="crazyflie",
        aliases=("cf2x",),
    ))

    # --- VaporX5 ---
    from source.isaac_pursuit_evasion.assets.vaporX5 import (
        VaporX5,
        fpv_camera_cfg as vx5_fpv_camera_cfg,
        fpv_camera_center_line as vx5_fpv_camera_center_line,
        transform_camera_line as vx5_transform_camera_line,
    )

    register_drone(DroneConfig(
        name="vaporx5",
        pursuer_cfg=VaporX5,
        evader_cfg=VaporX5,
        body_name="body",
        prop_joint_patterns=[
            ["m[1-4]_joint"],
            [".*prop.*"],
        ],
        fpv_camera_cfg_fn=vx5_fpv_camera_cfg,
        fpv_camera_center_line_fn=vx5_fpv_camera_center_line,
        transform_camera_line_fn=vx5_transform_camera_line,
        dynamics_name="vaporX5",
        aliases=("vapor_x5", "vapor"),
    ))


_register_builtins()
