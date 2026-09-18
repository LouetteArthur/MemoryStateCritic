# Copyright (c) 2026, the MemoryStateCritic authors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import torch
from isaaclab.utils.math import matrix_from_quat


class FrustumVisualizer:
    """Debug-draw pyramid approximating camera FOV."""

    def __init__(self, enabled: bool, cam_cfg, device: str, length: float = 5.0):
        self.enabled = enabled
        self.debug = None
        if enabled:
            from isaacsim.util.debug_draw import _debug_draw

            self.debug = _debug_draw.acquire_debug_draw_interface()
        self.device = device
        self.length = length
        self.fov_x, self.fov_y = self._estimate_fov(cam_cfg)

    def _estimate_fov(self, cam_cfg):
        spawn = cam_cfg.spawn
        if hasattr(spawn, "focal_length"):
            fov_x = 2.0 * math.atan(spawn.horizontal_aperture / (2.0 * spawn.focal_length))
            v_ap = getattr(spawn, "vertical_aperture", None) or (
                spawn.horizontal_aperture * cam_cfg.height / cam_cfg.width
            )
            fov_y = 2.0 * math.atan(v_ap / (2.0 * spawn.focal_length))
        else:
            fov = getattr(spawn, "fisheye_max_fov", 180.0)
            fov_x = fov_y = math.radians(fov)
        return fov_x, fov_y

    def draw(self, cam_pos: torch.Tensor, cam_quat: torch.Tensor):
        if not (self.enabled and self.debug):
            return
        clear_fn = getattr(self.debug, "clear_lines", None) or getattr(self.debug, "clear_all", None)
        if callable(clear_fn):
            clear_fn()
        cam_pos = cam_pos.detach().cpu()
        cam_quat = cam_quat.detach().cpu()
        half_x = math.tan(0.5 * self.fov_x) * self.length
        half_y = math.tan(0.5 * self.fov_y) * self.length
        corners = torch.tensor(
            [
                [self.length, -half_x, -half_y],
                [self.length, half_x, -half_y],
                [self.length, half_x, half_y],
                [self.length, -half_x, half_y],
            ],
            dtype=torch.float32,
        )
        rot = matrix_from_quat(cam_quat).view(3, 3)
        world_corners = (rot @ corners.t()).t() + cam_pos
        origin = cam_pos.tolist()
        starts, ends, colors, sizes = [], [], [], []
        color = (0.1, 0.8, 0.1, 1.0)
        for pt in world_corners:
            starts.append(origin)
            ends.append(pt.tolist())
            colors.append(color)
            sizes.append(1.5)
        for i in range(4):
            starts.append(world_corners[i].tolist())
            ends.append(world_corners[(i + 1) % 4].tolist())
            colors.append(color)
            sizes.append(1.5)
        self.debug.draw_lines(starts, ends, colors, sizes)
