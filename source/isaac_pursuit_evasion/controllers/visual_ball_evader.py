"""Visual-only evader implementation for fast vision training."""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.trajectories.trajectory import (
    TrajectoryBatchManager,
    TrajectorySpec,
    WallConfig,
)

if TYPE_CHECKING:
    from pxr import UsdGeom


class VisualBallEvaderData:
    """Mock data class that provides the same interface as ArticulationData for the visual ball evader."""

    def __init__(self, num_envs: int, device: torch.device):
        self.num_envs = num_envs
        self.device = device
        # State tensors matching ArticulationData interface
        self.root_pos_w = torch.zeros(num_envs, 3, device=device)
        self.root_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(num_envs, -1).clone()
        self.root_lin_vel_w = torch.zeros(num_envs, 3, device=device)
        self.root_lin_vel_b = torch.zeros(num_envs, 3, device=device)
        self.root_com_lin_vel_b = torch.zeros(num_envs, 3, device=device)
        self.root_ang_vel_w = torch.zeros(num_envs, 3, device=device)
        self.root_ang_vel_b = torch.zeros(num_envs, 3, device=device)
        # Combined state for compatibility
        self._update_root_state()

    def _update_root_state(self):
        """Update the combined root_state_w tensor."""
        self.root_state_w = torch.cat(
            [
                self.root_pos_w,
                self.root_quat_w,
                self.root_lin_vel_w,
                self.root_ang_vel_w,
            ],
            dim=-1,
        )
        # For the visual ball, body frame aligns with world frame (identity orientation).
        self.root_lin_vel_b = self.root_lin_vel_w
        self.root_com_lin_vel_b = self.root_lin_vel_w


class VisualBallEvader:
    """Visual ball evader that follows trajectories without physics simulation.

    This provides a faster alternative to simulating a full drone evader for
    vision-based policy training. The ball follows predefined trajectories
    (hover, circular, lemniscate) and is rendered as a colored sphere.
    """

    # Map trajectory names to class names used by TrajectoryBatchManager
    TRAJECTORY_MAP = {
        "hover": "HoverTrajectory",
        "circular": "CircularTrajectory",
        "lemniscate": "LemniscateTrajectory",
    }

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        arena_bounds: torch.Tensor,
        env_origins: torch.Tensor,
        trajectory_type: str = "hover",
        trajectory_groups: Optional[dict[str, torch.Tensor]] = None,
        trajectory_horizon: int = 1000,
        dt: float = 0.02,
        radius: float = 0.075,
        color: tuple = (1.0, 0.0, 0.0),
        wall_cfg: Optional[WallConfig] = None,
    ):
        """Initialize the visual ball evader.

        Args:
            num_envs: Number of environments
            device: Torch device
            arena_bounds: Arena bounds tensor (3, 2) with min/max for x, y, z
            env_origins: Environment origins tensor (num_envs, 3)
            trajectory_type: Type of trajectory ("hover", "circular", "lemniscate") when no groups provided
            trajectory_groups: Optional mapping of trajectory name to env ids for mixed trajectories
            trajectory_horizon: Number of steps in the trajectory
            dt: Time step for trajectory generation
            radius: Ball radius for visualization
            color: Ball color as RGB tuple
        """
        self.num_envs = num_envs
        self.device = device
        self.arena_bounds = arena_bounds
        self.env_origins = env_origins
        self.dt = dt
        self.radius = float(radius)
        self._radius = torch.full((num_envs,), float(radius), device=device, dtype=torch.float32)
        self.color = color
        self.trajectory_horizon = trajectory_horizon
        self.wall_cfg = wall_cfg

        # Mock data object for compatibility with articulation data access
        self.data = VisualBallEvaderData(num_envs, device)

        # Set up trajectory managers (support mixed trajectory groups).
        self._trajectory_groups: list[dict[str, torch.Tensor | TrajectoryBatchManager]] = []
        self._group_index = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self._local_index = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self._init_trajectory_groups(trajectory_type, trajectory_groups)

        # Pre-generate trajectory series
        self._horizon = trajectory_horizon
        self._step = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._series_pos = torch.zeros(self._horizon, num_envs, 3, device=device)
        self._series_vel = torch.zeros_like(self._series_pos)
        self._regenerate_trajectories()

        # Visual sphere prims will be created by the environment
        self._sphere_prims: list[UsdGeom.Sphere] = []
        self._sphere_xforms: list[UsdGeom.Xform] = []
        self._translate_ops: list = []  # Cached translate ops for fast updates

    def _init_trajectory_groups(
        self,
        trajectory_type: str,
        trajectory_groups: Optional[dict[str, torch.Tensor]],
    ) -> None:
        """Initialize trajectory managers and env group mappings."""
        arena_min = self.arena_bounds[:, 0]
        arena_max = self.arena_bounds[:, 1]
        if trajectory_groups is None or len(trajectory_groups) == 0:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            trajectory_groups = {trajectory_type: env_ids}

        group_idx = 0
        for name, env_ids in trajectory_groups.items():
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
            if env_ids.numel() == 0:
                continue
            traj_class_name = self.TRAJECTORY_MAP.get(str(name).lower(), str(name))
            spec = TrajectorySpec(name=traj_class_name, count=int(env_ids.numel()))
            manager = TrajectoryBatchManager(
                specs=[spec],
                device=self.device,
                arena_min=arena_min,
                arena_max=arena_max,
                wall_cfg=self.wall_cfg,
            )
            self._trajectory_groups.append({"env_ids": env_ids, "manager": manager})
            self._group_index[env_ids] = group_idx
            self._local_index[env_ids] = torch.arange(env_ids.numel(), device=self.device, dtype=torch.long)
            group_idx += 1

        # Assign any unassigned envs to hover by default.
        unassigned = self._group_index < 0
        if bool(torch.any(unassigned)):
            env_ids = torch.where(unassigned)[0]
            traj_class_name = self.TRAJECTORY_MAP.get("hover", "HoverTrajectory")
            spec = TrajectorySpec(name=traj_class_name, count=int(env_ids.numel()))
            manager = TrajectoryBatchManager(
                specs=[spec],
                device=self.device,
                arena_min=arena_min,
                arena_max=arena_max,
                wall_cfg=self.wall_cfg,
            )
            self._trajectory_groups.append({"env_ids": env_ids, "manager": manager})
            self._group_index[env_ids] = group_idx
            self._local_index[env_ids] = torch.arange(env_ids.numel(), device=self.device, dtype=torch.long)

    def _regenerate_trajectories(self, env_ids: torch.Tensor | None = None):
        """Regenerate trajectory series for specified environments."""
        if env_ids is not None:
            env_ids = env_ids.to(dtype=torch.long, device=self.device)
        for group_idx, group in enumerate(self._trajectory_groups):
            manager = group["manager"]
            if env_ids is None:
                env_ids_group = group["env_ids"]
                pos, vel, _ = manager.generate_series(self._horizon, self.dt)
                self._series_pos[:, env_ids_group] = pos
                self._series_vel[:, env_ids_group] = vel
                continue

            group_mask = self._group_index[env_ids] == group_idx
            if not torch.any(group_mask):
                continue
            env_ids_group = env_ids[group_mask]
            local_ids = self._local_index[env_ids_group]
            pos, vel, _ = manager.generate_series(self._horizon, self.dt)
            self._series_pos[:, env_ids_group] = pos[:, local_ids]
            self._series_vel[:, env_ids_group] = vel[:, local_ids]
        if env_ids is None:
            self._step.zero_()
        else:
            self._step[env_ids] = 0

    def _add_semantic_label(self, prim, label: str):
        """Add semantic label to a prim for semantic segmentation.

        Tries multiple methods in order of preference:
        1. Replicator semantic API
        2. Isaac Core semantics utility
        3. Direct USD Semantics schema
        """
        # Method 1: Try Replicator semantic API
        try:
            import omni.replicator.core as rep

            rep.modify.semantics([("class", label)], prim.GetPath().pathString)
            return
        except Exception:
            pass

        # Method 2: Try Isaac Core semantics utility
        try:
            from omni.isaac.core.utils.semantics import add_update_semantics

            add_update_semantics(prim, label, "class")
            return
        except Exception:
            pass

        # Method 3: Direct USD Semantics schema (Isaac Sim standard)
        try:
            from pxr import Semantics

            if not prim.HasAPI(Semantics.SemanticsAPI):
                Semantics.SemanticsAPI.Apply(prim, "Semantics")
            sem_api = Semantics.SemanticsAPI.Get(prim, "Semantics")
            sem_api.CreateSemanticTypeAttr().Set("class")
            sem_api.CreateSemanticDataAttr().Set(label)
            return
        except Exception:
            pass

        # Method 4: Manual attribute creation (fallback)
        try:
            from pxr import Sdf

            # Create semantic type attribute
            type_attr = prim.GetAttribute("semantics:Semantics:params:semanticType")
            if not type_attr:
                type_attr = prim.CreateAttribute(
                    "semantics:Semantics:params:semanticType", Sdf.ValueTypeNames.String
                )
            type_attr.Set("class")

            # Create semantic data attribute
            data_attr = prim.GetAttribute("semantics:Semantics:params:semanticData")
            if not data_attr:
                data_attr = prim.CreateAttribute(
                    "semantics:Semantics:params:semanticData", Sdf.ValueTypeNames.String
                )
            data_attr.Set(label)
        except Exception as e:
            print(f"Warning: Could not apply semantic label '{label}' to {prim.GetPath()}: {e}")

    def create_markers(self, prim_path: str = "/Visuals/BallEvader", semantic_label: str = "evader"):
        """Create real sphere prims for the ball evader with semantic labels.

        Unlike VisualizationMarkers which use Point Instancers (not compatible with
        semantic segmentation), this spawns actual UsdGeom.Sphere prims that can
        be properly labeled for semantic segmentation cameras.

        Args:
            prim_path: Base prim path for the spheres
            semantic_label: Semantic label for the evader ball (for segmentation)
        """
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage not available")

        self._semantic_label = semantic_label
        self._sphere_prims = []
        self._sphere_xforms = []

        # Create parent xform for organization
        parent_xform = UsdGeom.Xform.Define(stage, prim_path)

        # Create a shared material for all spheres
        material_path = f"{prim_path}/evader_material"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*self.color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.1)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        # Create one sphere per environment
        for i in range(self.num_envs):
            sphere_path = f"{prim_path}/sphere_{i}"

            # Create an Xform parent for the sphere to handle transforms
            xform = UsdGeom.Xform.Define(stage, sphere_path)
            self._sphere_xforms.append(xform)

            # Create the sphere geometry as a child
            geom_path = f"{sphere_path}/geom"
            sphere = UsdGeom.Sphere.Define(stage, geom_path)
            sphere.GetRadiusAttr().Set(self._radius[i].item())
            self._sphere_prims.append(sphere)

            # Bind material to sphere
            UsdShade.MaterialBindingAPI(sphere.GetPrim()).Bind(material)

            # Apply semantic label to the sphere geometry
            self._add_semantic_label(sphere.GetPrim(), semantic_label)

        # Cache translate ops for faster updates
        self._translate_ops = []
        for xform in self._sphere_xforms:
            xformable = UsdGeom.Xformable(xform)
            translate_op = xformable.AddTranslateOp()
            self._translate_ops.append(translate_op)

    def step(self):
        """Advance the ball position along the trajectory by one step."""
        step_idx = self._step.clamp(0, self._horizon - 1)

        # Get current position and velocity from trajectory
        # _series_pos shape: (horizon, num_envs, 3)
        pos_local = self._series_pos[step_idx, torch.arange(self.num_envs, device=self.device)]
        vel_local = self._series_vel[step_idx, torch.arange(self.num_envs, device=self.device)]

        # Update data (world frame = local + env_origins)
        self.data.root_pos_w = pos_local + self.env_origins
        self.data.root_lin_vel_w = vel_local
        self.data._update_root_state()

        # Advance step counter
        self._step = (self._step + 1) % self._horizon

    def reset(self, env_ids: torch.Tensor):
        """Reset the ball evader for specified environments."""
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(dtype=torch.long, device=self.device)

        # Reset trajectories for these environments
        group_ids = torch.unique(self._group_index[env_ids])
        for group_idx in group_ids.tolist():
            group = self._trajectory_groups[group_idx]
            local_ids = self._local_index[env_ids[self._group_index[env_ids] == group_idx]]
            manager = group["manager"]
            manager.reset(local_ids)
        self._regenerate_trajectories(env_ids)

        # Update initial position
        step_idx = self._step[env_ids].clamp(0, self._horizon - 1)
        pos_local = self._series_pos[step_idx, env_ids]
        self.data.root_pos_w[env_ids] = pos_local + self.env_origins[env_ids]
        self.data.root_lin_vel_w[env_ids] = 0.0
        self.data._update_root_state()

    def update_visuals(self):
        """Update sphere prim positions."""
        if not self._translate_ops:
            return

        from pxr import Gf

        positions = self.data.root_pos_w.cpu().numpy()

        for i, translate_op in enumerate(self._translate_ops):
            pos = positions[i]
            translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

    def set_radius(self, radius: float | torch.Tensor, env_ids: torch.Tensor | None = None):
        """Update ball radius (for domain randomization)."""
        if isinstance(radius, torch.Tensor):
            radius_tensor = radius.to(device=self.device, dtype=torch.float32).view(-1)
        else:
            radius_tensor = torch.full((self.num_envs,), float(radius), device=self.device, dtype=torch.float32)

        if env_ids is None:
            if radius_tensor.numel() == 1:
                self._radius = radius_tensor.expand(self.num_envs).clone()
            else:
                self._radius = radius_tensor
            # Update sphere prim radii if they exist
            if self._sphere_prims:
                radii = self._radius.cpu().numpy()
                for i, sphere in enumerate(self._sphere_prims):
                    sphere.GetRadiusAttr().Set(float(radii[i]))
        else:
            env_ids = env_ids.to(dtype=torch.long, device=self.device)
            if radius_tensor.numel() == 1:
                self._radius[env_ids] = radius_tensor.item()
            else:
                self._radius[env_ids] = radius_tensor
            # Update sphere prim radii for affected envs
            if self._sphere_prims:
                radii = self._radius.cpu().numpy()
                for idx in env_ids.cpu().numpy():
                    self._sphere_prims[idx].GetRadiusAttr().Set(float(radii[idx]))

    def sample_radius(self, env_ids: torch.Tensor, min_radius: float, max_radius: float) -> None:
        """Sample new radii for the specified environments."""
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        self._radius[env_ids] = torch.empty(env_ids.numel(), device=self.device).uniform_(min_radius, max_radius)

        # Update sphere prim radii for affected envs
        if self._sphere_prims:
            radii = self._radius.cpu().numpy()
            for idx in env_ids.cpu().numpy():
                self._sphere_prims[idx].GetRadiusAttr().Set(float(radii[idx]))
