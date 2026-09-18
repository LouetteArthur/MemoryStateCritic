"""Pursuit-evasion environment for training RL agents."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Literal, Optional
from collections.abc import Sequence

import carb
import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
import torch
import torch.distributions as D
from isaaclab.assets import ArticulationData
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import TiledCamera
from isaaclab.sensors.camera.utils import save_images_to_file
from isaaclab.utils.math import matrix_from_quat, quat_mul
from tensordict import TensorDict

from source.isaac_pursuit_evasion.assets.drone_registry import get_drone_config
from source.isaac_pursuit_evasion.controllers.crazy_controller import DEFAULT_GAINS
from source.isaac_pursuit_evasion.controllers.quadrotor_manager import (
    RL_KINDS,
    QuadrotorManager,
)
from source.isaac_pursuit_evasion.controllers.rl_controllers import (
    CrazyflieRLBodyRatesWrapper,
    CrazyflieRLVelocityWrapper,
)
from source.isaac_pursuit_evasion.controllers.visual_ball_evader import VisualBallEvader
from source.isaac_pursuit_evasion.dynamics.propellers import Drone_cfg, Propellers
from source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.trajectories.trajectory import WallConfig

# Import config classes and helpers from the cfg module
from source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.pursuit_evasion.pursuit_evasion_cfg import (
    DONE_REASON_LABELS,
    INTRINSICS_ORDER,
    ControllerSpec,
    PursuitEvasionEnvCfg,
    _infer_controller_kind,
    compute_image_obs_shape,
    compute_obs_dim,
    compute_state_dim,
    uses_image_observations,
)
from source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.pursuit_evasion.tools.frustum_viz import (
    FrustumVisualizer,
)
from source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.pursuit_evasion.tools.sampling import (
    min_separation_sampling,
    policy_sampling,
)
from source.isaac_pursuit_evasion.isaac_pursuit_evasion.tasks.direct.pursuit_evasion.tools.stats_tracker import (
    PursuitEvasionStatsTracker,
)


class PursuitEvasionEnv(DirectRLEnv):
    """Pursuit-evasion environment with two quadrotors.

    This environment simulates a pursuit-evasion game between two quadrotors:
    - Pursuer: Tries to capture the evader
    - Evader: Tries to escape from the pursuer

    The environment supports:
    - Population-based training with multiple controller types
    - Geometric controllers (PID, FRPN, APF, trajectories)
    - RL policy training for either pursuer or evader
    """

    cfg: PursuitEvasionEnvCfg
    DONE_REASON_MAP = DONE_REASON_LABELS

    def __init__(self, cfg: PursuitEvasionEnvCfg, **kwargs) -> None:
        """Initialize the pursuit-evasion environment.

        Args:
            cfg: Configuration for the environment
            **kwargs: Additional arguments for DirectRLEnv
        """
        # Compute observation space - Dict if images enabled, else flat vector
        vector_obs_dim = compute_obs_dim(cfg)
        if cfg.obs_include_segmap or cfg.obs_include_depth:
            # Dict observation space with image component
            channels_per_frame = 0
            if cfg.obs_include_segmap:
                channels_per_frame += 1
            if cfg.obs_include_depth:
                channels_per_frame += 1
            total_channels = channels_per_frame * cfg.obs_image_history

            if cfg.obs_image_only:
                # Image-only mode: actor gets segmap + past_actions (no vector state)
                # Inspired by "Demonstrating Agile Flight from Pixels" (Geles et al.)
                past_actions_dim = cfg.obs_num_past_actions * cfg.action_space
                cfg.observation_space = {
                    "image": [total_channels, cfg.camera_height, cfg.camera_width],
                    "past_actions": past_actions_dim,
                }
            else:
                # Mixed mode: actor gets both image and vector observations
                cfg.observation_space = {
                    "vector": vector_obs_dim,
                    "image": [total_channels, cfg.camera_height, cfg.camera_width],
                }
        else:
            cfg.observation_space = vector_obs_dim
        cfg.state_space = compute_state_dim(cfg)
        drone_spec = get_drone_config(cfg.drone_name)
        self._drone_spec = drone_spec

        # Visual ball evader replaces the evader drone with a simple marker.
        self._use_visual_ball_evader = bool(cfg.use_visual_ball_evader)
        if self._use_visual_ball_evader:
            if cfg.training_agent == "evader":
                raise ValueError("visual_ball_evader is incompatible with training_agent='evader'.")
            if cfg.enable_evader_cameras:
                carb.log_warn("[PursuitEvasion] Disabling evader cameras for visual_ball_evader.")
                cfg.enable_evader_cameras = False
            non_traj = [
                spec.name for spec in cfg.evader_controllers if spec.name.lower() not in VisualBallEvader.TRAJECTORY_MAP
            ]
            if non_traj:
                carb.log_warn(
                    "[PursuitEvasion] visual_ball_evader ignores non-trajectory evader controllers: "
                    + ", ".join(non_traj)
                )

        # Set up robot configurations if not provided
        if cfg.pursuer_robot is None:
            cfg.pursuer_robot = drone_spec.pursuer_cfg.replace(prim_path="/World/envs/env_.*/Pursuer")
        if cfg.evader_robot is None:
            cfg.evader_robot = drone_spec.evader_cfg.replace(prim_path="/World/envs/env_.*/Evader")

        # Auto-enable cameras if image observations are requested
        if cfg.obs_include_segmap or cfg.obs_include_depth:
            if not cfg.enable_cameras:
                carb.log_info("[PursuitEvasion] Enabling cameras for image-based observations.")
                cfg.enable_cameras = True

        if cfg.enable_cameras or cfg.obs_include_camera_angle:
            cam_cfg = self._build_fpv_camera_cfg(cfg)
            self._camera_cfg = cam_cfg if cam_cfg else None
            if self._drone_spec.fpv_camera_center_line_fn and self._camera_cfg is not None:
                self._cam_origin, self._cam_line = self._drone_spec.fpv_camera_center_line_fn(
                    length=5.0, device=cfg.sim.device
                )
            else:
                self._cam_origin, self._cam_line = None, None
            self._frustum_viz = FrustumVisualizer(cfg.flag_draw_camera_frustum, self._camera_cfg, device=cfg.sim.device)
            # use half FOV in radians for angle normalization
            half_fov = math.radians(getattr(self._camera_cfg.spawn, "fisheye_max_fov", 180.0)) * 0.5
            self._K_RHOANGLE = torch.tensor(half_fov, device=cfg.sim.device, dtype=torch.float32)
        else:
            self._camera_cfg = None
            self._cam_origin = None
            self._cam_line = None
            self._frustum_viz = None
            self._K_RHOANGLE = torch.pi

        # Must precede super().__init__(): it calls _setup_scene(), which reads
        # this attribute when the evader is the visual ball (the dataclass
        # default). Assigning it after super() left every such task dying with
        # AttributeError; the paper's ablation task sets use_visual_ball_evader
        # = False, which is why it went unnoticed.
        self._wall_cfg_for_trajectories = self._build_wall_cfg_for_trajectories(cfg)

        super().__init__(cfg, **kwargs)

        self._pursuer_camera = self.scene.sensors.get("pursuer_camera") if self.cfg.enable_cameras else None
        self._evader_camera = (
            self.scene.sensors.get("evader_camera")
            if (self.cfg.enable_cameras and self.cfg.enable_evader_cameras)
            else None
        )
        self._camera_save_stride = 1
        if self._camera_cfg is not None:
            update_period = float(getattr(self._camera_cfg, "update_period", 0.0))
            dt = float(self.sim.cfg.dt)
            if update_period > 0.0 and dt > 0.0:
                self._camera_save_stride = max(1, int(round(update_period / dt)))
        self._last_rho_camera: dict[str, torch.Tensor] = {
            "pursuer": torch.zeros(self.num_envs, device=self.device),
            "evader": torch.zeros(self.num_envs, device=self.device),
        }

        # Image observation buffers (for segmap/depth-based actor observations)
        self._use_image_obs = cfg.obs_include_segmap or cfg.obs_include_depth
        if self._use_image_obs:
            # Image dimensions
            self._img_h = cfg.camera_height
            self._img_w = cfg.camera_width
            self._img_history = cfg.obs_image_history
            # Determine number of channels per frame
            channels_per_frame = 0
            if cfg.obs_include_segmap:
                channels_per_frame += 1  # Single channel for semantic segmentation
            if cfg.obs_include_depth:
                channels_per_frame += 1  # Single channel for depth
            self._img_channels_per_frame = channels_per_frame
            self._img_total_channels = channels_per_frame * self._img_history
            # Frame history buffer: (num_envs, history * channels, H, W)
            self._image_history_buffer = torch.zeros(
                self.num_envs,
                self._img_total_channels,
                self._img_h,
                self._img_w,
                device=self.device,
                dtype=torch.float32,
            )
        else:
            self._image_history_buffer = None

        # Past actions buffer for image-only mode (Geles et al. architecture)
        self._use_image_only = cfg.obs_image_only and self._use_image_obs
        if self._use_image_only:
            self._num_past_actions = cfg.obs_num_past_actions
            # Buffer shape: (num_envs, num_past_actions, action_dim)
            self._past_actions_buffer = torch.zeros(
                self.num_envs, self._num_past_actions, cfg.action_space, device=self.device, dtype=torch.float32
            )
        else:
            self._past_actions_buffer = None

        # Opponent past-action buffers — needed when a loaded RL opponent
        # is a CNN+GRU vision policy (its forward pass takes both ``image``
        # and ``past_actions``). One buffer per role keeps the previous step's
        # action so it can be supplied as past_actions on the next step.
        # The training agent uses its own _past_actions_buffer above; these
        # ``_opp_past_actions`` buffers serve the opposing-role RL controllers.
        _opp_n = max(int(getattr(cfg, "obs_num_past_actions", 1)), 1)
        self._opp_num_past_actions = _opp_n
        self._opp_past_actions = {
            "pursuer": torch.zeros(self.num_envs, _opp_n, cfg.action_space, device=self.device, dtype=torch.float32),
            "evader": torch.zeros(self.num_envs, _opp_n, cfg.action_space, device=self.device, dtype=torch.float32),
        }

        # Action and state buffers
        self._pursuer_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._evader_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._pursuer_wrench = torch.zeros(self.num_envs, 4, device=self.device)
        self._evader_wrench = torch.zeros(self.num_envs, 4, device=self.device)
        self._pursuer_thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._pursuer_moment = torch.zeros_like(self._pursuer_thrust)
        self._evader_thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._evader_moment = torch.zeros_like(self._evader_thrust)

        # Normalization gains
        arena_extent = torch.tensor(cfg.arena_max, device=self.device) - torch.tensor(cfg.arena_min, device=self.device)
        if cfg.flag_obs_manual_normalization:
            self._K_P = arena_extent / 2.0
            self._K_V = torch.tensor([5.0, 5.0, 5.0], device=self.device)
            self._K_OMEGA = torch.tensor([5.0, 5.0, 5.0], device=self.device)
            self._K_RP = self._K_P * 2.0
            self._K_RV = self._K_V * 2.0
            self._K_EFFORT = torch.tensor(4.0, device=self.device)
        else:
            ones = torch.ones(3, device=self.device)
            self._K_P = ones
            self._K_V = ones
            self._K_OMEGA = ones
            self._K_RP = ones
            self._K_RV = ones
            self._K_EFFORT = torch.tensor(1.0, device=self.device)
        self._K_D = torch.norm(self._K_RP).clamp_min(1e-6)

        # Propellers for force/torque computation
        self._drone_cfg = Drone_cfg(self._drone_spec.dynamics_name or cfg.drone_name, device=self.device)
        self._pursuer_body_id = self._pursuer.find_bodies(self._drone_spec.body_name)[0]
        if self._use_visual_ball_evader:
            self._evader_body_id = None
        else:
            self._evader_body_id = self._evader.find_bodies(self._drone_spec.body_name)[0]
        masses = self._pursuer.root_physx_view.get_masses()[0].to(self.device)
        mass_total = masses.sum()
        inertia_body = (
            self._pursuer.root_physx_view.get_inertias()[0, self._pursuer_body_id, :].view(3, 3).to(self.device)
        )
        self._mass_total = mass_total
        self._inertia_body = inertia_body
        self._drone_cfg.set_physical_params(mass_total, inertia_body)
        self._pursuer_propellers = Propellers(
            self.num_envs, self._drone_cfg, self.sim.cfg.dt, use=True, device=self.device
        )
        if self._use_visual_ball_evader:
            self._evader_propellers = None
        else:
            self._evader_propellers = Propellers(
                self.num_envs, self._drone_cfg, self.sim.cfg.dt, use=True, device=self.device
            )
        self._pursuer_omega_ref = torch.zeros(self.num_envs, 4, device=self.device)
        self._evader_omega_ref = torch.zeros(self.num_envs, 4, device=self.device)
        self._pursuer_prop_joint_ids = self._find_prop_joints(self._pursuer)
        self._evader_prop_joint_ids = [] if self._use_visual_ball_evader else self._find_prop_joints(self._evader)

        # Arena bounds
        self._arena_min = torch.tensor(cfg.arena_min, device=self.device, dtype=torch.float32)
        self._arena_max = torch.tensor(cfg.arena_max, device=self.device, dtype=torch.float32)
        self._arena_bounds = torch.stack([self._arena_min, self._arena_max], dim=1)  # [3, 2]
        self._critic_pos_scale = (self._arena_max - self._arena_min).clamp_min(1e-6)
        self._collision_altitude = cfg.collision_altitude
        margin = torch.tensor(cfg.arena_margin, device=self.device, dtype=torch.float32)
        self._arena_min_safe = self._arena_min + margin
        self._arena_max_safe = self._arena_max - margin
        self._arena_min_safe[2] = torch.maximum(
            self._arena_min[2] + margin, torch.tensor(self._collision_altitude, device=self.device)
        )
        # Track last visibility status per agent (initialized early for obs construction).
        self._last_target_visible: dict[str, torch.Tensor] = {
            "pursuer": torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
            "evader": torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
        }
        # Cache for critic state construction.
        self._last_critic_states: torch.Tensor | None = None

        # Controller assignments
        self._pursuer_controller_assignment, self._evader_controller_assignment = self._assign_controllers()

        dt_ctrl = self.sim.cfg.dt * self.cfg.decimation
        self._pid_params = {
            "sim_rate_hz": float(self.cfg.sim_frequency),
            "pid_loop_rate_hz": float(self.cfg.pid_loop_rate_hz),
            "pid_posvel_loop_rate_hz": float(self.cfg.pid_posvel_loop_rate_hz),
        }
        self._init_domain_randomization()
        self.pursuer_manager = self._maybe_create_manager(
            agent="pursuer",
            assignment=self._pursuer_controller_assignment,
            dt_ctrl=dt_ctrl,
        )
        self.evader_manager = self._maybe_create_manager(
            agent="evader",
            assignment=self._evader_controller_assignment,
            dt_ctrl=dt_ctrl,
        )
        self._agent_managers = {"pursuer": self.pursuer_manager, "evader": self.evader_manager}

        self._training_wrappers = {
            "pursuer": self._build_training_action_wrapper("pursuer"),
            "evader": self._build_training_action_wrapper("evader"),
        }

        # One-hot heuristic identifier for the Markov critic state.
        # Built from evader controller assignments — identifies which scripted
        # heuristic drives the opponent.  Only valid when training_agent="pursuer".
        self._heuristic_names = ["hover", "circular", "lemniscate", "apf_evader", "rl"]
        self._heuristic_onehot = torch.zeros(self.num_envs, len(self._heuristic_names), device=self.device)
        self._build_heuristic_onehot()

        # Per-pool-member opponent identifier k_t for the paper's joint critic
        # e(k_t). Each distinct opponent controller name is assigned a unique
        # integer in [0, K). For heuristic ablation configs this collapses to
        # the heuristic-type index (functionally identical to the existing
        # one-hot in s); for AMSPB, each RL pool member gets its own id so the
        # critic can distinguish pool members (Prop. 2 injectivity).
        self._opp_pool_names: list[str] = []
        self._opp_pool_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._build_opp_pool_id()

        # Zero-sum per-step time cost = kappa_t / T (full-episode time pressure
        # = kappa_t for the pursuer, +kappa_t for the evader). Decoupled from
        # the capture magnitude R so that the +/-R terminal events dominate
        # the dense time-pressure signal — without this, V(s,h) was winning by
        # avoiding the heavy R-scaled time cost rather than by capturing.
        R = self.cfg.reward_catch
        kappa_t = self.cfg.reward_time_scale
        T = self.max_episode_length  # episode_length_s * policy_rate_hz
        self._reward_time_cost = kappa_t / T if T > 0 else 0.0

        # Variables to keep track of things, in this way we can still access them even after a reset call
        self._prev_distance = torch.zeros(self.num_envs, device=self.device)
        # Wall-aware distance used by the approach shaping reward (see _wall_geodesic_distance).
        # Equals Euclidean when obstacles are disabled or the line of sight is clear.
        self._prev_shaping_distance = torch.zeros(self.num_envs, device=self.device)
        self._last_pursuer_rewards = torch.zeros(self.num_envs, device=self.device)
        self._last_evader_rewards = torch.zeros(self.num_envs, device=self.device)
        self._last_done_reasons = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._last_reward_components: dict[str, dict[str, torch.Tensor]] = {"pursuer": {}, "evader": {}}
        self.extras["log"] = {}

        self._initial_reset_complete = False
        speed_agent = self.cfg.training_agent if self.cfg.training_agent else "pursuer"
        self._stats = PursuitEvasionStatsTracker(
            num_envs=self.num_envs,
            device=self.device,
            done_reason_labels=DONE_REASON_LABELS,
            speed_agent=speed_agent,
        )
        self._stats.set_heuristic_names(self._heuristic_names)

        # Visualization
        self._pursuer_vel_markers: VisualizationMarkers | None = None
        self._evader_vel_markers: VisualizationMarkers | None = None
        self._marker_align_quat = torch.tensor(
            [0.0, 0.70710677, 0.0, 0.70710677],
            device=self.device,
            dtype=torch.float32,
        )
        self._quat_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32)
        if self.cfg.debug_vis:
            if self.cfg.flag_draw_velocity_markers:
                self._setup_visualizers()
            self._setup_default_camera()
        if not carb.profiler.is_profiler_active():
            carb.log_info("carb.profiler is disabled; Tracy zones will appear only when profiler is active.")

    # -------------------------------------------------------------------------
    # Core RL interface
    # -------------------------------------------------------------------------
    @carb.profiler.profile
    def _get_states(self) -> torch.Tensor:
        """Compute critic states for asymmetric actor-critic training.

        Returns the pursuer observation with unmasked adversary information,
        always including relative velocity.
        """
        cached = self._last_critic_states
        if cached is not None:
            return cached
        # Fallback: rebuild from current data if cache is unavailable.
        agent = "evader" if self.cfg.training_agent == "evader" else "pursuer"
        return self._compute_unmasked_obs_tensor(agent)

    @carb.profiler.profile
    def _get_observations(self) -> dict:
        """Compute observations for the training agent.

        Returns a dict with:
        - "policy": actor observations (partial/FOV-gated)
        - "critic": critic states (privileged/full) if asymmetric_actor_critic is enabled
        """
        if self.cfg.training_agent == "pursuer":
            obs = self._get_pursuer_observations()
            # Include critic states only if asymmetric actor-critic is enabled
            if self.cfg.asymmetric_actor_critic:
                obs["critic"] = self._build_critic_states("pursuer", obs)
            self._populate_opponent_extras("evader")
            return obs
        if self.cfg.training_agent == "evader":
            obs = self._get_evader_observations()
            # Include critic states only if asymmetric actor-critic is enabled
            if self.cfg.asymmetric_actor_critic:
                obs["critic"] = self._build_critic_states("evader", obs)
            self._populate_opponent_extras("pursuer")
            return obs
        # Multi-agent case
        obs_both = {
            "pursuer": self._compute_obs_tensor("pursuer"),
            "evader": self._compute_obs_tensor("evader"),
        }
        obs_both["policy"] = torch.zeros(self.num_envs, self.cfg.observation_space, device=self.device)
        if self.cfg.asymmetric_actor_critic:
            agent = self.cfg.training_agent if self.cfg.training_agent else "pursuer"
            obs_both["critic"] = self._build_critic_states(agent, obs_both)
        return obs_both

    @carb.profiler.profile
    def _get_pursuer_observations(self) -> dict:
        """Get observations from the pursuer's perspective."""
        if self._use_image_only:
            # Image-only mode: actor gets segmap + past_actions (Geles et al. architecture)
            image_obs = self._get_image_observations("pursuer")
            # Flatten past actions buffer: (num_envs, num_past_actions * action_dim)
            past_actions = self._past_actions_buffer.view(self.num_envs, -1)
            return {"policy": {"image": image_obs, "past_actions": past_actions}}
        elif self._use_image_obs:
            # Mixed mode: vector + image
            vector_obs = self._compute_obs_tensor("pursuer")
            image_obs = self._get_image_observations("pursuer")
            return {"policy": {"vector": vector_obs, "image": image_obs}}
        else:
            # Vector-only mode
            vector_obs = self._compute_obs_tensor("pursuer")
            return {"policy": vector_obs}

    @carb.profiler.profile
    def _get_evader_observations(self) -> dict:
        """Get observations from the evader's perspective."""
        if self._use_image_only:
            # Image-only mode: actor gets segmap + past_actions (Geles et al. architecture)
            image_obs = self._get_image_observations("evader")
            # Flatten past actions buffer: (num_envs, num_past_actions * action_dim)
            past_actions = self._past_actions_buffer.view(self.num_envs, -1)
            return {"policy": {"image": image_obs, "past_actions": past_actions}}
        elif self._use_image_obs:
            # Mixed mode: vector + image
            vector_obs = self._compute_obs_tensor("evader")
            image_obs = self._get_image_observations("evader")
            return {"policy": {"vector": vector_obs, "image": image_obs}}
        else:
            # Vector-only mode
            vector_obs = self._compute_obs_tensor("evader")
            return {"policy": vector_obs}

    def _compute_obs_tensor(
        self, agent: Literal["pursuer", "evader"], env_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Shared observation builder for both agents."""
        other = "evader" if agent == "pursuer" else "pursuer"
        parts = []
        if self.cfg.obs_include_time_encoding:
            select = env_ids if env_ids is not None else slice(None)
            time_encoding = (self.episode_length_buf[select] / self.max_episode_length).unsqueeze(-1)
            parts.append(time_encoding)

        robot = self._agent_data(agent)
        quat = self._select(robot.data.root_quat_w, env_ids)
        # 6D rotation: first two columns of R_WB (Zhou et al., Geles et al.)
        rot_full = matrix_from_quat(quat)  # (N, 3, 3)
        rot_6d = rot_full[:, :, :2].reshape(quat.shape[0], -1)  # (N, 6)
        parts.append(rot_6d)

        vel_body = self._select(robot.data.root_com_lin_vel_b, env_ids)
        parts.append(vel_body / self._K_V)

        ang_vel = self._select(robot.data.root_ang_vel_b, env_ids)
        parts.append(ang_vel / self._K_OMEGA)

        pos_local = self._agent_position(agent, env_ids)
        # Always include local position for critic states
        parts.append(pos_local / self._K_P)

        if self.cfg.obs_include_prev_action:
            parts.append(self._get_prev_action(agent, env_ids))

        distance_unmasked = None
        if self.cfg.obs_include_camera_angle:
            cam_pos_w, _, forward, _ = self._camera_pose(agent, env_ids)
            other_world = self._select(self._agent_data(other).data.root_pos_w, env_ids)
            rel_cam = other_world - cam_pos_w
            rho_cam = self._camera_angle(agent, env_ids).unsqueeze(-1)
            rho_limit = self._K_RHOANGLE
            in_fov = rho_cam <= rho_limit
            # True visibility also requires the wall not to block line-of-sight.
            origins = (
                self._terrain.env_origins if env_ids is None else self._terrain.env_origins[env_ids]
            )
            cam_pos_local = cam_pos_w - origins
            other_local = self._select(self._agent_data(other).data.root_pos_w, env_ids) - origins
            occluded = self._wall_occludes_los(cam_pos_local, other_local).unsqueeze(-1)
            visible = in_fov & (~occluded)
            if env_ids is None:
                self._last_target_visible[agent] = visible.squeeze(-1)
            rel_used = torch.where(visible, rel_cam, torch.zeros_like(rel_cam))
            parts.append(visible.float())
            parts.append(rel_used / self._K_RP)
            parts.append((rho_cam / rho_limit).clamp(-1.0, 1.0))
            relative_pos = rel_used
            distance_unmasked = torch.norm(rel_cam, dim=-1, keepdim=True)
        else:
            visible = torch.ones(size=(self.num_envs, 1), dtype=bool, device=self.device)
            other_pos = self._agent_position(other, env_ids)
            relative_pos = other_pos - pos_local
            parts.append(relative_pos / self._K_RP)
            distance_unmasked = torch.norm(relative_pos, dim=-1, keepdim=True)

        if self.cfg.obs_include_relative_distance:
            if distance_unmasked is None:
                distance_unmasked = torch.norm(relative_pos, dim=-1, keepdim=True)
            distance = torch.where(visible, distance_unmasked, torch.ones_like(distance_unmasked))
            parts.append(distance / self._K_D)

        if self.cfg.obs_include_closing_velocity:
            self_vel = self._agent_linear_velocity(agent, env_ids)
            other_vel = self._agent_linear_velocity(other, env_ids)
            parts.append((other_vel - self_vel) / self._K_RV)

        return torch.cat(parts, dim=-1)

    def _compute_unmasked_obs_tensor(
        self, agent: Literal["pursuer", "evader"], env_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Observation-like critic state without masking adversary information."""
        other = "evader" if agent == "pursuer" else "pursuer"
        parts = []
        if self.cfg.obs_include_time_encoding:
            select = env_ids if env_ids is not None else slice(None)
            time_encoding = (self.episode_length_buf[select] / self.max_episode_length).unsqueeze(-1)
            parts.append(time_encoding)

        robot = self._agent_data(agent)
        quat = self._select(robot.data.root_quat_w, env_ids)
        # 6D rotation: first two columns of R_WB (Zhou et al., Geles et al.)
        rot_full = matrix_from_quat(quat)  # (N, 3, 3)
        rot_6d = rot_full[:, :, :2].reshape(quat.shape[0], -1)  # (N, 6)
        parts.append(rot_6d)

        vel_body = self._select(robot.data.root_com_lin_vel_b, env_ids)
        parts.append(vel_body / self._K_V)

        ang_vel = self._select(robot.data.root_ang_vel_b, env_ids)
        parts.append(ang_vel / self._K_OMEGA)

        pos_local = self._agent_position(agent, env_ids)
        parts.append(pos_local / self._K_P)

        if self.cfg.obs_include_prev_action:
            parts.append(self._get_prev_action(agent, env_ids))

        if self.cfg.obs_include_camera_angle:
            cam_pos_w, _, _, _ = self._camera_pose(agent, env_ids)
            other_world = self._select(self._agent_data(other).data.root_pos_w, env_ids)
            rel_cam = other_world - cam_pos_w
            rho_cam = self._camera_angle(agent, env_ids).unsqueeze(-1)
            rho_limit = self._K_RHOANGLE
            in_fov = rho_cam <= rho_limit
            origins = (
                self._terrain.env_origins if env_ids is None else self._terrain.env_origins[env_ids]
            )
            cam_pos_local = cam_pos_w - origins
            other_local = other_world - origins
            occluded = self._wall_occludes_los(cam_pos_local, other_local).unsqueeze(-1)
            visible = in_fov & (~occluded)
            if env_ids is None:
                self._last_target_visible[agent] = visible.squeeze(-1)
            parts.append(visible.float())
            parts.append(rel_cam / self._K_RP)  # unmasked
            parts.append((rho_cam / rho_limit).clamp(-1.0, 1.0))
            relative_pos = rel_cam
        else:
            other_pos = self._agent_position(other, env_ids)
            relative_pos = other_pos - pos_local
            parts.append(relative_pos / self._K_RP)

        distance = torch.norm(relative_pos, dim=-1, keepdim=True)
        parts.append(distance / self._K_D)

        self_vel = self._agent_linear_velocity(agent, env_ids)
        other_vel = self._agent_linear_velocity(other, env_ids)
        parts.append((other_vel - self_vel) / self._K_RV)

        if self.cfg.critic_include_propeller_speeds:
            omega_p = self._select(self._pursuer_propellers.omega, env_ids)  # (N, 4)
            parts.append(omega_p / self._pursuer_propellers.motor_speed_max)
            if not self._use_visual_ball_evader and self._evader_propellers is not None:
                omega_e = self._select(self._evader_propellers.omega, env_ids)  # (N, 4)
                parts.append(omega_e / self._evader_propellers.motor_speed_max)

        if self.cfg.critic_include_heuristic_state:
            # Markov state extras (32 dims) — makes P(s_{t+1}|s_t, a_t) deterministic
            # within an episode.  See _compute_state_vector_dim() for the dim breakdown.
            #   evader_rot6d(6) + evader_ang_vel(3) + agent_rate_pid(3)
            #   + opponent_pid(9) + heuristic_onehot(5) + traj_setpoint(6)
            opp_data = self._agent_data(other)
            opp_manager = self._agent_managers.get(other)

            # --- Evader 6D rotation (6 dims) ---
            opp_quat = self._select(opp_data.data.root_quat_w, env_ids)
            opp_rot = matrix_from_quat(opp_quat)  # (N, 3, 3)
            opp_rot6d = opp_rot[:, :, :2].reshape(opp_quat.shape[0], -1)  # (N, 6)
            parts.append(opp_rot6d)

            # --- Evader angular velocity (3 dims) ---
            opp_ang_vel = self._select(opp_data.data.root_ang_vel_b, env_ids)
            parts.append(opp_ang_vel / self._K_OMEGA)

            # --- Training agent rate PID integral (3 dims) ---
            # Body-rate RL controller uses only the rate PID (last 3 of the 9-dim
            # integral vector); vel/att PIDs stay zero because they are never called.
            agent_wrapper = self._training_wrappers.get(agent)
            if agent_wrapper is not None and hasattr(agent_wrapper, "pid"):
                agent_pid_int = agent_wrapper.pid.get_pid_integrals(num_envs=self.num_envs)
                parts.append(self._select(agent_pid_int[:, 6:], env_ids))
            else:
                n = self.num_envs if env_ids is None else env_ids.numel()
                parts.append(torch.zeros(n, 3, device=self.device))

            # --- Opponent PID integrals (9 dims) ---
            # Full cascade: vel(3) + att(3) + rate(3) for velocity-command controllers.
            if opp_manager is not None:
                opp_pid_int = opp_manager.get_pid_integrals()
                parts.append(self._select(opp_pid_int, env_ids))
            else:
                n = self.num_envs if env_ids is None else env_ids.numel()
                parts.append(torch.zeros(n, 9, device=self.device))

            # --- One-hot heuristic identifier (5 dims) ---
            # Order: hover, circular, lemniscate, apf_evader, rl
            parts.append(self._select(self._heuristic_onehot, env_ids))

            # --- Trajectory setpoint pos + vel (6 dims) ---
            # Zeros for non-trajectory controllers (APF, RL, etc.).
            if opp_manager is not None:
                tgt_pos, tgt_vel = opp_manager.get_trajectory_setpoint()
                parts.append(self._select(tgt_pos, env_ids) / self._K_P)
                parts.append(self._select(tgt_vel, env_ids) / self._K_V)
            else:
                n = self.num_envs if env_ids is None else env_ids.numel()
                parts.append(torch.zeros(n, 3, device=self.device))
                parts.append(torch.zeros(n, 3, device=self.device))

        return torch.cat(parts, dim=-1)

    def _build_critic_states(self, agent: str, obs: dict):
        """Build critic states for asymmetric actor-critic.

        For vision-based training (obs_image_only), returns a dict with the actor's
        observations (image + past_actions) AND the privileged state vector.
        This follows Baisero & Amato's unbiased asymmetric actor-critic: the critic
        must receive at least everything the actor sees to avoid biased gradients.

        For state-based training, returns the privileged state vector only (flat tensor).
        """
        state_vector = self._compute_unmasked_obs_tensor(agent)
        self._last_critic_states = state_vector

        if self._use_image_only and self.cfg.unbiased_critic:
            # Unbiased: critic gets state + actor's observations (image + past_actions)
            policy_obs = obs.get("policy", {})
            if isinstance(policy_obs, dict):
                return {
                    "image": policy_obs["image"],
                    "past_actions": policy_obs["past_actions"],
                    "state": state_vector,
                }

        return state_vector

    def _populate_opponent_extras(self, opponent_role: str) -> None:
        """Populate self.extras with opponent z^opp and obs/action for joint critics.

        Called after each observation step. Provides:
        - "z_opp": opponent's RNN hidden state (zeros for scripted/MLP opponents)
        - "opp_prev_action": opponent's last action
        - "opp_image": opponent's camera image (zeros if no opponent camera)
        """
        manager = self._agent_managers.get(opponent_role)

        if self.cfg.expose_opponent_z:
            if manager is not None:
                self.extras["z_opp"] = manager.get_opp_z()
            else:
                self.extras["z_opp"] = torch.zeros(
                    self.num_envs, self.cfg.opponent_z_dim, device=self.device
                )

        if self.cfg.expose_opp_id:
            self.extras["opp_id"] = self._opp_pool_id

        if self.cfg.expose_opponent_obs:
            # Opponent's previous action
            if manager is not None:
                self.extras["opp_prev_action"] = manager.get_opp_prev_action()
            else:
                self.extras["opp_prev_action"] = torch.zeros(self.num_envs, 4, device=self.device)

            # Opponent's camera image (zeros if cameras not enabled for opponent).
            # update_history=False is REQUIRED: this is an opponent query, and
            # the default (True) would append the opponent's frame to the
            # shared training-agent image-history buffer, corrupting its
            # temporal consistency for every critic that reads it.
            if self._use_image_only and hasattr(self, "_get_image_observations"):
                try:
                    opp_image = self._get_image_observations(opponent_role, update_history=False)
                    self.extras["opp_image"] = opp_image
                except Exception:
                    # Opponent camera not available — provide zeros
                    img_shape = self._image_obs_shape
                    self.extras["opp_image"] = torch.zeros(
                        self.num_envs, *img_shape, device=self.device
                    )
            else:
                # No image mode or no camera — provide zeros matching actor image shape
                if hasattr(self, "_image_obs_shape") and self._image_obs_shape is not None:
                    img_shape = self._image_obs_shape
                else:
                    img_shape = (2, 64, 64)  # default: depth + segmap
                self.extras["opp_image"] = torch.zeros(
                    self.num_envs, *img_shape, device=self.device
                )

    @carb.profiler.profile
    def _get_rewards(self) -> torch.Tensor:
        pursuer_state = self._pursuer.data.root_state_w
        evader_state = self._evader.data.root_state_w
        invalid_pursuer = ~torch.isfinite(pursuer_state).all(dim=-1)
        invalid_evader = ~torch.isfinite(evader_state).all(dim=-1)

        pursuer_pos = torch.nan_to_num(self._agent_position("pursuer"), nan=0.0, posinf=0.0, neginf=0.0)
        evader_pos = torch.nan_to_num(self._agent_position("evader"), nan=0.0, posinf=0.0, neginf=0.0)
        distance = torch.norm(evader_pos - pursuer_pos, dim=-1)
        # Wall-aware distance for approach shaping; equals Euclidean when no wall is between them.
        shaping_distance = self._wall_geodesic_distance(pursuer_pos, evader_pos)
        # Spatial-extent diagnostics: track how far each agent explores per episode.
        self._stats.update_position_extent(
            {"pursuer": pursuer_pos, "evader": evader_pos},
            arena_min=self._arena_min,
            arena_max=self._arena_max,
        )
        rho_camera_pursuer = self._camera_angle("pursuer") if self.cfg.obs_include_camera_angle else None
        if rho_camera_pursuer is not None:
            rho_camera_pursuer = torch.nan_to_num(rho_camera_pursuer, nan=0.0, posinf=0.0, neginf=0.0)
        pursuer_rewards = self._get_pursuer_rewards(
            distance,
            pursuer_pos,
            evader_pos,
            rho_camera=rho_camera_pursuer,
            invalid_mask=invalid_pursuer,
            shaping_distance=shaping_distance,
        )
        evader_rewards = self._get_evader_rewards(distance, pursuer_pos, evader_pos, invalid_mask=invalid_evader)
        self._last_pursuer_rewards = pursuer_rewards
        self._last_evader_rewards = evader_rewards
        speed_agent = self._stats.speed_agent
        speed_robot = self._agent_data(speed_agent)
        self._stats.update_speed_stats(
            speed_robot.data.root_lin_vel_w,
            speed_robot.data.root_ang_vel_b,
            self.step_dt,
        )
        if self.cfg.obs_include_camera_angle and self._camera_cfg is not None:
            rho_by_agent = {
                "pursuer": rho_camera_pursuer,  # reuse already-computed value
                "evader": self._camera_angle("evader"),
            }
            self._stats.update_rho_stats(rho_by_agent)

        if self.cfg.training_agent == "pursuer":
            rewards = pursuer_rewards
        elif self.cfg.training_agent == "evader":
            rewards = evader_rewards
        else:
            rewards = torch.zeros(self.num_envs, device=self.device)

        self._prev_distance = distance.clone()
        self._prev_shaping_distance = shaping_distance.clone()
        if self.cfg.save_camera_images:
            self._maybe_save_camera_images()
        return rewards

    @carb.profiler.profile
    def _get_pursuer_rewards(
        self,
        distance: torch.Tensor,
        pursuer_pos: torch.Tensor,
        evader_pos: torch.Tensor,
        rho_camera: torch.Tensor | None = None,
        invalid_mask: torch.Tensor | None = None,
        shaping_distance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        R = self.cfg.reward_catch
        rewards = torch.zeros(self.num_envs, device=self.device)
        components: dict[str, torch.Tensor] = {}

        # Time penalty: -R/T per step (incentivizes catching fast).
        time_cost = torch.full_like(rewards, -self._reward_time_cost)
        rewards += time_cost
        components["time"] = time_cost

        # Potential-based exponential approach shaping: w * (Φ(d_t) - Φ(d_{t-1}))
        # with Φ(d) = exp(-d / d_0).  Same Ng-Harada-Russell guarantee as the linear
        # potential, but the dense reward is concentrated near the capture radius
        # (>70% of cumulative shaping is collected in the last d_0 of approach).
        # This breaks the "stalker" equilibrium that the linear Φ = -d allows, where
        # the policy can collect most of the dense reward by maintaining moderate
        # proximity without committing to capture. Standard form in legged/aerial
        # robot RL (Hwangbo 2019; Lee 2020; Rudin 2022; Kaufmann 2023). Uses the
        # wall-aware geodesic distance so the gradient routes through the gap.
        if shaping_distance is None:
            shaping_distance = distance
        # Φ(d) = exp(-(d - r_c) / d_0) — shifted exponential potential. The
        # shift by the capture distance r_c bounds Φ at 1 (reached at the
        # capture boundary, where the episode terminates with the catch
        # reward) so the dense shaping signal cannot dwarf the terminal R.
        # We clamp d - r_c at 0 to keep Φ ≤ 1 if the agent briefly crosses
        # the capture boundary within a single physics step.
        d_0 = self.cfg.reward_approach_decay
        r_c = self.cfg.capture_distance
        phi_t = torch.exp(-(shaping_distance - r_c).clamp(min=0.0) / d_0)
        phi_prev = torch.exp(-(self._prev_shaping_distance - r_c).clamp(min=0.0) / d_0)
        approach_term = self.cfg.reward_approach * (phi_t - phi_prev)
        rewards += approach_term
        components["approach"] = approach_term

        # FPV centering bonus, gated by true visibility (in-FoV AND wall does not
        # occlude line-of-sight). Without the occlusion gate, the agent could earn
        # this reward by orienting toward the evader's position when the wall hides it.
        if rho_camera is not None and self.cfg.reward_perception_scale > 0:
            visible = self._last_target_visible.get("pursuer")
            if visible is None:
                visible = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            perception = visible.float() * self.cfg.reward_perception_scale * torch.exp(
                -self.cfg.reward_perception_angle_scale * rho_camera
            )
        else:
            perception = torch.zeros_like(rewards)
        rewards += perception
        components["perception"] = perception

        # Terminal events: +R for catch (or evader self-destructs via OOB / wall), -R for own OOB.
        captured = (distance < self.cfg.capture_distance) | self._out_of_bounds_mask(evader_pos)
        if self.cfg.enable_obstacles:
            captured = captured | self._check_wall_collision(evader_pos)
        capture_bonus = torch.zeros_like(rewards)
        capture_bonus[captured] = R
        rewards += capture_bonus
        components["capture"] = capture_bonus

        bounds_collision = self._out_of_bounds_mask(pursuer_pos)
        bounds_penalty = torch.zeros_like(rewards)
        bounds_penalty[bounds_collision] = -R
        rewards += bounds_penalty
        components["bounds"] = bounds_penalty

        # Regularization (sim-to-real, not part of the game).
        ang_vel_b = self._pursuer.data.root_ang_vel_b
        weights = torch.tensor([1.0, 1.0, 0.2], device=self.device, dtype=torch.float32)
        body_penalty = -self.cfg.reward_body_rates * torch.norm(ang_vel_b * weights, dim=-1)
        rewards += body_penalty
        components["body_rates"] = body_penalty

        prev_action = self._get_prev_action("pursuer")
        action_diff = self._pursuer_actions - prev_action
        smooth_penalty = -self.cfg.reward_action_smoothness * torch.sum(action_diff**2, dim=-1)
        rewards += smooth_penalty
        components["action_smoothness"] = smooth_penalty

        if self.cfg.enable_obstacles:
            wall_hit = self._check_wall_collision(pursuer_pos)
            wall_penalty = torch.zeros_like(rewards)
            wall_penalty[wall_hit] = -self.cfg.obstacle_collision_penalty
            rewards += wall_penalty
            components["wall_collision"] = wall_penalty

        if invalid_mask is not None:
            invalid_penalty = torch.zeros_like(rewards)
            invalid_penalty[invalid_mask] = -R
            rewards += invalid_penalty
            components["invalid_state"] = invalid_penalty

        components["total"] = rewards
        self._last_reward_components["pursuer"] = {k: v.clone() for k, v in components.items()}
        self._stats.accumulate_episode_reward("pursuer", components)
        return rewards

    @carb.profiler.profile
    def _get_evader_rewards(
        self,
        distance: torch.Tensor,
        pursuer_pos: torch.Tensor,
        evader_pos: torch.Tensor,
        invalid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        R = self.cfg.reward_catch
        rewards = torch.zeros(self.num_envs, device=self.device)
        components: dict[str, torch.Tensor] = {}

        # Survival reward: +R/T per step (zero-sum counterpart of pursuer time penalty).
        time_reward = torch.full_like(rewards, self._reward_time_cost)
        rewards += time_reward
        components["time"] = time_reward

        # Terminal events: -R for capture, +R when pursuer crashes (OOB or wall).
        captured = distance < self.cfg.capture_distance
        capture_term = torch.zeros_like(rewards)
        capture_term[captured] = -R
        pursuer_oob = self._out_of_bounds_mask(pursuer_pos)
        capture_term[pursuer_oob] += R
        if self.cfg.enable_obstacles:
            pursuer_wall_hit = self._check_wall_collision(pursuer_pos)
            capture_term[pursuer_wall_hit] += R
        rewards += capture_term
        components["capture"] = capture_term

        # Regularization (sim-to-real, not part of the game).
        ang_vel_b = self._evader.data.root_ang_vel_b
        weights = torch.tensor([1.0, 1.0, 0.2], device=self.device, dtype=torch.float32)
        body_penalty = -self.cfg.reward_body_rates * torch.norm(ang_vel_b * weights, dim=-1)
        rewards += body_penalty
        components["body_rates"] = body_penalty

        prev_action = self._get_prev_action("evader")
        action_diff = self._evader_actions - prev_action
        smooth_penalty = -self.cfg.reward_action_smoothness * torch.sum(action_diff**2, dim=-1)
        rewards += smooth_penalty
        components["action_smoothness"] = smooth_penalty

        out_of_bounds = self._out_of_bounds_mask(evader_pos)
        bounds_penalty = torch.zeros_like(rewards)
        bounds_penalty[out_of_bounds] = -R
        rewards += bounds_penalty
        components["bounds"] = bounds_penalty

        if self.cfg.enable_obstacles:
            wall_hit = self._check_wall_collision(evader_pos)
            wall_penalty = torch.zeros_like(rewards)
            wall_penalty[wall_hit] = -self.cfg.obstacle_collision_penalty
            rewards += wall_penalty
            components["wall_collision"] = wall_penalty

        if invalid_mask is not None:
            invalid_penalty = torch.zeros_like(rewards)
            invalid_penalty[invalid_mask] = -R
            rewards += invalid_penalty
            components["invalid_state"] = invalid_penalty

        components["total"] = rewards
        self._last_reward_components["evader"] = {k: v.clone() for k, v in components.items()}
        self._stats.accumulate_episode_reward("evader", components)
        return rewards

    def _out_of_bounds_mask(self, positions: torch.Tensor) -> torch.Tensor:
        return (
            (positions[:, 0] < self._arena_min_safe[0])
            | (positions[:, 0] > self._arena_max_safe[0])
            | (positions[:, 1] < self._arena_min_safe[1])
            | (positions[:, 1] > self._arena_max_safe[1])
            | (positions[:, 2] < self._arena_min_safe[2])
            | (positions[:, 2] > self._arena_max_safe[2])
        )

    @carb.profiler.profile
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check episode termination conditions."""
        pursuer_pos = self._agent_position("pursuer")
        evader_pos = self._agent_position("evader")
        distance = torch.norm(evader_pos - pursuer_pos, dim=-1)

        # Capture condition
        captured = distance < self.cfg.capture_distance

        # Out of bounds
        pursuer_oob = self._out_of_bounds_mask(pursuer_pos)
        evader_oob = self._out_of_bounds_mask(evader_pos)

        invalid_pursuer = ~torch.isfinite(self._pursuer.data.root_state_w).all(dim=-1)
        invalid_evader = ~torch.isfinite(self._evader.data.root_state_w).all(dim=-1)
        invalid_state = invalid_pursuer | invalid_evader

        # Obstacle collision (split per agent so callers can attribute who hit the wall)
        if self.cfg.enable_obstacles:
            pursuer_wall_hit = self._check_wall_collision(pursuer_pos)
            evader_wall_hit = self._check_wall_collision(evader_pos)
        else:
            pursuer_wall_hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            evader_wall_hit = torch.zeros_like(pursuer_wall_hit)

        terminated = captured | pursuer_oob | evader_oob | invalid_state | pursuer_wall_hit | evader_wall_hit
        timeout = self.episode_length_buf >= self.max_episode_length
        truncated = timeout & (~terminated)

        reasons = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        reasons[captured] = 1
        reasons[pursuer_oob] = 3
        reasons[evader_oob] = 4
        reasons[timeout] = 5
        reasons[invalid_state] = 6
        reasons[pursuer_wall_hit] = 7
        reasons[evader_wall_hit] = 8
        self._last_done_reasons = reasons

        return terminated, truncated

    def _init_domain_randomization(self) -> None:
        self._dr_cfg = self.cfg.domain_randomization
        self._intrinsics_names = INTRINSICS_ORDER

        self._dr_default_masses = {
            "pursuer": self._pursuer.root_physx_view.get_masses().clone().cpu(),
        }
        self._dr_default_inertias = {
            "pursuer": self._pursuer.root_physx_view.get_inertias().clone().cpu(),
        }
        if not self._use_visual_ball_evader:
            self._dr_default_masses["evader"] = self._evader.root_physx_view.get_masses().clone().cpu()
            self._dr_default_inertias["evader"] = self._evader.root_physx_view.get_inertias().clone().cpu()
        self._dr_masses = {
            "pursuer": self._dr_default_masses["pursuer"].clone(),
        }
        self._dr_inertias = {
            "pursuer": self._dr_default_inertias["pursuer"].clone(),
        }
        if not self._use_visual_ball_evader:
            self._dr_masses["evader"] = self._dr_default_masses["evader"].clone()
            self._dr_inertias["evader"] = self._dr_default_inertias["evader"].clone()

        self._dr_nominal_mass = float(self._mass_total)
        self._dr_nominal_inertia = torch.diagonal(self._inertia_body).clone().to(self.device)
        self._dr_nominal_k_eta = float(self._pursuer_propellers.k_eta[0, 0])
        self._dr_nominal_k_m = float(self._pursuer_propellers.k_m[0, 0])
        self._dr_nominal_tau = float(self._pursuer_propellers.tau_m[0, 0])
        self._dr_nominal_k_aero_xy = float(self._pursuer_propellers.K_aero[0, 0])
        self._dr_nominal_k_aero_z = float(self._pursuer_propellers.K_aero[0, 2])

        self._dr_rate_kp_nominal = self._resolve_rate_gain_nominal(
            "rate_kp", ("rollRateKp", "pitchRateKp", "yawRateKp"), DEFAULT_GAINS["rate"]["kp"]
        )
        self._dr_rate_ki_nominal = self._resolve_rate_gain_nominal(
            "rate_ki", ("rollRateKi", "pitchRateKi", "yawRateKi"), DEFAULT_GAINS["rate"]["ki"]
        )
        self._dr_rate_kd_nominal = self._resolve_rate_gain_nominal(
            "rate_kd", ("rollRateKd", "pitchRateKd", "yawRateKd"), DEFAULT_GAINS["rate"]["kd"]
        )

        nominal_parts = [
            torch.tensor([self._dr_nominal_mass], device=self.device, dtype=torch.float32),
            self._dr_nominal_inertia.view(3),
            torch.tensor([self._dr_nominal_k_eta], device=self.device, dtype=torch.float32),
            torch.tensor([self._dr_nominal_k_m], device=self.device, dtype=torch.float32),
            torch.tensor([self._dr_nominal_tau], device=self.device, dtype=torch.float32),
            torch.tensor([self._dr_nominal_k_aero_xy], device=self.device, dtype=torch.float32),
            torch.tensor([self._dr_nominal_k_aero_z], device=self.device, dtype=torch.float32),
            self._dr_rate_kp_nominal.view(3),
            self._dr_rate_ki_nominal.view(3),
            self._dr_rate_kd_nominal.view(3),
        ]
        self._intrinsics_nominal = torch.cat(nominal_parts, dim=-1)

        self._intrinsics = self._intrinsics_nominal.view(1, -1).repeat(self.num_envs, 1)

        self._dr_mass = torch.full((self.num_envs,), self._dr_nominal_mass, device=self.device)
        self._dr_inertia = self._dr_nominal_inertia.view(1, 3).repeat(self.num_envs, 1)
        self._dr_k_eta = torch.full((self.num_envs,), self._dr_nominal_k_eta, device=self.device)
        self._dr_k_m = torch.full((self.num_envs,), self._dr_nominal_k_m, device=self.device)
        self._dr_tau = torch.full((self.num_envs,), self._dr_nominal_tau, device=self.device)
        self._dr_k_aero_xy = torch.full((self.num_envs,), self._dr_nominal_k_aero_xy, device=self.device)
        self._dr_k_aero_z = torch.full((self.num_envs,), self._dr_nominal_k_aero_z, device=self.device)
        self._dr_rate_kp = self._dr_rate_kp_nominal.view(1, 3).repeat(self.num_envs, 1)
        self._dr_rate_ki = self._dr_rate_ki_nominal.view(1, 3).repeat(self.num_envs, 1)
        self._dr_rate_kd = self._dr_rate_kd_nominal.view(1, 3).repeat(self.num_envs, 1)
        self._dr_mass_inertia_initialized = False
        self._log_domain_randomization_config()
        self._maybe_init_mass_inertia_randomization()

    def _resolve_rate_gain_nominal(
        self,
        key: str,
        legacy_keys: tuple[str, str, str],
        default: list,
    ) -> torch.Tensor:
        params = self._pid_params or {}
        if key in params:
            values = params[key]
        elif any(legacy in params for legacy in legacy_keys):
            values = [params.get(legacy, default[idx]) for idx, legacy in enumerate(legacy_keys)]
        else:
            values = default
        return torch.as_tensor(values, device=self.device, dtype=torch.float32)

    def _dr_target_agent(self) -> Literal["pursuer", "evader"] | None:
        if self.cfg.training_agent == "pursuer":
            return "pursuer"
        if self.cfg.training_agent == "evader":
            return "evader"
        return None

    def _log_domain_randomization_config(self) -> None:
        def _emit(message: str) -> None:
            carb.log_info(message)

        if not self._dr_cfg.enable:
            _emit("[PursuitEvasion][DR] Domain randomization inactive.")
            return

        target = self._dr_target_agent()
        if target is None:
            _emit("[PursuitEvasion][DR] Enabled but training_agent is empty; no agent randomized.")
            return

        nominal = self._intrinsics_nominal.detach().to("cpu").tolist()
        pairs = ", ".join(f"{name}={value:.6g}" for name, value in zip(self._intrinsics_names, nominal))
        _emit(f"[PursuitEvasion][DR] Enabled for {target}. Nominal intrinsics: {pairs}")
        _emit(
            "[PursuitEvasion][DR] Scales: "
            f"mass/inertia/k_eta/k_m/tau={self._dr_cfg.scale_min:.2f}-{self._dr_cfg.scale_max:.2f}, "
            f"k_aero_xy={self._dr_cfg.k_aero_xy_min_scale:.2f}-{self._dr_cfg.k_aero_xy_max_scale:.2f}, "
            f"k_aero_z={self._dr_cfg.k_aero_z_min_scale:.2f}-{self._dr_cfg.k_aero_z_max_scale:.2f}, "
            f"rate_kp={self._dr_cfg.rate_kp_min_scale:.2f}-{self._dr_cfg.rate_kp_max_scale:.2f}, "
            f"rate_ki={self._dr_cfg.rate_ki_min_scale:.2f}-{self._dr_cfg.rate_ki_max_scale:.2f}, "
            f"rate_kd={self._dr_cfg.rate_kd_min_scale:.2f}-{self._dr_cfg.rate_kd_max_scale:.2f}"
        )

    def _maybe_init_mass_inertia_randomization(self) -> None:
        if self._dr_mass_inertia_initialized:
            return
        if not self._dr_cfg.enable:
            return
        if not (self._dr_cfg.randomize_mass or self._dr_cfg.randomize_inertia):
            return
        target = self._dr_target_agent()
        if target is None:
            return

        n = self.num_envs
        device = self.device
        scale_min = float(self._dr_cfg.scale_min)
        scale_max = float(self._dr_cfg.scale_max)

        mass_scale = torch.ones(n, device=device)
        inertia_scale = torch.ones(n, device=device)
        if self._dr_cfg.randomize_mass:
            mass_scale = torch.empty(n, device=device).uniform_(scale_min, scale_max)
        if self._dr_cfg.randomize_inertia:
            inertia_scale = torch.empty(n, device=device).uniform_(scale_min, scale_max)

        mass = mass_scale * self._dr_nominal_mass
        inertia = inertia_scale.view(-1, 1) * self._dr_nominal_inertia.view(1, 3)
        self._dr_mass = mass
        self._dr_inertia = inertia

        env_ids = torch.arange(n, device=device, dtype=torch.long)
        if target == "pursuer":
            self._apply_mass_inertia(self._pursuer, "pursuer", env_ids, mass_scale, inertia_scale)
        elif target == "evader":
            self._apply_mass_inertia(self._evader, "evader", env_ids, mass_scale, inertia_scale)

        self._intrinsics[:, 0] = mass
        self._intrinsics[:, 1:4] = inertia
        self._dr_mass_inertia_initialized = True

    def _spawn_fpv_cameras(self) -> None:
        """Create and register FPV cameras on the pursuer and evader bodies using the shared config."""
        if self._camera_cfg is not None:
            cam_cfg = copy.deepcopy(self._camera_cfg)
        else:
            cam_cfg = self._build_fpv_camera_cfg(self.cfg)
        if cam_cfg is None:
            return

        def _make(name: str, prim_path: str):
            cfg_copy = copy.deepcopy(cam_cfg)
            cfg_copy.prim_path = prim_path
            # Tiled camera batches per-env render products to reduce overhead.
            self.scene.sensors[name] = TiledCamera(cfg_copy)

        base_regex = self.scene.env_regex_ns
        _make("pursuer_camera", f"{base_regex}/Pursuer/body/fpv_camera")
        if self.cfg.enable_evader_cameras:
            _make("evader_camera", f"{base_regex}/Evader/body/fpv_camera")

    def _apply_semantic_labels(self) -> None:
        """Apply semantic labels to drone prims for segmentation cameras.

        Without explicit labels, drone meshes are invisible in the
        ``semantic_segmentation`` annotator output. We label the evader
        (and optionally the pursuer) root prims so all child meshes
        inherit the class.
        """
        import isaacsim.core.utils.prims as prim_utils

        _label_method_logged = False

        def _label_prim(prim_path: str, label: str) -> None:
            nonlocal _label_method_logged
            try:
                import omni.replicator.core as rep
                rep.modify.semantics([("class", label)], prim_path)
                if not _label_method_logged:
                    carb.log_info("[PursuitEvasion] Semantic labels applied via omni.replicator.core")
                    _label_method_logged = True
                return
            except Exception:
                pass
            try:
                from omni.isaac.core.utils.semantics import add_update_semantics
                prim = prim_utils.get_prim_at_path(prim_path)
                if prim.IsValid():
                    add_update_semantics(prim, label, "class")
                    if not _label_method_logged:
                        carb.log_info("[PursuitEvasion] Semantic labels applied via isaac.core.utils.semantics")
                        _label_method_logged = True
                return
            except Exception:
                pass
            try:
                from pxr import Semantics
                prim = prim_utils.get_prim_at_path(prim_path)
                if prim.IsValid():
                    if not prim.HasAPI(Semantics.SemanticsAPI):
                        Semantics.SemanticsAPI.Apply(prim, "Semantics")
                    sem = Semantics.SemanticsAPI.Get(prim, "Semantics")
                    sem.CreateSemanticTypeAttr().Set("class")
                    sem.CreateSemanticDataAttr().Set(label)
                    if not _label_method_logged:
                        carb.log_info("[PursuitEvasion] Semantic labels applied via pxr.Semantics")
                        _label_method_logged = True
            except Exception as e:
                carb.log_warn(f"[PursuitEvasion] Could not set semantic label on {prim_path}: {e}")

        # Label only the *opponent* for each camera so the agent doesn't
        # segment its own body (propeller arms at frame edges).
        base = self.scene.env_prim_paths[0].rsplit("/", 1)[0]  # e.g. /World/envs
        for i in range(self.scene.cfg.num_envs):
            env_path = f"{base}/env_{i}"
            # Pursuer camera should see evader, not itself
            _label_prim(f"{env_path}/Evader", "evader")
            # Evader camera (if enabled) should see pursuer, not itself
            if self.cfg.enable_evader_cameras:
                _label_prim(f"{env_path}/Pursuer", "pursuer")

    def _spawn_occlusion_walls(self) -> None:
        """Spawn lightweight visual-only walls around each environment to block cross-talk between cameras.

        IMPORTANT: Walls are spawned *under the env prim*, so translations must be in *env-local* coordinates.
        Do NOT add env_origins again, otherwise walls get double-offset from the arena.
        """
        span_x = (self.cfg.arena_max[0] - self.cfg.arena_min[0]) + 2 * self.cfg.wall_extra_margin
        span_y = (self.cfg.arena_max[1] - self.cfg.arena_min[1]) + 2 * self.cfg.wall_extra_margin
        height = self.cfg.arena_max[2] - self.cfg.arena_min[2]
        thickness = self.cfg.wall_thickness
        half_thickness = 0.5 * thickness

        # Center of arena in LOCAL env coordinates
        center_z_local = 0.5 * (self.cfg.arena_max[2] + self.cfg.arena_min[2])

        # Wall planes in LOCAL env coordinates (centered so inner faces match the requested clearance).
        x_pos_local = self.cfg.arena_max[0] + self.cfg.wall_extra_margin + half_thickness
        x_neg_local = self.cfg.arena_min[0] - self.cfg.wall_extra_margin - half_thickness
        y_pos_local = self.cfg.arena_max[1] + self.cfg.wall_extra_margin + half_thickness
        y_neg_local = self.cfg.arena_min[1] - self.cfg.wall_extra_margin - half_thickness

        # Brighter wall material (mocap room style - light gray with some reflectivity)
        wall_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.75, 0.75, 0.75),
            roughness=0.3,
            metallic=0.05,
        )

        wall_x_cfg = sim_utils.CuboidCfg(
            size=(thickness, span_y, height),
            visual_material=wall_material,
            copy_from_source=False,
        )
        wall_y_cfg = sim_utils.CuboidCfg(
            size=(span_x, thickness, height),
            visual_material=wall_material,
            copy_from_source=False,
        )

        for env_id in range(self.scene.cfg.num_envs):
            base = f"/World/envs/env_{env_id}/Walls"
            placements = [
                (f"{base}/WallXPos", (x_pos_local, 0.0, center_z_local)),
                (f"{base}/WallXNeg", (x_neg_local, 0.0, center_z_local)),
                (f"{base}/WallYPos", (0.0, y_pos_local, center_z_local)),
                (f"{base}/WallYNeg", (0.0, y_neg_local, center_z_local)),
            ]

            for path, translation in placements:
                if prim_utils.is_prim_path_valid(path):
                    continue
                cfg = wall_x_cfg if "WallX" in path else wall_y_cfg
                cfg.func(path, cfg, translation=translation)

    def _spawn_roof(self) -> None:
        """Spawn a lightweight transparent roof above each environment for bounded camera depth."""
        span_x = (
            (self.cfg.arena_max[0] - self.cfg.arena_min[0])
            + 2 * self.cfg.wall_extra_margin
            + 2 * self.cfg.wall_thickness
        )
        span_y = (
            (self.cfg.arena_max[1] - self.cfg.arena_min[1])
            + 2 * self.cfg.wall_extra_margin
            + 2 * self.cfg.wall_thickness
        )
        thickness = self.cfg.roof_thickness
        half_thickness = 0.5 * thickness
        roof_z_local = self.cfg.arena_max[2] + self.cfg.roof_height_offset + half_thickness

        roof_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.8, 0.8, 0.8),
            opacity=self.cfg.roof_opacity,
            roughness=0.4,
        )
        roof_cfg = sim_utils.CuboidCfg(
            size=(span_x, span_y, thickness),
            visual_material=roof_material,
            copy_from_source=False,
        )

        for env_id in range(self.scene.cfg.num_envs):
            path = f"/World/envs/env_{env_id}/Roof"
            if prim_utils.is_prim_path_valid(path):
                continue
            roof_cfg.func(path, roof_cfg, translation=(0.0, 0.0, roof_z_local))

    def _spawn_obstacle_wall(self) -> None:
        """Spawn a single wall along the Y axis at x=0, with gaps at ±y ends.

        Layout (top-down, arena 5m × 4m)::

            y_max +------ gap ------+
                  |                 |
                  |    =========    |   wall at x=0, spans y
                  |                 |
            y_min +------ gap ------+

        The wall breaks line-of-sight across the arena while leaving wide
        passages at both ends for the drones to fly around.
        Walls are visual-only — collision is checked in software.
        """
        cfg = self.cfg
        thickness = cfg.obstacle_wall_thickness
        height = cfg.obstacle_wall_height
        gap = cfg.obstacle_gap_size
        half_gap = gap / 2.0
        half_thickness = thickness / 2.0

        x_min, y_min, z_min = cfg.arena_min
        x_max, y_max, z_max = cfg.arena_max

        center_z = height / 2.0

        # Wall runs along Y at x=0, with gap at ±y ends
        wall_y_start = y_min + half_gap
        wall_y_end = y_max - half_gap
        wall_length = wall_y_end - wall_y_start
        wall_center_y = (wall_y_start + wall_y_end) / 2.0

        wall_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.5, 0.5, 0.55),
            roughness=0.6,
            metallic=0.05,
        )

        wall_cfg = sim_utils.CuboidCfg(
            size=(thickness, wall_length, height),
            visual_material=wall_material,
            copy_from_source=False,
        )

        for env_id in range(self.scene.cfg.num_envs):
            path = f"/World/envs/env_{env_id}/ObstacleWall"
            if prim_utils.is_prim_path_valid(path):
                continue
            wall_cfg.func(path, wall_cfg, translation=(0.0, wall_center_y, center_z))

        # Store wall geometry for software collision checks
        # Wall occupies: x in [-half_thickness, +half_thickness], y in [wall_y_start, wall_y_end]
        self._wall_y_range = (wall_y_start, wall_y_end)
        self._wall_half_thickness = half_thickness

    def _check_wall_collision(self, positions: torch.Tensor) -> torch.Tensor:
        """Check if positions collide with the obstacle wall.

        Returns a boolean mask of shape (num_envs,) indicating collision.
        """
        clearance = self.cfg.obstacle_drone_clearance
        half_t = self._wall_half_thickness + clearance
        x = positions[:, 0]
        y = positions[:, 1]

        y_lo, y_hi = self._wall_y_range

        # Wall at x=0: within y span AND close to x=0
        return (y >= y_lo) & (y <= y_hi) & (x.abs() < half_t)

    def _wall_occludes_los(self, p_a: torch.Tensor, p_b: torch.Tensor) -> torch.Tensor:
        """Boolean mask: True where the line segment p_a→p_b is blocked by the wall.

        Both arguments are env-local 3D positions (the wall is at x=0 spanning
        ``self._wall_y_range`` in env-local coords). Returns all-False when
        obstacles are disabled. Used to make camera visibility correct in the
        wall arena — without this, the actor's ``visible`` flag reports True
        when the wall hides the target.
        """
        if not self.cfg.enable_obstacles:
            return torch.zeros(p_a.shape[0], dtype=torch.bool, device=p_a.device)

        pax, pay = p_a[:, 0], p_a[:, 1]
        pbx, pby = p_b[:, 0], p_b[:, 1]
        dx = pbx - pax
        safe_dx = torch.where(dx.abs() < 1e-6, torch.ones_like(dx), dx)
        t = -pax / safe_dx
        crosses_plane = (t > 0.0) & (t < 1.0)
        y_at_x0 = pay + t * (pby - pay)
        y_lo, y_hi = self._wall_y_range
        return crosses_plane & (y_at_x0 >= y_lo) & (y_at_x0 <= y_hi)

    def _wall_geodesic_distance(self, p: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Wall-aware shortest-path distance between two env-local positions.

        When the straight segment p→e crosses the wall body (wall along Y at x=0,
        spanning ``self._wall_y_range``), route through the closer y-edge of the
        wall (top or bottom gap). Otherwise return the Euclidean distance.

        Used by the approach shaping reward so it does not pull the pursuer
        through the wall when the evader is on the opposite side.
        """
        euclid = torch.norm(e - p, dim=-1)
        if not self.cfg.enable_obstacles:
            return euclid

        px, py, pz = p[:, 0], p[:, 1], p[:, 2]
        ex, ey, ez = e[:, 0], e[:, 1], e[:, 2]

        y_lo, y_hi = self._wall_y_range
        margin = self._wall_half_thickness + self.cfg.obstacle_drone_clearance

        # Parameter t at which the line p→e crosses x=0 (only meaningful when sign(px)≠sign(ex)).
        dx = ex - px
        safe_dx = torch.where(dx.abs() < 1e-6, torch.ones_like(dx), dx)
        t = -px / safe_dx
        crosses_plane = (t > 0.0) & (t < 1.0)
        y_at_x0 = py + t * (ey - py)
        hits_wall_body = crosses_plane & (y_at_x0 >= y_lo) & (y_at_x0 <= y_hi)

        # Route through the gap edge points just outside each y-end of the wall.
        z_mid = 0.5 * (pz + ez)
        zeros = torch.zeros_like(pz)
        gap_top = torch.stack([zeros, torch.full_like(py, y_hi + margin), z_mid], dim=-1)
        gap_bot = torch.stack([zeros, torch.full_like(py, y_lo - margin), z_mid], dim=-1)
        via_top = torch.norm(p - gap_top, dim=-1) + torch.norm(gap_top - e, dim=-1)
        via_bot = torch.norm(p - gap_bot, dim=-1) + torch.norm(gap_bot - e, dim=-1)
        geodesic = torch.minimum(via_top, via_bot)
        return torch.where(hits_wall_body, geodesic, euclid)

    def _apply_domain_randomization(self, env_ids: torch.Tensor) -> None:
        if not self._dr_cfg.enable:
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        target = self._dr_target_agent()
        if target is None:
            return

        n = env_ids.shape[0]
        device = self.device
        scale_min = float(self._dr_cfg.scale_min)
        scale_max = float(self._dr_cfg.scale_max)

        def _sample_scale(enabled: bool) -> torch.Tensor:
            if not enabled:
                return torch.ones(n, device=device)
            return torch.empty(n, device=device).uniform_(scale_min, scale_max)

        k_eta_scale = _sample_scale(self._dr_cfg.randomize_k_eta)
        k_m_scale = _sample_scale(self._dr_cfg.randomize_k_m)
        tau_scale = _sample_scale(self._dr_cfg.randomize_tau)

        mass = self._dr_mass[env_ids]
        inertia = self._dr_inertia[env_ids]
        k_eta = k_eta_scale * self._dr_nominal_k_eta
        k_m = k_m_scale * self._dr_nominal_k_m
        tau_m = tau_scale * self._dr_nominal_tau

        if self._dr_cfg.randomize_k_aero:
            k_aero_xy = torch.empty(n, device=device).uniform_(
                self._dr_nominal_k_aero_xy * self._dr_cfg.k_aero_xy_min_scale,
                self._dr_nominal_k_aero_xy * self._dr_cfg.k_aero_xy_max_scale,
            )
            k_aero_z = torch.empty(n, device=device).uniform_(
                self._dr_nominal_k_aero_z * self._dr_cfg.k_aero_z_min_scale,
                self._dr_nominal_k_aero_z * self._dr_cfg.k_aero_z_max_scale,
            )
        else:
            k_aero_xy = torch.full((n,), self._dr_nominal_k_aero_xy, device=device)
            k_aero_z = torch.full((n,), self._dr_nominal_k_aero_z, device=device)

        if self._dr_cfg.randomize_rate_gains:
            kp_rp = (
                torch.empty(n, device=device).uniform_(self._dr_cfg.rate_kp_min_scale, self._dr_cfg.rate_kp_max_scale)
                * self._dr_rate_kp_nominal[0]
            )
            kp_y = (
                torch.empty(n, device=device).uniform_(self._dr_cfg.rate_kp_min_scale, self._dr_cfg.rate_kp_max_scale)
                * self._dr_rate_kp_nominal[2]
            )

            ki_rp = (
                torch.empty(n, device=device).uniform_(self._dr_cfg.rate_ki_min_scale, self._dr_cfg.rate_ki_max_scale)
                * self._dr_rate_ki_nominal[0]
            )
            ki_y = (
                torch.empty(n, device=device).uniform_(self._dr_cfg.rate_ki_min_scale, self._dr_cfg.rate_ki_max_scale)
                * self._dr_rate_ki_nominal[2]
            )

            kd_rp = (
                torch.empty(n, device=device).uniform_(self._dr_cfg.rate_kd_min_scale, self._dr_cfg.rate_kd_max_scale)
                * self._dr_rate_kd_nominal[0]
            )
            kd_y = (
                torch.empty(n, device=device).uniform_(self._dr_cfg.rate_kd_min_scale, self._dr_cfg.rate_kd_max_scale)
                * self._dr_rate_kd_nominal[2]
            )
        else:
            kp_rp = self._dr_rate_kp_nominal[0].expand(n)
            kp_y = self._dr_rate_kp_nominal[2].expand(n)
            ki_rp = self._dr_rate_ki_nominal[0].expand(n)
            ki_y = self._dr_rate_ki_nominal[2].expand(n)
            kd_rp = self._dr_rate_kd_nominal[0].expand(n)
            kd_y = self._dr_rate_kd_nominal[2].expand(n)

        rate_kp = torch.stack([kp_rp, kp_rp, kp_y], dim=1)
        rate_ki = torch.stack([ki_rp, ki_rp, ki_y], dim=1)
        rate_kd = torch.stack([kd_rp, kd_rp, kd_y], dim=1)

        self._dr_k_eta[env_ids] = k_eta
        self._dr_k_m[env_ids] = k_m
        self._dr_tau[env_ids] = tau_m
        self._dr_k_aero_xy[env_ids] = k_aero_xy
        self._dr_k_aero_z[env_ids] = k_aero_z
        self._dr_rate_kp[env_ids] = rate_kp
        self._dr_rate_ki[env_ids] = rate_ki
        self._dr_rate_kd[env_ids] = rate_kd

        intrinsics_parts = [
            mass.view(-1, 1),
            inertia,
            k_eta.view(-1, 1),
            k_m.view(-1, 1),
            tau_m.view(-1, 1),
            k_aero_xy.view(-1, 1),
            k_aero_z.view(-1, 1),
            rate_kp,
            rate_ki,
            rate_kd,
        ]
        intrinsics = torch.cat(intrinsics_parts, dim=-1)
        self._intrinsics[env_ids] = intrinsics

        k_aero = torch.stack([k_aero_xy, k_aero_xy, k_aero_z], dim=1)
        if target == "pursuer":
            self._pursuer_propellers.set_params(env_ids, k_eta=k_eta, k_m=k_m, tau_m=tau_m, k_aero=k_aero)
        elif target == "evader":
            self._evader_propellers.set_params(env_ids, k_eta=k_eta, k_m=k_m, tau_m=tau_m, k_aero=k_aero)

        self._apply_rate_gains(env_ids, target=target)

    def _apply_mass_inertia(
        self,
        robot: ArticulationData | Any,
        label: Literal["pursuer", "evader"],
        env_ids: torch.Tensor,
        mass_scale: torch.Tensor,
        inertia_scale: torch.Tensor,
    ) -> None:
        env_ids_cpu = env_ids.to(device="cpu", dtype=torch.int)
        mass_scale_cpu = mass_scale.detach().to("cpu").view(-1, 1)
        inertia_scale_cpu = inertia_scale.detach().to("cpu").view(-1, 1, 1)
        masses = self._dr_masses[label]
        inertias = self._dr_inertias[label]
        masses[env_ids_cpu] = self._dr_default_masses[label][env_ids_cpu] * mass_scale_cpu
        inertias[env_ids_cpu] = self._dr_default_inertias[label][env_ids_cpu] * inertia_scale_cpu
        # NOTE: controllers keep nominal mass/inertia; physics uses randomized values.
        robot.root_physx_view.set_masses(masses, env_ids_cpu)
        robot.root_physx_view.set_inertias(inertias, env_ids_cpu)

    def _build_fpv_camera_cfg(self, cfg: PursuitEvasionEnvCfg):
        """Build a drone-specific FPV camera config with safe overrides."""
        fn = self._drone_spec.fpv_camera_cfg_fn
        if fn is None:
            return None
        import inspect

        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            sig = None

        kwargs = {"tiled": True}
        override_map = {
            "camera_width": "width",
            "camera_height": "height",
            "camera_fx": "fx",
            "camera_fy": "fy",
            "camera_tilt_deg": "tilt_deg",
        }
        for cfg_key, arg in override_map.items():
            if hasattr(cfg, cfg_key):
                kwargs[arg] = getattr(cfg, cfg_key)

        if sig is not None:
            supported = sig.parameters.keys()
            kwargs = {k: v for k, v in kwargs.items() if k in supported}
            if "width" in kwargs and "height" in kwargs:
                if "cx" in supported and "cx" not in kwargs:
                    kwargs["cx"] = kwargs["width"] / 2.0
                if "cy" in supported and "cy" not in kwargs:
                    kwargs["cy"] = kwargs["height"] / 2.0

        try:
            cam_cfg = fn(**kwargs)
        except TypeError:
            # Fallback to minimal invocation for non-standard signatures
            cam_cfg = fn(tiled=True)
        return cam_cfg

    def _apply_rate_gains(
        self,
        env_ids: torch.Tensor,
        target: Literal["pursuer", "evader"] | None = None,
    ) -> None:
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        rate_kp = self._dr_rate_kp
        rate_ki = self._dr_rate_ki
        rate_kd = self._dr_rate_kd
        targets: list[str] = []
        if target is None or target == "pursuer":
            targets.append("pursuer")
        if target is None or target == "evader":
            targets.append("evader")

        for agent in targets:
            wrapper = self._training_wrappers.get(agent)
            if wrapper is not None:
                pid = getattr(wrapper, "pid", None)
                if pid is not None and hasattr(pid, "set_rate_gains"):
                    pid.set_rate_gains(
                        rate_kp=rate_kp[env_ids],
                        rate_ki=rate_ki[env_ids],
                        rate_kd=rate_kd[env_ids],
                        env_ids=env_ids,
                    )

        manager_pairs = []
        if target is None or target == "pursuer":
            manager_pairs.append(self.pursuer_manager)
        if target is None or target == "evader":
            manager_pairs.append(self.evader_manager)
        for manager in manager_pairs:
            if manager is None:
                continue
            if hasattr(manager, "set_rate_gains"):
                manager.set_rate_gains(env_ids, rate_kp, rate_ki, rate_kd)

    # -------------------------------------------------------------------------
    # IsaacLab interface implementation
    # -------------------------------------------------------------------------
    @carb.profiler.profile
    def _setup_scene(self) -> None:
        """Set up the scene with two quadrotors."""
        from isaaclab.assets import Articulation

        # Create pursuer and evader
        self._pursuer = Articulation(self.cfg.pursuer_robot)
        self.scene.articulations["pursuer"] = self._pursuer
        if not self._use_visual_ball_evader:
            self._evader = Articulation(self.cfg.evader_robot)
            self.scene.articulations["evader"] = self._evader
        else:
            self._evader = None
            self._visual_ball_evader = None

        # Cameras (spawn before cloning so they replicate to all envs)
        if self.cfg.enable_cameras:
            self._spawn_fpv_cameras()

        # Terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        if self.cfg.enable_occlusion_walls:
            self._spawn_occlusion_walls()
        if self.cfg.enable_roof:
            self._spawn_roof()
        if self.cfg.enable_obstacles:
            self._spawn_obstacle_wall()

        if self.cfg.enable_cameras:
            self._pursuer_camera = self.scene.sensors.get("pursuer_camera")
            self._evader_camera = self.scene.sensors.get("evader_camera") if self.cfg.enable_evader_cameras else None

        # Add semantic labels to drone prims so they appear in segmentation maps.
        # The visual ball evader handles its own labels; real drone meshes need
        # explicit annotation since the USD files don't include them.
        if self.cfg.enable_cameras and (self.cfg.obs_include_segmap or self.cfg.obs_include_depth):
            self._apply_semantic_labels()

        if self._use_visual_ball_evader:
            device = torch.device(self.cfg.sim.device)
            arena_min = torch.tensor(self.cfg.arena_min, device=device, dtype=torch.float32)
            arena_max = torch.tensor(self.cfg.arena_max, device=device, dtype=torch.float32)
            arena_bounds = torch.stack([arena_min, arena_max], dim=1)
            dt = float(self.cfg.sim.dt) * float(self.cfg.decimation)
            horizon = max(1, int(round(self.cfg.episode_length_s / dt)))
            trajectory_groups = self._build_visual_ball_trajectory_groups()
            self._visual_ball_evader = VisualBallEvader(
                num_envs=self.scene.cfg.num_envs,
                device=device,
                arena_bounds=arena_bounds,
                env_origins=self._terrain.env_origins.to(device),
                trajectory_groups=trajectory_groups,
                trajectory_horizon=horizon,
                dt=dt,
                radius=self.cfg.visual_ball_radius,
                color=self.cfg.visual_ball_color,
                wall_cfg=self._wall_cfg_for_trajectories,
            )
            self._visual_ball_evader.create_markers("/Visuals/PursuitEvasion/EvaderBall")
            self._evader = self._visual_ball_evader

        # Lighting - bright mocap room-like setup
        dome_light_cfg = sim_utils.DomeLightCfg(intensity=4000.0, color=(1.0, 1.0, 1.0))
        dome_light_cfg.func("/World/DomeLight", dome_light_cfg)

        # Add directional light for better shadows and depth perception
        distant_light_cfg = sim_utils.DistantLightCfg(
            intensity=2000.0,
            color=(1.0, 0.98, 0.95),
            angle=0.5,
        )
        distant_light_cfg.func("/World/DistantLight", distant_light_cfg, translation=(0, 0, 10))

    def _maybe_create_manager(
        self,
        agent: Literal["pursuer", "evader"],
        assignment: dict[str, dict],
        dt_ctrl: float,
    ) -> QuadrotorManager | None:
        if not assignment:
            return None
        if agent == "evader" and self._use_visual_ball_evader:
            return None

        if agent == self.cfg.training_agent:
            return None

        robot = self._pursuer if agent == "pursuer" else self._evader
        obs_dim = self._compute_obs_tensor(agent).shape[-1]
        return QuadrotorManager(
            robot=robot,
            drone_cfg=self._drone_cfg,
            dt=dt_ctrl,
            num_envs=self.num_envs,
            total_timesteps=self.cfg.total_timesteps,
            controller_assignment=assignment,
            device=self.device,
            role=agent,
            arena_bounds=self._arena_bounds,
            env_origins=self._terrain.env_origins,
            obs_dim=obs_dim,
            action_dim=self.cfg.action_space,
            pid_params=self._pid_params,
            pid_dt=self.sim.cfg.dt,
            wall_cfg=self._wall_cfg_for_trajectories,
        )

    @staticmethod
    def _build_wall_cfg_for_trajectories(cfg) -> WallConfig | None:
        """Build a WallConfig describing the obstacle wall, or None when no wall exists.

        Takes `cfg` explicitly and is static because it must run *before*
        `super().__init__()` — which is what sets `self.cfg` — since
        `_setup_scene()` consumes the result. Geometry must stay in sync with
        `_spawn_obstacle_wall`.
        """
        if not cfg.enable_obstacles:
            return None
        thickness = float(cfg.obstacle_wall_thickness)
        gap = float(cfg.obstacle_gap_size)
        half_gap = gap / 2.0
        _, y_min, _ = cfg.arena_min
        _, y_max, _ = cfg.arena_max
        wall_y_start = float(y_min) + half_gap
        wall_y_end = float(y_max) - half_gap
        return WallConfig(
            half_thickness=thickness / 2.0,
            y_range=(wall_y_start, wall_y_end),
            clearance=float(cfg.obstacle_drone_clearance),
        )

    def _build_training_action_wrapper(self, agent: Literal["pursuer", "evader"]):
        """Create the action-to-omega wrapper for the training agent."""
        if self.cfg.training_agent != agent:
            return None
        mode = self._action_mode(agent)
        dt = self.sim.cfg.dt * self.cfg.decimation

        # add tensordict kind of type to td?
        def passthrough(td):
            return td.get("rl_action")

        if mode == "velocity":
            return CrazyflieRLVelocityWrapper(
                num_envs=self.num_envs,
                drone_cfg=self._drone_cfg,
                policy=passthrough,
                dt=dt,
                pid_dt=self.sim.cfg.dt,
                device=self.device,
                action_key="rl_action",
                root_state_key="root_state",
                pid_params=self._pid_params,
            )
        elif mode == "body_rates":
            return CrazyflieRLBodyRatesWrapper(
                num_envs=self.num_envs,
                drone_cfg=self._drone_cfg,
                policy=passthrough,
                dt=dt,
                pid_dt=self.sim.cfg.dt,
                device=self.device,
                action_key="rl_action",
                root_state_key="root_state",
                body_rate_key="body_rate",
                pid_params=self._pid_params,
            )
        else:
            raise ValueError(f"Unsupported action mode '{mode}' for agent '{agent}'.")

    def _build_training_tensordict(self, agent: Literal["pursuer", "evader"], rl_action: torch.Tensor) -> TensorDict:
        robot = self._pursuer if agent == "pursuer" else self._evader
        # Minimal state used by the training wrappers.
        return TensorDict(
            {
                "rl_action": rl_action,
                "root_state": robot.data.root_state_w,
                "body_rate": robot.data.root_ang_vel_b,
            },
            batch_size=[self.num_envs],
            device=self.device,
        )

    def _compute_manager_actions(
        self,
        manager: QuadrotorManager | None,
        adversary_data: ArticulationData | None,
        rl_observations: dict[str, TensorDict],
    ) -> torch.Tensor:
        if manager is None:
            return torch.zeros((self.num_envs, 4), device=self.device)
        return manager.compute_action(adversary_data=adversary_data, rl_observations=rl_observations)

    def _build_rl_observations(self, agent: Literal["pursuer", "evader"]) -> dict[str, TensorDict]:
        manager = self._agent_managers.get(agent)
        if manager is None:
            return {}

        requests = manager.rl_env_assignments()
        if not requests:
            return {}
        robot = self._pursuer if agent == "pursuer" else self._evader

        # Recurrent vision opponents (CNN+GRU) need ``image`` and ``past_actions``
        # in the TD. Compute the opponent's FPV image once (it's the same for
        # every controller assigned to this role). Skip when cameras aren't
        # available for that role — the recurrent loader will then error out
        # with a clear message, but state-based loaded opponents still work.
        opp_image: torch.Tensor | None = None
        if self._use_image_obs:
            camera_available = (
                (agent == "pursuer") or self.cfg.enable_evader_cameras
            )
            if camera_available:
                try:
                    # update_history=False so opponent reads don't corrupt
                    # the training agent's image history buffer.
                    opp_image = self._get_image_observations(agent, update_history=False)
                except Exception:
                    opp_image = None
        # Per-env opponent past-actions buffer (flattened to (N, n_past*action_dim))
        opp_past = self._opp_past_actions.get(agent)

        result: dict[str, TensorDict] = {}
        for name, info in requests.items():
            env_ids = info["env_ids"]
            obs_tensor = self._compute_obs_tensor(agent, env_ids)
            td_dict: dict[str, torch.Tensor] = {
                "observation": obs_tensor,
                "root_state": robot.data.root_state_w[env_ids],
                "body_rate": robot.data.root_ang_vel_b[env_ids],
            }
            # Defensive indexing: ensure env_ids is on the same device and
            # is a long tensor before slicing. Silent device/dtype mismatch
            # would produce uninitialized memory reads → NaN through the
            # loaded recurrent policy's CNN → physics blow-up at step 1.
            if opp_image is not None:
                idx = env_ids.to(opp_image.device).long()
                td_dict["image"] = opp_image.index_select(0, idx).contiguous()
            if opp_past is not None:
                idx = env_ids.to(opp_past.device).long()
                td_dict["past_actions"] = opp_past.index_select(0, idx).view(idx.shape[0], -1).contiguous()
            td = TensorDict(
                td_dict,
                batch_size=[env_ids.shape[0]],
                device=self.device,
            )
            result[name] = td
        return result

    def _apply_action(self) -> None:
        """Apply forces and torques to both quadrotors."""
        if self.cfg.training_agent == "pursuer":
            wrapper = self._training_wrappers.get("pursuer")
            if wrapper is None:
                raise RuntimeError("No action wrapper configured for pursuer training.")
            self._pursuer_wrench = wrapper.wrench_from_command(self._pursuer.data.root_state_w, self._pursuer_actions)
        else:
            if self.pursuer_manager is not None:
                self._pursuer_wrench = self.pursuer_manager.compute_wrench(self._pursuer_actions)
            else:
                self._pursuer_wrench.zero_()

        if self.cfg.training_agent == "evader":
            if self._use_visual_ball_evader:
                raise RuntimeError("Training the evader is not supported with visual_ball_evader enabled.")
            wrapper = self._training_wrappers.get("evader")
            if wrapper is None:
                raise RuntimeError("No action wrapper configured for evader training.")
            self._evader_wrench = wrapper.wrench_from_command(self._evader.data.root_state_w, self._evader_actions)
        else:
            if self._use_visual_ball_evader:
                self._evader_wrench.zero_()
            elif self.evader_manager is not None:
                self._evader_wrench = self.evader_manager.compute_wrench(self._evader_actions)
            else:
                self._evader_wrench.zero_()

        self._pursuer_omega_ref = self._pursuer_propellers.compute_motor_speeds_from_wrench(self._pursuer_wrench)
        if self._use_visual_ball_evader:
            self._evader_omega_ref.zero_()
        else:
            self._evader_omega_ref = self._evader_propellers.compute_motor_speeds_from_wrench(self._evader_wrench)

        vel_pursuer_b = self._pursuer.data.root_lin_vel_b
        state_stub_pursuer = torch.zeros(self.num_envs, 6, device=self.device)
        state_stub_pursuer[:, 3:6] = vel_pursuer_b
        self._pursuer_propellers.compute_omega(self._pursuer_omega_ref)
        self._pursuer_thrust, self._pursuer_moment = self._pursuer_propellers.compute_force_and_torque(
            state_stub_pursuer
        )

        if self._use_visual_ball_evader:
            self._evader_thrust.zero_()
            self._evader_moment.zero_()
        else:
            vel_evader_b = self._evader.data.root_lin_vel_b
            state_stub_evader = torch.zeros(self.num_envs, 6, device=self.device)
            state_stub_evader[:, 3:6] = vel_evader_b
            self._evader_propellers.compute_omega(self._evader_omega_ref)
            self._evader_thrust, self._evader_moment = self._evader_propellers.compute_force_and_torque(
                state_stub_evader
            )

        self._pursuer.set_external_force_and_torque(
            self._pursuer_thrust, self._pursuer_moment, body_ids=self._pursuer_body_id
        )
        if not self._use_visual_ball_evader:
            self._evader.set_external_force_and_torque(
                self._evader_thrust, self._evader_moment, body_ids=self._evader_body_id
            )
        self._update_prop_visuals()

    def _find_prop_joints(self, drone) -> list[int]:
        import re

        joint_ids, joint_names = None, None
        # Try drone-specific joint naming conventions, then generic fallbacks
        patterns_to_try = list(self._drone_spec.prop_joint_patterns)
        for pattern in patterns_to_try:
            try:
                joint_ids, joint_names = drone.find_joints(pattern, preserve_order=True)
                if joint_ids:
                    break
            except ValueError:
                continue
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
        def _write(drone, joint_ids, omega):
            if not joint_ids:
                return
            count = min(len(joint_ids), omega.shape[1])
            vis = omega[:, :count].clone()
            if count > 1:
                vis[:, 0::2] *= -1.0
            drone.write_joint_velocity_to_sim(vis, joint_ids=joint_ids[:count])

        _write(self._pursuer, self._pursuer_prop_joint_ids, self._pursuer_propellers.omega)
        if not self._use_visual_ball_evader and self._evader_propellers is not None:
            _write(self._evader, self._evader_prop_joint_ids, self._evader_propellers.omega)

    def _agent_data(self, agent: Literal["pursuer", "evader"]):
        return self._pursuer if agent == "pursuer" else self._evader

    def _agent_manager(self, agent: Literal["pursuer", "evader"]) -> QuadrotorManager | None:
        return self._agent_managers.get(agent)

    def _action_mode(self, agent: Literal["pursuer", "evader"]) -> str:
        """Resolve action mode for the given agent."""
        return self.cfg.agent_action_mode

    @staticmethod
    def _select(tensor: torch.Tensor, env_ids: torch.Tensor | None):
        return tensor if env_ids is None else tensor[env_ids]

    def _get_prev_action(
        self, agent: Literal["pursuer", "evader"], env_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        manager = self._agent_manager(agent)
        if manager is not None:
            prev = manager.get_previous_action()
        else:
            prev = self._pursuer_actions if agent == "pursuer" else self._evader_actions
        return self._select(prev, env_ids)

    def _agent_position(self, agent: Literal["pursuer", "evader"], env_ids: torch.Tensor | None = None) -> torch.Tensor:
        pos = self._select(self._agent_data(agent).data.root_pos_w, env_ids)
        origins = self._terrain.env_origins if env_ids is None else self._terrain.env_origins[env_ids]
        return pos - origins

    def _agent_linear_velocity(
        self, agent: Literal["pursuer", "evader"], env_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self._select(self._agent_data(agent).data.root_lin_vel_w, env_ids)

    def _camera_pose(
        self, agent: Literal["pursuer", "evader"], env_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return camera world pose and forward direction for the requested agent."""
        if self._cam_origin is None or self._cam_line is None or self._camera_cfg is None:
            base = self._select(self._agent_data(agent).data.root_pos_w, env_ids)
            zeros = torch.zeros_like(base)
            ones = (
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32)
                .view(1, 4)
                .expand(base.shape[0], -1)
            )
            forward = (
                torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=torch.float32)
                .view(1, 3)
                .expand(base.shape[0], -1)
            )
            return zeros, ones, forward, zeros

        pos_w = self._select(self._agent_data(agent).data.root_pos_w, env_ids)
        quat_w = self._select(self._agent_data(agent).data.root_quat_w, env_ids)
        batch = pos_w.shape[0]
        origin = self._cam_origin.expand(batch, -1)
        line_end = self._cam_line.expand(batch, -1)
        start_w, end_w, cam_pos_w, cam_quat_w = self._drone_spec.transform_camera_line_fn(
            origin, line_end, pos_w, quat_w, self._camera_cfg
        )
        forward = end_w - start_w
        forward = forward / torch.norm(forward, dim=-1, keepdim=True).clamp_min(1e-6)
        return cam_pos_w, cam_quat_w, forward, start_w

    def _camera_angle(self, agent: Literal["pursuer", "evader"], env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Angle between the camera optical axis and the opponent direction."""
        other = "evader" if agent == "pursuer" else "pursuer"
        if self._cam_origin is None or self._camera_cfg is None:
            base = self._select(self._agent_data(agent).data.root_pos_w, env_ids)
            return torch.zeros(base.shape[0], device=self.device, dtype=torch.float32)

        cam_pos_w, _, forward, _ = self._camera_pose(agent, env_ids)
        target_pos = self._select(self._agent_data(other).data.root_pos_w, env_ids)
        vec = target_pos - cam_pos_w
        vec_norm = torch.norm(vec, dim=-1, keepdim=True).clamp_min(1e-6)
        direction = vec / vec_norm
        cos_theta = torch.sum(forward * direction, dim=-1).clamp(-1.0, 1.0)
        angle = torch.acos(cos_theta)
        if env_ids is None:
            self._last_rho_camera[agent] = angle
        return angle

    def _opponent_class_id(self, camera, agent: str) -> int | None:
        """Return the class ID of ``agent``'s OPPONENT, or None if labels
        aren't ready (first frame). With colorize_semantic_segmentation=False,
        the segmap is an int class-ID image; this helper resolves which ID
        corresponds to the opponent so we can mask self vs. opponent cleanly.
        """
        cache = getattr(self, "_segmap_class_ids", None)
        if cache is None:
            cache = {}
            self._segmap_class_ids = cache
        want = "evader" if agent == "pursuer" else "pursuer"
        if want in cache:
            return cache[want]

        info = getattr(camera.data, "info", None)
        if not info:
            return None

        def _harvest(payload) -> None:
            if not isinstance(payload, dict):
                return
            for raw_id, entry in payload.items():
                label = entry.get("class") if isinstance(entry, dict) else str(entry)
                if not label:
                    continue
                try:
                    cache[label.lower()] = int(raw_id)
                except (TypeError, ValueError):
                    continue

        if isinstance(info, dict):
            sem_info = info.get("semantic_segmentation")
            if isinstance(sem_info, dict):
                _harvest(sem_info.get("idToLabels") or sem_info.get("id_to_labels") or sem_info)
        else:
            for env_info in info:
                if not isinstance(env_info, dict):
                    continue
                sem_info = env_info.get("semantic_segmentation")
                if not isinstance(sem_info, dict):
                    continue
                _harvest(sem_info.get("idToLabels") or sem_info.get("id_to_labels") or sem_info)
                if want in cache:
                    break
        return cache.get(want)

    def _get_image_observations(
        self,
        agent: Literal["pursuer", "evader"],
        update_history: bool = True,
    ) -> torch.Tensor:
        """Get image observations (segmap/depth) for the specified agent.

        Returns a tensor of shape (num_envs, channels * history, H, W) containing
        the stacked image observations. The images are normalized to [0, 1].

        ``update_history=False`` returns just the current frame without
        touching the training-agent history buffer. Use that path when
        querying images for *opponent* policies (which expect a single
        frame as input) so they don't corrupt the training agent's
        history. The shared ``self._image_history_buffer`` is reserved for
        the training agent.
        """
        if not self._use_image_obs:
            return None

        camera = self._pursuer_camera if agent == "pursuer" else self._evader_camera
        if camera is None:
            carb.log_warn(f"[PursuitEvasion] Camera for {agent} not available for image observations")
            if update_history:
                return self._image_history_buffer.clone()
            return torch.zeros_like(self._image_history_buffer)

        # Gather current frame channels
        current_channels = []

        if self.cfg.obs_include_segmap:
            segmap = camera.data.output.get("semantic_segmentation", None)
            if segmap is not None:
                # Cameras run with colorize_semantic_segmentation=False, so
                # the segmap is an int class-ID image of shape (N, H, W, 1)
                # (or (N, H, W) in some Isaac Lab builds).
                if segmap.dim() == 3:
                    segmap = segmap.unsqueeze(1)  # (N, H, W) → (N, 1, H, W)
                elif segmap.dim() == 4 and segmap.shape[-1] == 1:
                    segmap = segmap.permute(0, 3, 1, 2)  # (N, H, W, 1) → (N, 1, H, W)
                # Mask in only the OPPONENT class. Replicator inherits the
                # semantic label from /Pursuer (/Evader) down to all child
                # prims, so the agent's own propeller meshes carry its own
                # class and are correctly excluded by this filter.
                opp_id = self._opponent_class_id(camera, agent)
                if opp_id is None:
                    # First frame fallback: any labeled pixel.
                    segmap = (segmap != 0).float()
                else:
                    segmap = (segmap == opp_id).float()
                current_channels.append(segmap)
            else:
                current_channels.append(
                    torch.zeros(self.num_envs, 1, self._img_h, self._img_w, device=self.device, dtype=torch.float32)
                )

        if self.cfg.obs_include_depth:
            depth = camera.data.output.get("depth", None)
            if depth is not None:
                depth = depth.float()
                # Output shape is (N, H, W, 1) — move to (N, 1, H, W)
                if depth.dim() == 4 and depth.shape[-1] == 1:
                    depth = depth.permute(0, 3, 1, 2)
                elif depth.dim() == 3:
                    depth = depth.unsqueeze(1)
                elif depth.dim() == 4 and depth.shape[1] != 1:
                    depth = depth.permute(0, 3, 1, 2)[:, :1]
                # Normalize: arena diagonal is ~6m, use 10m as max range
                # Inf/nan pixels (sky) are mapped to max distance
                max_depth = 10.0
                depth = torch.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=0.0)
                depth = (depth / max_depth).clamp(0.0, 1.0)
                current_channels.append(depth)
            else:
                current_channels.append(
                    torch.zeros(self.num_envs, 1, self._img_h, self._img_w, device=self.device, dtype=torch.float32)
                )

        # Concatenate current frame channels
        current_frame = torch.cat(current_channels, dim=1)  # (num_envs, channels_per_frame, H, W)

        if not update_history:
            # Opponent path: return single-frame stacked into the history shape
            # (zeros for older frames) without touching the training-agent buffer.
            if self._img_history <= 1:
                return current_frame
            out = torch.zeros_like(self._image_history_buffer)
            out[:, : self._img_channels_per_frame] = current_frame
            return out

        # Update history buffer (shift old frames and add new)
        if self._img_history > 1:
            # Shift: move channels [0:-(channels_per_frame)] to [channels_per_frame:]
            self._image_history_buffer[:, self._img_channels_per_frame :] = self._image_history_buffer[
                :, : -self._img_channels_per_frame
            ].clone()
        # Insert new frame at the beginning
        self._image_history_buffer[:, : self._img_channels_per_frame] = current_frame

        return self._image_history_buffer.clone()

    def _reset_image_history(self, env_ids: torch.Tensor) -> None:
        """Reset image history buffer for specified environments."""
        if self._image_history_buffer is not None:
            self._image_history_buffer[env_ids] = 0.0

    def _maybe_save_camera_images(self) -> None:
        """Persist per-environment RGB frames when enabled."""
        if not (self.cfg.enable_cameras and self.cfg.save_camera_images and self._pursuer_camera is not None):
            return
        if self._camera_save_stride > 1:
            if int(self.common_step_counter) % self._camera_save_stride != 0:
                return
        images = self._pursuer_camera.data.output.get("rgb", None)
        if images is None:
            return
        img = images.detach().clone()
        # ensure channel-last float in [0,1]
        if img.dim() == 5 and img.shape[1] == 1:
            img = img.squeeze(1)
        if img.dim() == 4 and img.shape[1] in (1, 3, 4):
            img = img.permute(0, 2, 3, 1)
        img = img.to(torch.float32)
        if img.max() > 1.0:
            img = img / 255.0
        img = img.clamp(0.0, 1.0)
        out_root = Path(self.cfg.camera_image_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        step_idx = int(self.common_step_counter)
        vis_flags = self._last_target_visible.get("pursuer")
        rho_last = self._last_rho_camera.get("pursuer")
        for env_id in range(img.shape[0]):
            env_dir = out_root / f"env_{env_id:03d}"
            env_dir.mkdir(parents=True, exist_ok=True)
            frame = img[env_id : env_id + 1].cpu()
            if self.cfg.camera_overlay_text and vis_flags is not None and rho_last is not None:
                import torchvision.transforms.functional as F
                from PIL import ImageDraw, ImageFont

                frame_np = frame[0]
                frame_np = frame_np.permute(2, 0, 1)
                pil_img = F.to_pil_image(frame_np)
                draw = ImageDraw.Draw(pil_img)
                visible = bool(vis_flags[env_id].item())
                rho_deg = float(torch.rad2deg(rho_last[env_id]).item())
                status = "Target locked" if visible else "Target lost"
                text = f"{status} | angle: {rho_deg:.1f} deg"
                draw.text((10, 10), text, fill=(255, 0, 0))
                pil_tensor = F.to_tensor(pil_img)  # C,H,W
                frame = pil_tensor.permute(1, 2, 0).unsqueeze(0)

            save_images_to_file(frame, str(env_dir / f"step_{step_idx:06d}.png"))

    @carb.profiler.profile
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specific environments."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._pursuer._ALL_INDICES
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        should_log = self._initial_reset_complete
        if not self._initial_reset_complete:
            self._initial_reset_complete = True
        if should_log and env_ids.numel() > 0:
            heuristic_ids = self._heuristic_onehot.argmax(dim=-1)
            log_data = self._stats.finalize_episode_metrics(
                env_ids=env_ids,
                episode_length_buf=self.episode_length_buf,
                step_dt=self.step_dt,
                last_done_reasons=self._last_done_reasons,
                heuristic_ids=heuristic_ids,
            )
            self.extras["log"] = log_data

        # Reset robots
        self._pursuer.reset(env_ids)
        if self._use_visual_ball_evader:
            self._evader.reset(env_ids)
            if self.cfg.visual_ball_randomize_radius:
                self._evader.sample_radius(
                    env_ids,
                    float(self.cfg.visual_ball_radius_min),
                    float(self.cfg.visual_ball_radius_max),
                )
            self._evader.update_visuals()
        else:
            self._evader.reset(env_ids)
        super()._reset_idx(env_ids)

        # Reset image history buffer for image-based observations
        if self._use_image_obs:
            self._reset_image_history(env_ids)

        # Reset past actions buffer for image-only mode
        if self._use_image_only and self._past_actions_buffer is not None:
            self._past_actions_buffer[env_ids] = 0.0
        # Reset opponent past-actions on episode boundary so a fresh episode
        # starts the loaded recurrent policy with zero context.
        for _buf in self._opp_past_actions.values():
            _buf[env_ids] = 0.0

        # Reset managers
        if self.pursuer_manager is not None:
            self.pursuer_manager.reset(env_ids)
        if self.evader_manager is not None:
            self.evader_manager.reset(env_ids)

        if self._use_visual_ball_evader:
            pos_evader_w = self._agent_position("evader", env_ids)
        else:
            pos_evader_w = D.Uniform(self._arena_min_safe, self._arena_max_safe).sample([
                len(env_ids),
            ])
            if self.evader_manager is not None:
                traj = getattr(self.evader_manager, "_trajectory", None)
                series = None if traj is None else traj.get("series")
                if series is not None and "pos" in series:
                    global_to_traj = traj["global_to_traj"]
                    local_ids = global_to_traj[env_ids]
                    mask = local_ids >= 0
                    if bool(mask.any()):
                        pos_traj = series["pos"][0, local_ids[mask]]
                        pos_evader_w[mask] = pos_traj

        # Sample initial positions with minimum separation
        min_separation = self.cfg.capture_distance * 2.0
        pos_pursuer_w = min_separation_sampling(
            self._arena_min_safe + 0.3, self._arena_max_safe - 0.3, pos_evader_w, min_separation
        )

        # Rejection-resample positions that land inside cross walls
        if self.cfg.enable_obstacles:
            for _ in range(50):
                bad_p = self._check_wall_collision(pos_pursuer_w)
                if not bad_p.any():
                    break
                pos_pursuer_w[bad_p] = min_separation_sampling(
                    self._arena_min_safe + 0.3,
                    self._arena_max_safe - 0.3,
                    pos_evader_w[bad_p],
                    min_separation,
                )
            if not self._use_visual_ball_evader:
                for _ in range(50):
                    bad_e = self._check_wall_collision(pos_evader_w)
                    if not bad_e.any():
                        break
                    pos_evader_w[bad_e] = D.Uniform(self._arena_min_safe, self._arena_max_safe).sample(
                        [int(bad_e.sum())]
                    )

        # Set pursuer state
        state_pursuer = self._pursuer.data.default_root_state[env_ids].clone()
        state_pursuer[:, :3] = pos_pursuer_w + self._terrain.env_origins[env_ids]
        # yaw toward evader
        delta = pos_evader_w - pos_pursuer_w
        yaw = torch.atan2(delta[:, 1], delta[:, 0]) * 0.5
        state_pursuer[:, 3] = torch.cos(yaw)
        state_pursuer[:, 6] = torch.sin(yaw)

        self._pursuer.write_root_pose_to_sim(state_pursuer[:, :7], env_ids)
        self._pursuer.write_root_velocity_to_sim(state_pursuer[:, 7:], env_ids)

        # Set evader state (if using physical evader)
        if not self._use_visual_ball_evader:
            state_evader = self._evader.data.default_root_state[env_ids].clone()
            state_evader[:, :3] = pos_evader_w + self._terrain.env_origins[env_ids]

            self._evader.write_root_pose_to_sim(state_evader[:, :7], env_ids)
            self._evader.write_root_velocity_to_sim(state_evader[:, 7:], env_ids)

        # Reset joint states
        pursuer_joint_pos = self._pursuer.data.default_joint_pos[env_ids]
        pursuer_joint_vel = self._pursuer.data.default_joint_vel[env_ids]
        self._pursuer.write_joint_state_to_sim(pursuer_joint_pos, pursuer_joint_vel, None, env_ids)

        if not self._use_visual_ball_evader:
            evader_joint_pos = self._evader.data.default_joint_pos[env_ids]
            evader_joint_vel = self._evader.data.default_joint_vel[env_ids]
            self._evader.write_joint_state_to_sim(evader_joint_pos, evader_joint_vel, None, env_ids)

        # Reset propellers
        self._pursuer_propellers.reset(env_ids)
        if not self._use_visual_ball_evader and self._evader_propellers is not None:
            self._evader_propellers.reset(env_ids)
        self._pursuer_omega_ref[env_ids] = 0.0
        self._evader_omega_ref[env_ids] = 0.0
        self._pursuer_actions[env_ids] = 0.0
        self._evader_actions[env_ids] = 0.0
        self._pursuer_wrench[env_ids] = 0.0
        self._evader_wrench[env_ids] = 0.0
        if self.cfg.training_agent == "pursuer":
            wrapper = self._training_wrappers.get("pursuer")
            if wrapper is not None:
                wrapper.reset(env_ids)
        elif self.cfg.training_agent == "evader":
            wrapper = self._training_wrappers.get("evader")
            if wrapper is not None:
                wrapper.reset(env_ids)

        # Apply per-episode domain randomization after resets.
        self._apply_domain_randomization(env_ids)

        # Reset distance tracking
        self._prev_distance[env_ids] = torch.norm(pos_evader_w - pos_pursuer_w, dim=-1)
        # Geodesic-aware shaping distance: must match what _get_rewards will compute next step
        # so the first approach term is ~0 instead of a spike.
        self._prev_shaping_distance[env_ids] = self._wall_geodesic_distance(pos_pursuer_w, pos_evader_w)
        self._stats.reset_episode_trackers(env_ids)

    @carb.profiler.profile
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Compute control for both agents before physics step."""
        if self.pursuer_manager is not None:
            self.pursuer_manager.set_curriculum_frame(int(self.common_step_counter))
        if self.evader_manager is not None:
            self.evader_manager.set_curriculum_frame(int(self.common_step_counter))
        pursuer_rl_obs = self._build_rl_observations("pursuer")
        evader_rl_obs = self._build_rl_observations("evader")

        if self.cfg.training_agent == "pursuer":
            wrapper = self._training_wrappers["pursuer"]
            if wrapper is None:
                raise RuntimeError("No action wrapper configured for pursuer training.")
            td = self._build_training_tensordict("pursuer", actions)
            self._pursuer_actions = wrapper.command(td)
            self._evader_actions = self._compute_manager_actions(self.evader_manager, self._pursuer.data, evader_rl_obs)
        elif self.cfg.training_agent == "evader":
            wrapper = self._training_wrappers["evader"]
            if wrapper is None:
                raise RuntimeError("No action wrapper configured for evader training.")
            td = self._build_training_tensordict("evader", actions)
            self._evader_actions = wrapper.command(td)
            self._pursuer_actions = self._compute_manager_actions(
                self.pursuer_manager, self._evader.data, pursuer_rl_obs
            )
        else:
            self._pursuer_actions = self._compute_manager_actions(
                self.pursuer_manager, self._evader.data, pursuer_rl_obs
            )
            self._evader_actions = self._compute_manager_actions(self.evader_manager, self._pursuer.data, evader_rl_obs)

        # Update past actions buffer for image-only mode (shift and add new action)
        if self._use_image_only and self._past_actions_buffer is not None:
            # Shift history: move actions forward (oldest dropped, newest added)
            # Buffer shape: (num_envs, num_past_actions, action_dim)
            if self._num_past_actions > 1:
                self._past_actions_buffer[:, :-1] = self._past_actions_buffer[:, 1:].clone()
            # Add current action as the newest (last position)
            self._past_actions_buffer[:, -1] = actions

        # Opponent past-actions: shift each role's buffer with the action that
        # was just computed for that role. The recurrent vision loader reads
        # this buffer on the next step's _build_rl_observations call.
        for _role, _act in (("pursuer", self._pursuer_actions), ("evader", self._evader_actions)):
            _buf = self._opp_past_actions.get(_role)
            if _buf is None:
                continue
            if self._opp_num_past_actions > 1:
                _buf[:, :-1] = _buf[:, 1:].clone()
            _buf[:, -1] = _act

        if self._use_visual_ball_evader:
            self._evader.step()
            self._evader.update_visuals()

        # Update visualizers
        self._update_visualizers()

    # -------------------------------------------------------------------------
    # Controller assignment and sampling
    # -------------------------------------------------------------------------

    def _build_visual_ball_trajectory_groups(self) -> dict[str, torch.Tensor]:
        """Build env groups for visual ball trajectories from evader controller specs."""
        traj_names = set(VisualBallEvader.TRAJECTORY_MAP.keys())
        weights: dict[str, float] = {}
        for spec in self.cfg.evader_controllers:
            name = spec.name.lower()
            if name not in traj_names:
                continue
            weight = spec.probability if spec.probability is not None else float(spec.count)
            weight = 0.0 if weight is None else float(weight)
            if weight <= 0:
                continue
            weights[name] = weight

        if not weights:
            weights = {"hover": 1.0}

        assignments = policy_sampling(weights, num_sampled=self.scene.cfg.num_envs)
        device = torch.device(self.cfg.sim.device)
        groups = {
            name: torch.as_tensor(env_ids, device=device, dtype=torch.long)
            for name, env_ids in assignments.items()
            if env_ids
        }
        return groups

    def _assign_controllers(self) -> tuple[dict[str, dict], dict[str, dict]]:
        """Assign controllers to environments using stratified sampling."""
        pursuer_assignment = self._build_controller_assignment(self.cfg.pursuer_controllers)
        evader_assignment = self._build_controller_assignment(self.cfg.evader_controllers)
        return pursuer_assignment, evader_assignment

    def _build_heuristic_onehot(self) -> None:
        """Build the one-hot heuristic identifier from evader controller assignments.

        Maps each evader controller to one of ``_heuristic_names`` by matching
        the controller's ``kind`` or ``name`` (exact match, case-insensitive).
        Called once at env init; for AMSPB with rotating opponents, call again
        after reassigning controllers.
        """
        self._heuristic_onehot.zero_()
        for i, heuristic in enumerate(self._heuristic_names):
            for ctrl_name, ctrl_cfg in self._evader_controller_assignment.items():
                kind = ctrl_cfg.get("kind", ctrl_name).lower()
                if kind == heuristic or ctrl_name.lower() == heuristic:
                    self._heuristic_onehot[ctrl_cfg["env_ids"], i] = 1.0

    def _build_opp_pool_id(self) -> None:
        """Assign each env a unique integer id per distinct opponent controller name.

        Implements the discrete identifier k_t ∈ {0, ..., K-1} from the paper.
        For Experiment 1 (heuristic pool) this collapses to one id per heuristic
        type. For Experiment 2 (AMSPB) every RL checkpoint has a unique
        controller name (e.g. ``rl_evader_pretrain``, ``rl_evader_stage1``, ...),
        so each pool member gets a distinct id and the embedding lookup
        e(k_t) is injective on the pool — required by Proposition 2.
        """
        if self.cfg.training_agent == "pursuer":
            assignment = self._evader_controller_assignment
        else:
            assignment = self._pursuer_controller_assignment
        self._opp_pool_names = []
        name_to_id: dict[str, int] = {}
        self._opp_pool_id.zero_()
        for ctrl_name, ctrl_cfg in assignment.items():
            if ctrl_name not in name_to_id:
                name_to_id[ctrl_name] = len(self._opp_pool_names)
                self._opp_pool_names.append(ctrl_name)
            self._opp_pool_id[ctrl_cfg["env_ids"]] = name_to_id[ctrl_name]

    def _compose_wandb_artifact_cfg(
        self, name: str, kind: str, payload: dict[str, Any], defaults: dict[str, Any]
    ) -> dict[str, Any]:
        if "wandb_artifact" in payload or kind not in RL_KINDS:
            return payload

        artifact_name = payload.pop("artifact_name", None) or payload.pop("wandb_artifact_name", None)
        alias = payload.pop("artifact_alias", None) or defaults.get("alias", "latest")
        entity = payload.pop("artifact_entity", None) or defaults.get("entity")
        project = payload.pop("artifact_project", None) or defaults.get("project")
        local_dir = payload.pop("artifact_local_dir", None) or defaults.get("dir")
        file_name = payload.pop("artifact_file", None) or payload.pop("wandb_artifact_file", None)

        if artifact_name is None and entity and project:
            artifact_name = name
        if artifact_name is None:
            return payload

        artifact_path = artifact_name
        if entity and project and "/" not in artifact_name:
            artifact_path = f"{entity}/{project}/{artifact_name}"
        if alias and ":" not in artifact_path:
            artifact_path = f"{artifact_path}:{alias}"

        artifact_cfg: dict[str, Any] = {"artifact": artifact_path}
        if file_name:
            artifact_cfg["file"] = file_name
        if local_dir:
            artifact_cfg["local_dir"] = local_dir
        payload["wandb_artifact"] = artifact_cfg
        return payload

    def _build_controller_assignment(self, specs: Sequence[ControllerSpec]) -> dict[str, dict]:
        if not specs:
            return {}
        weights: dict[str, float] = {}
        for spec in specs:
            weight = spec.probability if spec.probability is not None else float(spec.count)
            weight = 0.0 if weight is None else float(weight)
            if weight <= 0:
                continue
            weights[spec.name] = weight
        total = sum(weights.values())
        if total <= 0:
            return {}

        # probability-proportional sampling over the policies
        pool = {name: weight / total for name, weight in weights.items()}
        assignments_raw = policy_sampling(pool, num_sampled=self.num_envs)
        assignments: dict[str, dict] = {}
        wandb_defaults = getattr(self.cfg, "wandb_artifact_defaults", None) or {}
        for spec in specs:
            env_ids = assignments_raw.get(spec.name, [])
            if not env_ids:
                continue
            kind = _infer_controller_kind(spec.name, spec.kind)
            payload = copy.deepcopy(spec.config) if spec.config else {}
            # Controller-specific configuration (and optional WandB artifact info for RL controllers).
            if payload:
                payload = self._compose_wandb_artifact_cfg(spec.name, kind, payload, wandb_defaults)

            cfg: dict[str, Any] = {
                "env_ids": torch.as_tensor(env_ids, device=self.device, dtype=torch.long),
                "kind": kind,
            }
            if payload:
                if kind in RL_KINDS:
                    cfg.update(payload)
                elif any(key in payload for key in ("config", "lee_controller_cfg")):
                    cfg.update(payload)
                else:
                    cfg["config"] = payload
            # Forward config_overrides for deferred YAML loading
            if spec.config_overrides:
                cfg["config_overrides"] = copy.deepcopy(spec.config_overrides)
            if spec.name in {"hover", "circular", "lemniscate"}:
                cfg.setdefault("trajectory_horizon", self.max_episode_length)
            assignments[spec.name] = cfg
        return assignments

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------

    def _setup_default_camera(self) -> None:
        """Set up a default camera view for the viewer.

        The camera is positioned to provide a good overview of the arena,
        looking down at an angle from above.
        """
        arena_size = (self._arena_max).cpu().numpy()
        arena_center = ((self._arena_min + self._arena_max) / 2).cpu().numpy()

        camera_pos = arena_size
        camera_pos[0] *= 1.05
        camera_pos[1] *= 1.05
        camera_pos[2] *= 6

        camera_target = arena_center

        # Apply camera transform
        from isaacsim.core.utils.viewports import set_camera_view

        set_camera_view(eye=camera_pos, target=camera_target, camera_prim_path="/OmniverseKit_Persp")

    def _setup_visualizers(self) -> None:
        """Set up velocity arrow visualizers."""
        if not (self.cfg.debug_vis and self.cfg.flag_draw_velocity_markers):
            return

        import isaaclab.sim as sim_utils
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

        # Pursuer velocity arrows (blue)
        pursuer_vel_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/PursuitEvasion/PursuerVelocity",
            markers={
                "arrow": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.8, 0.2, 0.2),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.3, 1.0)),
                )
            },
        )
        self._pursuer_vel_markers = VisualizationMarkers(pursuer_vel_cfg)

        # Evader velocity arrows (red)
        evader_vel_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/PursuitEvasion/EvaderVelocity",
            markers={
                "arrow": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.8, 0.2, 0.2),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                )
            },
        )
        self._evader_vel_markers = VisualizationMarkers(evader_vel_cfg)

    def _update_visualizers(self) -> None:
        """Update velocity arrow visualizations."""
        if not self.cfg.debug_vis:
            return

        if self.cfg.flag_draw_velocity_markers:
            # Update pursuer attitude arrows (blue) aligned with body +X and lifted along body +Z
            if self._pursuer_vel_markers is not None:
                pursuer_pos = self._pursuer.data.root_pos_w
                pursuer_quat = self._pursuer.data.root_quat_w
                orientations, scales = self._attitude_marker_data(pursuer_quat, align_quat=self._quat_identity)
                translations = self._marker_translations(pursuer_pos, pursuer_quat, extra_offset=0.15)
                self._pursuer_vel_markers.visualize(translations=translations, orientations=orientations, scales=scales)

            # Update evader attitude arrows (red)
            if self._evader_vel_markers is not None:
                evader_pos = self._evader.data.root_pos_w
                evader_quat = self._evader.data.root_quat_w
                orientations, scales = self._attitude_marker_data(evader_quat, align_quat=self._quat_identity)
                translations = self._marker_translations(evader_pos, evader_quat, extra_offset=0.15)
                self._evader_vel_markers.visualize(translations=translations, orientations=orientations, scales=scales)

        if self.cfg.flag_draw_camera_frustum and self._frustum_viz is not None and self.cfg.enable_cameras:
            cam_pos_w, cam_quat_w, _, _ = self._camera_pose("pursuer")
            if cam_pos_w.numel() > 0:
                self._frustum_viz.draw(cam_pos_w[0], cam_quat_w[0])

    def _attitude_marker_data(
        self, quats: torch.Tensor, align_quat: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate marker orientations aligned with each quadrotor's body +Z axis."""
        if align_quat is None:
            align_quat = self._marker_align_quat
        align = align_quat.view(1, 4).expand(quats.shape[0], -1)
        orientations = quat_mul(quats, align)
        length = self.cfg.velocity_marker_length
        thickness = self.cfg.velocity_marker_radius
        scales = torch.ones(quats.shape[0], 3, device=self.device, dtype=torch.float32)
        scales[:, 0] = length
        scales[:, 1] = thickness
        scales[:, 2] = thickness
        return orientations, scales

    def _marker_translations(
        self, positions: torch.Tensor, quats: torch.Tensor, extra_offset: float = 0.0
    ) -> torch.Tensor:
        """Offset marker origins along body +Z to avoid intersecting the frame."""
        offset = self.cfg.velocity_marker_offset + extra_offset
        if offset <= 0.0:
            return positions
        body_z = self._body_z_axes(quats)
        return positions + body_z * offset

    def _body_z_axes(self, quats: torch.Tensor) -> torch.Tensor:
        rot = matrix_from_quat(quats).view(-1, 3, 3)
        return rot[:, :, 2]

    # -------------------------------------------------------------------------
    # Logging helpers
    # -------------------------------------------------------------------------
    def get_last_rewards(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the per-environment pursuer and evader rewards computed at the last step."""
        return self._last_pursuer_rewards, self._last_evader_rewards

    def get_last_done_reasons(self) -> torch.Tensor:
        """Return encoded termination reasons for the last step."""
        return self._last_done_reasons

    def get_last_reward_components(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return per-component reward contributions computed at the last step."""
        result = {}
        for agent, components in self._last_reward_components.items():
            result[agent] = {name: value.clone() for name, value in components.items()}
        return result
