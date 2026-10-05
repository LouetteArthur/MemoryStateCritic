# Copyright (c) 2026, the MemoryStateCritic authors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pre-configured pursuit-evasion environment configurations and base config classes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# =============================================================================
# Constants
# =============================================================================

DONE_REASON_LABELS = {
    0: "running",
    1: "pursuer_capture",
    3: "pursuer_out_of_bounds",
    4: "evader_out_of_bounds",
    5: "timeout",
    6: "invalid_state",
    7: "pursuer_wall_collision",
    8: "evader_wall_collision",
}

# Intrinsics vector ordering (per-env).
# k_eta maps to thrust coefficient (k_f); k_m maps to moment coefficient.
INTRINSICS_ORDER = (
    "mass",
    "inertia_x",
    "inertia_y",
    "inertia_z",
    "k_eta",
    "k_m",
    "tau_m",
    "k_aero_xy",
    "k_aero_z",
    "rate_kp_roll",
    "rate_kp_pitch",
    "rate_kp_yaw",
    "rate_ki_roll",
    "rate_ki_pitch",
    "rate_ki_yaw",
    "rate_kd_roll",
    "rate_kd_pitch",
    "rate_kd_yaw",
)


# =============================================================================
# Data Classes and Config Classes
# =============================================================================


@dataclass
class ControllerSpec:
    """Specification for controller assignment."""

    name: str
    count: int
    probability: float | None = None
    kind: str | None = None
    config: dict | None = None
    config_overrides: dict | None = None


@configclass
class DomainRandomizationCfg:
    """Domain randomization configuration."""

    enable: bool = False
    debug_checks: bool = True

    # Uniform scale range for mass/inertia/k_eta/k_m/tau.
    scale_min: float = 0.85
    scale_max: float = 1.15

    randomize_mass: bool = True
    randomize_inertia: bool = True
    randomize_k_eta: bool = True
    randomize_k_m: bool = True
    randomize_tau: bool = True
    randomize_k_aero: bool = True
    randomize_rate_gains: bool = True

    # Aerodynamics scaling (ma_quadcopter_env.py ranges).
    k_aero_xy_min_scale: float = 0.5
    k_aero_xy_max_scale: float = 2.0
    k_aero_z_min_scale: float = 0.5
    k_aero_z_max_scale: float = 2.0

    # Rate gains scaling (ma_quadcopter_env.py ranges).
    rate_kp_min_scale: float = 0.85
    rate_kp_max_scale: float = 1.15
    rate_ki_min_scale: float = 0.85
    rate_ki_max_scale: float = 1.15
    rate_kd_min_scale: float = 0.7
    rate_kd_max_scale: float = 1.2


@configclass
class PursuitEvasionEnvCfg(DirectRLEnvCfg):
    """Configuration for the pursuit-evasion environment."""

    # Simulation settings
    episode_length_s = 10.0  # truncation limit
    sim_frequency = 500  # prev 500
    policy_rate_hz = 25  # prev 50
    pid_loop_rate_hz = 500  # prev 500
    pid_posvel_loop_rate_hz = 100
    decimation = sim_frequency // policy_rate_hz
    action_space = 4

    sim: SimulationCfg = SimulationCfg(dt=1 / sim_frequency, render_interval=decimation)

    terrain: TerrainImporterCfg = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", debug_vis=False)

    # Arena bounds [min, max] for [x, y, z]
    arena_min = (-2.5, -2.0, 0.0)
    arena_max = (2.5, 2.0, 2.0)
    collision_altitude: float = 0.2
    arena_margin: float = 0.0

    enable_occlusion_walls: bool = True
    wall_thickness: float = 0.05
    wall_extra_margin: float = 0.5  # distance from arena bounds to inner wall faces (x/y)
    enable_roof: bool = False
    roof_height_offset: float = 0.5  # distance above arena_max z to the roof inner face
    roof_thickness: float = 0.05
    roof_opacity: float = 0.2

    # Cross-wall obstacle configuration
    enable_obstacles: bool = False
    obstacle_wall_thickness: float = 0.05
    obstacle_wall_height: float = 2.0  # full arena height
    obstacle_gap_size: float = 1.5  # gap at ends of each cross arm (meters)
    obstacle_collision_penalty: float = 10.0  # equal to reward_catch — wall hit = full game-value loss (zero-sum)
    obstacle_drone_clearance: float = 0.15  # min distance from wall surface for collision

    env_spacing = max(
        (arena_max[0] - arena_min[0]) + 2 * wall_extra_margin + 2 * wall_thickness,
        (arena_max[1] - arena_min[1]) + 2 * wall_extra_margin + 2 * wall_thickness,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=env_spacing, replicate_physics=True)

    # Robot configurations
    pursuer_robot = None  # Lazily populated
    evader_robot = None  # Lazily populated
    drone_name: str = "crazyflie_brushless"

    # Visual ball evader configuration (for vision training - faster than simulating second drone)
    use_visual_ball_evader: bool = True
    visual_ball_radius: float = 0.075  # Default radius (midpoint of 0.05-0.1)
    visual_ball_randomize_radius: bool = False
    visual_ball_radius_min: float = 0.05
    visual_ball_radius_max: float = 0.1
    visual_ball_color: tuple = (1.0, 0.0, 0.0)  # Red ball by default

    capture_distance = 0.2  # Distance for successful capture
    domain_randomization: DomainRandomizationCfg = DomainRandomizationCfg()

    # Asymmetric actor-critic: critic uses privileged full state, actor uses partial observations
    # When enabled, critic receives both agents' full states (pos, vel, quat, ang_vel)
    # and intrinsics if DR is also enabled
    asymmetric_actor_critic: bool = True

    # Unbiased asymmetric critic: when True and using vision + asymmetric, the critic
    # receives both the privileged state AND the actor's observations (image + past_actions).
    # This follows Baisero & Amato's result that V(s, o) is an unbiased baseline.
    # When False (default), the critic receives only the privileged state vector (biased).
    unbiased_critic: bool = False

    # Opponent feature exposure (for Stage 2 joint critics: V(s,z,z^opp), V(s,h,h^opp))
    expose_opponent_z: bool = False  # Expose opponent's RNN hidden state z^opp in extras
    expose_opponent_obs: bool = False  # Expose opponent's image and prev_action in extras
    opponent_z_dim: int = 256  # Dimension of opponent z (GRU hidden size)
    # Per-pool-member opponent identifier k_t for the paper's joint critic e(k_t).
    # When True, the env emits extras["opp_id"] = LongTensor[num_envs] holding a
    # unique integer per distinct opponent controller name. Disabled by default so
    # heuristic ablation configs (Vs / Vsz / Vsh / Vo / Vsoa) are unaffected.
    expose_opp_id: bool = False

    state_space = 0  # placeholder, computed dynamically in compute_state_dim()

    pursuer_controllers: Sequence[ControllerSpec] = ()
    evader_controllers: Sequence[ControllerSpec] = ()

    # Observation configuration
    obs_include_time_encoding: bool = False
    obs_include_relative_distance: bool = True
    obs_include_closing_velocity: bool = False
    obs_include_prev_action: bool = False
    obs_include_self_position: bool = True
    obs_include_camera_angle: bool = True

    flag_obs_manual_normalization: bool = False
    critic_include_propeller_speeds: bool = False  # append propeller omega to critic state (4 pursuer + 4 evader dims)
    critic_include_heuristic_state: bool = (
        False  # Markov state: evader rot6d+ang_vel, PID integrals, heuristic one-hot, traj setpoint (32 dims)
    )

    enable_cameras: bool = False
    enable_evader_cameras: bool = False
    camera_overlay_text: bool = False
    save_camera_images: bool = False
    camera_image_dir: str = "logs/pursuit_evasion/fpv/images"
    flag_draw_camera_frustum: bool = False

    # Camera configuration — wide FOV (~120° HFOV, practical limit for pinhole model)
    camera_width: int = 96
    camera_height: int = 96
    camera_fx: float = 27.7  # 96/2 / tan(60°) ≈ 27.7 → 120° HFOV
    camera_fy: float = 27.7
    camera_tilt_deg: float = 0.0  # Camera tilt angle in degrees (positive = look down)

    # Image-based observations for actor (segmap/depth)
    # When enabled, actor receives image observations in addition to (or instead of) vector obs
    obs_include_segmap: bool = False  # Include semantic segmentation image in actor observations
    obs_include_depth: bool = False  # Include depth image in actor observations
    obs_image_history: int = 1  # Number of consecutive frames to stack (1 = current frame only)

    # Image-only mode: actor gets ONLY image + past actions (no vector state)
    # Inspired by "Demonstrating Agile Flight from Pixels without State Estimation" (Geles et al.)
    obs_image_only: bool = False  # When True, actor only gets image + past_actions
    obs_num_past_actions: int = 3  # Number of past actions to include in observations

    # Zero-sum reward: pursuer minimizes capture time, evader maximizes survival.
    # R = reward_catch is the terminal game value.  Per-step time cost = R / T
    # where T = episode_length_s * policy_rate_hz (250 by default).
    # Potential-based approach shaping (Ng et al. 1999) preserves optimal policy.
    reward_catch: float = 10.0  # R: terminal value for catch / OOB events
    reward_time_scale: float = (
        1.0  # kappa_t: total time budget (per-step cost = kappa_t / T). Decoupled from R so terminal events (+/-R) dominate the dense time pressure.
    )
    reward_approach: float = 3.0  # exponential potential shaping weight in Φ(d) = exp(-(d - r_c) / d_0)
    reward_approach_decay: float = (
        0.5  # d_0 in Φ(d) = exp(-(d - r_c) / d_0). d_0 ≈ success-threshold scale matches reach-task practice (Hwangbo'19, Lee'20, Rudin'22).
    )
    # Note: the potential is shifted by capture_distance (= r_c) so Φ(r_c) = 1
    # — the agent can never see d < r_c (episode terminates with catch reward
    # first), so the unshifted potential's max of exp(0) at d=0 was unreachable
    # and only the magnitude exp(r_c / d_0) at the capture boundary was felt.
    reward_perception_scale: float = 0.02  # FPV centering bonus when evader is visible (occlusion-aware)
    reward_perception_angle_scale: float = 1.0  # decay rate of exp(-kappa_rho * rho) centering term
    reward_body_rates: float = 0.001  # angular rate regularization (sim-to-real)
    reward_action_smoothness: float = 0.0  # action smoothness regularization

    training_agent: Literal["pursuer", "evader", ""] = "pursuer"
    agent_action_mode: Literal["velocity", "body_rates"] = "velocity"

    # Debug visualization
    debug_vis = True
    flag_draw_velocity_markers: bool = True
    velocity_marker_length: float = 2.0
    velocity_marker_radius: float = 0.25
    velocity_marker_offset: float = 0.25

    observation_space = 0  # Lazily populated
    total_timesteps = episode_length_s * sim_frequency

    # Optional WandB defaults used to resolve artifact identifiers for RL controllers.
    wandb_artifact_defaults: dict | None = None

    # Optional run name used for WandB (and checkpoint naming) when training via skrl.
    wandb_run_name: str | None = None

    # Optional skrl agent configuration (passed from benchmark / play scripts) for rebuilding policies.
    skrl_agent_cfg: dict | None = None

    # Optional warmstart for the training agent (local checkpoint path or WandB artifact spec).
    training_warmstart: dict | str | None = None


# =============================================================================
# Helper Functions
# =============================================================================


def compute_obs_dim(cfg: PursuitEvasionEnvCfg) -> int:
    """Compute observation dimension from active flags for the training agent."""
    common = 0
    if cfg.obs_include_time_encoding:
        common += 1
    common += 6  # 6D rotation (first two columns of R_WB, Zhou et al.)
    common += 3  # linear velocity (body)
    common += 3  # angular velocity (body)
    if cfg.obs_include_self_position:
        common += 3  # absolute position
    if cfg.obs_include_prev_action:
        common += cfg.action_space

    common += 3  # relative position

    if cfg.obs_include_relative_distance:
        common += 1
    if cfg.obs_include_closing_velocity:
        common += 3
    if cfg.obs_include_camera_angle:
        common += 1  # visibility flag
        common += 1  # rho normalized

    return common


def _compute_state_vector_dim(cfg: PursuitEvasionEnvCfg) -> int:
    """Compute the privileged state vector dimension for the critic.

    This is the unmasked observation vector with forced extras.
    """
    dim = compute_obs_dim(cfg)
    if not cfg.obs_include_self_position:
        dim += 3
    if not cfg.obs_include_relative_distance:
        dim += 1
    if not cfg.obs_include_closing_velocity:
        dim += 3
    if cfg.critic_include_propeller_speeds:
        dim += 4  # pursuer propeller angular velocities
        if not cfg.use_visual_ball_evader:
            dim += 4  # evader propeller angular velocities
    if cfg.critic_include_heuristic_state:
        # Markov state extras (32 dims total):
        #   evader_rot6d(6) + evader_ang_vel(3) + pursuer_rate_pid(3)
        #   + evader_pid_integrals(9) + heuristic_onehot(5) + traj_setpoint(6)
        dim += 32
    return dim


def compute_state_dim(cfg: PursuitEvasionEnvCfg) -> int | dict:
    """Compute critic state dimension for asymmetric actor-critic training.

    Returns:
        0 if symmetric (critic uses same obs as actor).
        int if asymmetric without vision (critic gets privileged state vector).
        dict if asymmetric with vision (critic gets state + image + past_actions,
            following Baisero & Amato's unbiased asymmetric actor-critic).
    """
    if not cfg.asymmetric_actor_critic:
        return 0

    state_dim = _compute_state_vector_dim(cfg)

    # Unbiased vision + asymmetric: critic receives state + actor's observations
    if cfg.unbiased_critic:
        image_shape = compute_image_obs_shape(cfg)
        if image_shape is not None and cfg.obs_image_only:
            past_actions_dim = cfg.obs_num_past_actions * cfg.action_space
            return {
                "image": list(image_shape),
                "past_actions": past_actions_dim,
                "state": state_dim,
            }

    return state_dim


def compute_image_obs_shape(cfg: PursuitEvasionEnvCfg) -> tuple[int, int, int] | None:
    """Compute the image observation shape (C, H, W) if image observations are enabled.

    Returns:
        Tuple of (channels, height, width) if images enabled, None otherwise
    """
    if not (cfg.obs_include_segmap or cfg.obs_include_depth):
        return None

    channels_per_frame = 0
    if cfg.obs_include_segmap:
        channels_per_frame += 1
    if cfg.obs_include_depth:
        channels_per_frame += 1

    total_channels = channels_per_frame * cfg.obs_image_history
    return (total_channels, cfg.camera_height, cfg.camera_width)


def _infer_controller_kind(name: str, explicit: str | None = None) -> str:
    """Infer the controller kind from its name."""
    if explicit:
        return explicit
    lowered = name.lower()
    if lowered.startswith("rl_velocity"):
        return "rl_velocity"
    if lowered.startswith("rl_bodyrates"):
        return "rl_bodyrates"
    if lowered.startswith("rl_policy"):
        return "rl_velocity"
    return lowered


# =============================================================================
# Configuration Factory Functions
# =============================================================================


def get_base_config() -> PursuitEvasionEnvCfg:
    cfg = PursuitEvasionEnvCfg()
    return cfg


def _resolve_action_mode(action_mode: str | None) -> tuple[str, str]:
    """Return (rl_kind, agent_action_mode)."""
    mode = (action_mode or "rl_velocity").strip()
    if mode not in {"rl_velocity", "rl_bodyrates"}:
        mode = "rl_velocity"
    agent_action_mode = "body_rates" if mode == "rl_bodyrates" else "velocity"
    return mode, agent_action_mode


# =============================================================================
# Vision-based Training Configurations (Geles et al. architecture)
# Actor: segmap image + past actions only (no vector state)
# Critic: privileged full state (asymmetric actor-critic)
#
# Reference: "Demonstrating Agile Flight from Pixels without State Estimation"
# =============================================================================


def _vision_base_cfg(
    num_envs: int = 256,
    action_mode: str | None = None,
    num_past_actions: int = 3,
    sensor_mode: str = "depth",
) -> tuple[PursuitEvasionEnvCfg, str]:
    """Create base config for vision-based training.

    Args:
        sensor_mode: "depth", "segmap", or "both".

    Returns:
        Tuple of (config, rl_kind) for further customization.
    """
    cfg = get_base_config()
    cfg.scene.num_envs = num_envs
    cfg.training_agent = "pursuer"
    rl_kind, agent_action_mode = _resolve_action_mode(action_mode)
    cfg.agent_action_mode = agent_action_mode

    # Enable image-only mode (Geles et al. architecture)
    cfg.obs_include_segmap = sensor_mode in ("segmap", "both")
    cfg.obs_include_depth = sensor_mode in ("depth", "both")
    cfg.obs_image_history = 1  # Only current frame
    cfg.obs_image_only = True  # Actor gets ONLY image + past_actions
    cfg.obs_num_past_actions = num_past_actions
    cfg.enable_cameras = True

    # Camera configuration — wide FOV (~120° HFOV, practical limit for pinhole model)
    cfg.camera_width = 64
    cfg.camera_height = 64
    cfg.camera_fx = 18.5  # 64/2 / tan(60°) ≈ 18.5 → 120° HFOV
    cfg.camera_fy = 18.5

    # Critic gets propeller speeds and heuristic state (Markov s_t)
    cfg.critic_include_propeller_speeds = True
    cfg.critic_include_heuristic_state = True

    return cfg, rl_kind


# =============================================================================
# Critic ablation (the paper's experiment)
# Sensor mode and obstacle map are overridden via CLI (--sensor-mode, --enable-obstacles)
# =============================================================================


def ablation_vision_vs_trajectories_cfg(num_envs: int = 256) -> PursuitEvasionEnvCfg:
    """Stage 1 ablation: vision RL (body-rates) vs scripted evader mix.

    Evaders: hover(25%), circular(25%), lemniscate(25%), APF(25%).
    Critic state is Markov (64 dims with propeller speeds + heuristic state).

    Use ``--sensor-mode`` to select depth/segmap/both.
    Use ``--enable-obstacles`` to switch to cross-wall arena.
    Use ``--agent`` to select critic variant YAML.
    """
    cfg, rl_kind = _vision_base_cfg(num_envs, action_mode="rl_bodyrates", num_past_actions=1)
    cfg.wandb_run_name = "ablation_vision_vs_trajectories"
    cfg.drone_name = "crazyflie"  # brushed Crazyflie 2.x (matches real-world deployment)
    cfg.use_visual_ball_evader = False  # real drone mesh

    cfg.pursuer_controllers = [
        ControllerSpec(name=f"{rl_kind}_pursuer_pretrain", kind=rl_kind, count=num_envs),
    ]

    cfg.evader_controllers = [
        ControllerSpec(name="hover", count=num_envs, probability=0.25),
        ControllerSpec(name="circular", count=num_envs, probability=0.25),
        ControllerSpec(name="lemniscate", count=num_envs, probability=0.25),
        ControllerSpec(name="apf_evader", count=num_envs, probability=0.25, kind="apf_evader"),
    ]

    return cfg
