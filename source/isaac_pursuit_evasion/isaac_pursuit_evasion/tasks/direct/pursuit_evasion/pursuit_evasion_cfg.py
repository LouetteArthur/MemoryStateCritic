"""Pre-configured pursuit-evasion environment configurations and base config classes."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal, Optional
from collections.abc import Iterable, Sequence

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from source.isaac_pursuit_evasion.controllers.config import load_controller_config

# ---------------------------------------------------------------------------
# Pretrained-opponent artifacts
#
# These defaults point at the authors' private wandb project, which nobody else
# can read. They are only reached by the auxiliary tasks (pretrained opponents,
# warm-starts) -- **no experiment in the paper uses them**; the paper's task,
# Ablation-vision-vs-trajectories, trains against scripted evaders only.
#
# Override with PE_ARTIFACT_ENTITY / PE_ARTIFACT_PROJECT, or pass the artifact
# explicitly. Leaving them unset yields None, so the caller fails with a clear
# message instead of an opaque wandb 404.
# ---------------------------------------------------------------------------
PE_ARTIFACT_ENTITY = os.environ.get("PE_ARTIFACT_ENTITY")
PE_ARTIFACT_PROJECT = os.environ.get("PE_ARTIFACT_PROJECT")


def _artifact(name: str, alias: str = "latest") -> str | None:
    """Fully-qualified wandb artifact path, or None when no entity is configured."""
    if not PE_ARTIFACT_ENTITY or not PE_ARTIFACT_PROJECT:
        return None
    return f"{PE_ARTIFACT_ENTITY}/{PE_ARTIFACT_PROJECT}/{name}:{alias}"

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
    expose_opponent_z: bool = False      # Expose opponent's RNN hidden state z^opp in extras
    expose_opponent_obs: bool = False    # Expose opponent's image and prev_action in extras
    opponent_z_dim: int = 256            # Dimension of opponent z (GRU hidden size)
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
    critic_include_heuristic_state: bool = False  # Markov state: evader rot6d+ang_vel, PID integrals, heuristic one-hot, traj setpoint (32 dims)

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
    reward_time_scale: float = 1.0  # kappa_t: total time budget (per-step cost = kappa_t / T). Decoupled from R so terminal events (+/-R) dominate the dense time pressure.
    reward_approach: float = 3.0  # exponential potential shaping weight in Φ(d) = exp(-(d - r_c) / d_0)
    reward_approach_decay: float = 0.5  # d_0 in Φ(d) = exp(-(d - r_c) / d_0). d_0 ≈ success-threshold scale matches reach-task practice (Hwangbo'19, Lee'20, Rudin'22).
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


def uses_image_observations(cfg: PursuitEvasionEnvCfg) -> bool:
    """Check if the configuration uses image-based observations."""
    return cfg.obs_include_segmap or cfg.obs_include_depth


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


def _apply_domain_randomization(cfg: PursuitEvasionEnvCfg, enable: bool | None) -> None:
    if enable is None:
        return
    cfg.domain_randomization.enable = bool(enable)


def _resolve_action_mode(action_mode: str | None) -> tuple[str, str]:
    """Return (rl_kind, agent_action_mode)."""
    _AMSPB_ACTION_MODE_ENV = "AMSPB_ACTION_MODE"
    mode = action_mode or os.environ.get(_AMSPB_ACTION_MODE_ENV, "rl_velocity")
    mode = mode.strip()
    if mode not in {"rl_velocity", "rl_bodyrates"}:
        mode = "rl_velocity"
    agent_action_mode = "body_rates" if mode == "rl_bodyrates" else "velocity"
    return mode, agent_action_mode


def pretrain_rl_vs_trajectories_cfg(
    num_envs: int = 512,
    action_mode: str | None = None,
    domain_randomization: bool | None = None,
) -> PursuitEvasionEnvCfg:
    """Pursuer (RL) vs Evader following fixed trajectories."""
    hover_count = num_envs // 3
    circular_count = num_envs // 3
    lemniscate_count = num_envs - (hover_count + circular_count)

    cfg = get_base_config()
    cfg.scene.num_envs = num_envs
    rl_kind, agent_action_mode = _resolve_action_mode(action_mode)
    cfg.training_agent = "pursuer"
    cfg.agent_action_mode = agent_action_mode
    cfg.wandb_run_name = f"pretrain_{rl_kind}_vs_trajectories"

    # Pursuer uses RL policy
    cfg.pursuer_controllers = [
        ControllerSpec(name=f"{rl_kind}_pursuer_pretrain", kind=rl_kind, count=num_envs),
    ]

    # Evader uses fixed trajectories
    cfg.evader_controllers = [
        ControllerSpec(name="hover", count=hover_count),
        ControllerSpec(name="circular", count=circular_count),
        ControllerSpec(name="lemniscate", count=lemniscate_count),
    ]

    _apply_domain_randomization(cfg, domain_randomization)

    return cfg


def pretrain_rl_rate_vs_trajectories_cfg(
    num_envs: int = 512,
    domain_randomization: bool | None = None,
) -> PursuitEvasionEnvCfg:
    """Wrapper to pretrain RL body-rates controller vs trajectories."""
    return pretrain_rl_vs_trajectories_cfg(
        num_envs=num_envs,
        action_mode="rl_bodyrates",
        domain_randomization=domain_randomization,
    )


def pretrain_rl_vs_hover_cfg(
    num_envs: int = 512,
    action_mode: str | None = None,
    domain_randomization: bool | None = None,
) -> PursuitEvasionEnvCfg:
    """Pursuer (RL) vs evader hovering in place."""
    cfg = get_base_config()
    cfg.scene.num_envs = num_envs
    cfg.training_agent = "pursuer"
    rl_kind, agent_action_mode = _resolve_action_mode(action_mode)
    cfg.agent_action_mode = agent_action_mode
    cfg.wandb_run_name = f"pretrain_{rl_kind}_vs_hover"

    cfg.pursuer_controllers = [
        ControllerSpec(name=f"{rl_kind}_pursuer_pretrain", kind=rl_kind, count=num_envs),
    ]
    cfg.evader_controllers = [ControllerSpec(name="hover", count=num_envs)]
    _apply_domain_randomization(cfg, domain_randomization)
    return cfg


def pretrain_rl_rate_vs_hover_cfg(num_envs: int = 512) -> PursuitEvasionEnvCfg:
    """Wrapper to pretrain RL body-rates controller vs hovering evader."""
    return pretrain_rl_vs_hover_cfg(num_envs=num_envs, action_mode="rl_bodyrates")


def pretrain_rl_vs_circ_lemniscate_cfg(
    num_envs: int = 512,
    action_mode: str | None = None,
) -> PursuitEvasionEnvCfg:
    """Pursuer (RL) vs evader following circles and lemniscates (50/50 split)."""
    cfg = get_base_config()
    cfg.scene.num_envs = num_envs
    cfg.training_agent = "pursuer"
    rl_kind, agent_action_mode = _resolve_action_mode(action_mode)
    cfg.agent_action_mode = agent_action_mode
    cfg.wandb_run_name = f"pretrain_{rl_kind}_vs_circ_lemniscate"

    cfg.pursuer_controllers = [
        ControllerSpec(name=f"{rl_kind}_pursuer_pretrain", kind=rl_kind, count=num_envs),
    ]
    share = 0.5
    cfg.evader_controllers = [
        ControllerSpec(name="circular", count=num_envs, probability=share),
        ControllerSpec(name="lemniscate", count=num_envs, probability=1.0 - share),
    ]
    return cfg


def pretrain_rl_rate_vs_circ_lemniscate_cfg(num_envs: int = 512) -> PursuitEvasionEnvCfg:
    """Wrapper to pretrain RL body-rates controller vs circle/lemniscate evaders."""
    return pretrain_rl_vs_circ_lemniscate_cfg(num_envs=num_envs, action_mode="rl_bodyrates")


def bench_rl_vs_trajectories_cfg(
    pursuer_artifact: str | None = _artifact("pretrain_rl_vel_vs_trajectories"),
    action_mode: str | None = None,
) -> PursuitEvasionEnvCfg:
    """Benchmark pre-trained RL pursuer against pre-trained RL evader (artifacts from WandB)."""

    cfg = pretrain_rl_vs_trajectories_cfg(action_mode=action_mode)
    cfg.training_agent = ""
    cfg.wandb_artifact_defaults = {"entity": PE_ARTIFACT_ENTITY, "project": PE_ARTIFACT_PROJECT, "alias": "latest"}
    rl_kind, _ = _resolve_action_mode(action_mode)

    pursuer_payload = {"wandb_artifact": {"artifact": pursuer_artifact}}

    cfg.pursuer_controllers = [
        ControllerSpec(
            name=f"{rl_kind}_pursuer_pretrain",
            kind=rl_kind,
            count=cfg.scene.num_envs,
            config=pursuer_payload or None,
        )
    ]

    return cfg


def bench_rl_rate_vs_trajectories_cfg(
    pursuer_artifact: str | None = _artifact("pretrain_rl_bodyrates_vs_trajectories"),
) -> PursuitEvasionEnvCfg:
    """Body-rates variant of bench_rl_vs_trajectories_cfg."""
    return bench_rl_vs_trajectories_cfg(pursuer_artifact=pursuer_artifact, action_mode="rl_bodyrates")


def pretrain_frpn_vs_rl_cfg(
    num_envs: int = 1024,
    action_mode: str | None = None,
    drone_name: str = "crazyflie_brushless",
) -> PursuitEvasionEnvCfg:

    cfg = get_base_config()
    cfg.drone_name = drone_name
    cfg.scene.num_envs = num_envs
    cfg.training_agent = "evader"
    rl_kind, agent_action_mode = _resolve_action_mode(action_mode)
    cfg.agent_action_mode = agent_action_mode
    cfg.wandb_run_name = f"pretrain_frpn_vs_{rl_kind}"

    cfg.pursuer_controllers = [
        ControllerSpec(
            name="frpn_pursuer",
            count=num_envs,
            config_overrides={
                "curriculum": {"enabled": True, "start_fraction": 0.10, "ramp_fraction": 0.75},
            },
        )
    ]

    cfg.evader_controllers = [ControllerSpec(name=f"{rl_kind}_evader_pretrain", kind=rl_kind, count=num_envs)]

    return cfg


def pretrain_frpn_vs_rl_vel_cfg(num_envs: int = 1024) -> PursuitEvasionEnvCfg:
    """Wrapper to train RL velocity evader vs FRPN pursuer."""
    return pretrain_frpn_vs_rl_cfg(num_envs=num_envs, action_mode="rl_velocity")


def pretrain_frpn_vs_rl_rate_cfg(num_envs: int = 1024) -> PursuitEvasionEnvCfg:
    """Wrapper to train RL body-rates evader vs FRPN pursuer."""
    return pretrain_frpn_vs_rl_cfg(num_envs=num_envs, action_mode="rl_bodyrates")


def _checkpoint_payload(value: Any) -> dict[str, Any]:
    """Build controller config supporting local checkpoints or WandB artifacts."""
    if isinstance(value, dict):
        if "wandb_artifact" in value:
            return {"wandb_artifact": value["wandb_artifact"]}
        if "artifact" in value:
            return {"wandb_artifact": {"artifact": value["artifact"]}}
        if "checkpoint" in value:
            return {"checkpoint": value["checkpoint"]}
    if isinstance(value, str):
        from pathlib import Path

        path = Path(value).expanduser()
        if path.exists():
            return {"checkpoint": str(path)}
        return {"wandb_artifact": {"artifact": value}}
    raise ValueError("Unsupported checkpoint specification; provide a path or WandB artifact.")


def pretrain_frpn_vs_rl_warmstart_cfg(
    num_envs: int = 1024,
    action_mode: str | None = None,
    warmstart_checkpoint: str | None = _artifact("pretrain_frpn_vs_rl_velocity", "v25"),
) -> PursuitEvasionEnvCfg:
    """Evader pretrain vs FRPN pursuer but start from a pretrained evader checkpoint."""

    cfg = pretrain_frpn_vs_rl_cfg(num_envs=num_envs, action_mode=action_mode)
    if warmstart_checkpoint:
        cfg.training_warmstart = _checkpoint_payload(warmstart_checkpoint)
    return cfg


def bench_slowfrpn_vs_apf_cfg(
    num_envs: int = 1024,
    drone_name: str = "crazyflie_brushless",
) -> PursuitEvasionEnvCfg:

    cfg = get_base_config()
    cfg.drone_name = drone_name
    cfg.scene.num_envs = num_envs
    cfg.training_agent = ""

    cfg.pursuer_controllers = [
        ControllerSpec(
            name="frpn_pursuer",
            count=num_envs,
            config_overrides={
                "curriculum": {"enabled": True, "start_fraction": 0, "ramp_fraction": 0.99},
            },
        )
    ]

    cfg.evader_controllers = [ControllerSpec(name="apf_evader", count=num_envs)]
    return cfg


def bench_frpn_vs_apf_cfg(
    num_envs: int = 1024,
    drone_name: str = "crazyflie_brushless",
) -> PursuitEvasionEnvCfg:

    cfg = get_base_config()
    cfg.drone_name = drone_name
    cfg.scene.num_envs = num_envs
    cfg.training_agent = ""

    cfg.pursuer_controllers = [
        ControllerSpec(
            name="frpn_pursuer",
            count=num_envs,
            config_overrides={
                "curriculum": {"enabled": False, "start_fraction": 0.99, "ramp_fraction": 0.99},
            },
        )
    ]

    cfg.evader_controllers = [ControllerSpec(name="apf_evader", count=num_envs)]
    return cfg


def bench_pursuit_evasion_vision_cfg(num_envs: int = 128) -> PursuitEvasionEnvCfg:
    """Open-loop trajectories to debug FPV perception: half circles, half lemniscates for both agents."""
    cfg = get_base_config()
    cfg.scene.num_envs = num_envs
    cfg.training_agent = ""
    cfg.flag_draw_velocity_markers = True
    cfg.enable_cameras = True

    half = num_envs // 2
    traj_cfg = {"trajectory_horizon": 1024}

    # cfg.pursuer_controllers = [
    #     ControllerSpec(name="circular", count=half, config=traj_cfg),
    #     ControllerSpec(name="lemniscate", count=num_envs - half, config=traj_cfg),
    # ]

    curriculum_overrides = {
        "enabled": False,
        "start_fraction": 0.99,
        "ramp_fraction": 0.99,
    }
    frpn_config = None
    try:
        frpn_config = load_controller_config("frpn_pursuer", cfg.drone_name)
        frpn_config["curriculum"] = curriculum_overrides
    except FileNotFoundError:
        frpn_config = None
    cfg.pursuer_controllers = [
        ControllerSpec(
            name="frpn_pursuer",
            count=num_envs,
            config=frpn_config,
            config_overrides=None if frpn_config else curriculum_overrides,
        )
    ]

    cfg.evader_controllers = [
        ControllerSpec(name="circular", count=half, config=traj_cfg),
        ControllerSpec(name="lemniscate", count=num_envs - half, config=traj_cfg),
    ]
    return cfg


def bench_frpn_vs_hover_cfg(
    num_envs: int = 1024,
    drone_name: str = "crazyflie_brushless",
) -> PursuitEvasionEnvCfg:

    cfg = get_base_config()
    cfg.drone_name = drone_name
    cfg.scene.num_envs = num_envs
    cfg.training_agent = ""

    cfg.pursuer_controllers = [
        ControllerSpec(
            name="frpn_pursuer",
            count=num_envs,
            config_overrides={
                "curriculum": {"enabled": False, "start_fraction": 0.99, "ramp_fraction": 0.99},
            },
        )
    ]

    cfg.evader_controllers = [ControllerSpec(name="hover", count=num_envs)]
    return cfg


def bench_frpn_vs_trajectories_cfg(
    num_envs: int = 1024,
    drone_name: str = "crazyflie_brushless",
) -> PursuitEvasionEnvCfg:

    cfg = get_base_config()
    cfg.drone_name = drone_name
    cfg.scene.num_envs = num_envs
    cfg.training_agent = ""

    cfg.pursuer_controllers = [
        ControllerSpec(
            name="frpn_pursuer",
            count=num_envs,
            config_overrides={
                "curriculum": {"enabled": False, "start_fraction": 0.99, "ramp_fraction": 0.99},
            },
        )
    ]

    hover_count = num_envs // 3
    circular_count = num_envs // 3
    lemniscate_count = num_envs - (hover_count + circular_count)

    cfg.evader_controllers = [
        ControllerSpec(name="hover", count=hover_count),
        ControllerSpec(name="circular", count=circular_count),
        ControllerSpec(name="lemniscate", count=lemniscate_count),
    ]
    return cfg


def bench_rl_vs_apf_cfg(
    num_envs: int = 1024,
    pursuer_policy_artifact: str | None = None,
    frpn_curriculum: bool = False,
    frpn_curriculum_start_fraction: float = 0.0,
    frpn_curriculum_ramp_fraction: float = 0.75,
    drone_name: str = "crazyflie_brushless",
) -> PursuitEvasionEnvCfg:

    cfg = get_base_config()
    cfg.drone_name = drone_name
    cfg.scene.num_envs = num_envs
    cfg.training_agent = ""

    if pursuer_policy_artifact:
        cfg.pursuer_controllers = [
            ControllerSpec(
                name=pursuer_policy_artifact,
                kind="rl_velocity",
                count=num_envs,
                config={
                    "artifact_name": pursuer_policy_artifact,
                    "action_key": "action",
                    "root_state_key": "root_state",
                },
            )
        ]
    else:
        overrides = {}
        if frpn_curriculum:
            overrides["curriculum"] = {
                "enabled": True,
                "start_fraction": frpn_curriculum_start_fraction,
                "ramp_fraction": frpn_curriculum_ramp_fraction,
            }

        cfg.pursuer_controllers = [
            ControllerSpec(name="frpn_pursuer", count=num_envs, config_overrides=overrides or None)
        ]

    cfg.evader_controllers = [ControllerSpec(name="apf_evader", count=num_envs)]
    return cfg


def bench_pretrained_rl_vs_rl_cfg(
    num_envs: int = 512,
    pursuer_artifact: str | None = _artifact("pretrain_rl_vel_vs_trajectories"),
    evader_artifact: str | None = _artifact("pretrain_frpn_vs_rl_vel"),
    action_mode: str | None = None,
) -> PursuitEvasionEnvCfg:
    """Benchmark pre-trained RL pursuer against pre-trained RL evader (artifacts from WandB)."""

    cfg = get_base_config()
    cfg.scene.num_envs = num_envs
    cfg.training_agent = ""
    cfg.wandb_artifact_defaults = {"entity": PE_ARTIFACT_ENTITY, "project": PE_ARTIFACT_PROJECT, "alias": "latest"}
    rl_kind, _ = _resolve_action_mode(action_mode)

    pursuer_payload = {"wandb_artifact": {"artifact": pursuer_artifact}} if pursuer_artifact else {}
    evader_payload = {"wandb_artifact": {"artifact": evader_artifact}} if evader_artifact else {}

    cfg.pursuer_controllers = [
        ControllerSpec(
            name=f"{rl_kind}_pursuer_pretrain",
            kind=rl_kind,
            count=num_envs,
            config=pursuer_payload or None,
        )
    ]
    cfg.evader_controllers = [
        ControllerSpec(
            name=f"{rl_kind}_evader_pretrain",
            kind=rl_kind,
            count=num_envs,
            config=evader_payload or None,
        )
    ]

    return cfg


# =============================================================================
# AMSPB population configs
# =============================================================================
_AMSPB_CHECKPOINTS_ENV = "AMSPB_CHECKPOINTS"
_AMSPB_PROB_ENV = "AMSPB_BASELINE_PROB"
_AMSPB_ACTION_MODE_ENV = "AMSPB_ACTION_MODE"


def _load_amspb_checkpoint_map() -> dict[str, str]:
    raw = os.environ.get(_AMSPB_CHECKPOINTS_ENV, "")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    cleaned: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            cleaned[str(k)] = v
        else:
            cleaned[str(k)] = str(v)
    return cleaned


def _optional_checkpoint_payload(checkpoints: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Return a checkpoint payload if available; otherwise, return None without raising."""
    if not key:
        return None
    if key not in checkpoints:
        return None
    try:
        return _checkpoint_payload(checkpoints[key])
    except Exception:
        return None


def _require_checkpoint(checkpoints: dict[str, str], key: str) -> str:
    value = checkpoints.get(key)
    if isinstance(value, dict):
        path = value.get("checkpoint")
    else:
        path = value
    if path:
        return _checkpoint_payload(value)
    raise ValueError(
        f"Missing AMSPB checkpoint '{key}'. Set {_AMSPB_CHECKPOINTS_ENV} with a JSON map of stage->checkpoint."
    )


def _resolve_amspb_prob(probability: float | None) -> float:
    if probability is None:
        try:
            probability = float(os.environ.get(_AMSPB_PROB_ENV, 0.5))
        except Exception:
            probability = 0.5
    probability = max(0.0, min(1.0, float(probability)))
    return probability


def _previous_stage_key(training_agent: str, rl_kind: str, stage_index: int) -> str | None:
    """Return the checkpoint key for the prior stage (or pretrain) for warmstarts."""
    if stage_index <= 0:
        return None
    if stage_index == 1:
        return f"{training_agent}_{rl_kind}_pretrain"
    return f"{training_agent}_{rl_kind}_stage{stage_index - 1}"


def _build_frpn_spec(
    num_envs: int, probability: float, curriculum: bool = True, drone_name: str = "crazyflie_brushless"
) -> ControllerSpec:
    if curriculum:
        curriculum_cfg = {"enabled": True, "start_fraction": 0.10, "ramp_fraction": 0.75}
    else:
        curriculum_cfg = {"enabled": False, "start_fraction": 0.0, "ramp_fraction": 0.0}
    return ControllerSpec(
        name="frpn_pursuer",
        count=num_envs,
        probability=probability,
        config_overrides={"curriculum": curriculum_cfg},
    )


def _build_rl_spec(
    name: str, checkpoint_config: dict[str, Any], num_envs: int, probability: float, rl_kind: str
) -> ControllerSpec:
    # AMSPB pools always load CNN+GRU vision RNN policies as opponents.
    # Without ``recurrent_actor_cfg`` in the spec, the quadrotor manager falls
    # back to the MLP SimpleGaussianActor loader and tries to feed it the
    # actor's image obs (RuntimeError: mat1 and mat2 shapes can't be
    # multiplied (...x21 and 24x512)).  Inject the recurrent-actor descriptor
    # so the manager picks ``load_recurrent_actor_from_checkpoint``.
    config = dict(checkpoint_config or {})
    config.setdefault("recurrent_actor_cfg", {
        "image_channels": 2,        # depth + segmap (sensor-mode=both)
        "image_height": 64,
        "image_width": 64,
        "past_actions_size": 4,     # num_past_actions=1 × action_dim=4
        "action_dim": 4,            # body rates: roll/pitch/yaw/thrust
        "cnn_feature_size": 128,
        "rnn_hidden_size": 256,
        "rnn_num_layers": 1,
    })
    return ControllerSpec(
        name=name,
        count=num_envs,
        probability=probability,
        kind=rl_kind,
        config=config,
    )


def _trajectory_controllers(num_envs: int, total_probability: float) -> list[ControllerSpec]:
    if total_probability <= 0.0:
        return []
    share = total_probability / 3.0
    return [
        ControllerSpec(name="hover", count=num_envs, probability=share),
        ControllerSpec(name="circular", count=num_envs, probability=share),
        ControllerSpec(name="lemniscate", count=num_envs, probability=share),
    ]


def _amspb_pursuer_pool(
    stage_index: int,
    rl_kind: str,
    num_envs: int,
    baseline_prob: float,
    checkpoints: dict[str, Any],
    drone_name: str = "crazyflie_brushless",
) -> list[ControllerSpec]:
    """Opponents when training the evader (pursuer controllers).

    Stage 1: FRPN (baseline_prob) + pretrained pursuer (1 - baseline_prob).
    Stage N>1: FRPN (baseline_prob/2) + pretrained pursuer (baseline_prob/2) +
               latest pursuer checkpoint (1 - baseline_prob).
    """
    if stage_index == 1:
        return [
            _build_frpn_spec(num_envs, baseline_prob, curriculum=False, drone_name=drone_name),
            _build_rl_spec(
                name=f"{rl_kind}_pursuer_pretrain",
                checkpoint_config=_require_checkpoint(checkpoints, f"pursuer_{rl_kind}_pretrain"),
                num_envs=num_envs,
                probability=1.0 - baseline_prob,
                rl_kind=rl_kind,
            ),
        ]
    # Stage N>1: baseline mix + latest pursuer from previous stage
    shared = baseline_prob * 0.5
    latest_key = f"pursuer_{rl_kind}_stage{stage_index - 1}"
    return [
        _build_frpn_spec(num_envs, shared, curriculum=False, drone_name=drone_name),
        _build_rl_spec(
            name=f"{rl_kind}_pursuer_pretrain",
            checkpoint_config=_require_checkpoint(checkpoints, f"pursuer_{rl_kind}_pretrain"),
            num_envs=num_envs,
            probability=shared,
            rl_kind=rl_kind,
        ),
        _build_rl_spec(
            name=f"{rl_kind}_pursuer_stage{stage_index - 1}",
            checkpoint_config=_require_checkpoint(checkpoints, latest_key),
            num_envs=num_envs,
            probability=1.0 - baseline_prob,
            rl_kind=rl_kind,
        ),
    ]


def _amspb_evader_pool(
    stage_index: int, rl_kind: str, num_envs: int, baseline_prob: float, checkpoints: dict[str, Any]
) -> list[ControllerSpec]:
    """Opponents when training the pursuer (evader controllers).

    Stage N: trajectories (baseline_prob) + latest evader checkpoint (1 - baseline_prob).
    The evader checkpoint key is evader_{rl_kind}_stage{N} (same stage index).
    """
    latest_key = f"evader_{rl_kind}_stage{stage_index}"
    controllers = _trajectory_controllers(num_envs, baseline_prob)
    controllers.append(
        _build_rl_spec(
            name=f"{rl_kind}_evader_stage{stage_index}",
            checkpoint_config=_require_checkpoint(checkpoints, latest_key),
            num_envs=num_envs,
            probability=1.0 - baseline_prob,
            rl_kind=rl_kind,
        )
    )
    return controllers


def _amspb_cfg(
    stage_index: int,
    training_agent: str,
    num_envs: int,
    baseline_prob: float | None = None,
    action_mode: str | None = None,
    drone_name: str = "crazyflie_brushless",
) -> PursuitEvasionEnvCfg:
    """Generic AMSPB stage builder (expects checkpoints via env AMSPB_CHECKPOINTS)."""
    rl_kind, agent_action_mode = _resolve_action_mode(action_mode)
    stage_name = f"{training_agent}_{rl_kind}_stage{stage_index}"
    cfg = get_base_config()
    cfg.drone_name = drone_name
    cfg.scene.num_envs = num_envs
    cfg.training_agent = training_agent
    cfg.agent_action_mode = agent_action_mode
    cfg.wandb_run_name = f"amspb_{stage_name}"

    prob = _resolve_amspb_prob(baseline_prob)
    checkpoints = _load_amspb_checkpoint_map()
    warmstart_key = _previous_stage_key(training_agent, rl_kind, stage_index)
    if cfg.training_warmstart is None:
        cfg.training_warmstart = _optional_checkpoint_payload(checkpoints, warmstart_key)

    rl_spec = ControllerSpec(name=f"{rl_kind}_{training_agent}_train", kind=rl_kind, count=num_envs)
    if training_agent == "pursuer":
        cfg.pursuer_controllers = [rl_spec]
        cfg.evader_controllers = _amspb_evader_pool(stage_index, rl_kind, num_envs, prob, checkpoints)
    elif training_agent == "evader":
        cfg.evader_controllers = [rl_spec]
        cfg.pursuer_controllers = _amspb_pursuer_pool(
            stage_index, rl_kind, num_envs, prob, checkpoints, drone_name=drone_name
        )
    else:
        raise ValueError(f"Unsupported training agent '{training_agent}' for AMSPB.")
    return cfg


def amspb_evader_stage1_cfg(
    num_envs: int = 512, baseline_prob: float | None = None, action_mode: str | None = None
) -> PursuitEvasionEnvCfg:
    """Train evader vs FRPN and pretrained pursuer (probabilities controlled by baseline_prob / env)."""
    return _amspb_cfg(1, "evader", num_envs=num_envs, baseline_prob=baseline_prob, action_mode=action_mode)


def amspb_pursuer_stage1_cfg(
    num_envs: int = 512, baseline_prob: float | None = None, action_mode: str | None = None
) -> PursuitEvasionEnvCfg:
    """Train pursuer vs trajectories and evader from AMSPB stage 1."""
    return _amspb_cfg(1, "pursuer", num_envs=num_envs, baseline_prob=baseline_prob, action_mode=action_mode)


def amspb_evader_stage2_cfg(
    num_envs: int = 512, baseline_prob: float | None = None, action_mode: str | None = None
) -> PursuitEvasionEnvCfg:
    """Train evader vs FRPN, old pursuer, and the latest pursuer (stage 1)."""
    return _amspb_cfg(2, "evader", num_envs=num_envs, baseline_prob=baseline_prob, action_mode=action_mode)


def amspb_pursuer_stage2_cfg(
    num_envs: int = 512, baseline_prob: float | None = None, action_mode: str | None = None
) -> PursuitEvasionEnvCfg:
    """Train pursuer vs trajectories and evader from AMSPB stage 2."""
    return _amspb_cfg(2, "pursuer", num_envs=num_envs, baseline_prob=baseline_prob, action_mode=action_mode)


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


# -----------------------------------------------------------------------------
# Pretrain vs Hover
# -----------------------------------------------------------------------------


def pretrain_rl_vision_vs_hover_cfg(
    num_envs: int = 256,
    action_mode: str | None = None,
    num_past_actions: int = 3,
) -> PursuitEvasionEnvCfg:
    """Train vision-based RL vs hovering evader."""
    cfg, rl_kind = _vision_base_cfg(num_envs, action_mode, num_past_actions)

    cfg.wandb_run_name = f"pretrain_{rl_kind}_vision_vs_hover"

    cfg.pursuer_controllers = [
        ControllerSpec(name=f"{rl_kind}_pursuer_pretrain", kind=rl_kind, count=num_envs),
    ]
    cfg.evader_controllers = [ControllerSpec(name="hover", count=num_envs)]

    return cfg


def pretrain_rl_rate_vision_vs_hover_cfg(num_envs: int = 256) -> PursuitEvasionEnvCfg:
    """Body-rates RL with vision vs hovering evader."""
    return pretrain_rl_vision_vs_hover_cfg(num_envs=num_envs, action_mode="rl_bodyrates")


def pretrain_rl_rate_vision_rnn_vs_hover_cfg(num_envs: int = 256) -> PursuitEvasionEnvCfg:
    """Body-rates RL with CNN+GRU vision vs hovering evader.

    Uses depth-only image input.  GRU provides temporal memory so only 1
    past action is kept (for env compatibility; the model ignores it).
    """
    cfg = pretrain_rl_vision_vs_hover_cfg(num_envs=num_envs, action_mode="rl_bodyrates", num_past_actions=1)
    cfg.wandb_run_name = "pretrain_rl_bodyrates_vision_rnn_vs_hover"
    return cfg


def pretrain_rl_rate_vision_rnn_vs_trajectories_cfg(num_envs: int = 256) -> PursuitEvasionEnvCfg:
    """Body-rates RL with CNN+GRU vision vs trajectory evaders."""
    cfg = pretrain_rl_vision_vs_trajectories_cfg(num_envs=num_envs, action_mode="rl_bodyrates", num_past_actions=1)
    cfg.wandb_run_name = "pretrain_rl_bodyrates_vision_rnn_vs_trajectories"
    return cfg


def pretrain_rl_rate_vision_symmetric_vs_hover_cfg(num_envs: int = 256) -> PursuitEvasionEnvCfg:
    """Body-rates RL with vision vs hovering evader — symmetric (no privileged critic)."""
    cfg = pretrain_rl_vision_vs_hover_cfg(num_envs=num_envs, action_mode="rl_bodyrates")
    cfg.asymmetric_actor_critic = False
    cfg.wandb_run_name = "pretrain_rl_bodyrates_vision_symmetric_vs_hover"
    return cfg


def pretrain_rl_rate_vision_unbiased_vs_hover_cfg(num_envs: int = 256) -> PursuitEvasionEnvCfg:
    """Body-rates RL with vision vs hover — unbiased asymmetric critic V(s, o)."""
    cfg = pretrain_rl_vision_vs_hover_cfg(num_envs=num_envs, action_mode="rl_bodyrates")
    cfg.unbiased_critic = True
    cfg.wandb_run_name = "pretrain_rl_bodyrates_vision_unbiased_vs_hover"
    return cfg


def pretrain_rl_vel_vision_vs_hover_cfg(num_envs: int = 256) -> PursuitEvasionEnvCfg:
    """Velocity RL with vision vs hovering evader."""
    return pretrain_rl_vision_vs_hover_cfg(num_envs=num_envs, action_mode="rl_velocity")


# -----------------------------------------------------------------------------
# Pretrain vs Trajectories (hover, circular, lemniscate mix)
# -----------------------------------------------------------------------------


def pretrain_rl_vision_vs_trajectories_cfg(
    num_envs: int = 256,
    action_mode: str | None = None,
    num_past_actions: int = 3,
    domain_randomization: bool | None = None,
) -> PursuitEvasionEnvCfg:
    """Train vision-based RL vs trajectory-following evader (hover/circular/lemniscate)."""
    cfg, rl_kind = _vision_base_cfg(num_envs, action_mode, num_past_actions)

    cfg.wandb_run_name = f"pretrain_{rl_kind}_vision_vs_trajectories"

    cfg.pursuer_controllers = [
        ControllerSpec(name=f"{rl_kind}_pursuer_pretrain", kind=rl_kind, count=num_envs),
    ]

    # Trajectory mix (same as pretrain_rl_vs_trajectories_cfg)
    hover_share, circular_share = 1 / 3, 1 / 3
    cfg.evader_controllers = [
        ControllerSpec(name="hover", count=num_envs, probability=hover_share),
        ControllerSpec(name="circular", count=num_envs, probability=circular_share),
        ControllerSpec(name="lemniscate", count=num_envs, probability=1.0 - hover_share - circular_share),
    ]

    _apply_domain_randomization(cfg, domain_randomization)
    return cfg


def pretrain_rl_rate_vision_vs_trajectories_cfg(
    num_envs: int = 256,
    domain_randomization: bool | None = None,
) -> PursuitEvasionEnvCfg:
    """Body-rates RL with vision vs trajectories."""
    return pretrain_rl_vision_vs_trajectories_cfg(
        num_envs=num_envs,
        action_mode="rl_bodyrates",
        domain_randomization=domain_randomization,
    )


def pretrain_rl_rate_vision_unbiased_vs_trajectories_cfg(
    num_envs: int = 256,
    domain_randomization: bool | None = None,
) -> PursuitEvasionEnvCfg:
    """Body-rates RL with vision vs trajectories — unbiased asymmetric critic V(s, o)."""
    cfg = pretrain_rl_vision_vs_trajectories_cfg(
        num_envs=num_envs,
        action_mode="rl_bodyrates",
        domain_randomization=domain_randomization,
    )
    cfg.unbiased_critic = True
    cfg.wandb_run_name = "pretrain_rl_bodyrates_vision_unbiased_vs_trajectories"
    return cfg


def pretrain_rl_rate_vision_symmetric_vs_trajectories_cfg(
    num_envs: int = 256,
    domain_randomization: bool | None = None,
) -> PursuitEvasionEnvCfg:
    """Body-rates RL with vision vs trajectories — symmetric (no privileged critic)."""
    cfg = pretrain_rl_vision_vs_trajectories_cfg(
        num_envs=num_envs,
        action_mode="rl_bodyrates",
        domain_randomization=domain_randomization,
    )
    cfg.asymmetric_actor_critic = False
    cfg.wandb_run_name = "pretrain_rl_bodyrates_vision_symmetric_vs_trajectories"
    return cfg


def pretrain_rl_vel_vision_vs_trajectories_cfg(
    num_envs: int = 256,
    domain_randomization: bool | None = None,
) -> PursuitEvasionEnvCfg:
    """Velocity RL with vision vs trajectories."""
    return pretrain_rl_vision_vs_trajectories_cfg(
        num_envs=num_envs,
        action_mode="rl_velocity",
        domain_randomization=domain_randomization,
    )


# -----------------------------------------------------------------------------
# Pretrain vs Circular + Lemniscate (no hover)
# -----------------------------------------------------------------------------


def pretrain_rl_vision_vs_circ_lemniscate_cfg(
    num_envs: int = 256,
    action_mode: str | None = None,
    num_past_actions: int = 3,
) -> PursuitEvasionEnvCfg:
    """Train vision-based RL vs circular and lemniscate trajectories (no hover)."""
    cfg, rl_kind = _vision_base_cfg(num_envs, action_mode, num_past_actions)

    cfg.wandb_run_name = f"pretrain_{rl_kind}_vision_vs_circ_lemniscate"

    cfg.pursuer_controllers = [
        ControllerSpec(name=f"{rl_kind}_pursuer_pretrain", kind=rl_kind, count=num_envs),
    ]

    cfg.evader_controllers = [
        ControllerSpec(name="circular", count=num_envs, probability=0.5),
        ControllerSpec(name="lemniscate", count=num_envs, probability=0.5),
    ]

    return cfg


def pretrain_rl_rate_vision_vs_circ_lemniscate_cfg(num_envs: int = 256) -> PursuitEvasionEnvCfg:
    """Body-rates RL with vision vs circular/lemniscate."""
    return pretrain_rl_vision_vs_circ_lemniscate_cfg(num_envs=num_envs, action_mode="rl_bodyrates")


def pretrain_rl_vel_vision_vs_circ_lemniscate_cfg(num_envs: int = 256) -> PursuitEvasionEnvCfg:
    """Velocity RL with vision vs circular/lemniscate."""
    return pretrain_rl_vision_vs_circ_lemniscate_cfg(num_envs=num_envs, action_mode="rl_velocity")


# -----------------------------------------------------------------------------
# AMSPB Vision Stages (Alternating Multi-Stage Population-Based training)
# -----------------------------------------------------------------------------


_AMSPB_STAGE_ENV = "AMSPB_STAGE"


def _amspb_vision_cfg(
    stage: int | None = None,
    training_agent: str = "pursuer",
    num_envs: int = 256,
    baseline_prob: float | None = None,
    action_mode: str | None = None,
    num_past_actions: int = 3,
    sensor_mode: str = "both",
) -> PursuitEvasionEnvCfg:
    """Create vision-based AMSPB stage configuration.

    Loads opponent checkpoints from the AMSPB_CHECKPOINTS env var (same
    mechanism as state-based AMSPB).  Stage index can be passed directly
    or read from the AMSPB_STAGE env var.
    """
    if stage is None:
        stage = int(os.environ.get(_AMSPB_STAGE_ENV, "1"))

    cfg, rl_kind = _vision_base_cfg(num_envs, action_mode, num_past_actions, sensor_mode)
    cfg.training_agent = training_agent
    cfg.drone_name = "crazyflie"
    cfg.use_visual_ball_evader = False
    cfg.wandb_run_name = f"amspb_vision_{training_agent}_stage{stage}"
    # AMSPB pool members must be distinguishable to the joint memory-state
    # critic — see Proposition 2 in paper v22 (injectivity of e(k_t)).
    cfg.expose_opp_id = True
    # The SZZ joint memory-state critic concatenates z_opp (the opponent's
    # GRU hidden state) into its MLP head. Without this flag the env never
    # populates extras["z_opp"], the agent reads None, the critic falls
    # back to zeros, and V(s, z, z_opp, e(k)) silently degrades to
    # V(s, z, 0, e(k)) — NOT the joint critic the paper claims.
    cfg.expose_opponent_z = True
    # The SHH joint history-state critic's opp_branch re-encodes the opponent's
    # camera image (extras["opp_image"]). REQUIRED for the same reason as
    # expose_opponent_z above: without it the env never populates
    # extras["opp_image"]/["opp_prev_action"], the agent registers an
    # opp_image memory tensor but never writes it, and the critic reads
    # UNINITIALISED memory (NaN) during BPTT — cascading into invalid_state=1.
    # SZZ was unaffected because it uses z_opp, not opp_image. (Root cause of
    # the SHH NaN divergence.)
    cfg.expose_opponent_obs = True

    prob = _resolve_amspb_prob(baseline_prob)
    checkpoints = _load_amspb_checkpoint_map()
    warmstart_key = _previous_stage_key(training_agent, rl_kind, stage)
    if cfg.training_warmstart is None:
        cfg.training_warmstart = _optional_checkpoint_payload(checkpoints, warmstart_key)

    rl_spec = ControllerSpec(name=f"{rl_kind}_{training_agent}_train", kind=rl_kind, count=num_envs)

    # AMSPB pools contain CNN+GRU vision policies in both roles. The loaded
    # RL opponent needs its own FPV camera image at runtime, so always enable
    # cameras for both agents during AMSPB stages (regardless of who's
    # training). Without this the recurrent-loader path raises at first step.
    cfg.enable_evader_cameras = True
    if training_agent == "evader":
        cfg.evader_controllers = [rl_spec]
        # Forward our drone_name to the pursuer pool builder so the FRPN
        # baseline opponent uses the same dynamics as the training drone
        # (default was crazyflie_brushless — mismatch with our crazyflie
        # caused step-1 NaN in mixed pools).
        cfg.pursuer_controllers = _amspb_pursuer_pool(
            stage, rl_kind, num_envs, prob, checkpoints, drone_name=cfg.drone_name
        )
    elif training_agent == "pursuer":
        cfg.pursuer_controllers = [rl_spec]
        cfg.evader_controllers = _amspb_evader_pool(stage, rl_kind, num_envs, prob, checkpoints)
    else:
        raise ValueError(f"Unsupported training agent '{training_agent}' for AMSPB.")

    return cfg


def amspb_vision_evader_stage1_cfg(
    num_envs: int = 256,
    baseline_prob: float | None = None,
    action_mode: str | None = None,
    num_past_actions: int = 3,
) -> PursuitEvasionEnvCfg:
    """Vision AMSPB Stage 1: Train evader vs FRPN pursuer."""
    return _amspb_vision_cfg(1, "evader", num_envs, baseline_prob, action_mode, num_past_actions)


def amspb_vision_pursuer_stage1_cfg(
    num_envs: int = 256,
    baseline_prob: float | None = None,
    action_mode: str | None = None,
    num_past_actions: int = 3,
) -> PursuitEvasionEnvCfg:
    """Vision AMSPB Stage 1: Train pursuer vs trajectories."""
    return _amspb_vision_cfg(1, "pursuer", num_envs, baseline_prob, action_mode, num_past_actions)


def amspb_vision_evader_stage2_cfg(
    num_envs: int = 256,
    baseline_prob: float | None = None,
    action_mode: str | None = None,
    num_past_actions: int = 3,
) -> PursuitEvasionEnvCfg:
    """Vision AMSPB Stage 2: Train evader vs FRPN + pursuer checkpoint."""
    return _amspb_vision_cfg(2, "evader", num_envs, baseline_prob, action_mode, num_past_actions)


def amspb_vision_pursuer_stage2_cfg(
    num_envs: int = 256,
    baseline_prob: float | None = None,
    action_mode: str | None = None,
    num_past_actions: int = 3,
) -> PursuitEvasionEnvCfg:
    """Vision AMSPB Stage 2: Train pursuer vs trajectories + evader checkpoint."""
    return _amspb_vision_cfg(2, "pursuer", num_envs, baseline_prob, action_mode, num_past_actions)


# Body-rates wrappers for AMSPB
def amspb_vision_rate_evader_stage1_cfg(num_envs: int = 256) -> PursuitEvasionEnvCfg:
    """Body-rates vision AMSPB Stage 1: Train evader."""
    return amspb_vision_evader_stage1_cfg(num_envs=num_envs, action_mode="rl_bodyrates")


def amspb_vision_rate_pursuer_stage1_cfg(num_envs: int = 256) -> PursuitEvasionEnvCfg:
    """Body-rates vision AMSPB Stage 1: Train pursuer."""
    return amspb_vision_pursuer_stage1_cfg(num_envs=num_envs, action_mode="rl_bodyrates")


def amspb_vision_rate_evader_stage2_cfg(num_envs: int = 256) -> PursuitEvasionEnvCfg:
    """Body-rates vision AMSPB Stage 2: Train evader."""
    return amspb_vision_evader_stage2_cfg(num_envs=num_envs, action_mode="rl_bodyrates")


def amspb_vision_rate_pursuer_stage2_cfg(num_envs: int = 256) -> PursuitEvasionEnvCfg:
    """Body-rates vision AMSPB Stage 2: Train pursuer."""
    return amspb_vision_pursuer_stage2_cfg(num_envs=num_envs, action_mode="rl_bodyrates")


# Velocity wrappers for AMSPB
def amspb_vision_vel_evader_stage1_cfg(num_envs: int = 256) -> PursuitEvasionEnvCfg:
    """Velocity vision AMSPB Stage 1: Train evader."""
    return amspb_vision_evader_stage1_cfg(num_envs=num_envs, action_mode="rl_velocity")


def amspb_vision_vel_pursuer_stage1_cfg(num_envs: int = 256) -> PursuitEvasionEnvCfg:
    """Velocity vision AMSPB Stage 1: Train pursuer."""
    return amspb_vision_pursuer_stage1_cfg(num_envs=num_envs, action_mode="rl_velocity")


def amspb_vision_vel_evader_stage2_cfg(num_envs: int = 256) -> PursuitEvasionEnvCfg:
    """Velocity vision AMSPB Stage 2: Train evader."""
    return amspb_vision_evader_stage2_cfg(num_envs=num_envs, action_mode="rl_velocity")


def amspb_vision_vel_pursuer_stage2_cfg(num_envs: int = 256) -> PursuitEvasionEnvCfg:
    """Velocity vision AMSPB Stage 2: Train pursuer."""
    return amspb_vision_pursuer_stage2_cfg(num_envs=num_envs, action_mode="rl_velocity")


# -----------------------------------------------------------------------------
# Vision + RNN AMSPB (parametric stage from AMSPB_STAGE env var)
# CNN+GRU actor with num_past_actions=1, stage read at runtime.
# -----------------------------------------------------------------------------


def amspb_vision_rnn_pursuer_stage_cfg(num_envs: int = 256) -> PursuitEvasionEnvCfg:
    """Vision+RNN AMSPB: train pursuer (stage from AMSPB_STAGE env var)."""
    return _amspb_vision_cfg(
        stage=None, training_agent="pursuer", num_envs=num_envs, num_past_actions=1,
    )


def amspb_vision_rnn_evader_stage_cfg(num_envs: int = 256) -> PursuitEvasionEnvCfg:
    """Vision+RNN AMSPB: train evader (stage from AMSPB_STAGE env var)."""
    return _amspb_vision_cfg(
        stage=None, training_agent="evader", num_envs=num_envs, num_past_actions=1,
    )


def pretrain_vision_rnn_evader_cfg(num_envs: int = 256) -> PursuitEvasionEnvCfg:
    """Pretrain evader with vision+RNN against FRPN pursuer.

    Used as AMSPB stage 0: evader learns basic evasion before co-evolution.
    """
    cfg, rl_kind = _vision_base_cfg(num_envs, action_mode=None, num_past_actions=1, sensor_mode="both")
    cfg.training_agent = "evader"
    cfg.drone_name = "crazyflie"
    cfg.use_visual_ball_evader = False
    cfg.enable_evader_cameras = True
    cfg.wandb_run_name = "pretrain_vision_rnn_evader"

    cfg.evader_controllers = [
        ControllerSpec(name=f"{rl_kind}_evader_pretrain", kind=rl_kind, count=num_envs),
    ]
    cfg.pursuer_controllers = [
        _build_frpn_spec(num_envs, probability=1.0, curriculum=False, drone_name="crazyflie"),
    ]
    return cfg


# =============================================================================
# Ablation Study Configurations
# 4 algorithms × 3 sensors × 2 maps = 24 experiments
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
    cfg.use_visual_ball_evader = False  # real drone mesh (consistent with AMSPB stages)

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
