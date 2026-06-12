"""Evaluate a pursuer checkpoint against an evader for N episodes.

Called by run_tournament.py to evaluate one (pursuer, evader) match.
Outputs a single JSON line to stdout with episode outcome counts.

Usage:
    python scripts/skrl/eval_match.py \\
        --task Ablation-vision-vs-trajectories \\
        --agent skrl_ppo_vision_rnn_cfg_entry_point \\
        --pursuer-checkpoint path/to/best_agent.pt \\
        --evader-type hover \\
        --episodes 100 \\
        --num_envs 64 \\
        --headless --enable_cameras
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os
import sys

# Auto-add project root to PYTHONPATH.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a pursuer vs evader match.")
parser.add_argument("--task", type=str, required=True, help="Task name")
parser.add_argument("--agent", type=str, required=True, help="Agent config entry point")
parser.add_argument("--pursuer-checkpoint", type=str, required=True, help="Path to pursuer .pt checkpoint")
parser.add_argument(
    "--evader-type",
    type=str,
    required=True,
    help="Evader type: hover, circular, lemniscate, frpn, or checkpoint (uses --evader-checkpoint).",
)
parser.add_argument("--evader-checkpoint", type=str, default=None, help="Path to evader .pt checkpoint (if evader-type=checkpoint)")
parser.add_argument(
    "--evader-rl-kind",
    type=str,
    default="rl_bodyrates",
    choices=["rl_bodyrates", "rl_velocity"],
    help="RL controller kind for the evader checkpoint (must match how it was trained).",
)
parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to run")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--sensor-mode", type=str, default="segmap", choices=["depth", "segmap", "both"])
parser.add_argument("--enable-obstacles", action="store_true", default=False, help="Enable wall map")
parser.add_argument(
    "--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"],
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import gymnasium as gym

from isaac_pursuit_evasion.skrl_ext import CustomRunner as Runner
from isaaclab.envs import DirectRLEnvCfg, DirectMARLEnv, DirectMARLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
import isaac_pursuit_evasion.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

agent_cfg_entry_point = args_cli.agent


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Run evaluation match."""
    # Override env config
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = args_cli.seed

    # Configure sensor mode
    if hasattr(env_cfg, "obs_include_depth") and hasattr(env_cfg, "obs_include_segmap"):
        if args_cli.sensor_mode == "depth":
            env_cfg.obs_include_depth = True
            env_cfg.obs_include_segmap = False
        elif args_cli.sensor_mode == "segmap":
            env_cfg.obs_include_depth = False
            env_cfg.obs_include_segmap = True
        elif args_cli.sensor_mode == "both":
            env_cfg.obs_include_depth = True
            env_cfg.obs_include_segmap = True

    if args_cli.enable_obstacles:
        env_cfg.enable_obstacles = True

    # Override evader controllers based on --evader-type / --evader-checkpoint.
    # Otherwise eval inherits whatever the task cfg defaults to (often a mix).
    from isaac_pursuit_evasion.tasks.direct.pursuit_evasion.pursuit_evasion_cfg import (
        ControllerSpec,
        _checkpoint_payload,
    )

    n = args_cli.num_envs
    evader_type = args_cli.evader_type
    if evader_type in ("hover", "circular", "lemniscate"):
        env_cfg.evader_controllers = [ControllerSpec(name=evader_type, count=n)]
    elif evader_type == "frpn":
        env_cfg.evader_controllers = [ControllerSpec(name="frpn_pursuer", count=n)]
    elif evader_type == "checkpoint":
        if not args_cli.evader_checkpoint:
            raise ValueError("--evader-checkpoint is required when --evader-type=checkpoint")
        rl_kind = args_cli.evader_rl_kind
        env_cfg.evader_controllers = [
            ControllerSpec(
                name=f"{rl_kind}_evader_eval",
                kind=rl_kind,
                count=n,
                config=_checkpoint_payload(args_cli.evader_checkpoint),
            )
        ]
    else:
        raise ValueError(f"Unsupported evader-type '{evader_type}'.")

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Get reference to unwrapped env for done reasons
    raw_env = env.unwrapped

    # Wrap for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # Configure agent (no logging, no checkpoints)
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    experiment_cfg["seed"] = args_cli.seed
    runner = Runner(env, experiment_cfg)

    # Load pursuer checkpoint
    runner.agent.load(args_cli.pursuer_checkpoint)
    runner.agent.set_running_mode("eval")

    # Run episodes and count outcomes
    num_envs = args_cli.num_envs
    target_episodes = args_cli.episodes
    completed_episodes = 0

    # Outcome counters
    counts = {
        "captures": 0,                # reason 1
        "pursuer_oob": 0,             # reason 3
        "evader_oob": 0,              # reason 4
        "timeouts": 0,                # reason 5
        "invalid": 0,                 # reason 6
        "pursuer_wall_collision": 0,  # reason 7
        "evader_wall_collision": 0,   # reason 8
    }

    reason_to_key = {
        1: "captures",
        3: "pursuer_oob",
        4: "evader_oob",
        5: "timeouts",
        6: "invalid",
        7: "pursuer_wall_collision",
        8: "evader_wall_collision",
    }

    obs, _ = env.reset()

    while completed_episodes < target_episodes and simulation_app.is_running():
        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, terminated, truncated, infos = env.step(actions)

        # Check which envs just finished
        dones = terminated | truncated
        if dones.any():
            done_reasons = raw_env.get_last_done_reasons()
            done_mask = dones.squeeze(-1) if dones.dim() > 1 else dones
            done_envs = torch.where(done_mask)[0]

            for env_id in done_envs:
                if completed_episodes >= target_episodes:
                    break
                reason = int(done_reasons[env_id].item())
                key = reason_to_key.get(reason)
                if key:
                    counts[key] += 1
                completed_episodes += 1

    counts["episodes"] = completed_episodes

    # Print progress to stderr (not captured as result)
    print(f"[EVAL] Completed {completed_episodes} episodes", file=sys.stderr)

    # Output JSON result on stdout (parsed by run_tournament.py)
    print(json.dumps(counts))

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
