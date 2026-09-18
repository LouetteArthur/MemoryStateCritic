# Copyright (c) 2026, the MemoryStateCritic authors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode-level stats tracking for pursuit-evasion tasks."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import torch

logger = logging.getLogger(__name__)


class PursuitEvasionStatsTracker:
    """Collect per-episode stats for pursuit-evasion rollouts."""

    def __init__(
        self,
        *,
        num_envs: int,
        device: torch.device,
        done_reason_labels: dict[int, str],
        speed_agent: str,
        agents: Iterable[str] = ("pursuer", "evader"),
        termination_rate_window: int = 10240,
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self.speed_agent = speed_agent
        self.agents = tuple(agents)
        self._done_reason_labels = done_reason_labels

        self.episode_reward_sums: dict[str, dict[str, torch.Tensor]] = {agent: {} for agent in self.agents}
        speed_keys = ("linear", "angular", "acceleration")
        self.speed_stats = {self.speed_agent: {key: self._create_speed_tracker() for key in speed_keys}}
        self.prev_lin_vel = {self.speed_agent: torch.zeros(self.num_envs, 3, device=self.device)}
        self.rho_stats = {
            agent: {
                "sum": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
                "count": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            }
            for agent in self.agents
        }
        # Spatial-extent tracking: per-env running max |x|, max |y|, max z, min z
        # plus running min distance from any arena boundary. Cleared per episode.
        # Used to diagnose whether the policy explores the arena or hugs the spawn
        # region (Position/Agent/MaxAbsX/mean tells you the typical x reach).
        inf = float("inf")
        self.position_extent_stats = {
            agent: {
                "max_abs_x": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
                "max_abs_y": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
                "max_z": torch.full((self.num_envs,), -inf, device=self.device, dtype=torch.float32),
                "min_z": torch.full((self.num_envs,), inf, device=self.device, dtype=torch.float32),
                "min_boundary_dist": torch.full((self.num_envs,), inf, device=self.device, dtype=torch.float32),
                "count": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            }
            for agent in self.agents
        }

        max_reason = max(self._done_reason_labels.keys()) if self._done_reason_labels else 0
        self.termination_rate_window = max(int(termination_rate_window), 1)
        self.window_done_reason_counts = torch.zeros(max_reason + 1, dtype=torch.float32, device=self.device)
        self.window_episode_count = 0
        self.duration_total = 0.0
        self.duration_count = 0

        # Termination rate history for heatmap visualization
        # Each entry: (cumulative_episodes, {label: pct, ...})
        self._term_rate_history: list[tuple[int, dict[str, float]]] = []
        self._cumulative_episodes = 0
        # Names of the heuristic / RL evader controllers (set by env via
        # set_heuristic_names). Used to emit TerminationRate / TerminationStep
        # split per heuristic, so the paper can report "Vs catches 85% of
        # hover but only 30% of APF" without staring at raw episode data.
        self._heuristic_names: list[str] = []

    def _create_speed_tracker(self) -> dict[str, torch.Tensor]:
        inf = float("inf")
        return {
            "sum": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "count": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "min": torch.full((self.num_envs,), inf, device=self.device, dtype=torch.float32),
            "max": torch.full((self.num_envs,), -inf, device=self.device, dtype=torch.float32),
        }

    def _accumulate_speed_sample(self, key: str, values: torch.Tensor) -> None:
        tracker = self.speed_stats[self.speed_agent][key]
        tracker["sum"] += values
        tracker["count"] += 1.0
        tracker["max"] = torch.maximum(tracker["max"], values)
        tracker["min"] = torch.minimum(tracker["min"], values)

    def update_speed_stats(self, lin_vel_w: torch.Tensor, ang_vel_b: torch.Tensor, step_dt: float) -> None:
        dt = max(float(step_dt), 1e-6)
        lin_speed = torch.norm(lin_vel_w, dim=-1)
        ang_speed = torch.norm(ang_vel_b, dim=-1)
        prev_lin = self.prev_lin_vel[self.speed_agent]
        accel_mag = torch.norm((lin_vel_w - prev_lin) / dt, dim=-1)
        self._accumulate_speed_sample("linear", lin_speed)
        self._accumulate_speed_sample("angular", ang_speed)
        self._accumulate_speed_sample("acceleration", accel_mag)
        self.prev_lin_vel[self.speed_agent] = lin_vel_w.clone()

    def update_rho_stats(self, rho_by_agent: dict[str, torch.Tensor]) -> None:
        for agent, rho in rho_by_agent.items():
            if agent not in self.rho_stats:
                continue
            tracker = self.rho_stats[agent]
            tracker["sum"] += rho
            tracker["count"] += 1.0

    def update_position_extent(
        self,
        positions_by_agent: dict[str, torch.Tensor],
        arena_min: torch.Tensor,
        arena_max: torch.Tensor,
    ) -> None:
        """Track per-episode max |x|, |y|, z and min z + min boundary distance.

        ``positions_by_agent`` maps agent → (num_envs, 3) env-local positions.
        ``arena_min`` and ``arena_max`` are 3-vectors with the env-local bounds
        used for the boundary-distance metric.
        """
        for agent, pos in positions_by_agent.items():
            tracker = self.position_extent_stats.get(agent)
            if tracker is None:
                continue
            abs_x = pos[:, 0].abs()
            abs_y = pos[:, 1].abs()
            z = pos[:, 2]
            tracker["max_abs_x"] = torch.maximum(tracker["max_abs_x"], abs_x)
            tracker["max_abs_y"] = torch.maximum(tracker["max_abs_y"], abs_y)
            tracker["max_z"] = torch.maximum(tracker["max_z"], z)
            tracker["min_z"] = torch.minimum(tracker["min_z"], z)
            # min distance to any of the 6 boundary planes (positive when inside)
            margins = torch.stack(
                [
                    pos[:, 0] - arena_min[0],
                    arena_max[0] - pos[:, 0],
                    pos[:, 1] - arena_min[1],
                    arena_max[1] - pos[:, 1],
                    pos[:, 2] - arena_min[2],
                    arena_max[2] - pos[:, 2],
                ],
                dim=-1,
            )
            min_margin = margins.min(dim=-1).values
            tracker["min_boundary_dist"] = torch.minimum(tracker["min_boundary_dist"], min_margin)
            tracker["count"] += 1.0

    def accumulate_episode_reward(self, agent: str, components: dict[str, torch.Tensor]) -> None:
        store = self.episode_reward_sums.setdefault(agent, {})
        for name, tensor in components.items():
            if name not in store:
                store[name] = torch.zeros_like(tensor)
            store[name] += tensor

    def reset_episode_trackers(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        for agent_store in self.episode_reward_sums.values():
            for tensor in agent_store.values():
                tensor[env_ids] = 0.0
        for tracker in self.speed_stats[self.speed_agent].values():
            tracker["sum"][env_ids] = 0.0
            tracker["count"][env_ids] = 0.0
            tracker["min"][env_ids] = float("inf")
            tracker["max"][env_ids] = float("-inf")
        self.prev_lin_vel[self.speed_agent][env_ids] = 0.0
        for tracker in self.rho_stats.values():
            tracker["sum"][env_ids] = 0.0
            tracker["count"][env_ids] = 0.0
        for tracker in self.position_extent_stats.values():
            tracker["max_abs_x"][env_ids] = 0.0
            tracker["max_abs_y"][env_ids] = 0.0
            tracker["max_z"][env_ids] = float("-inf")
            tracker["min_z"][env_ids] = float("inf")
            tracker["min_boundary_dist"][env_ids] = float("inf")
            tracker["count"][env_ids] = 0.0

    def gather_speed_stats(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        entries: dict[str, torch.Tensor] = {}
        labels = {"linear": "LinearSpeed", "angular": "AngularSpeed", "acceleration": "Acceleration"}
        for key, label in labels.items():
            tracker = self.speed_stats[self.speed_agent][key]
            counts = tracker["count"][env_ids]
            if not bool(torch.any(counts > 0)):
                continue
            valid_ids = env_ids[counts > 0]
            per_episode_mean = tracker["sum"][valid_ids] / tracker["count"][valid_ids]
            prefix = f"Speed/{self.speed_agent}"
            entries[f"{prefix}/{label}/mean"] = per_episode_mean.mean()
            entries[f"{prefix}/{label}/min"] = tracker["min"][valid_ids].min()
            entries[f"{prefix}/{label}/max"] = tracker["max"][valid_ids].max()
        return entries

    def gather_rho_stats(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        entries: dict[str, torch.Tensor] = {}
        for agent, tracker in self.rho_stats.items():
            counts = tracker["count"][env_ids]
            valid = counts > 0
            if not bool(torch.any(valid)):
                continue
            per_episode_mean = tracker["sum"][env_ids][valid] / counts[valid]
            entries[f"Camera/{agent}/Rho/mean"] = per_episode_mean.mean()
        return entries

    def set_heuristic_names(self, names) -> None:
        """Provide the ordered list of evader-controller labels (length must
        match the heuristic one-hot dimension)."""
        self._heuristic_names = list(names)

    def gather_per_heuristic_termination_stats(
        self,
        env_ids: torch.Tensor,
        episode_length_buf: torch.Tensor,
        last_done_reasons: torch.Tensor,
        heuristic_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """For each (heuristic, reason) pair that occurred in this reset batch,
        emit TerminationRate/<heuristic>/<reason> (fraction over episodes
        against that heuristic) and TerminationStep/<heuristic>/<reason>/mean.
        Skips heuristics with no episodes in the batch.
        """
        if not self._heuristic_names or heuristic_ids is None:
            return {}
        entries: dict[str, torch.Tensor] = {}
        h_ids = heuristic_ids[env_ids].to(torch.long)
        reasons = last_done_reasons[env_ids].to(torch.long)
        lengths = episode_length_buf[env_ids].to(torch.float32)
        for h_idx, h_name in enumerate(self._heuristic_names):
            h_mask = h_ids == h_idx
            n_h = int(h_mask.sum().item())
            if n_h == 0:
                continue
            entries[f"Heuristic/{h_name}/episodes"] = torch.tensor(float(n_h), device=self.device, dtype=torch.float32)
            for r_idx, r_label in self._done_reason_labels.items():
                if r_idx == 0:
                    continue
                rh_mask = h_mask & (reasons == r_idx)
                n_rh = int(rh_mask.sum().item())
                if n_rh == 0:
                    continue
                pct = torch.tensor(n_rh / n_h, device=self.device, dtype=torch.float32)
                entries[f"TerminationRate/{h_name}/{r_label}"] = pct
                entries[f"TerminationStep/{h_name}/{r_label}/mean"] = lengths[rh_mask].mean()
        return entries

    def gather_position_extent_stats(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Mean over the env_ids batch of the per-episode spatial extent stats."""
        entries: dict[str, torch.Tensor] = {}
        for agent, tracker in self.position_extent_stats.items():
            counts = tracker["count"][env_ids]
            valid = counts > 0
            if not bool(torch.any(valid)):
                continue
            valid_ids = env_ids[valid]
            prefix = f"Position/{agent.capitalize()}"
            entries[f"{prefix}/MaxAbsX/mean"] = tracker["max_abs_x"][valid_ids].mean()
            entries[f"{prefix}/MaxAbsY/mean"] = tracker["max_abs_y"][valid_ids].mean()
            entries[f"{prefix}/MaxZ/mean"] = tracker["max_z"][valid_ids].mean()
            entries[f"{prefix}/MinZ/mean"] = tracker["min_z"][valid_ids].mean()
            entries[f"{prefix}/MinBoundaryDist/mean"] = tracker["min_boundary_dist"][valid_ids].mean()
        return entries

    def finalize_episode_metrics(
        self,
        *,
        env_ids: torch.Tensor,
        episode_length_buf: torch.Tensor,
        step_dt: float,
        last_done_reasons: torch.Tensor,
        heuristic_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return {}

        log_data: dict[str, torch.Tensor] = {}
        durations = episode_length_buf[env_ids].float() * float(step_dt)
        if durations.numel() > 0:
            log_data["Episode/Duration/mean"] = durations.mean()
            self.duration_total += float(durations.sum().item())
            self.duration_count += int(durations.numel())
            if self.duration_count > 0:
                log_data["Episode/Duration/cumulative_mean"] = torch.tensor(
                    self.duration_total / self.duration_count,
                    dtype=torch.float32,
                    device=self.device,
                )

        reason_vals = last_done_reasons[env_ids].to(torch.long)
        start = 0
        total_reasons = reason_vals.numel()
        while start < total_reasons:
            remaining = self.termination_rate_window - self.window_episode_count
            take = min(remaining, total_reasons - start)
            batch_vals = reason_vals[start : start + take]
            reason_counts = torch.bincount(
                batch_vals,
                minlength=self.window_done_reason_counts.shape[0],
            ).to(torch.float32)
            self.window_done_reason_counts += reason_counts
            self.window_episode_count += int(take)
            if self.window_episode_count >= self.termination_rate_window:
                total_eps = max(float(self.window_episode_count), 1.0)
                snapshot: dict[str, float] = {}
                for reason_idx, label in self._done_reason_labels.items():
                    if reason_idx == 0:
                        continue
                    pct = float(self.window_done_reason_counts[reason_idx] / total_eps)
                    log_data[f"TerminationRate/{label}"] = torch.tensor(pct, device=self.device)
                    snapshot[label] = pct
                self._cumulative_episodes += self.window_episode_count
                self._term_rate_history.append((self._cumulative_episodes, snapshot))
                self._log_termination_heatmap()
                self.window_done_reason_counts.zero_()
                self.window_episode_count = 0
            start += take
        for agent, components in self.episode_reward_sums.items():
            for name, tensor in components.items():
                if tensor.numel() == 0:
                    continue
                vals = tensor[env_ids]
                log_data[f"Reward/{agent.capitalize()}/{name}"] = vals.mean()

        # Per-reason termination-step distribution: for each done reason that
        # occurred in this batch, the mean step at which it fired. Tells us
        # whether OOB / wall hits happen immediately on spawn or after the
        # drone has been chasing for a while.
        reasons_long = last_done_reasons[env_ids].to(torch.long)
        lengths = episode_length_buf[env_ids].to(torch.float32)
        for reason_idx, label in self._done_reason_labels.items():
            if reason_idx == 0:
                continue
            mask = reasons_long == reason_idx
            if bool(mask.any()):
                log_data[f"TerminationStep/{label}/mean"] = lengths[mask].mean()

        speed_logs = self.gather_speed_stats(env_ids)
        if speed_logs:
            log_data.update(speed_logs)
        rho_logs = self.gather_rho_stats(env_ids)
        if rho_logs:
            log_data.update(rho_logs)
        position_logs = self.gather_position_extent_stats(env_ids)
        if position_logs:
            log_data.update(position_logs)

        if heuristic_ids is not None and self._heuristic_names:
            heuristic_logs = self.gather_per_heuristic_termination_stats(
                env_ids=env_ids,
                episode_length_buf=episode_length_buf,
                last_done_reasons=last_done_reasons,
                heuristic_ids=heuristic_ids,
            )
            if heuristic_logs:
                log_data.update(heuristic_logs)

        return log_data

    def _log_termination_heatmap(self) -> None:
        """Generate a termination-rate heatmap and log it to wandb."""
        if len(self._term_rate_history) < 2:
            return
        try:
            import wandb

            if wandb.run is None:
                return
        except ImportError:
            return

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            logger.debug("matplotlib not available, skipping termination heatmap")
            return

        try:
            # Build data matrix: rows = reasons, cols = windows
            labels = [lbl for idx, lbl in sorted(self._done_reason_labels.items()) if idx != 0]
            n_windows = len(self._term_rate_history)
            data = np.zeros((len(labels), n_windows))
            x_ticks = []
            for col, (cum_eps, snapshot) in enumerate(self._term_rate_history):
                x_ticks.append(f"{cum_eps // 1000}k" if cum_eps >= 1000 else str(cum_eps))
                for row, lbl in enumerate(labels):
                    data[row, col] = snapshot.get(lbl, 0.0) * 100.0  # percent

            fig, ax = plt.subplots(figsize=(max(6, n_windows * 0.5), max(3, len(labels) * 0.6)))
            im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)

            ax.set_xticks(range(n_windows))
            ax.set_xticklabels(x_ticks, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel("Cumulative Episodes")
            ax.set_title("Termination Rate (%)")

            # Annotate cells with percentage values
            for row in range(len(labels)):
                for col in range(n_windows):
                    val = data[row, col]
                    color = "white" if val > 50 else "black"
                    ax.text(col, row, f"{val:.0f}", ha="center", va="center", fontsize=7, color=color)

            fig.colorbar(im, ax=ax, label="%", shrink=0.8)
            fig.tight_layout()

            wandb.log({"Termination/heatmap": wandb.Image(fig)}, commit=False)
            plt.close(fig)
        except Exception:
            logger.debug("Failed to log termination heatmap", exc_info=True)
