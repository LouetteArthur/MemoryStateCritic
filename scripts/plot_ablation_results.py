#!/usr/bin/env python3
"""Generate paper-quality ablation plots from wandb logs.

Pulls completed runs matching the critic-ablation naming convention
``<Critic>_<arena>_s<seed>`` (e.g. ``Vsz_wall_s42``) from a wandb
project, aggregates mean / std across seeds, and saves three figures
suitable for the CoRL submission:

  1. ``reward_evolution.{pdf,png}`` — pursuer episode return over training,
     one panel per arena (wall, open). Methods in legend, shaded ±1 std.
  2. ``termination_wall.{pdf,png}`` — termination breakdown on the wall
     arena (pursuer_capture, pursuer_out_of_bounds, pursuer_wall_collision,
     timeout). 2×2 panels, shaded ±1 std.
  3. ``win_rate.{pdf,png}`` and ``win_rate_summary.csv`` — final pursuer
     win rate (capture + evader self-destruct) per critic × arena, with
     std-of-mean error bars.

Usage::

    python scripts/plot_ablation_results.py \\
        --entity louettearthur --project critic_ablation \\
        --output-dir figures/ablation

Requires::

    pip install wandb pandas matplotlib seaborn

Caching: pass ``--cache figures/ablation/runs.pkl`` and the first call
will materialise the wandb history to disk; subsequent calls re-use it
so you can iterate on plots without paying API latency each time.
"""

from __future__ import annotations

import argparse
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


def _si_step_formatter(x: float, _pos: int | None = None) -> str:
    """Format axis ticks with k / M SI suffix: 512000 → '512k', 51.2e6 → '51.2M'."""
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 1e6:
        return f"{x / 1e6:.1f}".rstrip("0").rstrip(".") + "M"
    if ax >= 1e3:
        return f"{x / 1e3:.1f}".rstrip("0").rstrip(".") + "k"
    return f"{int(x)}"


_TICK_FORMATTER = mticker.FuncFormatter(_si_step_formatter)

# --------------------------------------------------------------------------
# Configuration: critic order, labels, colours, metric keys
# --------------------------------------------------------------------------

CRITICS = ["Vs", "Vsh", "Vsz", "Vsoa"]

# Labels follow the paper's notation update: critic-side recurrent encoding is
# z^c (was h); actor-memory encoding is z^a (was z). V(o) is no longer shown.
CRITIC_LABELS = {
    "Vs":   r"$V(s)$",
    "Vsh":  r"$V(s, z^c)$",
    "Vsz":  r"$V(s, z^a)$",
    "Vsoa": r"$V(s, o, a)$",
}

# Colour-blind-friendly palette (Wong 2011, reordered)
CRITIC_COLORS = {
    "Vs":   "#0072B2",  # blue
    "Vsh":  "#009E73",  # green   — history-state V(s, z^c)
    "Vsz":  "#E69F00",  # orange  — memory-state V(s, z^a)
    "Vsoa": "#CC79A7",  # reddish purple
}

ARENAS = ["wall", "open"]
ARENA_COLORS = {"wall": "#4C72B0", "open": "#DD8452"}

# Wandb metric keys (the skrl trainer prefixes env extras with "Info / ").
KEY_REWARD     = "Info / Reward/Pursuer/total"
KEY_TR_CAPTURE = "Info / TerminationRate/pursuer_capture"
KEY_TR_POOB    = "Info / TerminationRate/pursuer_out_of_bounds"
KEY_TR_EOOB    = "Info / TerminationRate/evader_out_of_bounds"
KEY_TR_PWALL   = "Info / TerminationRate/pursuer_wall_collision"
KEY_TR_EWALL   = "Info / TerminationRate/evader_wall_collision"
KEY_TR_TIMEOUT = "Info / TerminationRate/timeout"

# Heuristics paired with the corresponding evader controller in the env.
# Order matches PursuitEvasionEnv._heuristic_names, but RL is omitted because
# the open-/wall-arena sweep doesn't pit the pursuer against an RL evader.
HEURISTICS = ["hover", "circular", "lemniscate", "apf_evader"]
HEURISTIC_LABELS = {
    "hover":       "Hover",
    "circular":    "Circular",
    "lemniscate":  "Lemniscate",
    "apf_evader":  "APF",
    "rl":          "RL",
}

METRIC_KEYS = [
    KEY_REWARD, KEY_TR_CAPTURE, KEY_TR_POOB, KEY_TR_EOOB,
    KEY_TR_PWALL, KEY_TR_EWALL, KEY_TR_TIMEOUT,
]

NAME_RE = re.compile(r"^(Vs|Vsz|Vsh|Vo|Vsoa)_(wall|open)_s(\d+)$")


@dataclass
class RunData:
    critic: str
    arena: str
    seed: int
    history: pd.DataFrame  # columns: _step + the metric keys present in this run


# --------------------------------------------------------------------------
# Data acquisition
# --------------------------------------------------------------------------


def fetch_runs(
    entity: str,
    project: str,
    num_samples: int = 500,
    require_config: dict[str, float] | None = None,
) -> list[RunData]:
    """Pull the most-recent *finished* run for each (critic, arena, seed) triple.

    ``require_config`` is an optional dict of (Config/<key> -> required value);
    runs that don't have the field, or whose value doesn't match (within 1e-6),
    are silently skipped. Use this to make sure the plotted runs all share the
    same reward/training configuration.
    """
    import wandb

    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    require_config = require_config or {}

    n_skip_state = n_skip_config = 0
    latest: dict[tuple[str, str, int], object] = {}
    for run in runs:
        match = NAME_RE.match(run.name or "")
        if match is None:
            continue
        if run.state != "finished":
            n_skip_state += 1
            continue
        # config filter — require_config maps "Config/<name>" -> expected value
        skip = False
        for cfg_key, expected in require_config.items():
            got = run.summary.get(cfg_key)
            if got is None or abs(float(got) - float(expected)) > 1e-6:
                skip = True
                break
        if skip:
            n_skip_config += 1
            continue
        key = (match.group(1), match.group(2), int(match.group(3)))
        prev = latest.get(key)
        if prev is None or run.created_at > prev.created_at:
            latest[key] = run

    if require_config:
        print(f"  config filter {require_config}: kept {len(latest)} runs, "
              f"skipped {n_skip_config} (config mismatch) + {n_skip_state} (not finished)")

    out: list[RunData] = []
    for (critic, arena, seed), run in sorted(latest.items()):
        # Fetch one full series per metric via scan_history(keys=[_step, K]).
        # Each call returns every event where K was logged (dense for K).
        # We then OUTER-JOIN them on _step into a single sparse frame —
        # each row has one column non-null. Aggregate downstream handles
        # the NaN gaps via dropna(subset=[metric]).
        #
        # Background: scan_history() (no keys) silently truncates / misses
        # events for sparse metrics. history(samples=N) drops rows when
        # multiple sparse columns don't align. Neither was acceptable.
        per_metric = []
        for key in METRIC_KEYS:
            try:
                rows = list(run.scan_history(keys=["_step", key]))
            except Exception as exc:
                continue
            if not rows:
                continue
            per_metric.append(pd.DataFrame(rows))
        if not per_metric:
            print(f"[WARN] no metric history retrievable for {run.name}")
            continue
        # Outer-join on _step
        hist = per_metric[0]
        for df in per_metric[1:]:
            hist = hist.merge(df, on="_step", how="outer", suffixes=(None, "_dup"))
            # Drop any "_dup" columns introduced by overlapping keys (shouldn't
            # happen since each scan returns a distinct metric, but defensive).
            hist = hist.loc[:, ~hist.columns.str.endswith("_dup")]
        hist = hist.sort_values("_step").reset_index(drop=True)
        out.append(RunData(critic=critic, arena=arena, seed=seed, history=hist))
        n_metrics = sum(1 for c in hist.columns if c != "_step")
        print(f"  pulled {run.name} ({len(hist)} samples, {n_metrics} metrics)")
    return out


# --------------------------------------------------------------------------
# Aggregation utilities
# --------------------------------------------------------------------------


def aggregate_metric(
    data: list[RunData],
    metric: str,
    arena: str,
    n_grid: int = 200,
    env_timesteps_per_run: int = 100_000,
    aggregation: str = "mean_std",
) -> dict[str, dict[str, np.ndarray | int]]:
    """For each critic, return seed-aggregated curves on a common step grid.

    The x-axis is rescaled from wandb's internal `_step` (which is "number of
    skrl log-flushes") to **environment timesteps** assuming every run trained
    for ``env_timesteps_per_run`` steps. We do this per-run (each run's max
    `_step` maps to `env_timesteps_per_run`), then linearly interpolate onto a
    shared grid so seeds with slightly different flush counts can be averaged.

    ``aggregation`` selects how the per-seed curves are reduced:

    - ``"mean_std"``: central line = sample mean across seeds, band = ±1 std.
      Sensitive to outlier seeds in the small-N regime.
    - ``"iqm_range"``: central line = **interquartile mean** (Agarwal et al.,
      NeurIPS 2021, §4.3), defined as the 25%-trimmed mean: discard the
      ``floor(n/4)`` worst- and best-performing seeds — ranked by the last
      10% of training on this very metric — and average the surviving
      curves. Band = pointwise min and max of the surviving curves (so the
      shaded region is the spread of the seeds that actually got kept).
      For N=5 this drops 1 seed from each tail and reports the mean of the
      middle 3, which matches Agarwal et al.'s recommended robust statistic
      for the few-run regime.
    """
    traces: dict[str, list[pd.DataFrame]] = {c: [] for c in CRITICS}
    for rd in data:
        if rd.arena != arena or metric not in rd.history.columns:
            continue
        sub = rd.history[["_step", metric]].dropna()
        if len(sub) < 2:
            continue
        max_step = float(sub["_step"].max())
        if max_step <= 0:
            continue
        # Rescale this run's _step axis to env timesteps
        scale = float(env_timesteps_per_run) / max_step
        sub = sub.assign(env_step=sub["_step"].to_numpy() * scale)
        traces[rd.critic].append(sub)

    if not any(traces.values()):
        return {}
    grid = np.linspace(0.0, float(env_timesteps_per_run), n_grid)
    out: dict[str, dict] = {}
    for critic in CRITICS:
        runs = traces[critic]
        if not runs:
            continue
        interp = np.stack(
            [np.interp(grid, r["env_step"].to_numpy(), r[metric].to_numpy()) for r in runs]
        )
        n = interp.shape[0]
        if aggregation == "iqm_range" and n >= 3:
            # Rank seeds by their final-window mean of this metric, then drop
            # floor(n/4) entire seeds from each tail (Agarwal et al. 2021 §4.3).
            # For n=5 → trim 1 from each tail → 3 seeds kept (mean of middle 3).
            window = max(1, n_grid // 10)
            final_score = interp[:, -window:].mean(axis=1)
            order = np.argsort(final_score)               # ascending: worst -> best
            trim = int(np.floor(n / 4))
            kept_idx = order[trim : n - trim] if trim > 0 else order
            kept = interp[kept_idx]                       # (n_kept, n_grid)
            out[critic] = {
                "step": grid,
                "mean": kept.mean(axis=0),                # IQM curve
                "lower": kept.min(axis=0),                # worst surviving seed
                "upper": kept.max(axis=0),                # best  surviving seed
                "n": n,
                "n_kept": int(kept.shape[0]),
            }
        elif aggregation == "top3_range" and n >= 1:
            # Keep only the 3 best-performing seeds (ranked by last-10% mean
            # of this metric). Visual style matches iqm_range: central line is
            # the mean of the kept curves, band is their pointwise [min, max].
            # NOTE: this is a *biased upward* estimator (no top-tail trim) and
            # is NOT the IQM. Useful as a best-case showcase, not a central
            # tendency.
            window = max(1, n_grid // 10)
            final_score = interp[:, -window:].mean(axis=1)
            order = np.argsort(final_score)               # ascending
            k = min(3, n)
            kept_idx = order[-k:]                          # top-k
            kept = interp[kept_idx]
            out[critic] = {
                "step": grid,
                "mean": kept.mean(axis=0),
                "lower": kept.min(axis=0),
                "upper": kept.max(axis=0),
                "n": n,
                "n_kept": int(kept.shape[0]),
            }
        else:
            out[critic] = {
                "step": grid,
                "mean": interp.mean(axis=0),
                "std": interp.std(axis=0),
                "n": n,
            }
    return out


def compute_final_win_rates(data: list[RunData], last_n: int = 5) -> pd.DataFrame:
    """Final pursuer win rate per (critic, arena, seed), averaged over the last
    ``last_n`` logged termination snapshots so the result isn't a single noisy
    point. Win = capture + evader_oob + evader_wall_collision (zero-sum).
    """
    rows = []
    for rd in data:
        # Add zero columns for keys missing in open-arena runs (no wall events).
        h = rd.history.copy()
        for k in (KEY_TR_CAPTURE, KEY_TR_EOOB, KEY_TR_EWALL):
            if k not in h.columns:
                h[k] = 0.0
        sub = h[["_step", KEY_TR_CAPTURE, KEY_TR_EOOB, KEY_TR_EWALL]]
        sub = sub.dropna(subset=[KEY_TR_CAPTURE])
        if sub.empty:
            continue
        tail = sub.tail(last_n)
        win = (
            tail[KEY_TR_CAPTURE].fillna(0.0)
            + tail[KEY_TR_EOOB].fillna(0.0)
            + tail[KEY_TR_EWALL].fillna(0.0)
        ).mean()
        rows.append(
            {"critic": rd.critic, "arena": rd.arena, "seed": rd.seed, "win_rate": float(win)}
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------


def _apply_paper_style() -> None:
    sns.set_context("paper", font_scale=1.15)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def _plot_metric_panels(
    data: list[RunData],
    panels: list[tuple[str, str, str]],
    output_path: Path,
    ylim: tuple[float, float] | None = None,
    ylabel: str = "",
    suptitle: str = "",
    ncols: int = 2,
    env_timesteps: int = 100_000,
    aggregation: str = "mean_std",
):
    """Generic helper: a grid of (metric, arena, title) panels."""
    _apply_paper_style()
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.0 * ncols, 3.4 * nrows), sharex=False, sharey=(ylim is not None)
    )
    axes_flat = np.array(axes).reshape(-1)

    # Collect line handles across all panels so the figure legend has every
    # critic that appears in any panel (not just the first).
    legend_lines: dict[str, plt.Line2D] = {}
    for ax, (metric, arena, title) in zip(axes_flat, panels):
        agg = aggregate_metric(
            data, metric, arena,
            env_timesteps_per_run=env_timesteps,
            aggregation=aggregation,
        )
        for critic in CRITICS:
            if critic not in agg:
                continue
            a = agg[critic]
            color = CRITIC_COLORS[critic]
            (line,) = ax.plot(
                a["step"], a["mean"], color=color, linewidth=1.8,
                label=CRITIC_LABELS[critic],
            )
            if "lower" in a:
                lo, hi = a["lower"], a["upper"]
            else:
                lo = a["mean"] - a["std"]
                hi = a["mean"] + a["std"]
            ax.fill_between(
                a["step"], lo, hi,
                color=color, alpha=0.18, linewidth=0,
            )
            legend_lines.setdefault(critic, line)
        ax.set_title(title)
        ax.set_xlabel("Environment timesteps")
        ax.xaxis.set_major_formatter(_TICK_FORMATTER)
        if ylim is not None:
            ax.set_ylim(*ylim)
    # Order the legend by the canonical CRITICS order
    handles_ordered = [legend_lines[c] for c in CRITICS if c in legend_lines]
    labels_ordered = [CRITIC_LABELS[c] for c in CRITICS if c in legend_lines]
    legend_handles_labels = (handles_ordered, labels_ordered) if handles_ordered else None

    for ax in axes_flat[: nrows * ncols]:
        ax.set_ylabel(ylabel)
    # blank unused panels
    for ax in axes_flat[len(panels):]:
        ax.set_visible(False)

    if legend_handles_labels is not None:
        fig.legend(
            *legend_handles_labels,
            loc="upper center",
            ncol=min(len(CRITICS), 5),
            frameon=False,
            bbox_to_anchor=(0.5, 1.04),
        )
    if suptitle:
        fig.suptitle(suptitle, y=1.10)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_reward_evolution(
    data: list[RunData], output_dir: Path, env_timesteps: int = 100_000,
    aggregation: str = "mean_std",
) -> None:
    panels = [(KEY_REWARD, arena, f"{arena.capitalize()} arena") for arena in ARENAS]
    suptitle = "Pursuer episode return"
    _plot_metric_panels(
        data, panels,
        output_path=output_dir / "reward_evolution",
        ylabel="Pursuer episode return",
        suptitle=suptitle,
        ncols=2,
        env_timesteps=env_timesteps,
        aggregation=aggregation,
    )


def plot_capture_rate_evolution(
    data: list[RunData], output_dir: Path, env_timesteps: int = 100_000,
    aggregation: str = "mean_std",
) -> None:
    """Same layout as plot_reward_evolution, but the y-axis is the pursuer
    capture rate (∈ [0, 1]) instead of the per-episode return. Useful as a
    standalone training-progress signal that's directly comparable across
    reward formulations (a reward change shifts the return axis; capture
    rate is invariant)."""
    panels = [(KEY_TR_CAPTURE, arena, f"{arena.capitalize()} arena") for arena in ARENAS]
    suptitle = "Pursuer capture rate"
    _plot_metric_panels(
        data, panels,
        output_path=output_dir / "capture_rate_evolution",
        ylim=(-0.02, 1.02),
        ylabel="Pursuer capture rate",
        suptitle=suptitle,
        ncols=2,
        env_timesteps=env_timesteps,
        aggregation=aggregation,
    )


def plot_terminations_wall(
    data: list[RunData], output_dir: Path, env_timesteps: int = 100_000,
    aggregation: str = "mean_std",
) -> None:
    panels = [
        (KEY_TR_CAPTURE, "wall", "Pursuer capture"),
        (KEY_TR_POOB,    "wall", "Pursuer out-of-bounds"),
        (KEY_TR_PWALL,   "wall", "Pursuer wall collision"),
        (KEY_TR_TIMEOUT, "wall", "Timeout"),
    ]
    _plot_metric_panels(
        data, panels,
        output_path=output_dir / "termination_wall",
        ylim=(-0.02, 1.02),
        ylabel="Rate",
        suptitle="Wall arena: termination outcomes",
        ncols=2,
        env_timesteps=env_timesteps,
        aggregation=aggregation,
    )


REWARD_COMPONENTS = [
    ("time", "Time"),
    ("approach", "Approach"),
    ("perception", "Perception"),
    ("capture", "Capture"),
    ("bounds", "OOB"),
    ("wall_collision", "Wall hit"),
    ("body_rates", "Body rates"),
]


def plot_reward_component_magnitudes(
    data: list[RunData], output_dir: Path, entity: str, project: str
) -> None:
    """Per-(critic,arena) bar chart of mean ± std of each reward component's
    final per-episode value.

    Component magnitudes are not cached in ``data`` (only ``Reward/Pursuer/
    total`` and termination rates are kept in the history), so we re-query
    ``run.summary`` directly here — small payload, no full history needed.

    Tells you which terms actually dominate the total: e.g. if ``bounds`` is
    -9 and ``capture`` is +0.5, the policy is mostly being penalised for OOB
    and never catches, independently of how the "total" line looks.
    """
    import wandb

    api = wandb.Api()
    # Build a (critic, arena, seed) -> run id index from the cached data so we
    # query the same runs the rest of the figures are aggregated over.
    keys = {(rd.critic, rd.arena, rd.seed) for rd in data}
    rows: list[dict] = []
    for run in api.runs(f"{entity}/{project}"):
        m = NAME_RE.match(run.name or "")
        if not m or run.state != "finished":
            continue
        critic, arena, seed = m.group(1), m.group(2), int(m.group(3))
        if (critic, arena, seed) not in keys:
            continue
        for key, _ in REWARD_COMPONENTS:
            v = run.summary.get(f"Info / Reward/Pursuer/{key}")
            if v is None:
                continue
            rows.append({"critic": critic, "arena": arena, "seed": seed,
                         "component": key, "value": float(v)})
    df = pd.DataFrame(rows)
    if df.empty:
        print("[WARN] no reward-component data; skipping magnitude plot")
        return
    summary = (
        df.groupby(["arena", "critic", "component"])["value"]
          .agg(["mean", "std", "count"]).reset_index()
    )
    summary.to_csv(output_dir / "reward_component_magnitudes.csv", index=False)

    _apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    component_keys = [c for c, _ in REWARD_COMPONENTS]
    component_labels = [lbl for _, lbl in REWARD_COMPONENTS]
    x = np.arange(len(component_keys))
    width = 0.16

    for ax, arena in zip(axes, ARENAS):
        for i, critic in enumerate(CRITICS):
            sub = summary[(summary["arena"] == arena) & (summary["critic"] == critic)].set_index("component")
            means = np.array([sub.loc[c, "mean"] if c in sub.index else 0.0 for c in component_keys])
            stds = np.array([sub.loc[c, "std"] if c in sub.index else 0.0 for c in component_keys])
            offset = (i - (len(CRITICS) - 1) / 2.0) * width
            ax.bar(x + offset, means, width, yerr=stds, capsize=2,
                   color=CRITIC_COLORS[critic], edgecolor="black", linewidth=0.4,
                   label=CRITIC_LABELS[critic])
        ax.set_title(f"{arena.capitalize()} arena")
        ax.set_xticks(x)
        ax.set_xticklabels(component_labels, rotation=30, ha="right")
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.7)
    axes[0].set_ylabel("Cumulative pursuer reward (per episode)")
    axes[0].legend(ncol=3, fontsize=8, loc="lower center", frameon=True)
    fig.suptitle("Per-episode reward components", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "reward_components.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "reward_components.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_per_heuristic_capture(
    data: list[RunData], output_dir: Path, entity: str, project: str
) -> None:
    """Per-(critic, arena, heuristic) bar chart of pursuer win rate, plus
    mean episode step at capture vs at non-capture termination.

    Pulls ``Info / TerminationRate/<heuristic>/<reason>`` and
    ``Info / TerminationStep/<heuristic>/<reason>/mean`` from each run's
    summary, then aggregates over seeds. Win rate per heuristic =
    pursuer_capture + evader_out_of_bounds + evader_wall_collision (zero-sum,
    same as the global win-rate definition).
    """
    import wandb

    api = wandb.Api()
    keys = {(rd.critic, rd.arena, rd.seed) for rd in data}
    win_keys = ("pursuer_capture", "evader_out_of_bounds", "evader_wall_collision")
    rate_rows: list[dict] = []
    step_rows: list[dict] = []
    for run in api.runs(f"{entity}/{project}"):
        m = NAME_RE.match(run.name or "")
        if not m or run.state != "finished":
            continue
        critic, arena, seed = m.group(1), m.group(2), int(m.group(3))
        if (critic, arena, seed) not in keys:
            continue
        for h in HEURISTICS:
            win_rate = 0.0
            saw_any = False
            for reason in win_keys:
                v = run.summary.get(f"Info / TerminationRate/{h}/{reason}")
                if v is not None:
                    win_rate += float(v)
                    saw_any = True
            if saw_any:
                rate_rows.append({
                    "critic": critic, "arena": arena, "seed": seed,
                    "heuristic": h, "win_rate": win_rate,
                })
            for outcome in ("pursuer_capture", "timeout"):
                v = run.summary.get(f"Info / TerminationStep/{h}/{outcome}/mean")
                if v is not None:
                    step_rows.append({
                        "critic": critic, "arena": arena, "seed": seed,
                        "heuristic": h, "outcome": outcome, "step_mean": float(v),
                    })
    if not rate_rows:
        print("[WARN] no per-heuristic data; skipping per-heuristic plots "
              "(re-run training after the stats_tracker update)")
        return

    rate_df = pd.DataFrame(rate_rows)
    rate_summary = (
        rate_df.groupby(["arena", "critic", "heuristic"])["win_rate"]
        .agg(["mean", "std", "count"]).reset_index()
    )
    rate_summary.to_csv(output_dir / "per_heuristic_win_rate.csv", index=False)

    _apply_paper_style()
    fig, axes = plt.subplots(1, len(ARENAS), figsize=(6.5 * len(ARENAS), 4.0), sharey=True)
    if len(ARENAS) == 1:
        axes = [axes]
    x = np.arange(len(HEURISTICS))
    width = 0.16
    for ax, arena in zip(axes, ARENAS):
        for i, critic in enumerate(CRITICS):
            sub = rate_summary[
                (rate_summary["arena"] == arena) & (rate_summary["critic"] == critic)
            ].set_index("heuristic")
            means = np.array([sub.loc[h, "mean"] if h in sub.index else np.nan for h in HEURISTICS])
            stds = np.array([sub.loc[h, "std"] if h in sub.index else 0.0 for h in HEURISTICS])
            offset = (i - (len(CRITICS) - 1) / 2.0) * width
            ax.bar(
                x + offset, np.nan_to_num(means), width, yerr=stds, capsize=2,
                color=CRITIC_COLORS[critic], edgecolor="black", linewidth=0.4,
                label=CRITIC_LABELS[critic],
            )
        ax.set_title(f"{arena.capitalize()} arena")
        ax.set_xticks(x)
        ax.set_xticklabels([HEURISTIC_LABELS[h] for h in HEURISTICS])
        ax.set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Pursuer win rate")
    axes[0].legend(ncol=3, fontsize=8, loc="upper right", frameon=True)
    fig.suptitle("Pursuer win rate per evader heuristic", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "per_heuristic_win_rate.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "per_heuristic_win_rate.png", bbox_inches="tight", dpi=200)
    plt.close(fig)

    if step_rows:
        step_df = pd.DataFrame(step_rows)
        step_summary = (
            step_df.groupby(["arena", "critic", "heuristic", "outcome"])["step_mean"]
            .agg(["mean", "std", "count"]).reset_index()
        )
        step_summary.to_csv(output_dir / "per_heuristic_episode_step.csv", index=False)
        print(f"  per_heuristic_episode_step.csv saved ({len(step_summary)} rows)")


def plot_termination_heatmap(data: list[RunData], output_dir: Path, last_n: int = 5) -> None:
    """Two-panel heatmap of termination-reason percentages per critic, one
    panel per arena. Cells annotated as ``mean% ± std%`` across seeds.

    Open panel columns: pursuer_capture, pursuer_oob, timeout.
    Wall panel adds: pursuer_wall_collision (pursuer crashed into wall).

    Reasons excluded by request: evader_out_of_bounds, evader_wall_collision,
    invalid_state — these are zero in the open arena and tell us little for
    the paper's win-rate question.
    """
    REASONS_OPEN = [
        (KEY_TR_CAPTURE, "Capture"),
        (KEY_TR_POOB,    "P. OOB"),
        (KEY_TR_TIMEOUT, "Timeout"),
    ]
    REASONS_WALL = [
        (KEY_TR_CAPTURE, "Capture"),
        (KEY_TR_POOB,    "P. OOB"),
        (KEY_TR_PWALL,   "P. wall hit"),
        (KEY_TR_TIMEOUT, "Timeout"),
    ]

    # Build the per-(critic, arena, seed, reason) final-rate table from the
    # last ``last_n`` history rows (same approach as compute_final_win_rates).
    rows: list[dict] = []
    for rd in data:
        h = rd.history.copy()
        reasons_for_arena = REASONS_WALL if rd.arena == "wall" else REASONS_OPEN
        for key, _label in reasons_for_arena:
            if key not in h.columns:
                # The metric never fired in this run; treat as 0.
                rows.append(
                    {"critic": rd.critic, "arena": rd.arena, "seed": rd.seed,
                     "reason": key, "rate": 0.0}
                )
                continue
            sub = h[["_step", key]].dropna(subset=[key])
            if sub.empty:
                rows.append(
                    {"critic": rd.critic, "arena": rd.arena, "seed": rd.seed,
                     "reason": key, "rate": 0.0}
                )
                continue
            rate = float(sub.tail(last_n)[key].mean())
            rows.append(
                {"critic": rd.critic, "arena": rd.arena, "seed": rd.seed,
                 "reason": key, "rate": rate}
            )
    if not rows:
        print("[WARN] no termination data; skipping heatmap")
        return
    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["arena", "critic", "reason"])["rate"]
          .agg(["mean", "std", "count"]).reset_index()
    )
    summary["std"] = summary["std"].fillna(0.0)  # std is NaN if only 1 seed
    summary.to_csv(output_dir / "termination_summary.csv", index=False)
    print(f"  termination_summary.csv saved ({len(summary)} rows)")

    _apply_paper_style()
    fig, axes = plt.subplots(1, len(ARENAS), figsize=(7.2 * len(ARENAS), 4.0))
    if len(ARENAS) == 1:
        axes = [axes]
    seed_count = len({r["seed"] for r in rows})
    for ax, arena in zip(axes, ARENAS):
        reasons = REASONS_WALL if arena == "wall" else REASONS_OPEN
        n_critic = len(CRITICS)
        n_reason = len(reasons)
        mean_matrix = np.full((n_critic, n_reason), np.nan, dtype=float)
        std_matrix = np.zeros((n_critic, n_reason), dtype=float)
        for i, critic in enumerate(CRITICS):
            for j, (key, _label) in enumerate(reasons):
                row = summary[(summary["arena"] == arena)
                              & (summary["critic"] == critic)
                              & (summary["reason"] == key)]
                if row.empty:
                    continue
                mean_matrix[i, j] = row["mean"].iloc[0]
                std_matrix[i, j] = row["std"].iloc[0]
        # Plot heatmap as imshow with viridis (paper-friendly + colour-blind safe).
        im = ax.imshow(mean_matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
        # Annotate every cell with "XX% ± YY%" (or "—" for NaN cells)
        for i in range(n_critic):
            for j in range(n_reason):
                m = mean_matrix[i, j]
                s = std_matrix[i, j]
                if np.isnan(m):
                    txt = "—"
                else:
                    txt = f"{m*100:.0f}%\n±{s*100:.0f}%" if seed_count > 1 else f"{m*100:.0f}%"
                # White text on dark cells (low values are dark in viridis)
                colour = "white" if (np.isnan(m) or m < 0.55) else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=9, color=colour)
        ax.set_xticks(range(n_reason))
        ax.set_xticklabels([lbl for _, lbl in reasons])
        ax.set_yticks(range(n_critic))
        ax.set_yticklabels([CRITIC_LABELS[c] for c in CRITICS])
        ax.set_title(f"{arena.capitalize()} arena")
        ax.tick_params(axis="x", labelrotation=0)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Termination rate")
    fig.suptitle("Termination outcomes per critic", y=1.01)
    fig.savefig(output_dir / "termination_heatmap.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "termination_heatmap.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def _iqm_summary(values: np.ndarray) -> tuple[float, float, float, int, int]:
    """Return (IQM, lower, upper, n, n_kept) for a 1-D array of per-seed scores.

    25%-trimmed mean per Agarwal et al. 2021 §4.3: drop floor(n/4) worst and
    best seeds, take the mean of the rest. Lower/upper are the min/max of
    the kept seeds (i.e. the range across the seeds that weren't trimmed).
    """
    v = np.sort(values)
    n = v.size
    if n < 3:
        return float(v.mean()), float(v.min()), float(v.max()), n, n
    trim = int(np.floor(n / 4))
    kept = v[trim : n - trim] if trim > 0 else v
    return float(kept.mean()), float(kept.min()), float(kept.max()), n, int(kept.size)


def plot_win_rate_bars(
    win_df: pd.DataFrame, output_dir: Path, aggregation: str = "mean_std",
) -> None:
    if win_df.empty:
        print("[WARN] no win-rate data; skipping bar plot")
        return
    _apply_paper_style()

    if aggregation in ("iqm_range", "top3_range"):
        rows = []
        for (critic, arena), grp in win_df.groupby(["critic", "arena"]):
            vals = grp["win_rate"].to_numpy()
            if aggregation == "iqm_range":
                m, lo, hi, n, n_kept = _iqm_summary(vals)
            else:
                v = np.sort(vals)
                k = min(3, v.size)
                kept = v[-k:]
                m, lo, hi, n, n_kept = float(kept.mean()), float(kept.min()), float(kept.max()), v.size, int(k)
            rows.append({"critic": critic, "arena": arena, "mean": m,
                         "lower": lo, "upper": hi, "count": n, "n_kept": n_kept})
        summary = pd.DataFrame(rows)
    else:
        summary = (
            win_df.groupby(["critic", "arena"])["win_rate"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
    summary.to_csv(output_dir / "win_rate_summary.csv", index=False)
    print(f"  win_rate_summary.csv saved ({len(summary)} rows)")

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(CRITICS))
    width = 0.38
    for i, arena in enumerate(ARENAS):
        sub = summary[summary["arena"] == arena].set_index("critic")
        means = np.array([sub.loc[c, "mean"] if c in sub.index else np.nan for c in CRITICS])
        if aggregation in ("iqm_range", "top3_range"):
            lo = np.array([sub.loc[c, "lower"] if c in sub.index else 0.0 for c in CRITICS])
            hi = np.array([sub.loc[c, "upper"] if c in sub.index else 0.0 for c in CRITICS])
            yerr = np.vstack([np.maximum(0.0, means - lo),
                              np.maximum(0.0, hi - means)])
        else:
            stds = np.array([sub.loc[c, "std"] if c in sub.index else 0.0 for c in CRITICS])
            yerr = stds
        offset = (-0.5 + i) * width
        ax.bar(
            x + offset, np.nan_to_num(means), width,
            yerr=yerr, capsize=4, label=arena.capitalize(),
            color=ARENA_COLORS[arena], edgecolor="black", linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([CRITIC_LABELS[c] for c in CRITICS])
    ax.set_ylabel("Pursuer win rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Pursuer win rate by critic and arena")
    ax.legend(title="Arena", loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "win_rate.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "win_rate.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def tag_runs(entity: str, project: str) -> None:
    """Add critic / arena / seed wandb tags + config keys to every ablation run.

    Idempotent — tags are deduplicated and config keys are overwritten.
    After running this once, the wandb UI lets you Group-by ``critic`` or
    filter by ``arena`` natively.
    """
    import wandb
    api = wandb.Api()
    n_updated = 0
    for run in api.runs(f"{entity}/{project}"):
        match = NAME_RE.match(run.name or "")
        if match is None:
            continue
        critic, arena, seed = match.group(1), match.group(2), int(match.group(3))
        new_tags = sorted({*(run.tags or []), f"critic:{critic}", f"arena:{arena}", f"seed:{seed}"})
        run.tags = new_tags
        run.config["critic"] = critic
        run.config["arena"] = arena
        run.config["seed_label"] = seed
        run.update()
        n_updated += 1
    print(f"  tagged {n_updated} runs")


def upload_figures(output_dir: Path, entity: str, project: str) -> None:
    """Create / overwrite a wandb run named 'ablation_summary' that contains
    the generated PDFs + PNGs + win_rate_summary.csv as artifacts.
    """
    import wandb
    figures = [
        "reward_evolution",
        "termination_wall",
        "reward_components",
        "win_rate",
    ]
    run = wandb.init(
        entity=entity, project=project, name="ablation_summary",
        job_type="summary", reinit=True, settings=wandb.Settings(silent=True),
    )
    images = {}
    for fig in figures:
        png = output_dir / f"{fig}.png"
        if png.exists():
            images[f"figures/{fig}"] = wandb.Image(str(png))
    if images:
        run.log(images)
    csv_path = output_dir / "win_rate_summary.csv"
    if csv_path.exists():
        # log as table for nicer rendering
        df = pd.read_csv(csv_path)
        run.log({"tables/win_rate_summary": wandb.Table(dataframe=df)})
    # attach the PDFs as artifact files for download
    artifact = wandb.Artifact("ablation_figures", type="figures")
    for fig in figures:
        pdf = output_dir / f"{fig}.pdf"
        if pdf.exists():
            artifact.add_file(str(pdf))
    if csv_path.exists():
        artifact.add_file(str(csv_path))
    run.log_artifact(artifact)
    run.finish()
    print(f"  figures pushed to wandb run 'ablation_summary' ({len(images)} images)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entity", required=True, help="wandb entity (username or team)")
    parser.add_argument("--project", default="critic_ablation", help="wandb project name")
    parser.add_argument("--output-dir", default="figures/ablation", help="where to save figures")
    parser.add_argument("--cache", default=None, help="pickle path to cache fetched runs")
    parser.add_argument("--samples", type=int, default=500, help="wandb history sample count per run")
    parser.add_argument(
        "--env-timesteps", type=int, default=100_000,
        help="Environment timesteps per run (TOTAL_FRAMES / NUM_ENVS). "
             "Used to rescale wandb's internal _step axis to a physical quantity.",
    )
    parser.add_argument(
        "--tag-runs", action="store_true",
        help="Set wandb tags + config fields (critic, arena, seed) on each "
             "ablation run so you can group/filter by critic in the wandb UI.",
    )
    parser.add_argument(
        "--upload-figures", action="store_true",
        help="Upload generated figures (reward_evolution, termination_wall, "
             "reward_components, win_rate) to a wandb run called "
             "'ablation_summary' for browsing alongside the per-run data.",
    )
    parser.add_argument(
        "--require-reward-approach", type=float, default=None,
        help="Only include runs whose Config/reward_approach matches this "
             "value (e.g. 3.0 for the v7 exponential sweep). Stale runs from "
             "earlier reward variants are dropped from the plots.",
    )
    parser.add_argument(
        "--require-reward-approach-decay", type=float, default=None,
        help="Only include runs whose Config/reward_approach_decay matches "
             "this value (e.g. 0.5 for the v7 exponential sweep).",
    )
    parser.add_argument(
        "--require-reward-time-scale", type=float, default=None,
        help="Only include runs whose Config/reward_time_scale matches this "
             "value (e.g. 1.0 for the v8 decoupled-time sweep). v7 and earlier "
             "runs have no such config and will be dropped.",
    )
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated list of seeds to keep (e.g. '42' or '42,123'). "
             "Default: keep all seeds present in the project. Captions adapt "
             "to single-seed mode automatically.",
    )
    parser.add_argument(
        "--seeds-open", type=str, default=None,
        help="Per-arena override: comma-separated seeds to keep for the open "
             "arena only. Takes precedence over --seeds for open runs.",
    )
    parser.add_argument(
        "--seeds-wall", type=str, default=None,
        help="Per-arena override: comma-separated seeds to keep for the wall "
             "arena only. Takes precedence over --seeds for wall runs.",
    )
    parser.add_argument(
        "--exclude-critics", type=str, default="",
        help="Comma-separated critics to drop from the plot (e.g. 'Vsoa,Vo'). "
             "Useful for in-progress runs where some critics don't yet have "
             "enough seeds to aggregate cleanly.",
    )
    parser.add_argument(
        "--aggregation", choices=["mean_std", "iqm_range", "top3_range"], default="mean_std",
        help=(
            "Per-cell aggregation across seeds. "
            "'mean_std' (default) plots sample mean ± 1 std. "
            "'iqm_range' plots the 25%%-trimmed mean (Agarwal et al., NeurIPS "
            "2021, §4.3): with N=5 seeds, drop the best- and worst-performing "
            "seed (ranked by the last 10%% of training on the plotted metric) "
            "and report the mean of the middle 3, with the band/error bar "
            "spanning the min and max of the 3 kept seeds. Robust to a single "
            "outlier seed in the few-run regime."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data: list[RunData]
    require_config: dict[str, float] = {}
    if args.require_reward_approach is not None:
        require_config["Config/reward_approach"] = args.require_reward_approach
    if args.require_reward_approach_decay is not None:
        require_config["Config/reward_approach_decay"] = args.require_reward_approach_decay
    if args.require_reward_time_scale is not None:
        require_config["Config/reward_time_scale"] = args.require_reward_time_scale

    if args.cache and Path(args.cache).exists():
        print(f"Loading cached runs from {args.cache}")
        with open(args.cache, "rb") as fh:
            data = pickle.load(fh)
    else:
        print(f"Fetching runs from {args.entity}/{args.project}...")
        data = fetch_runs(
            args.entity, args.project,
            num_samples=args.samples,
            require_config=require_config,
        )
        if args.cache:
            with open(args.cache, "wb") as fh:
                pickle.dump(data, fh)
            print(f"  cached {len(data)} runs to {args.cache}")

    if not data:
        raise SystemExit("No usable runs found. Check entity/project and run names.")

    if args.exclude_critics:
        excluded = {c.strip() for c in args.exclude_critics.split(",") if c.strip()}
        before = len(data)
        data = [rd for rd in data if rd.critic not in excluded]
        global CRITICS
        CRITICS = [c for c in CRITICS if c not in excluded]
        print(f"Exclude-critics {sorted(excluded)}: kept {len(data)}/{before} runs, "
              f"plotting {CRITICS}.")

    if args.seeds is not None:
        wanted_seeds = {int(s.strip()) for s in args.seeds.split(",") if s.strip()}
        before = len(data)
        data = [rd for rd in data if rd.seed in wanted_seeds]
        print(f"Seed filter '{args.seeds}': kept {len(data)}/{before} runs.")

    def _parse_seed_csv(arg):
        return {int(s.strip()) for s in arg.split(",") if s.strip()} if arg else None

    seeds_open = _parse_seed_csv(args.seeds_open)
    seeds_wall = _parse_seed_csv(args.seeds_wall)
    if seeds_open is not None or seeds_wall is not None:
        before = len(data)
        kept = []
        for rd in data:
            if rd.arena == "open" and seeds_open is not None and rd.seed not in seeds_open:
                continue
            if rd.arena == "wall" and seeds_wall is not None and rd.seed not in seeds_wall:
                continue
            kept.append(rd)
        data = kept
        print(f"Per-arena seed filter (open={seeds_open}, wall={seeds_wall}): "
              f"kept {len(data)}/{before} runs.")
        if not data:
            raise SystemExit(f"No runs match --seeds={args.seeds}.")

    print(f"Aggregating {len(data)} runs across "
          f"{len({(r.critic, r.arena) for r in data})} cells and "
          f"{len({r.seed for r in data})} seeds.")

    print(f"Aggregation mode: {args.aggregation}")

    print("Plotting reward evolution...")
    plot_reward_evolution(data, output_dir, env_timesteps=args.env_timesteps,
                          aggregation=args.aggregation)
    print("  reward_evolution.{pdf,png} saved")

    print("Plotting capture-rate evolution...")
    plot_capture_rate_evolution(data, output_dir, env_timesteps=args.env_timesteps,
                                aggregation=args.aggregation)
    print("  capture_rate_evolution.{pdf,png} saved")

    print("Plotting termination heatmap...")
    plot_termination_heatmap(data, output_dir)
    print("  termination_heatmap.{pdf,png} saved")

    print("Plotting wall-arena terminations...")
    plot_terminations_wall(data, output_dir, env_timesteps=args.env_timesteps,
                           aggregation=args.aggregation)
    print("  termination_wall.{pdf,png} saved")

    print("Plotting reward-component magnitudes...")
    plot_reward_component_magnitudes(data, output_dir, entity=args.entity, project=args.project)
    print("  reward_components.{pdf,png} saved")

    print("Computing win rates...")
    win_df = compute_final_win_rates(data)
    plot_win_rate_bars(win_df, output_dir, aggregation=args.aggregation)
    print(f"  win_rate.{{pdf,png}} saved")

    print("Plotting per-heuristic capture rates...")
    plot_per_heuristic_capture(data, output_dir, entity=args.entity, project=args.project)
    print("  per_heuristic_win_rate.{pdf,png} saved (if data was present)")

    print(f"\nAll figures written to {output_dir}/")

    if args.tag_runs:
        print("Tagging wandb runs with critic / arena / seed...")
        tag_runs(args.entity, args.project)

    if args.upload_figures:
        print("Uploading figures to wandb 'ablation_summary' run...")
        upload_figures(output_dir, args.entity, args.project)


if __name__ == "__main__":
    main()
