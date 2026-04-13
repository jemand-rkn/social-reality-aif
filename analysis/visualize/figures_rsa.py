from __future__ import annotations

from typing import Any, Dict, List

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _moving_average(series: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(series, np.ones(window) / window, mode="same")


def _smooth_series(series: np.ndarray, window: int) -> np.ndarray:
    pad = window // 2
    padded = np.pad(series, (pad, pad), mode="edge")
    smooth = np.convolve(padded, np.ones(window) / window, mode="valid")
    return smooth[: len(series)]


def render_rsa_clusters(data: Dict[str, Any]) -> matplotlib.figure.Figure:
    sns.set()
    steps = np.asarray(data["steps"])
    cluster_rsa = np.asarray(data["cluster_rsa"])  # [T, C]

    smooth_window = 25
    tab10 = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(12, 6))
    if cluster_rsa.ndim == 2 and len(steps) > 0:
        num_clusters = cluster_rsa.shape[1]
        for c in range(num_clusters):
            series = cluster_rsa[:, c]
            valid = ~np.isnan(series)
            if not valid.any():
                continue
            color = tab10(c % 10)
            label = f"Cluster {c}"
            # raw (low alpha)
            sns.lineplot(x=steps[valid], y=series[valid], label="_nolegend_",
                         color=color, linewidth=1.1, alpha=0.25, ax=ax)
            # smoothed (high alpha)
            pad = smooth_window // 2
            padded = np.pad(series, (pad, pad), mode="edge")
            smooth = np.convolve(padded, np.ones(smooth_window) / smooth_window, mode="valid")
            smooth = smooth[:len(series)]
            smooth[~valid] = np.nan
            sns.lineplot(x=steps[valid], y=smooth[valid], label=label,
                         color=color, linewidth=3.0, alpha=1.0, ax=ax)

    ax.set_xlabel("Step", fontsize=20)
    ax.set_ylabel("Cluster-level\nRepresentational Similarity", fontsize=16)
    ax.tick_params(axis="both", which="both", top=True, right=True, labelsize=16)
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.legend(fontsize=14, loc="best")
    fig.tight_layout()
    return fig


def render_rsa_agent(data: Dict[str, Any], agent_idx: int) -> matplotlib.figure.Figure:
    sns.set()
    steps = np.asarray(data["steps"])
    agent_rsa = np.asarray(data["agent_rsa"])  # [T, A]

    smooth_window = 25

    fig, ax = plt.subplots(figsize=(12, 4.5))
    if agent_rsa.ndim == 2 and agent_idx < agent_rsa.shape[1] and len(steps) > 0:
        series = agent_rsa[:, agent_idx]
        valid = ~np.isnan(series)
        if valid.any():
            color = "steelblue"
            # raw (low alpha)
            sns.lineplot(x=steps[valid], y=series[valid], label="_nolegend_",
                         color=color, linewidth=1.1, alpha=0.25, ax=ax)
            # smoothed (high alpha)
            pad = smooth_window // 2
            padded = np.pad(series, (pad, pad), mode="edge")
            smooth = np.convolve(padded, np.ones(smooth_window) / smooth_window, mode="valid")
            smooth = smooth[:len(series)]
            smooth[~valid] = np.nan
            sns.lineplot(x=steps[valid], y=smooth[valid], label=f"Agent {agent_idx}",
                         color=color, linewidth=3.0, alpha=1.0, ax=ax)

    ax.set_xlabel("Step", fontsize=20)
    ax.set_ylabel("Representational Similarity", fontsize=16)
    ax.set_title(f"Agent {agent_idx} RSA over time", fontsize=14)
    ax.tick_params(axis="both", which="both", top=True, right=True, labelsize=16)
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.legend(fontsize=14, loc="best")
    fig.tight_layout()
    return fig


def render_rsa_clusters_seeds(data: Dict[str, Any]) -> matplotlib.figure.Figure:
    sns.set()
    steps = np.asarray(data["steps"])
    cluster_rsa_seeds = np.asarray(data["cluster_rsa_seeds"])
    cluster_rsa_mean = np.asarray(data["cluster_rsa_mean"])
    cluster_rsa_std = np.asarray(data["cluster_rsa_std"])

    smooth_window = 25
    tab10 = plt.get_cmap("tab10")
    num_seeds = cluster_rsa_seeds.shape[0] if cluster_rsa_seeds.ndim == 3 else 0

    fig, ax = plt.subplots(figsize=(12, 4.5))
    if cluster_rsa_mean.ndim == 2 and cluster_rsa_std.ndim == 2 and len(steps) > 0:
        num_clusters = cluster_rsa_mean.shape[1]
        for c in range(num_clusters):
            color = tab10(c % 10)
            if cluster_rsa_seeds.ndim == 3:
                for seed_idx in range(cluster_rsa_seeds.shape[0]):
                    seed_series = cluster_rsa_seeds[seed_idx, :, c]
                    valid_seed = ~np.isnan(seed_series)
                    if not valid_seed.any():
                        continue
                    sns.lineplot(
                        x=steps[valid_seed],
                        y=seed_series[valid_seed],
                        label="_nolegend_",
                        color=color,
                        linewidth=0.8,
                        alpha=0.2,
                        ax=ax,
                    )

            smooth_mean = _smooth_series(cluster_rsa_mean[:, c], smooth_window)
            smooth_std = _smooth_series(cluster_rsa_std[:, c], smooth_window)
            valid = ~np.isnan(smooth_mean)
            if not valid.any():
                continue
            ax.fill_between(
                steps[valid],
                (smooth_mean - smooth_std)[valid],
                (smooth_mean + smooth_std)[valid],
                color=color,
                alpha=0.25,
            )
            sns.lineplot(
                x=steps[valid],
                y=smooth_mean[valid],
                label=f"Cluster {c} (mean)",
                color=color,
                linewidth=3.0,
                alpha=1.0,
                ax=ax,
            )

    ax.set_xlabel("Step", fontsize=20)
    ax.set_ylabel("Cluster-level\nRepresentational Similarity", fontsize=16)
    ax.set_title(f"N={num_seeds} seeds", fontsize=13, loc="right")
    ax.tick_params(axis="both", which="both", top=True, right=True, labelsize=16)
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.legend(fontsize=14, loc="best")
    fig.tight_layout()
    return fig


def render_rsa_agent_seeds(data: Dict[str, Any], agent_idx: int) -> matplotlib.figure.Figure:
    sns.set()
    steps = np.asarray(data["steps"])
    agent_rsa_seeds = np.asarray(data["agent_rsa_seeds"])
    agent_rsa_mean = np.asarray(data["agent_rsa_mean"])
    agent_rsa_std = np.asarray(data["agent_rsa_std"])

    smooth_window = 25
    color = "steelblue"
    num_seeds = agent_rsa_seeds.shape[0] if agent_rsa_seeds.ndim == 3 else 0

    fig, ax = plt.subplots(figsize=(12, 4.5))
    if agent_rsa_mean.ndim == 2 and agent_rsa_std.ndim == 2 and agent_idx < agent_rsa_mean.shape[1] and len(steps) > 0:
        if agent_rsa_seeds.ndim == 3 and agent_idx < agent_rsa_seeds.shape[2]:
            for seed_idx in range(agent_rsa_seeds.shape[0]):
                seed_series = agent_rsa_seeds[seed_idx, :, agent_idx]
                valid_seed = ~np.isnan(seed_series)
                if not valid_seed.any():
                    continue
                sns.lineplot(
                    x=steps[valid_seed],
                    y=seed_series[valid_seed],
                    label="_nolegend_",
                    color=color,
                    linewidth=0.8,
                    alpha=0.2,
                    ax=ax,
                )

        smooth_mean = _smooth_series(agent_rsa_mean[:, agent_idx], smooth_window)
        smooth_std = _smooth_series(agent_rsa_std[:, agent_idx], smooth_window)
        valid = ~np.isnan(smooth_mean)
        if valid.any():
            ax.fill_between(
                steps[valid],
                (smooth_mean - smooth_std)[valid],
                (smooth_mean + smooth_std)[valid],
                color=color,
                alpha=0.25,
            )
            sns.lineplot(
                x=steps[valid],
                y=smooth_mean[valid],
                label=f"Agent {agent_idx} (mean)",
                color=color,
                linewidth=3.0,
                alpha=1.0,
                ax=ax,
            )

    ax.set_xlabel("Step", fontsize=20)
    ax.set_ylabel("Representational Similarity", fontsize=16)
    ax.set_title(f"Agent {agent_idx} RSA (N={num_seeds} seeds)", fontsize=14)
    ax.tick_params(axis="both", which="both", top=True, right=True, labelsize=16)
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.legend(fontsize=14, loc="best")
    fig.tight_layout()
    return fig


def render_rsa_clusters_conditions(
    conditions_data: List[Dict[str, Any]],
) -> matplotlib.figure.Figure:
    """Render rsa_clusters comparing multiple conditions (e.g. w/ creation vs w/o creation).

    Each element of conditions_data must have:
        "label"             : str               condition name
        "steps"             : array [T]
        "cluster_rsa_seeds" : array [S, T, C]   per-seed values
        "cluster_rsa_mean"  : array [T, C]      mean across seeds
        "cluster_rsa_std"   : array [T, C]      std  across seeds

    Color encodes cluster; linestyle encodes condition.
    """
    sns.set()
    smooth_window = 25
    tab20 = plt.get_cmap("tab20")
    linestyles = ["-", "--", ":", "-."]

    fig, ax = plt.subplots(figsize=(12, 6))

    for cond_idx, cond in enumerate(conditions_data):
        label = cond["label"]
        steps = np.asarray(cond["steps"])
        cluster_rsa_mean = np.asarray(cond["cluster_rsa_mean"])    # [T, C]
        cluster_rsa_std = np.asarray(cond["cluster_rsa_std"])      # [T, C]
        ls = linestyles[cond_idx % len(linestyles)]
        normalized_label = label.lower()
        use_light_color = "w/o" in normalized_label or "without" in normalized_label

        if cluster_rsa_mean.ndim != 2 or len(steps) == 0:
            continue
        num_clusters = cluster_rsa_mean.shape[1]

        for c in range(num_clusters):
            pair_base = (c % 10) * 2
            dark_color = tab20(pair_base)
            light_color = tab20(pair_base + 1)
            color = light_color if use_light_color else dark_color

            # smoothed mean ± std
            smooth_mean = _smooth_series(cluster_rsa_mean[:, c], smooth_window)
            smooth_std = _smooth_series(cluster_rsa_std[:, c], smooth_window)
            valid = ~np.isnan(smooth_mean)
            if not valid.any():
                continue
            ax.fill_between(
                steps[valid],
                (smooth_mean - smooth_std)[valid],
                (smooth_mean + smooth_std)[valid],
                color=color,
                alpha=0.2,
            )
            ax.plot(
                steps[valid],
                smooth_mean[valid],
                label=f"{label} - cluster {c}",
                color=color,
                linewidth=3.0,
                alpha=1.0,
                linestyle=ls,
            )

    ax.set_xlabel("Step", fontsize=20)
    ax.set_ylabel("Cluster-level\nRepresentational Similarity", fontsize=16)
    ax.tick_params(axis="both", which="both", top=True, right=True, labelsize=16)
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.legend(fontsize=14, loc="best")
    fig.tight_layout()
    return fig
