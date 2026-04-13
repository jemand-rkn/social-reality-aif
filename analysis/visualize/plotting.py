from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.colors as mcolors
import matplotlib.figure
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _get_jet_agent_colors(num_agents: int) -> List[Tuple[float, float, float]]:
    if num_agents <= 0:
        return []
    if num_agents == 1:
        return [plt.cm.jet(0.0)[:3]]
    return [plt.cm.jet(v)[:3] for v in np.linspace(0.0, 1.0, num_agents)]


def _get_cluster_color(cluster_id: int, num_clusters: int) -> Tuple[float, float, float, float]:
    colors = list(mcolors.TABLEAU_COLORS.values())
    if num_clusters > len(colors):
        palette = plt.cm.tab20(np.linspace(0, 1, num_clusters))
        return palette[cluster_id]
    return mcolors.to_rgba(colors[cluster_id % len(colors)])



def _auto_detect_hub_agents(
    agent_to_cluster: List[int],
    num_clusters: int,
    all_acc: List[Optional[Dict]],
) -> List[int]:
    if num_clusters <= 1 or not all_acc:
        return []
    acc = next((item for item in all_acc if item is not None), None)
    if acc is None:
        return []
    adj = np.asarray(acc["adjacency_matrix"])
    num_agents = adj.shape[0]
    hub_ids: List[int] = []
    for tgt in range(num_agents):
        connected_clusters = set()
        for src in range(num_agents):
            if src != tgt and adj[src, tgt] == 1 and agent_to_cluster:
                connected_clusters.add(agent_to_cluster[src])
        if len(connected_clusters) >= 2:
            hub_ids.append(tgt)
    return hub_ids


def _cluster_scatter_marker(cluster_id: int) -> str:
    if cluster_id == 0:
        return "o"
    if cluster_id == 1:
        return "x"
    return "o"


def _set_square_axes_box(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    try:
        ax.set_box_aspect(1)
    except AttributeError:
        ax.set_aspect("equal", adjustable="box")


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size < window or window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="valid")


def _trajectory_annotation_indices(steps: List[int], step_interval: int) -> List[int]:
    if step_interval <= 0 or not steps:
        return []
    step_values = np.asarray(steps, dtype=np.int64)
    targets = np.asarray([2000, 3000, 4000], dtype=np.int64)
    if targets.size == 0:
        return []

    indices: List[int] = []
    for target in targets:
        idx = int(np.argmin(np.abs(step_values - target)))
        if idx <= 0 or idx >= len(step_values) - 1:
            continue
        if indices and idx == indices[-1]:
            continue
        indices.append(idx)
    return indices


def _flatten_reference_obs(reference_obs: np.ndarray) -> np.ndarray:
    ref_np = np.asarray(reference_obs, dtype=np.float64)
    if ref_np.ndim == 3:
        ref_np = ref_np.reshape(-1, ref_np.shape[-1])
    elif ref_np.ndim != 2:
        ref_np = ref_np.reshape(ref_np.shape[0], -1)
    return ref_np


def _as_obs_2d(reference_obs: np.ndarray) -> np.ndarray:
    ref_np = _flatten_reference_obs(reference_obs)
    if ref_np.shape[1] >= 2:
        return ref_np[:, :2]
    padded = np.zeros((ref_np.shape[0], 2), dtype=np.float64)
    padded[:, 0] = ref_np[:, 0]
    return padded


def _smooth_trajectory(traj: np.ndarray, window: int) -> np.ndarray:
    if traj.shape[0] < 3 or window <= 1:
        return traj
    window = min(window, traj.shape[0])
    if window % 2 == 0:
        window = max(1, window - 1)
    if window <= 2:
        return traj
    try:
        from scipy.signal import savgol_filter

        polyorder = min(3, window - 1)
        smoothed = np.zeros_like(traj)
        for dim in range(traj.shape[1]):
            smoothed[:, dim] = savgol_filter(traj[:, dim], window_length=window, polyorder=polyorder, mode="mirror")
        return smoothed
    except Exception:
        kernel = np.ones(window, dtype=np.float64) / float(window)
        pad = window // 2
        padded = np.pad(traj, ((pad, pad), (0, 0)), mode="edge")
        out = np.zeros_like(traj)
        for dim in range(traj.shape[1]):
            out[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
        return out


def _set_square_axes_box_auto(ax: plt.Axes) -> None:
    ax.set_aspect("auto")
    try:
        ax.set_box_aspect(1)
    except AttributeError:
        pass


def _darken_color(color: Tuple[float, float, float], factor: float = 0.7) -> Tuple[float, float, float]:
    return tuple(max(0.0, min(1.0, c * factor)) for c in color)  # type: ignore[return-value]


def _lighten_color(color: Tuple[float, float, float], factor: float = 0.25) -> Tuple[float, float, float]:
    return tuple(max(0.0, min(1.0, c + (1.0 - c) * factor)) for c in color)  # type: ignore[return-value]


def _get_clustered_agent_colors(
    num_agents: int,
    agent_to_cluster: List[int],
    hub_agent_ids: Optional[Sequence[int]] = None,
) -> List[Tuple[float, float, float]]:
    tab20 = plt.cm.get_cmap("tab20")
    hub_agent_id_set = {int(agent_id) for agent_id in (hub_agent_ids or [])}
    colors: List[Tuple[float, float, float]] = []
    for agent_id in range(num_agents):
        cluster_id = agent_to_cluster[agent_id] if agent_to_cluster and agent_id < len(agent_to_cluster) else 0
        pair_base = (cluster_id % 10) * 2
        light_color = tuple(tab20(pair_base + 1)[:3])
        dark_color = tuple(tab20(pair_base)[:3])
        colors.append(dark_color if agent_id in hub_agent_id_set else light_color)
    return colors


class DataVisualizer:
    def __init__(self, obs_shape: Sequence[int]):
        self.obs_shape = tuple(obs_shape)
        self.colors = list(mcolors.TABLEAU_COLORS.values())

    def _get_color(self, agent_id: int, num_agents: int):
        if num_agents > len(self.colors):
            colors = plt.cm.tab20(np.linspace(0, 1, num_agents))
            return colors[agent_id]
        return self.colors[agent_id % len(self.colors)]

    def _draw_cluster_legend(
        self,
        ax: plt.Axes,
        agent_to_cluster: List[int],
        agent_colors: List[Tuple[float, float, float]],
        title: Optional[str] = None,
        font_scale: float = 1.0,
        row_gap_scale: float = 1.1,
    ) -> None:
        clusters: Dict[int, List[int]] = {}
        for agent_id, cluster_id in enumerate(agent_to_cluster):
            clusters.setdefault(cluster_id, []).append(agent_id)
        ax.axis("off")
        if title:
            ax.set_title(title, fontweight="bold", fontsize=24 * font_scale)
        if not clusters:
            return
        row_gap = (1.0 / (len(clusters) + 0.9)) * row_gap_scale
        label_block = 0.14
        label_to_agent_gap = 0.02
        agent_block = 0.095
        agent_text_offset = 0.009
        for row_idx, cluster_id in enumerate(sorted(clusters)):
            y = 0.5 - (row_idx + 1) * row_gap
            members = clusters[cluster_id]
            total_span = label_block + label_to_agent_gap + agent_block * len(members)
            x_left = max(0.01, 0.5 - total_span / 2.0)
            ax.text(
                x_left,
                y,
                f"cluster {cluster_id}:",
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontweight="bold",
                fontsize=24 * font_scale,
            )
            x = x_left + label_block + label_to_agent_gap
            for agent_id in members:
                marker = _cluster_scatter_marker(cluster_id)
                ax.scatter([x], [y], s=90, color=agent_colors[agent_id], marker=marker, transform=ax.transAxes, clip_on=False)
                ax.text(
                    x + agent_text_offset,
                    y,
                    f"agent {agent_id}",
                    transform=ax.transAxes,
                    ha="left",
                    va="center",
                    fontsize=24 * font_scale,
                )
                x += agent_block

    def plot_clustered_agent_legend(self, data: Dict, color_mode: str, title: Optional[str] = None) -> matplotlib.figure.Figure:
        observations = data.get("observations", [])
        num_agents = len(observations)
        agent_to_cluster = list(data.get("agent_to_cluster", [])) or [0] * num_agents
        num_clusters = int(data.get("num_clusters", max(agent_to_cluster) + 1 if agent_to_cluster else 1))

        if color_mode == "agent":
            agent_colors = [self._get_color(agent_id, num_agents) for agent_id in range(num_agents)]
        elif color_mode == "cluster":
            agent_colors = [_get_cluster_color(agent_to_cluster[agent_id], max(1, num_clusters)) for agent_id in range(num_agents)]
        elif color_mode == "jet":
            agent_colors = _get_jet_agent_colors(num_agents)
        else:
            raise ValueError(f"Unsupported legend color_mode: {color_mode}")

        fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * max(1, num_agents)), max(1.6, 0.65 * max(1, num_clusters))))
        self._draw_cluster_legend(ax, agent_to_cluster, agent_colors, title=title)
        fig.tight_layout()
        return fig

    def _plot_scatter_pairs(
        self,
        ax: plt.Axes,
        observations: List[np.ndarray],
        creations: Optional[List[np.ndarray]],
        colors: List,
        step: int,
        markers: Optional[List[str]] = None,
        observation_alpha: float = 0.1,
        shuffle_observation_point_order: bool = False,
        font_scale: float = 1.0,
        step_label_y: float = -0.30,
    ) -> None:
        if shuffle_observation_point_order:
            point_cloud: List[Tuple[float, float, Any, str]] = []
            for agent_id, obs in enumerate(observations):
                obs_np = np.asarray(obs)
                if obs_np.size == 0:
                    continue
                obs_2d = obs_np.reshape(obs_np.shape[0], -1)
                marker = markers[agent_id] if markers is not None and agent_id < len(markers) else "o"
                color = colors[agent_id]
                for x, y in obs_2d[:, :2]:
                    point_cloud.append((float(x), float(y), color, marker))

            if point_cloud:
                rng = np.random.default_rng(int(step))
                order = rng.permutation(len(point_cloud))
                shuffled = [point_cloud[int(idx)] for idx in order]

                marker_groups: Dict[str, List[Tuple[float, float, Any]]] = {}
                for x, y, color, marker in shuffled:
                    marker_groups.setdefault(marker, []).append((x, y, color))

                marker_order = list(marker_groups.keys())
                marker_order = [marker_order[int(i)] for i in rng.permutation(len(marker_order))]
                for marker in marker_order:
                    entries = marker_groups[marker]
                    xs = [entry[0] for entry in entries]
                    ys = [entry[1] for entry in entries]
                    cs = [entry[2] for entry in entries]
                    ax.scatter(xs, ys, c=cs, alpha=observation_alpha, s=45, marker=marker)
        else:
            for agent_id, obs in enumerate(observations):
                obs_np = np.asarray(obs)
                if obs_np.size == 0:
                    continue
                obs_2d = obs_np.reshape(obs_np.shape[0], -1)
                marker = markers[agent_id] if markers is not None and agent_id < len(markers) else "o"
                ax.scatter(obs_2d[:, 0], obs_2d[:, 1], c=[colors[agent_id]], alpha=observation_alpha, s=45, marker=marker)
        if creations is not None:
            for agent_id, created in enumerate(creations):
                created_np = np.asarray(created)
                if created_np.size == 0:
                    continue
                created_2d = created_np.reshape(created_np.shape[0], -1)
                marker = markers[agent_id] if markers is not None and agent_id < len(markers) else "o"
                ax.scatter(
                    created_2d[:, 0],
                    created_2d[:, 1],
                    c=[colors[agent_id]],
                    alpha=0.5,
                    s=110,
                    edgecolors="black",
                    linewidths=0.5,
                    marker=marker,
                )
        ax.set_xlabel("Dimension 0", fontsize=12 * font_scale)
        ax.set_ylabel("Dimension 1", fontsize=12 * font_scale)
        ax.tick_params(axis="both", labelsize=20 * font_scale)
        ax.grid(True, alpha=0.3)
        _set_square_axes_box(ax)
        ax.text(
            0.5,
            step_label_y,
            f"step={step}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontweight="bold",
            fontsize=24 * font_scale,
        )

    def plot_observations_and_creations_clustered(self, data: Dict, step: int) -> matplotlib.figure.Figure:
        observations = data.get("observations", [])
        agent_to_cluster = list(data.get("agent_to_cluster", []))
        num_clusters = int(data.get("num_clusters", 1))
        colors = [_get_cluster_color(agent_to_cluster[agent_id], max(1, num_clusters)) for agent_id in range(len(observations))]
        fig, ax = plt.subplots(figsize=(8, 8))
        self._plot_scatter_pairs(ax, observations, data.get("creations"), colors, step)
        fig.tight_layout()
        return fig

    def _plot_observations_and_creations_agents_clustered_on_ax(self, data: Dict, step: int, ax: plt.Axes) -> None:
        observations = data.get("observations", [])
        colors = _get_jet_agent_colors(len(observations))
        self._plot_scatter_pairs(ax, observations, data.get("creations"), colors, step)

    def plot_observations_and_creations_agents_clustered(self, data: Dict, step: int) -> matplotlib.figure.Figure:
        fig, ax = plt.subplots(figsize=(8, 8))
        self._plot_observations_and_creations_agents_clustered_on_ax(data, step, ax)
        fig.tight_layout()
        return fig

    def _plot_observations_agents_clustered_on_ax(
        self,
        data: Dict,
        step: int,
        ax: plt.Axes,
        font_scale: float = 1.0,
        step_label_y: float = -0.20,
    ) -> None:
        observations = data.get("observations", [])
        agent_to_cluster = list(data.get("agent_to_cluster", []))
        colors = _get_jet_agent_colors(len(observations))
        markers = [_cluster_scatter_marker(agent_to_cluster[i] if i < len(agent_to_cluster) else 0) for i in range(len(observations))]
        self._plot_scatter_pairs(
            ax,
            observations,
            None,
            colors,
            step,
            markers=markers,
            observation_alpha=0.3,
            shuffle_observation_point_order=True,
            font_scale=font_scale,
            step_label_y=step_label_y,
        )

    def plot_observations_agents_clustered(self, data: Dict, step: int) -> matplotlib.figure.Figure:
        fig, ax = plt.subplots(figsize=(8, 8))
        self._plot_observations_agents_clustered_on_ax(data, step, ax)
        fig.tight_layout()
        return fig

    def _fgw_similarity_plot_config(self, alpha: float) -> Tuple[str, str, float, float]:
        if alpha == 0.0:
            return "distance_matrix_00", "Wasserstein Distance", 0.0, 11.0
            # return "distance_matrix_00", "Wasserstein Distance", None, None
        if alpha == 1.0:
            return "distance_matrix_10", "Gromov-Wasserstein Distance", 0.0, 16.0
            # return "distance_matrix_10", "Gromov-Wasserstein Distance", None, None
        raise ValueError(f"Unsupported alpha for FGW matrix plot: {alpha}")

    def _plot_fgw_similarity_matrix_on_ax(
        self,
        ax: plt.Axes,
        data: Dict,
        alpha: float,
        step: Any,
        *,
        show_ylabel: bool = True,
        font_scale: float = 1.0,
        step_label_y: float = -0.16,
        step_label_formatter: Optional[Callable[[Any], str]] = None,
        vmin_override: Optional[float] = None,
        vmax_override: Optional[float] = None,
    ):
        matrix_key, _colorbar_label, vmin, vmax = self._fgw_similarity_plot_config(alpha)
        if vmin_override is not None:
            vmin = vmin_override
        if vmax_override is not None:
            vmax = vmax_override
        matrix = np.asarray(data[matrix_key], dtype=np.float64)
        num_agents = matrix.shape[0]

        im = ax.imshow(matrix, cmap="viridis_r", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xlabel("Agent ID", fontsize=16 * font_scale)
        ax.set_ylabel("Agent ID" if show_ylabel else "", fontsize=16 * font_scale)
        ax.set_xticks(range(num_agents))
        ax.set_yticks(range(num_agents))
        ax.tick_params(axis="both", labelsize=12 * font_scale)
        ax.set_xticks(np.arange(num_agents) - 0.5, minor=True)
        ax.set_yticks(np.arange(num_agents) - 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)
        if step_label_formatter is None:
            step_text = f"step = {step}"
        else:
            step_text = step_label_formatter(step)
        ax.text(
            0.5,
            step_label_y,
            step_text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontweight="bold",
            fontsize=16 * font_scale,
        )
        return im

    def plot_fgw_similarity_matrix(self, data: Dict, alpha: float, step: int) -> matplotlib.figure.Figure:
        _matrix_key, colorbar_label, _vmin, _vmax = self._fgw_similarity_plot_config(alpha)
        fig, ax = plt.subplots(figsize=(6, 6))
        im = self._plot_fgw_similarity_matrix_on_ax(ax, data, alpha, step)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.08)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(colorbar_label, fontsize=14)
        fig.tight_layout()
        return fig

    def plot_fgw_similarity_snapshot(
        self,
        panel_items: List[Tuple[Any, Dict]],
        alpha: float,
        font_scale: float = 1.0,
        step_label_y: float = -0.16,
        step_label_formatter: Optional[Callable[[Any], str]] = None,
        vmin_override: Optional[float] = None,
        vmax_override: Optional[float] = None,
    ) -> matplotlib.figure.Figure:
        if not panel_items:
            raise ValueError("panel_items is empty")

        _matrix_key, colorbar_label, _vmin, _vmax = self._fgw_similarity_plot_config(alpha)
        num_panels = len(panel_items)
        fig_w = max(7.0, 4.6 * num_panels + 0.7)
        fig, axes = plt.subplots(1, num_panels, figsize=(fig_w, 4.8), squeeze=False, constrained_layout=True)
        axes_list = list(axes[0])

        im = None
        for idx, ((step, data), ax) in enumerate(zip(panel_items, axes_list)):
            im = self._plot_fgw_similarity_matrix_on_ax(
                ax,
                data,
                alpha,
                step,
                show_ylabel=(idx == 0),
                font_scale=font_scale,
                step_label_y=step_label_y,
                step_label_formatter=step_label_formatter,
                vmin_override=vmin_override,
                vmax_override=vmax_override,
            )

        assert im is not None
        cbar = fig.colorbar(im, ax=axes_list[-1], fraction=0.046, pad=0.04)
        cbar.set_label(colorbar_label, fontsize=14 * font_scale)
        cbar.ax.tick_params(labelsize=12 * font_scale)
        return fig

    def plot_social_network(self, data: Dict, step: int) -> matplotlib.figure.Figure:
        adjacency_matrix = np.asarray(data.get("adjacency_matrix"))
        agent_to_cluster = list(data.get("agent_to_cluster", []))
        num_clusters = int(data.get("num_clusters", 0))
        num_agents = adjacency_matrix.shape[0]
        hub_agent_ids_raw = data.get("hub_agent_ids")
        if hub_agent_ids_raw is None:
            hub_agent_ids = _auto_detect_hub_agents(agent_to_cluster, num_clusters, [data])
        else:
            hub_agent_ids = [int(agent_id) for agent_id in np.asarray(hub_agent_ids_raw).tolist()]
        graph, pos = self._social_network_graph_and_layout(adjacency_matrix)

        fig, (ax_graph, ax_matrix) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 0.9]})
        node_colors = _get_clustered_agent_colors(num_agents, agent_to_cluster, hub_agent_ids)
        nx.draw_networkx_nodes(graph, pos, node_size=900, node_color=node_colors, ax=ax_graph)
        nx.draw_networkx_edges(graph, pos, edge_color="gray", alpha=0.6, width=2.5, ax=ax_graph)
        nx.draw_networkx_labels(graph, pos, font_size=18, font_color="black", ax=ax_graph)
        ax_graph.text(0.0, 0.9, "A", transform=ax_graph.transAxes, ha="left", va="bottom", fontweight="bold", fontsize=26, clip_on=False)
        # ax_graph.set_title(f"Social Network (step={step})")
        ax_graph.axis("off")
        im = ax_matrix.imshow(adjacency_matrix, cmap="Blues", aspect="equal")
        ax_matrix.text(-0.12, 0.94, "B", transform=ax_matrix.transAxes, ha="left", va="bottom", fontweight="bold", fontsize=26, clip_on=False)
        ax_matrix.set_xlabel("Agent ID", fontsize=17)
        ax_matrix.set_ylabel("Agent ID", fontsize=17)
        ax_matrix.set_xticks(range(num_agents))
        ax_matrix.set_yticks(range(num_agents))
        ax_matrix.tick_params(axis="both", labelsize=15)
        colorbar = fig.colorbar(im, ax=ax_matrix, fraction=0.046, pad=0.04)
        colorbar.ax.tick_params(labelsize=15)
        fig.tight_layout()
        return fig

    def _social_network_graph_and_layout(self, adjacency_matrix: np.ndarray) -> Tuple[nx.Graph, Dict[int, np.ndarray]]:
        graph = nx.from_numpy_array(np.asarray(adjacency_matrix))
        try:
            pos = nx.spring_layout(graph, seed=42)
        except Exception:
            pos = nx.circular_layout(graph)
        return graph, pos

    def plot_latents_from_reference(self, data: Dict, step: int) -> matplotlib.figure.Figure:
        ref_2d = _as_obs_2d(np.asarray(data["reference_obs"]))
        latents_2d_list = [np.asarray(item, dtype=np.float64) for item in data["latents_2d"]]
        if not latents_2d_list:
            raise ValueError("latents_2d is empty")

        x = ref_2d[:, 0]
        y = ref_2d[:, 1]
        eps = 1e-8
        x_norm = (x - x.min()) / (x.max() - x.min() + eps)
        y_norm = (y - y.min()) / (y.max() - y.min() + eps)
        base_rgb = plt.cm.jet(x_norm)[:, :3]
        hsv = mcolors.rgb_to_hsv(base_rgb)
        hsv[:, 1] = y_norm
        hsv[:, 2] = 1.0
        colors = mcolors.hsv_to_rgb(hsv)

        all_latents = np.concatenate(latents_2d_list, axis=0)
        x_min, x_max = float(all_latents[:, 0].min()), float(all_latents[:, 0].max())
        y_min, y_max = float(all_latents[:, 1].min()), float(all_latents[:, 1].max())
        x_pad = 0.05 * (x_max - x_min + 1e-8)
        y_pad = 0.05 * (y_max - y_min + 1e-8)

        total = len(latents_2d_list) + 1
        n_cols = int(np.ceil(np.sqrt(total)))
        n_rows = int(np.ceil(total / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
        axes_flat = axes.flatten()

        axes_flat[0].scatter(ref_2d[:, 0], ref_2d[:, 1], c=colors, s=20, alpha=0.8)
        axes_flat[0].set_title("Reference Obs", fontweight="bold")
        axes_flat[0].set_xlabel("obs dim 0")
        axes_flat[0].set_ylabel("obs dim 1")
        axes_flat[0].grid(True, alpha=0.3)

        for agent_id, lat_2d in enumerate(latents_2d_list):
            ax = axes_flat[agent_id + 1]
            ax.scatter(lat_2d[:, 0], lat_2d[:, 1], c=colors, s=20, alpha=0.8)
            ax.set_title(f"Agent {agent_id} Latent", fontweight="bold")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.grid(True, alpha=0.3)

        for idx in range(total, len(axes_flat)):
            axes_flat[idx].axis("off")

        fig.suptitle(f"Reference Obs and Latents (shared PCA) - Step {step}", fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig

    def _plot_acceptance_matrix(self, data: Dict, step: int, title_prefix: str) -> matplotlib.figure.Figure:
        accept_rates = np.asarray(data["accept_rates"], dtype=float)
        num_agents = accept_rates.shape[0]
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(np.ma.masked_invalid(accept_rates), cmap="viridis", vmin=0, vmax=1, aspect="equal")
        ax.set_xlabel("Source Agent")
        ax.set_ylabel("Target Agent")
        ax.set_xticks(range(num_agents))
        ax.set_yticks(range(num_agents))
        ax.set_title(f"{title_prefix} Acceptance Rate\nstep = {step}")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.08)
        fig.colorbar(im, cax=cax).set_label("Acceptance Rate")
        fig.tight_layout()
        return fig

    def _plot_acceptance_flow_network_on_ax(
        self,
        fig: matplotlib.figure.Figure,
        ax: plt.Axes,
        data: Dict,
        *,
        show_colorbar: bool = True,
        node_size: float = 900,
        position_scale: float = 1.0,
        node_label_font_size: float = 18,
        edge_width_scale: float = 1.0,
    ) -> None:
        raw_rates = np.asarray(data["accept_rates"], dtype=float)
        accept_rates = np.nan_to_num(raw_rates, nan=0.0)
        adjacency_matrix = np.asarray(data.get("adjacency_matrix"))
        proposed_counts_raw = data.get("proposed_counts")
        proposed_counts = np.asarray(proposed_counts_raw) if proposed_counts_raw is not None else None
        zero_accept_fraction_raw = data.get("zero_accept_fraction")
        zero_accept_fraction = np.asarray(zero_accept_fraction_raw, dtype=float) if zero_accept_fraction_raw is not None else None
        agent_to_cluster = list(data.get("agent_to_cluster", []))
        num_clusters = int(data.get("num_clusters", 0))
        num_agents = accept_rates.shape[0]

        social_graph, pos = self._social_network_graph_and_layout(adjacency_matrix)
        if position_scale != 1.0:
            pos = {node: np.asarray(coord, dtype=float) * float(position_scale) for node, coord in pos.items()}
        graph = nx.DiGraph()
        graph.add_nodes_from(range(num_agents))

        edge_weight_map: Dict[Tuple[int, int], float] = {}
        all_edge_weights: List[float] = []

        for src in range(num_agents):
            for tgt in range(num_agents):
                if src == tgt:
                    continue
                if adjacency_matrix[src, tgt] == 0:
                    continue

                if proposed_counts is not None:
                    if proposed_counts[src, tgt] <= 0:
                        continue
                    weight = float(accept_rates[src, tgt])
                else:
                    if not np.isfinite(raw_rates[src, tgt]):
                        continue
                    weight = float(raw_rates[src, tgt])

                if weight <= 0.0:
                    continue

                graph.add_edge(src, tgt, weight=weight)
                all_edge_weights.append(weight)
                edge_weight_map[(src, tgt)] = weight

        if agent_to_cluster:
            hub_agent_ids = _auto_detect_hub_agents(agent_to_cluster, num_clusters, [{"adjacency_matrix": adjacency_matrix}])
            node_colors = _get_clustered_agent_colors(num_agents, agent_to_cluster, hub_agent_ids)
        else:
            node_colors = ["tab:blue"] * num_agents

        nx.draw_networkx_nodes(social_graph, pos, ax=ax, node_color=node_colors, node_size=node_size)
        nx.draw_networkx_labels(social_graph, pos, ax=ax, font_size=node_label_font_size, font_color="black")

        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

        if all_edge_weights:
            for (src, tgt), weight in edge_weight_map.items():
                has_reverse = (tgt, src) in edge_weight_map
                connectionstyle = "arc3,rad=0.12" if has_reverse else "arc3,rad=0.0"
                min_target_margin = 16 if has_reverse else 22

                if zero_accept_fraction is not None and zero_accept_fraction.shape == accept_rates.shape:
                    frac = float(np.clip(zero_accept_fraction[src, tgt], 0.0, 1.0))
                    # Fully transparent when at least one-third of interval steps are zero-accept.
                    alpha = 0.95 * max(0.0, 1.0 - (frac / (1.0 / 3.0)))
                else:
                    alpha = 0.95

                nx.draw_networkx_edges(
                    graph,
                    pos,
                    edgelist=[(src, tgt)],
                    ax=ax,
                    width=(1.5 + 6.0 * float(np.clip(weight, 0.0, 1.0))) * edge_width_scale,
                    edge_color=[weight],
                    edge_cmap=plt.cm.plasma,
                    edge_vmin=0.0,
                    edge_vmax=1.0,
                    connectionstyle=connectionstyle,
                    arrows=True,
                    arrowstyle="-|>",
                    arrowsize=20,
                    min_source_margin=10,
                    min_target_margin=min_target_margin,
                    alpha=alpha,
                )

        if show_colorbar:
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.plasma)
            if all_edge_weights:
                mappable.set_array(np.asarray(all_edge_weights, dtype=float))
            else:
                mappable.set_array(np.asarray([0.0, 1.0], dtype=float))
            divider_right = make_axes_locatable(ax)
            cax_right = divider_right.append_axes("right", size="5%", pad=0.08)
            cbar = fig.colorbar(mappable, cax=cax_right)
            cbar.set_label("Edge Acceptance Rate", fontsize=15)
            cbar.ax.tick_params(labelsize=13)

        ax.axis("off")

    def plot_acceptance_flow_network(self, data: Dict, step: int, title_prefix: str, net_flow_threshold: float = 0.05) -> matplotlib.figure.Figure:
        fig, ax = plt.subplots(1, 1, figsize=(9, 7.5))
        self._plot_acceptance_flow_network_on_ax(fig, ax, data)

        ax.text(
            0.5,
            -0.08,
            f"step={step}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontweight="bold",
            fontsize=18,
        )

        fig.tight_layout()
        return fig

    def plot_both_acceptance_networks(self, mhng_data: Dict, memorize_data: Dict, step: int) -> matplotlib.figure.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
        self._plot_acceptance_flow_network_on_ax(fig, axes[0], mhng_data, show_colorbar=False)
        self._plot_acceptance_flow_network_on_ax(fig, axes[1], memorize_data, show_colorbar=True)

        axes[0].text(0.5, 1.01, "Social representations acceptance", transform=axes[0].transAxes, ha="center", va="bottom", fontweight="bold", fontsize=17)
        axes[1].text(0.5, 1.01, "Creations acceptance", transform=axes[1].transAxes, ha="center", va="bottom", fontweight="bold", fontsize=17)
        fig.text(0.5, 0.02, f"step={step}", ha="center", va="top", fontweight="bold", fontsize=18)
        fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
        return fig


def _plot_fgw_mds_trajectories(
    positions: np.ndarray,
    steps: List[int],
    scatter_steps: int,
    smooth_window: int,
    contrib: Optional[np.ndarray],
    agent_to_cluster: Optional[List[int]] = None,
    axis_prefix: str = "MDS",
    contrib_note: Optional[str] = None,
) -> matplotlib.figure.Figure:
    sns.set_theme()
    positions = np.asarray(positions, dtype=np.float64)
    num_steps, num_agents, _ = positions.shape
    colors = _get_jet_agent_colors(num_agents)

    fig, ax = plt.subplots(figsize=(9, 7))

    if num_steps == 0 or num_agents == 0:
        ax.set_xlabel(f"{axis_prefix}-1", fontsize=16)
        ax.set_ylabel(f"{axis_prefix}-2", fontsize=16)
        ax.grid(True, alpha=0.3)
        _set_square_axes_box_auto(ax)
        plt.tight_layout()
        return fig

    smoothed_trajs = [_smooth_trajectory(positions[:, agent_id, :], smooth_window) for agent_id in range(num_agents)]
    smoothed_all = np.concatenate(smoothed_trajs, axis=0)
    x_min = float(np.nanmin(smoothed_all[:, 0]))
    x_max = float(np.nanmax(smoothed_all[:, 0]))
    y_min = float(np.nanmin(smoothed_all[:, 1]))
    y_max = float(np.nanmax(smoothed_all[:, 1]))
    x_span = (x_max - x_min) if np.isfinite(x_max - x_min) and (x_max - x_min) > 0 else 1.0
    y_span = (y_max - y_min) if np.isfinite(y_max - y_min) and (y_max - y_min) > 0 else 1.0
    pad = 0.02 * max(x_span, y_span)
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)

    label_positions: List[np.ndarray] = []
    step_label_offset = 0.012 * max(x_span, y_span)
    annotation_indices = _trajectory_annotation_indices(steps, scatter_steps)

    for agent_id in range(num_agents):
        smoothed = smoothed_trajs[agent_id]
        linestyle = "-"
        if agent_to_cluster and agent_id < len(agent_to_cluster) and agent_to_cluster[agent_id] % 2 == 1:
            linestyle = "--"
        ax.plot(smoothed[:, 0], smoothed[:, 1], color=colors[agent_id], linewidth=2.8, linestyle=linestyle, alpha=1.0)

        if annotation_indices:
            annotation_points = smoothed[annotation_indices]
            ax.scatter(annotation_points[:, 0], annotation_points[:, 1], color=colors[agent_id], s=14, alpha=0.95, zorder=5)
            for idx in annotation_indices:
                ax.text(
                    smoothed[idx, 0],
                    smoothed[idx, 1] + step_label_offset,
                    f"{steps[idx]}",
                    color="black",
                    fontsize=7,
                    ha="center",
                    va="bottom",
                    zorder=6,
                )

        candidate = smoothed[-1]
        label_positions.append(candidate)

        ax.text(
            candidate[0],
            candidate[1] + step_label_offset,
            f"{steps[-1]}",
            color="black",
            fontsize=7,
            ha="center",
            va="bottom",
            zorder=6,
        )

        ax.text(
            candidate[0], candidate[1], f"{agent_id}",
            color=colors[agent_id], fontsize=14, va="center", ha="center", zorder=6,
            path_effects=[path_effects.withStroke(linewidth=1.5, foreground="black")],
        )

    # Axis labels with contribution %
    # contrib_arr = np.asarray(contrib, dtype=np.float64) if contrib is not None else None
    # if contrib_arr is not None and contrib_arr.size >= 2:
    #     xlabel = f"{axis_prefix}-1 ({contrib_arr[0] * 100:.1f}%)" if np.isfinite(contrib_arr[0]) and contrib_arr[0] > 0 else f"{axis_prefix}-1"
    #     ylabel = f"{axis_prefix}-2 ({contrib_arr[1] * 100:.1f}%)" if np.isfinite(contrib_arr[1]) and contrib_arr[1] > 0 else f"{axis_prefix}-2"
    # else:
    #     xlabel = f"{axis_prefix}-1"
    #     ylabel = f"{axis_prefix}-2"
    xlabel = f"{axis_prefix}-1"
    ylabel = f"{axis_prefix}-2"
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)

    if contrib_note:
        ax.text(0.01, 0.99, contrib_note, transform=ax.transAxes, ha="left", va="top", fontsize=9,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"})

    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=14)
    _set_square_axes_box_auto(ax)

    legend_handles = []
    for i in range(num_agents):
        linestyle = "-"
        if agent_to_cluster and i < len(agent_to_cluster) and agent_to_cluster[i] % 2 == 1:
            linestyle = "--"
        legend_handles.append(Line2D([0], [0], color=colors[i], lw=2.4, label=f"agent {i}", linestyle=linestyle))
    if legend_handles:
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=True)

    fig.subplots_adjust(right=0.78)
    plt.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    return fig


def _plot_fgw_mds_trajectories_clustered(
    positions: np.ndarray,
    steps: List[int],
    scatter_steps: int,
    smooth_window: int,
    contrib: Optional[np.ndarray],
    agent_to_cluster: List[int],
    num_clusters: int,
    hub_agent_ids: Optional[List[int]] = None,
    axis_prefix: str = "MDS",
    contrib_note: Optional[str] = None,
) -> matplotlib.figure.Figure:
    sns.set_theme()
    positions = np.asarray(positions, dtype=np.float64)
    num_steps, num_agents, _ = positions.shape

    colors = _get_clustered_agent_colors(num_agents, agent_to_cluster, hub_agent_ids)
    hub_agent_id_set = set(hub_agent_ids or [])

    fig, ax = plt.subplots(figsize=(9, 7))

    if num_steps == 0 or num_agents == 0:
        ax.set_xlabel(f"{axis_prefix}-1", fontsize=16)
        ax.set_ylabel(f"{axis_prefix}-2", fontsize=16)
        ax.grid(True, alpha=0.3)
        _set_square_axes_box_auto(ax)
        plt.tight_layout()
        return fig

    smoothed_trajs = [_smooth_trajectory(positions[:, agent_id, :], smooth_window) for agent_id in range(num_agents)]
    smoothed_all = np.concatenate(smoothed_trajs, axis=0)
    x_min = float(np.nanmin(smoothed_all[:, 0]))
    x_max = float(np.nanmax(smoothed_all[:, 0]))
    y_min = float(np.nanmin(smoothed_all[:, 1]))
    y_max = float(np.nanmax(smoothed_all[:, 1]))
    x_span = (x_max - x_min) if np.isfinite(x_max - x_min) and (x_max - x_min) > 0 else 1.0
    y_span = (y_max - y_min) if np.isfinite(y_max - y_min) and (y_max - y_min) > 0 else 1.0
    pad = 0.02 * max(x_span, y_span)
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)

    label_positions: List[np.ndarray] = []
    step_label_offset = 0.012 * max(x_span, y_span)
    annotation_indices = _trajectory_annotation_indices(steps, scatter_steps)
    draw_order = sorted(range(num_agents), key=lambda agent_id: (agent_id in hub_agent_id_set, agent_id))
    for agent_id in draw_order:
        smoothed = smoothed_trajs[agent_id]
        ax.plot(smoothed[:, 0], smoothed[:, 1], color=colors[agent_id], linewidth=2.8, alpha=1.0)

        if annotation_indices:
            annotation_points = smoothed[annotation_indices]
            ax.scatter(annotation_points[:, 0], annotation_points[:, 1], color=colors[agent_id], s=14, alpha=0.95, zorder=5)
            for idx in annotation_indices:
                ax.text(
                    smoothed[idx, 0],
                    smoothed[idx, 1] + step_label_offset,
                    f"{steps[idx]}",
                    color="black",
                    fontsize=7,
                    ha="center",
                    va="bottom",
                    zorder=6,
                )

        candidate = smoothed[-1]
        label_positions.append(candidate)

        ax.text(
            candidate[0],
            candidate[1] + step_label_offset,
            f"{steps[-1]}",
            color="black",
            fontsize=7,
            ha="center",
            va="bottom",
            zorder=6,
        )

        ax.text(
            candidate[0], candidate[1], f"{agent_id}",
            color=colors[agent_id], fontsize=14, va="center", ha="center", zorder=6,
            path_effects=[path_effects.withStroke(linewidth=1.5, foreground="black")],
        )

    # contrib_arr = np.asarray(contrib, dtype=np.float64) if contrib is not None else None
    # if contrib_arr is not None and contrib_arr.size >= 2:
    #     xlabel = f"{axis_prefix}-1 ({contrib_arr[0] * 100:.1f}%)" if np.isfinite(contrib_arr[0]) and contrib_arr[0] > 0 else f"{axis_prefix}-1"
    #     ylabel = f"{axis_prefix}-2 ({contrib_arr[1] * 100:.1f}%)" if np.isfinite(contrib_arr[1]) and contrib_arr[1] > 0 else f"{axis_prefix}-2"
    # else:
    #     xlabel = f"{axis_prefix}-1"
    #     ylabel = f"{axis_prefix}-2"
    xlabel = f"{axis_prefix}-1"
    ylabel = f"{axis_prefix}-2"
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)

    if contrib_note:
        ax.text(0.01, 0.99, contrib_note, transform=ax.transAxes, ha="left", va="top", fontsize=9,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"})

    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=14)
    _set_square_axes_box_auto(ax)

    legend_handles = [Line2D([0], [0], color=colors[i], lw=2.4, label=f"agent {i}") for i in range(num_agents)]
    if legend_handles:
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=True)

    fig.subplots_adjust(right=0.78)
    plt.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    return fig


def _plot_fgw_mds_trajectories_clustered_segmented(
    positions: np.ndarray,
    steps: List[int],
    segment_boundaries: List[int],
    smooth_window: int,
    agent_to_cluster: List[int],
    num_clusters: int,
    hub_agent_ids: Optional[List[int]] = None,
    axis_prefix: str = "MDS",
) -> matplotlib.figure.Figure:
    """Plot GW-MDS trajectories in separate axes per step-interval segment.

    Given ``segment_boundaries`` (e.g. [100, 1000, 5000]) the trajectory is
    split into intervals ``[first_step, b0], (b0, b1], (b1, b2], …`` and each
    interval is plotted in its own subplot.  All axes share the same spatial
    limits as the full trajectory so that positions are comparable across
    panels.  The legend is placed in the first empty grid cell; if the grid is
    completely filled it is placed outside the last axis.
    """
    sns.set_theme()
    positions = np.asarray(positions, dtype=np.float64)
    num_total_steps, num_agents, _ = positions.shape

    colors = _get_clustered_agent_colors(num_agents, agent_to_cluster, hub_agent_ids)
    hub_agent_id_set = set(hub_agent_ids or [])
    draw_order = sorted(range(num_agents), key=lambda a: (a in hub_agent_id_set, a))

    # Smooth full trajectories (consistent smoothing across panels)
    smoothed_trajs = [_smooth_trajectory(positions[:, agent_id, :], smooth_window) for agent_id in range(num_agents)]

    # Global axis limits (same as full-trajectory figure)
    if num_total_steps > 0 and num_agents > 0:
        smoothed_all = np.concatenate(smoothed_trajs, axis=0)
        x_min = float(np.nanmin(smoothed_all[:, 0]))
        x_max = float(np.nanmax(smoothed_all[:, 0]))
        y_min = float(np.nanmin(smoothed_all[:, 1]))
        y_max = float(np.nanmax(smoothed_all[:, 1]))
        x_span = (x_max - x_min) if np.isfinite(x_max - x_min) and (x_max - x_min) > 0 else 1.0
        y_span = (y_max - y_min) if np.isfinite(y_max - y_min) and (y_max - y_min) > 0 else 1.0
        pad = 0.02 * max(x_span, y_span)
        xlim = (x_min - pad, x_max + pad)
        ylim = (y_min - pad, y_max + pad)
    else:
        xlim = (-1.0, 1.0)
        ylim = (-1.0, 1.0)

    # Build step-index segments from sorted boundaries
    steps_arr = np.asarray(steps, dtype=np.int64)
    sorted_bounds = sorted(segment_boundaries)

    segments: List[tuple] = []  # (indices_array, start_step_label, end_step_label)
    prev_boundary = -1
    for boundary in sorted_bounds:
        mask = (steps_arr > prev_boundary) & (steps_arr <= boundary)
        idxs = np.where(mask)[0]
        if len(idxs) > 0:
            segments.append((idxs, int(steps_arr[idxs[0]]), int(steps_arr[idxs[-1]])))
        prev_boundary = boundary

    n_segments = len(segments)
    if n_segments == 0:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlabel(f"{axis_prefix}-1", fontsize=16)
        ax.set_ylabel(f"{axis_prefix}-2", fontsize=16)
        ax.grid(True, alpha=0.3)
        return fig

    n_cols = math.ceil(math.sqrt(n_segments))
    n_rows = math.ceil(n_segments / n_cols)
    n_cells = n_rows * n_cols
    has_empty_cell = n_cells > n_segments

    ax_size = 5
    fig, axes_arr = plt.subplots(n_rows, n_cols, figsize=(n_cols * ax_size, n_rows * ax_size * 0.85), squeeze=False)
    axes_flat: List[plt.Axes] = list(axes_arr.flat)

    for seg_idx, (idxs, start_label, end_label) in enumerate(segments):
        ax = axes_flat[seg_idx]
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        for agent_id in draw_order:
            smoothed = smoothed_trajs[agent_id]
            seg_traj = smoothed[idxs]
            if len(seg_traj) == 0:
                continue
            ax.plot(seg_traj[:, 0], seg_traj[:, 1], color=colors[agent_id], linewidth=2.8, alpha=1.0)
            # Mark the start (circle) and end (diamond) of each segment
            ax.scatter([seg_traj[0, 0]], [seg_traj[0, 1]], color=colors[agent_id], s=20, alpha=0.8, zorder=5, marker="o")
            ax.scatter([seg_traj[-1, 0]], [seg_traj[-1, 1]], color=colors[agent_id], s=30, alpha=0.95, zorder=5, marker="D")
            # Agent ID label at the end of the segment
            end_pos = seg_traj[-1]
            ax.text(
                end_pos[0], end_pos[1], f"{agent_id}",
                color=colors[agent_id], fontsize=14, va="center", ha="center", zorder=6,
                path_effects=[path_effects.withStroke(linewidth=1.5, foreground="black")],
            )

        ax.set_xlabel(f"{axis_prefix}-1", fontsize=14)
        ax.set_ylabel(f"{axis_prefix}-2", fontsize=14)
        caption = f"{start_label}–{end_label} step" if start_label != end_label else f"{start_label} step"
        ax.text(
            0.5,
            -0.17,
            caption,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontweight="bold",
            fontsize=16,
        )
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
        _set_square_axes_box_auto(ax)

    # Build legend handles grouped by cluster (one cluster per column)
    if num_clusters > 1 and agent_to_cluster:
        from collections import defaultdict
        cluster_to_agents: Dict[int, List[int]] = defaultdict(list)
        for agent_id in range(num_agents):
            cid = agent_to_cluster[agent_id] if agent_id < len(agent_to_cluster) else 0
            cluster_to_agents[cid].append(agent_id)
        sorted_clusters = sorted(cluster_to_agents.keys())
        max_per_col = max(len(cluster_to_agents[c]) for c in sorted_clusters)
        legend_handles: List[Line2D] = []
        for cid in sorted_clusters:
            for agent_id in cluster_to_agents[cid]:
                legend_handles.append(Line2D([0], [0], color=colors[agent_id], lw=2.4, label=f"agent {agent_id}"))
            # Pad shorter columns with invisible entries so columns stay aligned
            for _ in range(max_per_col - len(cluster_to_agents[cid])):
                legend_handles.append(Line2D([0], [0], color="none", lw=0, label=""))
        n_legend_cols = len(sorted_clusters)
    else:
        legend_handles = [Line2D([0], [0], color=colors[i], lw=2.4, label=f"agent {i}") for i in range(num_agents)]
        n_legend_cols = 1

    if has_empty_cell:
        legend_ax = axes_flat[n_segments]
        legend_ax.axis("off")
        legend_ax.legend(handles=legend_handles, loc="center", frameon=True, fontsize=15, ncols=n_legend_cols, handlelength=1.9, handleheight=1.1)
        for i in range(n_segments + 1, len(axes_flat)):
            axes_flat[i].axis("off")
    else:
        if legend_handles:
            axes_flat[n_segments - 1].legend(
                handles=legend_handles,
                bbox_to_anchor=(1.02, 1.0),
                loc="upper left",
                frameon=True,
                fontsize=15,
                ncols=n_legend_cols,
                handlelength=1.9,
                handleheight=1.1,
            )

    plt.tight_layout(pad=0.9, h_pad=2.0, w_pad=0.4)
    return fig


def _plot_fgw_mds_sequences(
    positions: np.ndarray,
    steps: List[int],
    title_prefix: str,
    smooth_window: int,
    scatter_steps: int,
    contrib: Optional[np.ndarray],
    agent_to_cluster: List[int],
    num_clusters: int,
) -> matplotlib.figure.Figure:
    sns.set_theme()
    positions = np.asarray(positions, dtype=np.float64)
    num_steps, num_agents, _ = positions.shape

    tab20 = plt.cm.get_cmap("tab20")
    colors = []
    for agent_id in range(num_agents):
        cluster_id = agent_to_cluster[agent_id] if agent_to_cluster and agent_id < len(agent_to_cluster) else 0
        pair_base = (cluster_id % 10) * 2
        light_color = tuple(tab20(pair_base + 1)[:3])
        dark_color = tuple(tab20(pair_base)[:3])
        colors.append(dark_color if agent_id < 1 else light_color)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4.8))

    if num_steps == 0 or num_agents == 0:
        ax.set_ylabel("MDS-1")
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    step_vals = np.asarray(steps, dtype=np.float64)
    raw_last_y = np.zeros(num_agents, dtype=np.float64)

    for agent_id in range(num_agents):
        raw = positions[:, agent_id, 0]
        ax.plot(step_vals, raw, color=colors[agent_id], linewidth=3.0)
        raw_last_y[agent_id] = raw[-1]

    # Right-edge agent labels with collision avoidance
    y_span = float(np.nanmax(raw_last_y) - np.nanmin(raw_last_y))
    y_span = y_span if np.isfinite(y_span) and y_span > 0 else 1.0
    min_sep = 0.03 * y_span
    label_order = np.argsort(raw_last_y)
    label_y = raw_last_y.copy()
    for k in range(1, len(label_order)):
        prev = label_order[k - 1]
        cur = label_order[k]
        if label_y[cur] - label_y[prev] < min_sep:
            label_y[cur] = label_y[prev] + min_sep

    x_span = float(step_vals[-1] - step_vals[0]) if len(step_vals) > 1 else 1.0
    x_span = x_span if x_span > 0 else 1.0
    x_end = step_vals[-1] + 0.02 * x_span
    for agent_id in range(num_agents):
        ax.text(x_end, label_y[agent_id], f"{agent_id}", color=colors[agent_id], fontsize=11,
                va="center", ha="left",
                path_effects=[path_effects.withStroke(linewidth=1.2, foreground="black")])

    contrib_arr = np.asarray(contrib, dtype=np.float64) if contrib is not None else None
    if contrib_arr is not None and contrib_arr.size >= 1 and np.isfinite(contrib_arr[0]) and contrib_arr[0] > 0:
        ylabel = f"MDS-1 ({contrib_arr[0] * 100:.1f}%)"
    else:
        ylabel = "MDS-1"
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel("Step", fontsize=14)
    ax.set_title(f"{title_prefix} MDS-1 over steps", fontsize=13)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig