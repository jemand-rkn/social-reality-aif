from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from .plotting import DataVisualizer, _get_jet_agent_colors


def _infer_obs_shape(data: Dict[str, Any]) -> Tuple[int, ...]:
    observations = data.get("observations", [])
    if not observations:
        return ()
    first = np.asarray(observations[0])
    if first.ndim >= 2:
        return tuple(first.shape[1:])
    if first.ndim == 1:
        return (1,)
    return ()


def render_observations_agents_clustered_panel(
    panel_items: List[Tuple[int, Dict[str, Any]]],
) -> matplotlib.figure.Figure:
    if not panel_items:
        raise ValueError("panel_items is empty")

    visualizer = DataVisualizer(_infer_obs_shape(panel_items[0][1]))

    num_panels = len(panel_items)
    fig_w = max(10.0, 4.5 * num_panels)
    fig_h = 6.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(2, num_panels, figure=fig, height_ratios=[1.0, 0.2])

    for idx, (step, data) in enumerate(panel_items):
        ax = fig.add_subplot(gs[0, idx])
        visualizer._plot_observations_agents_clustered_on_ax(
            data,
            step,
            ax,
            font_scale=1.4,
            step_label_y=-0.3,
        )

    legend_ax = fig.add_subplot(gs[1, :])
    legend_data = panel_items[0][1]
    num_agents = len(legend_data.get("observations", []))
    agent_to_cluster = list(legend_data.get("agent_to_cluster", []))
    if not agent_to_cluster:
        agent_to_cluster = [0] * num_agents
    visualizer._draw_cluster_legend(
        legend_ax,
        agent_to_cluster=agent_to_cluster,
        agent_colors=_get_jet_agent_colors(num_agents),
        title=None,
        font_scale=1.4,
        row_gap_scale=1.6,
    )

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.22, wspace=0.35, bottom=0.0)
    return fig


def render_wasserstein_similarity_snapshot(
    panel_items: List[Tuple[int, Dict[str, Any]]],
) -> matplotlib.figure.Figure:
    return DataVisualizer(()).plot_fgw_similarity_snapshot(
        panel_items,
        alpha=0.0,
        font_scale=1.6,
        step_label_y=-0.23,
    )


def render_gromov_wasserstein_similarity_snapshot(
    panel_items: List[Tuple[int, Dict[str, Any]]],
) -> matplotlib.figure.Figure:
    return DataVisualizer(()).plot_fgw_similarity_snapshot(panel_items, alpha=1.0)


def render_wasserstein_similarity_average_snapshot(
    panel_items: List[Tuple[str, Dict[str, Any]]],
) -> matplotlib.figure.Figure:
    return DataVisualizer(()).plot_fgw_similarity_snapshot(
        panel_items,
        alpha=0.0,
        font_scale=1.6,
        step_label_y=-0.23,
        step_label_formatter=lambda label: str(label),
        vmin_override=0.0,
        vmax_override=12.0,
    )


def render_gromov_wasserstein_similarity_average_snapshot(
    panel_items: List[Tuple[str, Dict[str, Any]]],
) -> matplotlib.figure.Figure:
    return DataVisualizer(()).plot_fgw_similarity_snapshot(
        panel_items,
        alpha=1.0,
        step_label_formatter=lambda label: str(label),
    )


def _render_both_acceptance_snapshot_common(
    panel_items: List[Tuple[Any, Dict[str, Any]]],
    label_formatter: Callable[[Any, Dict[str, Any]], str],
) -> matplotlib.figure.Figure:
    if not panel_items:
        raise ValueError("panel_items is empty")

    visualizer = DataVisualizer(())
    num_cols = len(panel_items)
    fig_w = max(12.0, 4.6 * num_cols)
    fig_h = 10.0
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Layout: [left label col] + [step columns...]
    gs = GridSpec(
        2,
        num_cols + 1,
        figure=fig,
        width_ratios=[0.16] + [1.0] * num_cols,
        height_ratios=[1.0, 1.0],
        wspace=0.08,
        hspace=0.16,
    )

    top_label_ax = fig.add_subplot(gs[0, 0])
    top_label_ax.axis("off")
    top_label_ax.text(
        0.55,
        0.5,
        "Social representations\nacceptance",
        rotation=90,
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=24,
    )

    bottom_label_ax = fig.add_subplot(gs[1, 0])
    bottom_label_ax.axis("off")
    bottom_label_ax.text(
        0.55,
        0.5,
        "Creations\nacceptance",
        rotation=90,
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=24,
    )

    for col_idx, (panel_key, data_pair) in enumerate(panel_items, start=1):
        top_ax = fig.add_subplot(gs[0, col_idx])
        bottom_ax = fig.add_subplot(gs[1, col_idx])
        show_bottom_colorbar = col_idx == num_cols

        visualizer._plot_acceptance_flow_network_on_ax(
            fig,
            top_ax,
            data_pair["mhng"],
            show_colorbar=False,
            node_size=450,
            position_scale=3.0,
            node_label_font_size=20,
            edge_width_scale=0.9,
        )
        visualizer._plot_acceptance_flow_network_on_ax(
            fig,
            bottom_ax,
            data_pair["memorize"],
            show_colorbar=show_bottom_colorbar,
            node_size=450,
            position_scale=3.0,
            node_label_font_size=20,
            edge_width_scale=0.9,
        )

        # Step label per column under bottom row.
        bottom_ax.text(
            0.5,
            -0.08,
            label_formatter(panel_key, data_pair),
            transform=bottom_ax.transAxes,
            ha="center",
            va="top",
            fontweight="bold",
            fontsize=24,
        )

    fig.tight_layout()
    return fig


def render_both_acceptance_network_snapshot(
    panel_items: List[Tuple[int, Dict[str, Any]]],
) -> matplotlib.figure.Figure:
    return _render_both_acceptance_snapshot_common(
        panel_items,
        label_formatter=lambda step, _data_pair: f"step={step}",
    )


def render_both_acceptance_average_network_snapshot(
    panel_items: List[Tuple[str, Dict[str, Any]]],
) -> matplotlib.figure.Figure:
    return _render_both_acceptance_snapshot_common(
        panel_items,
        label_formatter=lambda window, _data_pair: window,
    )
