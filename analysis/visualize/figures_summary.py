from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from .plotting import (
    DataVisualizer,
    _detect_hub_agents,
    _fade_nonhub_colors,
    _get_cluster_color,
    _get_jet_agent_colors,
)


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


def render_observations_clustered_panel(
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
        visualizer._plot_observations_clustered_on_ax(
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
    num_clusters = int(legend_data.get("num_clusters", max(agent_to_cluster) + 1 if agent_to_cluster else 1))
    agent_colors = [_get_cluster_color(agent_to_cluster[i], max(1, num_clusters)) for i in range(num_agents)]
    visualizer._draw_cluster_legend(
        legend_ax,
        agent_to_cluster=agent_to_cluster,
        agent_colors=agent_colors,
        title=None,
        font_scale=1.4,
        row_gap_scale=1.6,
    )

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.22, wspace=0.35, bottom=0.0)
    return fig


def render_observations_clustered_hubhil_panel(
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
        visualizer._plot_observations_clustered_hubhil_on_ax(
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
    num_clusters = int(legend_data.get("num_clusters", max(agent_to_cluster) + 1 if agent_to_cluster else 1))
    hub_ids = _detect_hub_agents(legend_data.get("adjacency_matrix"), agent_to_cluster)
    base_colors = [_get_cluster_color(agent_to_cluster[i], max(1, num_clusters)) for i in range(num_agents)]
    agent_colors = _fade_nonhub_colors(base_colors, hub_ids, 0.3) if hub_ids else base_colors
    visualizer._draw_cluster_legend(
        legend_ax,
        agent_to_cluster=agent_to_cluster,
        agent_colors=agent_colors,
        title=None,
        font_scale=1.4,
        row_gap_scale=1.6,
    )

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.22, wspace=0.35, bottom=0.0)
    return fig


def render_observations_agents_clustered_hubhil_panel(
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
        visualizer._plot_observations_agents_clustered_hubhil_on_ax(
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
    hub_ids = _detect_hub_agents(legend_data.get("adjacency_matrix"), agent_to_cluster)
    base_colors = _get_jet_agent_colors(num_agents)
    agent_colors = _fade_nonhub_colors(base_colors, hub_ids, 0.3) if hub_ids else base_colors
    visualizer._draw_cluster_legend(
        legend_ax,
        agent_to_cluster=agent_to_cluster,
        agent_colors=agent_colors,
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
    # fig_h = 10.0  # old setting (networks are drawn to fill their axes, so this height sets the network's reference size)
    fig_h = 8.0  # shrink the image vertically; network height is kept identical to fig_h=10.0 via gs.update below
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
        "Social\nrepresentations\nacceptance",
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
            -0.02,  # old: -0.08 (raised to tighten the gap between the creations row and the step label, shrinking the image height)
            label_formatter(panel_key, data_pair),
            transform=bottom_ax.transAxes,
            ha="center",
            va="top",
            fontweight="bold",
            fontsize=24,
        )

    # Leave the horizontal layout (column widths, colorbar position, etc.) to tight_layout.
    fig.tight_layout()
    # ===== Vertical tightening of the image =====
    # To revert: set fig_h back to 10.0 above and comment out this gs.update block.
    #
    # Because of auto-aspect, lowering fig_h would squash the networks themselves vertically.
    # So we keep each network axis height (= drawn size) fixed at ≈3.565in, the same as the
    # original setting (fig_h=10), and shrink only the inter-row gap and the top/bottom margins.
    # Note: since the GridSpec is created with an hspace, fig.subplots_adjust(hspace=...) has no
    #   effect (the GridSpec-local hspace takes precedence). Update the GridSpec itself.
    target_axes_h_in = 3.5648  # per-panel height measured at fig_h=10 (= original network size)
    # Inter-row gap (originally ≈0.57in). Each axis has dead autoscale margin (≈0.18in top/bottom),
    # so a negative value overlaps the axes into that dead margin (the networks do not collide).
    gap_in = -0.20  # old: -0.15
    bottom_in = 0.40           # bottom margin (for the per-column step labels)
    hs = gap_in / target_axes_h_in
    span_in = (2.0 + hs) * target_axes_h_in  # vertical span from bottom to top (2 rows + gap)
    gs.update(
        top=(bottom_in + span_in) / fig_h,
        bottom=bottom_in / fig_h,
        hspace=hs,
    )
    # ===== Right-edge alignment of the row labels =====
    # To revert: comment out this fig.canvas.draw()...set_x block.
    #
    # The row labels are rotation=90 texts; after rotation their "right edge" (= the bottom side of
    # the text box, the "acceptance" line) is not aligned between rows (top=3 lines / bottom=2 lines,
    # center-aligned, so the right edges differ). Using the bottom "Creations acceptance" as the
    # reference, shift the top label horizontally to match its right edge. To avoid magic numbers,
    # measure both labels' right edges after drawing and move only by the difference.
    fig.canvas.draw()
    _r = fig.canvas.get_renderer()
    _top_t = top_label_ax.texts[0]
    _top_x1 = _top_t.get_window_extent(_r).transformed(top_label_ax.transAxes.inverted()).x1
    _bot_x1 = bottom_label_ax.texts[0].get_window_extent(_r).transformed(bottom_label_ax.transAxes.inverted()).x1
    _top_t.set_x(_top_t.get_position()[0] - (_top_x1 - _bot_x1))
    # ===== End of right-edge alignment =====
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
