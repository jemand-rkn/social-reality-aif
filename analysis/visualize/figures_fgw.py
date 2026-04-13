from __future__ import annotations

from typing import Any, Dict, List

import matplotlib.figure
import numpy as np

from .plotting import (
    _auto_detect_hub_agents,
    _plot_fgw_mds_trajectories,
    _plot_fgw_mds_trajectories_clustered,
    _plot_fgw_mds_trajectories_clustered_segmented,
)


def _as_array(payload: Dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(payload[key], dtype=np.float64)


def _as_steps(payload: Dict[str, Any]) -> List[int]:
    return [int(v) for v in np.asarray(payload["steps"], dtype=np.int64).tolist()]


def _format_contrib_note(mds_contrib: np.ndarray, axis_contrib: np.ndarray) -> str:
    def _fmt(v: float) -> str:
        if np.isfinite(v):
            return f"{100.0 * v:.1f}%"
        return "n/a"

    return (
        f"MDS mean contrib: ({_fmt(float(mds_contrib[0]))}, {_fmt(float(mds_contrib[1]))})\\n"
        f"PC axis contrib: ({_fmt(float(axis_contrib[0]))}, {_fmt(float(axis_contrib[1]))})"
    )


def _require_pc_key(payload: Dict[str, Any], key: str) -> np.ndarray:
    if key not in payload:
        raise ValueError(
            f"Missing '{key}' in global metric payload. "
            "Recompute global metrics with the current pipeline so PC outputs are stored in layer 3."
        )
    return _as_array(payload, key)


def render_mds_trajectory(payload: Dict[str, Any], cluster_data: Dict[str, Any] | None = None) -> matplotlib.figure.Figure:
    contrib = np.nanmean(_as_array(payload, "contrib"), axis=0)
    agent_to_cluster = None
    if cluster_data:
        agent_to_cluster = list(cluster_data.get("agent_to_cluster", []))
    return _plot_fgw_mds_trajectories(
        positions=_as_array(payload, "positions"),
        steps=_as_steps(payload),
        # scatter_steps=10,
        scatter_steps=500,
        smooth_window=15,
        contrib=contrib,
        agent_to_cluster=agent_to_cluster,
    )


def render_mds_trajectory_clustered(payload: Dict[str, Any], cluster_data: Dict[str, Any]) -> matplotlib.figure.Figure:
    contrib = np.nanmean(_as_array(payload, "contrib"), axis=0)
    agent_to_cluster = list(cluster_data.get("agent_to_cluster", []))
    num_clusters = int(cluster_data.get("num_clusters", 0))
    hub_agent_ids = _auto_detect_hub_agents(agent_to_cluster, num_clusters, [cluster_data])
    return _plot_fgw_mds_trajectories_clustered(
        positions=_as_array(payload, "positions"),
        steps=_as_steps(payload),
        # scatter_steps=10,
        scatter_steps=500,
        smooth_window=15,
        contrib=contrib,
        agent_to_cluster=agent_to_cluster,
        num_clusters=num_clusters,
        hub_agent_ids=hub_agent_ids,
    )


def render_mds_pc_trajectory(payload: Dict[str, Any], cluster_data: Dict[str, Any] | None = None) -> matplotlib.figure.Figure:
    positions_pc = _require_pc_key(payload, "positions_pc")
    pc_contrib = _require_pc_key(payload, "contrib_pc")
    # mds_contrib = np.nanmean(_as_array(payload, "contrib"), axis=0)
    agent_to_cluster = None
    if cluster_data:
        agent_to_cluster = list(cluster_data.get("agent_to_cluster", []))
    display_note = None
    return _plot_fgw_mds_trajectories(
        positions=positions_pc,
        steps=_as_steps(payload),
        # scatter_steps=10,
        scatter_steps=500,
        smooth_window=15,
        contrib=pc_contrib,
        agent_to_cluster=agent_to_cluster,
        # axis_prefix="PC",
        axis_prefix="MDS",
        contrib_note=display_note,
        # contrib_note=_format_contrib_note(mds_contrib, pc_contrib),
    )


def render_mds_pc_trajectory_clustered_segmented(
    payload: Dict[str, Any],
    cluster_data: Dict[str, Any],
    segment_boundaries: List[int],
) -> matplotlib.figure.Figure:
    positions_pc = _require_pc_key(payload, "positions_pc")
    agent_to_cluster = list(cluster_data.get("agent_to_cluster", []))
    num_clusters = int(cluster_data.get("num_clusters", 0))
    hub_agent_ids = _auto_detect_hub_agents(agent_to_cluster, num_clusters, [cluster_data])
    return _plot_fgw_mds_trajectories_clustered_segmented(
        positions=positions_pc,
        steps=_as_steps(payload),
        segment_boundaries=segment_boundaries,
        smooth_window=15,
        agent_to_cluster=agent_to_cluster,
        num_clusters=num_clusters,
        hub_agent_ids=hub_agent_ids,
        axis_prefix="MDS",
    )


def render_mds_pc_trajectory_clustered(payload: Dict[str, Any], cluster_data: Dict[str, Any]) -> matplotlib.figure.Figure:
    positions_pc = _require_pc_key(payload, "positions_pc")
    pc_contrib = _require_pc_key(payload, "contrib_pc")
    # mds_contrib = np.nanmean(_as_array(payload, "contrib"), axis=0)
    agent_to_cluster = list(cluster_data.get("agent_to_cluster", []))
    num_clusters = int(cluster_data.get("num_clusters", 0))
    hub_agent_ids = _auto_detect_hub_agents(agent_to_cluster, num_clusters, [cluster_data])
    display_note = None
    return _plot_fgw_mds_trajectories_clustered(
        positions=positions_pc,
        steps=_as_steps(payload),
        # scatter_steps=10,
        scatter_steps=500,
        smooth_window=15,
        contrib=pc_contrib,
        agent_to_cluster=agent_to_cluster,
        num_clusters=num_clusters,
        hub_agent_ids=hub_agent_ids,
        # axis_prefix="PC",
        axis_prefix="MDS",
        contrib_note=display_note,
        # contrib_note=_format_contrib_note(mds_contrib, pc_contrib),
    )
