from __future__ import annotations

from typing import Any, Dict, Tuple

import matplotlib.figure
import numpy as np

from .plotting import DataVisualizer


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


def _visualizer_for(data: Dict[str, Any]) -> DataVisualizer:
    return DataVisualizer(_infer_obs_shape(data))


def render_observations_and_creations_clustered(data: Dict[str, Any], step: int) -> matplotlib.figure.Figure:
    return _visualizer_for(data).plot_observations_and_creations_clustered(data, step)


def render_observations_and_creations_clustered_legend(data: Dict[str, Any]) -> matplotlib.figure.Figure:
    return _visualizer_for(data).plot_clustered_agent_legend(data, color_mode="cluster")


def render_observations_and_creations_agents_clustered(data: Dict[str, Any], step: int) -> matplotlib.figure.Figure:
    return _visualizer_for(data).plot_observations_and_creations_agents_clustered(data, step)


def render_observations_and_creations_agents_clustered_legend(data: Dict[str, Any]) -> matplotlib.figure.Figure:
    return _visualizer_for(data).plot_clustered_agent_legend(data, color_mode="jet")


def render_observations_agents_clustered(data: Dict[str, Any], step: int) -> matplotlib.figure.Figure:
    return _visualizer_for(data).plot_observations_agents_clustered(data, step)


def render_wasserstein_similarity(data: Dict[str, Any], step: int) -> matplotlib.figure.Figure:
    return _visualizer_for(data).plot_fgw_similarity_matrix(data, alpha=0.0, step=step)


def render_gromov_wasserstein_similarity(data: Dict[str, Any], step: int) -> matplotlib.figure.Figure:
    return _visualizer_for(data).plot_fgw_similarity_matrix(data, alpha=1.0, step=step)


def render_latent_scatter_from_reference(data: Dict[str, Any], step: int) -> matplotlib.figure.Figure:
    return _visualizer_for(data).plot_latents_from_reference(data, step)


def render_social_network(data: Dict[str, Any], step: int) -> matplotlib.figure.Figure:
    return _visualizer_for(data).plot_social_network(data, step)


def render_mhng_acceptance_matrix(data: Dict[str, Any], step: int) -> matplotlib.figure.Figure:
    return _visualizer_for(data)._plot_acceptance_matrix(data, step, "MHNG")


def render_mhng_acceptance_network(data: Dict[str, Any], step: int) -> matplotlib.figure.Figure:
    return _visualizer_for(data).plot_acceptance_flow_network(data, step, "MHNG")


def render_memorize_acceptance_matrix(data: Dict[str, Any], step: int) -> matplotlib.figure.Figure:
    return _visualizer_for(data)._plot_acceptance_matrix(data, step, "Memorize")


def render_memorize_acceptance_network(data: Dict[str, Any], step: int) -> matplotlib.figure.Figure:
    return _visualizer_for(data).plot_acceptance_flow_network(data, step, "Memorize")


def render_both_acceptance_network(data: Dict[str, Any], step: int) -> matplotlib.figure.Figure:
    return _visualizer_for(data["mhng"]).plot_both_acceptance_networks(data["mhng"], data["memorize"], step)


def compose_fgw_pair(
    wasserstein_data: Dict[str, Any],
    gromov_wasserstein_data: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "distance_matrix_00": np.asarray(wasserstein_data["distance_matrix"]),
        "distance_matrix_10": np.asarray(gromov_wasserstein_data["distance_matrix"]),
    }


def compose_acceptance_pair(
    mhng_data: Dict[str, Any],
    memorize_data: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "mhng": mhng_data,
        "memorize": memorize_data,
    }


def with_cluster_meta(data: Dict[str, Any], cluster_data: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(data)
    if "agent_to_cluster" in cluster_data:
        result["agent_to_cluster"] = list(cluster_data["agent_to_cluster"])
    if "num_clusters" in cluster_data:
        result["num_clusters"] = int(cluster_data["num_clusters"])
    return result


