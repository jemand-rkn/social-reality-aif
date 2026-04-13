from __future__ import annotations

from typing import Any, Dict

import numpy as np
import rsatoolbox as rsa
from sklearn.metrics import pairwise_distances


def compute_rsa_within_clusters_from_reference(
    reference_obs: np.ndarray,
    reference_latent_mu: np.ndarray,
    agent_to_cluster: list[int],
    num_clusters: int,
) -> Dict[str, Any]:
    """Compute clustered RSA from extracted reference observations/latents.

    This keeps the cluster-pool idea of the legacy implementation, but uses
    extracted reference materials instead of rebuilding a society.
    """
    if reference_obs.ndim != 3:
        raise ValueError("reference_obs must be 3D [A, S, D].")
    if reference_latent_mu.ndim != 3:
        raise ValueError("reference_latent_mu must be 3D [A, R, L].")

    num_agents, samples_per_agent, obs_dim = reference_obs.shape
    num_refs = num_agents * samples_per_agent
    if reference_latent_mu.shape[0] != num_agents:
        raise ValueError(
            f"reference_obs and reference_latent_mu agent mismatch: obs={num_agents}, latent={reference_latent_mu.shape[0]}"
        )
    if reference_latent_mu.shape[1] != num_refs:
        raise ValueError(
            f"reference_obs and reference_latent_mu sample mismatch: obs={num_refs}, latent={reference_latent_mu.shape[1]}"
        )

    if len(agent_to_cluster) != num_agents:
        if len(agent_to_cluster) < num_agents:
            agent_to_cluster = list(agent_to_cluster) + [0] * (num_agents - len(agent_to_cluster))
        else:
            agent_to_cluster = list(agent_to_cluster[:num_agents])

    if num_clusters <= 0:
        num_clusters = max(agent_to_cluster) + 1 if agent_to_cluster else 1

    clusters = [[] for _ in range(num_clusters)]
    for agent_id, cluster_id in enumerate(agent_to_cluster):
        if 0 <= cluster_id < num_clusters:
            clusters[cluster_id].append(agent_id)

    flat_reference_obs = reference_obs.reshape(num_refs, obs_dim)
    source_cluster_ids = np.array(
        [
            agent_to_cluster[agent_id]
            for agent_id in range(num_agents)
            for _ in range(samples_per_agent)
        ],
        dtype=np.int64,
    )

    agent_rsa = np.full(num_agents, np.nan, dtype=float)
    cluster_rsa = np.full(num_clusters, np.nan, dtype=float)
    cluster_agent_rsa = np.full((num_clusters, num_agents), np.nan, dtype=float)
    cluster_cluster_rsa = np.full((num_clusters, num_clusters), np.nan, dtype=float)

    for cluster_idx in range(num_clusters):
        sample_idx = np.where(source_cluster_ids == cluster_idx)[0]
        if sample_idx.size < 2:
            continue

        pooled_obs = flat_reference_obs[sample_idx]
        obs_distances = pairwise_distances(pooled_obs, metric="euclidean")
        obs_rdm = rsa.rdm.RDMs(obs_distances[np.newaxis, :, :])

        for agent_id in range(num_agents):
            latents_np = reference_latent_mu[agent_id, sample_idx, :]
            latent_distances = pairwise_distances(latents_np, metric="euclidean")
            latent_rdm = rsa.rdm.RDMs(latent_distances[np.newaxis, :, :])
            rsa_value = rsa.rdm.compare(latent_rdm, obs_rdm, method="rho-a")[0, 0]
            cluster_agent_rsa[cluster_idx, agent_id] = float(rsa_value)

    for agent_id in range(num_agents):
        cluster_idx = agent_to_cluster[agent_id] if agent_id < len(agent_to_cluster) else -1
        if 0 <= cluster_idx < num_clusters:
            agent_rsa[agent_id] = cluster_agent_rsa[cluster_idx, agent_id]

    for cluster_idx, members in enumerate(clusters):
        if not members:
            continue
        values = cluster_agent_rsa[cluster_idx, members]
        if np.all(np.isnan(values)):
            continue
        cluster_rsa[cluster_idx] = float(np.nanmean(values))

    for cluster_i, members in enumerate(clusters):
        if not members:
            continue
        for cluster_j in range(num_clusters):
            values = cluster_agent_rsa[cluster_j, members]
            if np.all(np.isnan(values)):
                continue
            cluster_cluster_rsa[cluster_i, cluster_j] = float(np.nanmean(values))

    return {
        "clusters": [sorted(cluster) for cluster in clusters],
        "agent_rsa": agent_rsa,
        "cluster_rsa": cluster_rsa,
        "cluster_agent_rsa": cluster_agent_rsa,
        "cluster_cluster_rsa": cluster_cluster_rsa,
        "cluster_agents": {cluster_idx: sorted(cluster) for cluster_idx, cluster in enumerate(clusters)},
    }


def compute_neighbor_latent_rsa_from_reference(
    reference_latent_mu: np.ndarray,
    adjacency_matrix: np.ndarray,
) -> Dict[str, Any]:
    """Compute latent-latent RSA between connected neighbors and average per target agent."""
    if reference_latent_mu.ndim != 3:
        raise ValueError("reference_latent_mu must be 3D [A, R, L].")

    adjacency = np.asarray(adjacency_matrix)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency_matrix must be 2D square [A, A].")

    num_agents = int(reference_latent_mu.shape[0])
    if adjacency.shape[0] != num_agents:
        raise ValueError(
            f"reference_latent_mu and adjacency_matrix agent mismatch: latent={num_agents}, adjacency={adjacency.shape[0]}"
        )

    latent_rdms = []
    for agent_id in range(num_agents):
        latents_np = np.asarray(reference_latent_mu[agent_id], dtype=float)
        latent_distances = pairwise_distances(latents_np, metric="euclidean")
        latent_rdms.append(rsa.rdm.RDMs(latent_distances[np.newaxis, :, :]))

    pair_rsa = np.full((num_agents, num_agents), np.nan, dtype=float)
    for agent_i in range(num_agents):
        pair_rsa[agent_i, agent_i] = 1.0
        for agent_j in range(agent_i + 1, num_agents):
            rsa_value = rsa.rdm.compare(latent_rdms[agent_i], latent_rdms[agent_j], method="rho-a")[0, 0]
            pair_rsa[agent_i, agent_j] = float(rsa_value)
            pair_rsa[agent_j, agent_i] = float(rsa_value)

    neighbor_mask = adjacency > 0
    np.fill_diagonal(neighbor_mask, False)
    neighbor_counts = np.sum(neighbor_mask, axis=1).astype(int)
    neighbor_latent_rsa = np.full(num_agents, np.nan, dtype=float)
    for agent_id in range(num_agents):
        neighbors = np.flatnonzero(neighbor_mask[agent_id])
        if neighbors.size == 0:
            continue
        values = pair_rsa[agent_id, neighbors]
        if np.isfinite(values).any():
            neighbor_latent_rsa[agent_id] = float(np.nanmean(values))

    return {
        "neighbor_latent_rsa": neighbor_latent_rsa,
        "latent_pair_rsa": pair_rsa,
        "neighbor_counts": neighbor_counts,
    }
