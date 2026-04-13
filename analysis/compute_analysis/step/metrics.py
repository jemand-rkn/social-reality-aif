from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy.stats import pearsonr

from analysis.core.fgw import compute_fgw_distance_vectorized
from analysis.core.rsa import compute_neighbor_latent_rsa_from_reference, compute_rsa_within_clusters_from_reference
import torch


def _to_numpy(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _num_reference_samples(reference_obs: Optional[np.ndarray]) -> int:
    if reference_obs is None:
        return 0
    if reference_obs.ndim == 3:
        return int(reference_obs.shape[0] * reference_obs.shape[1])
    return int(reference_obs.shape[0])


def _cache_get(runtime: Optional[Dict[str, Any]], key: str):
    if runtime is None:
        return None
    cache = runtime.setdefault("cache", {})
    return cache.get(key)


def _cache_set(runtime: Optional[Dict[str, Any]], key: str, value):
    if runtime is None:
        return value
    cache = runtime.setdefault("cache", {})
    cache[key] = value
    return value


def _get_adjacency_with_self(step_payload: Dict[str, Any]) -> Optional[np.ndarray]:
    adj_with_self = _to_numpy(step_payload.get("adjacency_matrix_with_self"))
    if adj_with_self is not None:
        return adj_with_self
    adj = _to_numpy(step_payload.get("adjacency_matrix"))
    if adj is None:
        return None
    adj_with_self = np.array(adj, copy=True)
    np.fill_diagonal(adj_with_self, 1)
    return adj_with_self


def _compute_distance_tensors(step_payload: Dict[str, Any]):
    mu = _to_numpy(step_payload.get("reference_latent_mu"))
    std = _to_numpy(step_payload.get("reference_latent_std"))
    if mu is None or std is None:
        return None
    if mu.size == 0 or std.size == 0:
        return None

    mu_diff = mu[:, :, None, :] - mu[:, None, :, :]
    std_diff = std[:, :, None, :] - std[:, None, :, :]
    rdm_all = np.sqrt(np.sum(mu_diff**2 + std_diff**2, axis=-1))

    mu_diff_agents = mu[:, None, :, None, :] - mu[None, :, None, :, :]
    std_diff_agents = std[:, None, :, None, :] - std[None, :, None, :, :]
    m_all = np.sqrt(np.sum(mu_diff_agents**2 + std_diff_agents**2, axis=-1))
    return rdm_all, m_all


def _compute_network_vs_corr(adjacency_matrix: np.ndarray, distance_matrix: np.ndarray) -> float:
    mask = np.triu(np.ones_like(adjacency_matrix), k=1).astype(bool)
    adj_upper = adjacency_matrix[mask]
    dist_upper = distance_matrix[mask]
    try:
        corr, _ = pearsonr(adj_upper, dist_upper)
    except Exception:
        corr = float("nan")
    return float(corr)


def _compute_wasserstein_similarity(step_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    tensors = _compute_distance_tensors(step_payload)
    adjacency_matrix = _get_adjacency_with_self(step_payload)
    if tensors is None or adjacency_matrix is None:
        return None
    rdm_all, m_all = tensors
    distance_matrix = compute_fgw_distance_vectorized(rdm_all, m_all, alpha=0.0)
    corr = _compute_network_vs_corr(adjacency_matrix, distance_matrix)
    reference_obs = _to_numpy(step_payload.get("reference_obs"))
    num_refs = _num_reference_samples(reference_obs)
    return {
        "adjacency_matrix": adjacency_matrix,
        "distance_matrix": distance_matrix,
        "network_vs_corr": corr,
        "num_refs": num_refs,
    }


def _compute_gromov_wasserstein_similarity(step_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    tensors = _compute_distance_tensors(step_payload)
    adjacency_matrix = _get_adjacency_with_self(step_payload)
    if tensors is None or adjacency_matrix is None:
        return None
    rdm_all, m_all = tensors
    distance_matrix = compute_fgw_distance_vectorized(rdm_all, m_all, alpha=1.0)
    corr = _compute_network_vs_corr(adjacency_matrix, distance_matrix)
    reference_obs = _to_numpy(step_payload.get("reference_obs"))
    num_refs = _num_reference_samples(reference_obs)
    return {
        "adjacency_matrix": adjacency_matrix,
        "distance_matrix": distance_matrix,
        "network_vs_corr": corr,
        "num_refs": num_refs,
    }


def _compute_latent_scatter(step_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    mu = _to_numpy(step_payload.get("reference_latent_mu"))
    reference_obs = _to_numpy(step_payload.get("reference_obs"))
    if mu is None or reference_obs is None:
        return None
    if mu.size == 0 or reference_obs.size == 0:
        return None

    latents_list = [mu[i] for i in range(mu.shape[0])]
    latent_dim = latents_list[0].shape[1]
    if latent_dim == 2:
        latents_2d_list = [lat for lat in latents_list]
        components = None
    else:
        all_latents = np.concatenate(latents_list, axis=0)
        latents_center = np.mean(all_latents, axis=0, keepdims=True)
        x = all_latents - latents_center
        _, _, vh = np.linalg.svd(x, full_matrices=False)
        components = vh[:2, :]
        latents_2d_list = []
        for lat in latents_list:
            lat_2d = (lat - latents_center) @ components.T
            latents_2d_list.append(lat_2d)
    return {
        "reference_obs": reference_obs,
        "latents_2d": latents_2d_list,
        "pca_components": components,
    }


def _compute_rsa_from_reference_latent(step_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    reference_obs = _to_numpy(step_payload.get("reference_obs"))
    reference_latent_mu = _to_numpy(step_payload.get("reference_latent_mu"))
    adjacency_matrix = _to_numpy(step_payload.get("adjacency_matrix"))
    agent_to_cluster = list(step_payload.get("agent_to_cluster", []))
    num_clusters = int(step_payload.get("num_clusters", 0))

    if reference_obs is None or reference_latent_mu is None:
        return None
    if reference_obs.ndim != 3 or reference_latent_mu.ndim != 3:
        return None

    rsa_items = compute_rsa_within_clusters_from_reference(
        reference_obs=reference_obs,
        reference_latent_mu=reference_latent_mu,
        agent_to_cluster=agent_to_cluster,
        num_clusters=num_clusters,
    )
    if adjacency_matrix is None:
        return rsa_items

    neighbor_items = compute_neighbor_latent_rsa_from_reference(
        reference_latent_mu=reference_latent_mu,
        adjacency_matrix=adjacency_matrix,
    )
    return {
        **rsa_items,
        **neighbor_items,
    }


def _compute_network_vs_similarity_from_latent(step_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    mu = _to_numpy(step_payload.get("reference_latent_mu"))
    std = _to_numpy(step_payload.get("reference_latent_std"))
    adjacency_matrix = _get_adjacency_with_self(step_payload)
    if mu is None or std is None or adjacency_matrix is None:
        return None
    if mu.size == 0 or std.size == 0:
        return None

    eps = 1e-8
    std_safe = np.clip(std, eps, None)
    mu_i = mu[:, None, :, :]
    mu_j = mu[None, :, :, :]
    std_i = std_safe[:, None, :, :]
    std_j = std_safe[None, :, :, :]

    mu_diff_sq = (mu_i - mu_j) ** 2
    std_diff_sq = (std_i - std_j) ** 2
    emd = np.sqrt(mu_diff_sq + std_diff_sq)
    emd = np.sum(emd, axis=-1)
    avg_similarity_matrix = np.mean(emd, axis=-1)

    mask = np.triu(np.ones_like(adjacency_matrix), k=1).astype(bool)
    adj_upper = adjacency_matrix[mask]
    sim_upper = avg_similarity_matrix[mask]
    try:
        corr, p_value = pearsonr(adj_upper, sim_upper)
    except Exception:
        corr = float("nan")
        p_value = float("nan")

    reference_obs = _to_numpy(step_payload.get("reference_obs"))
    num_refs = _num_reference_samples(reference_obs)
    return {
        "adjacency_matrix": adjacency_matrix,
        "avg_similarity_matrix": avg_similarity_matrix,
        "corr": float(corr),
        "p_value": float(p_value),
        "num_refs": num_refs,
    }


def _compute_mhng_edge_acceptance(step_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    mhng_results = step_payload.get("mhng_results")
    num_agents = int(step_payload.get("num_agents", 0))
    if mhng_results is None or num_agents <= 0:
        return None

    try:
        mhng_metadata = mhng_results[2]
        mhng_accepted_indices = mhng_results[3]
    except Exception:
        return None

    proposed_counts = np.zeros((num_agents, num_agents), dtype=np.int64)
    accepted_counts = np.zeros((num_agents, num_agents), dtype=np.int64)

    for target_id in range(num_agents):
        if target_id >= len(mhng_metadata) or target_id >= len(mhng_accepted_indices):
            continue
        metadata = mhng_metadata[target_id]
        accepted_mask = mhng_accepted_indices[target_id]

        latent_source_ids = None
        if metadata is None:
            continue
        if hasattr(metadata, "get"):
            latent_source_ids = metadata.get("latent_source_ids")
        if latent_source_ids is None:
            continue
        if hasattr(latent_source_ids, "numel") and latent_source_ids.numel() == 0:
            continue

        latent_source_np = _to_numpy(latent_source_ids).astype(np.int64)
        accepted_mask_np = _to_numpy(accepted_mask).astype(bool)
        if latent_source_np.size == 0:
            continue
        if accepted_mask_np.size != latent_source_np.size:
            accepted_mask_np = accepted_mask_np.reshape(-1)[: latent_source_np.size]
            if accepted_mask_np.size != latent_source_np.size:
                continue

        proposed = np.bincount(latent_source_np, minlength=num_agents)
        accepted = np.bincount(latent_source_np[accepted_mask_np], minlength=num_agents)
        proposed_counts[:, target_id] += proposed
        accepted_counts[:, target_id] += accepted

    with np.errstate(divide="ignore", invalid="ignore"):
        accept_rates = accepted_counts / proposed_counts
        accept_rates = np.where(proposed_counts > 0, accept_rates, np.nan)

    return {
        "adjacency_matrix": _to_numpy(step_payload.get("adjacency_matrix")),
        "proposed_counts": proposed_counts,
        "accepted_counts": accepted_counts,
        "accept_rates": accept_rates,
    }


def _compute_memorize_edge_acceptance(step_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    memorize_results = step_payload.get("memorize_results")
    num_agents = int(step_payload.get("num_agents", 0))
    if memorize_results is None or num_agents <= 0:
        return None

    try:
        memorize_masks, metadatas = memorize_results
    except Exception:
        return None

    proposed_counts = np.zeros((num_agents, num_agents), dtype=np.int64)
    accepted_counts = np.zeros((num_agents, num_agents), dtype=np.int64)

    for target_id in range(num_agents):
        if target_id >= len(metadatas) or target_id >= len(memorize_masks):
            continue
        metadata = metadatas[target_id]
        accept_mask = memorize_masks[target_id]

        obs_source_ids = None
        from_buffer = None
        if metadata is None:
            continue
        if hasattr(metadata, "get"):
            obs_source_ids = metadata.get("obs_source_ids")
            from_buffer = metadata.get("from_buffer")
        if obs_source_ids is None:
            continue
        if hasattr(obs_source_ids, "numel") and obs_source_ids.numel() == 0:
            continue

        obs_source_np = _to_numpy(obs_source_ids).astype(np.int64)
        accept_mask_np = _to_numpy(accept_mask).astype(bool)
        if from_buffer is not None:
            from_buffer_np = _to_numpy(from_buffer).astype(bool)
        else:
            from_buffer_np = None

        if obs_source_np.size == 0:
            continue
        if accept_mask_np.size != obs_source_np.size:
            accept_mask_np = accept_mask_np.reshape(-1)[: obs_source_np.size]
            if accept_mask_np.size != obs_source_np.size:
                continue
        if from_buffer_np is not None and from_buffer_np.size != obs_source_np.size:
            from_buffer_np = from_buffer_np.reshape(-1)[: obs_source_np.size]
            if from_buffer_np.size != obs_source_np.size:
                from_buffer_np = None

        proposal_mask = ~from_buffer_np if from_buffer_np is not None else np.ones_like(obs_source_np, dtype=bool)
        proposed = np.bincount(obs_source_np[proposal_mask], minlength=num_agents)
        accepted_proposal_mask = accept_mask_np & proposal_mask
        accepted = np.bincount(obs_source_np[accepted_proposal_mask], minlength=num_agents)
        proposed_counts[:, target_id] += proposed
        accepted_counts[:, target_id] += accepted

    with np.errstate(divide="ignore", invalid="ignore"):
        accept_rates = accepted_counts / proposed_counts
        accept_rates = np.where(proposed_counts > 0, accept_rates, np.nan)

    return {
        "adjacency_matrix": _to_numpy(step_payload.get("adjacency_matrix")),
        "proposed_counts": proposed_counts,
        "accepted_counts": accepted_counts,
        "accept_rates": accept_rates,
    }


def metric_observations_and_creations_clustered(step_payload: Dict[str, Any], runtime: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "observations": step_payload.get("observations", []),
        "creations": step_payload.get("creations", []),
        "adjacency_matrix": step_payload.get("adjacency_matrix"),
        "agent_to_cluster": step_payload.get("agent_to_cluster", []),
        "num_clusters": step_payload.get("num_clusters", 0),
    }


def metric_network_vs_similarity_latent(step_payload: Dict[str, Any], runtime: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    cached = _cache_get(runtime, "network_vs_similarity_latent")
    if cached is not None:
        return cached
    return _cache_set(runtime, "network_vs_similarity_latent", _compute_network_vs_similarity_from_latent(step_payload))


def metric_wasserstein_similarity(step_payload: Dict[str, Any], runtime: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    cached = _cache_get(runtime, "wasserstein_similarity")
    if cached is not None:
        return cached
    return _cache_set(runtime, "wasserstein_similarity", _compute_wasserstein_similarity(step_payload))


def metric_gromov_wasserstein_similarity(step_payload: Dict[str, Any], runtime: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    cached = _cache_get(runtime, "gromov_wasserstein_similarity")
    if cached is not None:
        return cached
    return _cache_set(runtime, "gromov_wasserstein_similarity", _compute_gromov_wasserstein_similarity(step_payload))


def metric_network_vs_wasserstein_similarity(step_payload: Dict[str, Any], runtime: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    items = metric_wasserstein_similarity(step_payload, runtime)
    if items is None:
        return None
    return {
        "adjacency_matrix": items["adjacency_matrix"],
        "distance_matrix": items["distance_matrix"],
        "corr": items["network_vs_corr"],
        "num_refs": items["num_refs"],
    }


def metric_network_vs_gromov_wasserstein_similarity(step_payload: Dict[str, Any], runtime: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    items = metric_gromov_wasserstein_similarity(step_payload, runtime)
    if items is None:
        return None
    return {
        "adjacency_matrix": items["adjacency_matrix"],
        "distance_matrix": items["distance_matrix"],
        "corr": items["network_vs_corr"],
        "num_refs": items["num_refs"],
    }


def metric_latent_scatter_from_reference(step_payload: Dict[str, Any], runtime: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    cached = _cache_get(runtime, "latent_scatter_from_reference")
    if cached is not None:
        return cached
    return _cache_set(runtime, "latent_scatter_from_reference", _compute_latent_scatter(step_payload))


def metric_rsa_within_clusters(step_payload: Dict[str, Any], runtime: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    cached = _cache_get(runtime, "rsa_within_clusters")
    if cached is not None:
        return cached
    return _cache_set(runtime, "rsa_within_clusters", _compute_rsa_from_reference_latent(step_payload))


def metric_mhng_edge_acceptance(step_payload: Dict[str, Any], runtime: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    cached = _cache_get(runtime, "mhng_edge_acceptance")
    if cached is not None:
        return cached
    return _cache_set(runtime, "mhng_edge_acceptance", _compute_mhng_edge_acceptance(step_payload))


def metric_memorize_edge_acceptance(step_payload: Dict[str, Any], runtime: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    cached = _cache_get(runtime, "memorize_edge_acceptance")
    if cached is not None:
        return cached
    return _cache_set(runtime, "memorize_edge_acceptance", _compute_memorize_edge_acceptance(step_payload))


DEFAULT_STEP_METRICS = {
    "observations_and_creations_clustered": metric_observations_and_creations_clustered,
    "network_vs_similarity_latent": metric_network_vs_similarity_latent,
    "wasserstein_similarity": metric_wasserstein_similarity,
    "gromov_wasserstein_similarity": metric_gromov_wasserstein_similarity,
    "network_vs_wasserstein_similarity": metric_network_vs_wasserstein_similarity,
    "network_vs_gromov_wasserstein_similarity": metric_network_vs_gromov_wasserstein_similarity,
    "latent_scatter_from_reference": metric_latent_scatter_from_reference,
    "rsa_within_clusters": metric_rsa_within_clusters,
    "mhng_edge_acceptance": metric_mhng_edge_acceptance,
    "memorize_edge_acceptance": metric_memorize_edge_acceptance,
}
