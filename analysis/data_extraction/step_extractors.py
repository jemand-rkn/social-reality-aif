from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from analysis.core.registry import ExtractorRegistry, ExtractionRuntimeContext

def extract_observation_creation_views(
    context: ExtractionRuntimeContext,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    payload["observations"] = context.eval_context.get_observations()
    payload["creations"] = context.eval_context.get_creations()
    return payload


def extract_network_views(
    context: ExtractionRuntimeContext,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    agent_to_cluster, num_clusters = context.eval_context.get_agent_to_cluster()
    payload["adjacency_matrix"] = context.eval_context.get_adjacency_matrix(include_self=False)
    payload["adjacency_matrix_with_self"] = context.eval_context.get_adjacency_matrix(include_self=True)
    payload["agent_to_cluster"] = agent_to_cluster
    payload["num_clusters"] = num_clusters
    payload["network"] = {
        "adjacency_matrix": payload["adjacency_matrix"],
        "adjacency_matrix_with_self": payload["adjacency_matrix_with_self"],
        "agent_to_cluster": agent_to_cluster,
        "num_clusters": num_clusters,
    }

    context.store.save_shared_step(
        "adjacency_and_clusters",
        context.step_context.step,
        {
            "adjacency_matrix": np.asarray(payload["adjacency_matrix"]),
            "adjacency_matrix_with_self": np.asarray(payload["adjacency_matrix_with_self"]),
            "agent_to_cluster": np.asarray(agent_to_cluster, dtype=np.int64),
            "num_clusters": np.asarray([num_clusters], dtype=np.int64),
        },
    )
    return payload


def extract_reference_views(
    context: ExtractionRuntimeContext,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    reference_obs_grouped_t = context.eval_context.get_reference_observations_grouped()
    reference_obs = reference_obs_grouped_t.numpy()

    payload["reference_obs"] = reference_obs
    context.store.save_shared_step(
        "reference_obs",
        context.step_context.step,
        {
            "reference_obs": reference_obs,
        },
    )
    return payload


def extract_latent_stats(
    context: ExtractionRuntimeContext,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    reference_obs = payload.get("reference_obs")
    if reference_obs is None:
        raise ValueError("reference_obs must be extracted before reference_latent_stats.")

    reference_obs_np = np.asarray(reference_obs)
    if reference_obs_np.ndim == 3:
        reference_obs_flat = reference_obs_np.reshape(-1, reference_obs_np.shape[-1])
    elif reference_obs_np.ndim == 2:
        reference_obs_flat = reference_obs_np
    else:
        raise ValueError(f"Invalid reference_obs shape: {reference_obs_np.shape}")

    mu, std = context.eval_context.get_latent_stats(torch.from_numpy(reference_obs_flat))
    mu_np = mu.detach().cpu().numpy() if hasattr(mu, "detach") else np.asarray(mu)
    std_np = std.detach().cpu().numpy() if hasattr(std, "detach") else np.asarray(std)
    if mu_np.ndim >= 2 and std_np.ndim >= 2:
        num_refs = int(reference_obs_flat.shape[0])
        if mu_np.shape[1] != num_refs or std_np.shape[1] != num_refs:
            raise ValueError(
                f"reference_obs and reference_latent mismatch at step={context.step_context.step}: "
                f"reference_obs={num_refs}, reference_latent_mu={mu_np.shape[1]}, reference_latent_std={std_np.shape[1]}"
            )

    payload["reference_latent_mu"] = mu_np
    payload["reference_latent_std"] = std_np
    payload["reference_latent_stats"] = {
        "mu": mu_np,
        "std": std_np,
    }
    payload["reference_latent_aligned_with_reference_obs"] = True
    context.store.save_shared_step(
        "reference_latent_stats",
        context.step_context.step,
        {
            "mu": mu_np,
            "std": std_np,
        },
    )
    return payload


def extract_event_payloads(
    context: ExtractionRuntimeContext,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    payload["mhng_results"] = context.data_payload.get("mhng_results")
    payload["memorize_results"] = context.data_payload.get("memorize_results")
    payload["events"] = {
        "mhng_results": payload["mhng_results"],
        "memorize_results": payload["memorize_results"],
    }
    return payload


def build_default_registry() -> ExtractorRegistry:
    registry = ExtractorRegistry()
    registry.register("observations_creations", extract_observation_creation_views)
    registry.register("network", extract_network_views)
    registry.register("reference_obs", extract_reference_views)
    registry.register("reference_latent_stats", extract_latent_stats)
    registry.register("events", extract_event_payloads)
    return registry
