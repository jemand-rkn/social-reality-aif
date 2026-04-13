from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from srcaif.society import Society


class EvalContext:
    """Step-level extraction context.

    This class is intentionally minimal and only contains utilities required by
    layer-1 data extraction.
    """

    def __init__(
        self,
        society: Society,
        creations: Optional[List[torch.Tensor]],
        mhng_results: Optional[Any],
        memorize_results: Optional[Any],
        step: int,
        obs_shape: Tuple[int, ...],
        num_references: int,
    ):
        self.society = society
        self.creations = creations
        self.mhng_results = mhng_results
        self.memorize_results = memorize_results
        self.step = step
        self.obs_shape = tuple(obs_shape)
        self.num_references = num_references
        self._cache: Dict[str, Any] = {}

    def _is_vector_obs(self) -> bool:
        return len(self.obs_shape) == 1 and self.obs_shape[0] >= 2

    def get_observations(self) -> List[np.ndarray]:
        if "observations" in self._cache:
            return self._cache["observations"]
        observations = []
        for agent in self.society.agents:
            buffer_obs = agent.buffer.get_all().detach().cpu().numpy()
            observations.append(buffer_obs)
        self._cache["observations"] = observations
        return observations

    def get_creations(self) -> List[np.ndarray]:
        if "creations" in self._cache:
            return self._cache["creations"]
        creations = []
        if self.creations is None:
            for _ in self.society.agents:
                creations.append(np.empty((0, *self.obs_shape), dtype=np.float32))
        else:
            for created_obs in self.creations:
                creations.append(created_obs.detach().cpu().numpy())
        self._cache["creations"] = creations
        return creations

    def get_adjacency_matrix(self, include_self: bool = True) -> np.ndarray:
        key = f"adjacency_matrix_{include_self}"
        if key in self._cache:
            return self._cache[key]
        num_agents = len(self.society.agents)
        adjacency_matrix = nx.to_numpy_array(self.society.graph, nodelist=range(num_agents))
        if include_self:
            np.fill_diagonal(adjacency_matrix, 1)
        self._cache[key] = adjacency_matrix
        return adjacency_matrix

    def get_agent_to_cluster(self) -> Tuple[List[int], int]:
        if "agent_to_cluster" in self._cache:
            return self._cache["agent_to_cluster"], self._cache["num_clusters"]
        num_agents = len(self.society.agents)
        try:
            from networkx.algorithms import community

            clusters = list(community.greedy_modularity_communities(self.society.graph))
        except Exception:
            clusters = [{i} for i in range(num_agents)]
        agent_to_cluster = [0] * num_agents
        for cluster_id, cluster in enumerate(clusters):
            for agent_id in cluster:
                agent_to_cluster[agent_id] = cluster_id
        num_clusters = len(clusters)
        self._cache["agent_to_cluster"] = agent_to_cluster
        self._cache["num_clusters"] = num_clusters
        return agent_to_cluster, num_clusters

    def get_reference_observations(self) -> torch.Tensor:
        if "reference_obs_flat" in self._cache:
            return self._cache["reference_obs_flat"]
        grouped = self.get_reference_observations_grouped()
        if grouped.numel() == 0:
            flat = torch.empty(0, *self.obs_shape)
        else:
            flat = grouped.reshape(-1, *self.obs_shape)
        self._cache["reference_obs_flat"] = flat
        return flat

    def get_reference_observations_grouped(self) -> torch.Tensor:
        if "reference_obs_grouped" in self._cache:
            return self._cache["reference_obs_grouped"]

        if not self._is_vector_obs():
            raise ValueError("Evaluator currently supports vector observations only.")
        num_agents = len(self.society.agents)
        if num_agents == 0:
            reference_obs_grouped = torch.empty(0, 0, *self.obs_shape)
            self._cache["reference_obs_grouped"] = reference_obs_grouped
            return reference_obs_grouped

        samples_per_agent = max(1, self.num_references // num_agents)
        obs_dim = self.obs_shape[0]
        grouped_samples = []
        for agent in self.society.agents:
            n_samples = samples_per_agent
            if not agent.buffer.is_empty():
                available = min(n_samples, len(agent.buffer))
                samples = agent.buffer.sample(available)
                if available < n_samples:
                    pad = torch.rand(n_samples - available, obs_dim, device=agent.device) * 2 - 1
                    samples = torch.cat([samples, pad], dim=0)
            else:
                samples = torch.rand(n_samples, obs_dim, device=agent.device) * 2 - 1
            grouped_samples.append(samples)

        reference_obs_grouped = torch.stack(grouped_samples, dim=0)  # [A, S, D]
        reference_obs_grouped = reference_obs_grouped.detach().cpu()
        self._cache["reference_obs_grouped"] = reference_obs_grouped
        return reference_obs_grouped

    def get_latent_stats(self, reference_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = f"latent_stats_{reference_obs.shape}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        if reference_obs.numel() == 0:
            empty = (torch.empty(0), torch.empty(0))
            self._cache[cache_key] = empty
            return empty
        mu_list = []
        std_list = []
        for agent in self.society.agents:
            with torch.no_grad():
                ref_obs_device = reference_obs.to(agent.device)
                latent_dist = agent.encoder(ref_obs_device)
                base_dist = latent_dist.base_dist
                mu_list.append(base_dist.loc)
                std_list.append(base_dist.scale)
        device = mu_list[0].device
        mu = torch.stack([m.to(device) for m in mu_list], dim=0)
        std = torch.stack([s.to(device) for s in std_list], dim=0)
        self._cache[cache_key] = (mu, std)
        return mu, std
