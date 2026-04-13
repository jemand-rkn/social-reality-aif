import torch
import numpy as np
from tensordict import TensorDict
import networkx as nx
import matplotlib.pyplot as plt
import time
from typing import List, Dict, Tuple, Optional
from logging import getLogger
from concurrent.futures import ThreadPoolExecutor
import wandb

from .agent import Agent


logger = getLogger(__name__)


class Society:
    def __init__(
        self,
        agents: List[Agent],
        network_type: str = 'small_world',
        network_params: dict = {}
    ):
        self.agents = agents
        self.num_agents = len(agents)
        self.graph = self._create_network(network_type, network_params)
        
        self.social_latents: List[torch.Tensor] = None
    
    # =========================================================================
    # Network Creation and Management
    # =========================================================================
    def _create_network(self, network_type: str, params: dict) -> nx.Graph:
        """Create network graph based on specified type."""
        n = self.num_agents
        
        if network_type == 'small_world':
            # Watts-Strogatz small-world
            k = params.get('k', 4)  # each node connected to k nearest neighbors
            p = params.get('p', 0.1)  # rewiring probability
            graph = nx.watts_strogatz_graph(n, k, p)
        
        elif network_type == 'scale_free':
            # Barabási-Albert scale-free
            m = params.get('m', 2)  # number of edges to attach
            graph = nx.barabasi_albert_graph(n, m)
        
        elif network_type == 'connected_caveman':
            """
            Connected caveman graph: clusters of complete graphs connected in a ring.
            Each cluster is a clique, with one bridge node to adjacent clusters.
            
            This is a standard topology for studying community structure and
            information diffusion across isolated communities.
            """
            num_clusters = params.get('num_clusters', 3)
            cluster_size = params.get('cluster_size', n // num_clusters)
            if cluster_size is None:
                cluster_size = n // num_clusters
            
            graph = nx.connected_caveman_graph(num_clusters, cluster_size)
        
        elif network_type == 'fully_connected':
            # Complete graph
            graph = nx.complete_graph(n)
        
        elif network_type == 'ring':
            # Ring/cycle graph
            graph = nx.cycle_graph(n)
        
        elif network_type == 'grid':
            # 2D grid
            rows = params.get('rows', int(np.sqrt(n)))
            cols = params.get('cols', int(np.ceil(n / rows)))
            graph = nx.grid_2d_graph(rows, cols)
            # Relabel nodes to integers
            graph = nx.convert_node_labels_to_integers(graph)
        
        elif network_type == 'random':
            # Erdős-Rényi random graph
            p = params.get('p', 0.1)  # edge probability
            graph = nx.erdos_renyi_graph(n, p)
        
        else:
            raise ValueError(f"Unknown network type: {network_type}")
        
        return graph
    
    def get_neighbors(self, agent_id: int) -> List[Agent]:
        """
        Get neighbor agents
        
        Args:
            agent_id: agent ID
        Returns:
            neighbors: list of neighbor agents
        """
        neighbor_ids = list(self.graph.neighbors(agent_id))
        neighbors = [self.agents[i] for i in neighbor_ids]
        return neighbors
    
    def add_edge(self, agent_id_1: int, agent_id_2: int) -> None:
        """Add edge between two agents"""
        self.graph.add_edge(agent_id_1, agent_id_2)
    
    def remove_edge(self, agent_id_1: int, agent_id_2: int) -> None:
        """Remove edge between two agents"""
        if self.graph.has_edge(agent_id_1, agent_id_2):
            self.graph.remove_edge(agent_id_1, agent_id_2)
    
    def _visualize_network(
        self,
        figsize: Tuple[int, int] = (10, 8),
        node_size: int = 500,
        node_color: str = 'lightblue',
        edge_color: str = 'gray',
        with_labels: bool = True,
        font_size: int = 10,
        layout: str = 'spring',
        title: Optional[str] = None,
        **layout_kwargs
    ) -> plt.Figure:
        """
        Visualize the social network structure.
        
        Args:
            figsize: Figure size
            node_size: Size of nodes
            node_color: Color of nodes
            edge_color: Color of edges
            with_labels: Whether to show node labels
            font_size: Font size for labels
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai', 'shell', 'spectral')
            title: Figure title
            **layout_kwargs: Additional arguments for layout algorithm
        
        Returns:
            fig: matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Choose layout algorithm
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, **layout_kwargs)
        elif layout == 'circular':
            pos = nx.circular_layout(self.graph, **layout_kwargs)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.graph, **layout_kwargs)
        elif layout == 'shell':
            pos = nx.shell_layout(self.graph, **layout_kwargs)
        elif layout == 'spectral':
            pos = nx.spectral_layout(self.graph, **layout_kwargs)
        else:
            raise ValueError(f"Unknown layout: {layout}")
        
        # Draw network
        nx.draw_networkx_nodes(
            self.graph, pos, 
            node_size=node_size, 
            node_color=node_color,
            ax=ax
        )
        nx.draw_networkx_edges(
            self.graph, pos,
            edge_color=edge_color,
            ax=ax
        )
        
        if with_labels:
            nx.draw_networkx_labels(
                self.graph, pos,
                font_size=font_size,
                ax=ax
            )
        
        # Set title
        if title is None:
            title = f"Social Network Structure (n={self.num_agents})"
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def network_visualization(
        self,
        figsize: Tuple[int, int] = (10, 8),
        node_size: int = 500,
        node_color: str = 'lightblue',
        edge_color: str = 'gray',
        with_labels: bool = True,
        font_size: int = 10,
        layout: str = 'spring',
        title: Optional[str] = None,
        **layout_kwargs
    ) -> plt.Figure:
        """
        Visualize the social network and log to WandB.
        
        Args:
            figsize: Figure size
            node_size: Size of nodes
            node_color: Color of nodes
            edge_color: Color of edges
            with_labels: Whether to show node labels
            font_size: Font size for labels
            layout: Layout algorithm
            title: Figure title
            **layout_kwargs: Additional arguments for layout algorithm
        """
        fig = self._visualize_network(
            figsize=figsize,
            node_size=node_size,
            node_color=node_color,
            edge_color=edge_color,
            with_labels=with_labels,
            font_size=font_size,
            layout=layout,
            title=title,
            **layout_kwargs
        )
        return fig
    
    # =========================================================================
    # Data Management
    # =========================================================================
    def set_data_to_agents(self, data_list: List[torch.Tensor]):
        """
        Set initial buffer data to agents.
        
        Args:
            data_list: list of tensors, each with shape (buffer_capacity, *obs_shape)
        """
        assert len(data_list) == self.num_agents, "Data list length must match number of agents"
        for agent, data in zip(self.agents, data_list):
            creator_ids = torch.full((data.size(0),), agent.id, device=data.device, dtype=torch.long)
            agent.add_to_buffer(data, creator_ids=creator_ids)
    
    # =========================================================================
    # Main Functionalities
    # =========================================================================
    def create(
        self,
        num_creations: int,
        num_iterations: int,
        social_prior_latents: List[torch.Tensor],
        action_lr: float,
        creation_std: float,
    ) -> Tuple[List[torch.Tensor], List[Dict[str, float]]]:
        """Create new observations for all agents (parallel)."""
        def _create_agent(agent: Agent, social_prior_latent: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
            init_o = agent.buffer.sample(num_creations)
            created_o, create_loss = agent.create(init_o, num_iterations, social_prior_latent, action_lr, creation_std)
            # created_o = agent.create_action(init_o, num_iterations, action_lr)
            # created_o = agent.create_entropy(init_o, num_iterations, action_lr)
            return created_o, create_loss
        
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            create_results = list(executor.map(_create_agent, self.agents, social_prior_latents))

        if len(create_results) == 0:
            return [], []

        creations, create_losses = map(list, zip(*create_results))
        return creations, create_losses
    
    def communicate(self, creations: Optional[List[torch.Tensor]], num_samples_from_buffer: int) -> None:
        """Share created observations among neighboring agents (parallel)."""
        # Collect all (agent_id, neighbor) pairs and infer latents in parallel
        inference_tasks: List[Tuple[int, Agent, torch.Tensor]] = []
        for agent_id in range(self.num_agents):
            neighbors = self.get_neighbors(agent_id)
            created_o = creations[agent_id] if creations is not None else torch.empty(0, *self.agents[agent_id].obs_shape, device=self.agents[agent_id].device)
            sampled_o = self.agents[agent_id].buffer.sample(num_samples_from_buffer)
            all_o = torch.concat([created_o, sampled_o], dim=0)
            from_buffer = torch.concat([torch.zeros(created_o.size(0)), torch.ones(num_samples_from_buffer)]).bool()
            for neighbor in neighbors:
                inference_tasks.append((agent_id, neighbor, all_o, from_buffer))
        
        # Parallel inference: each neighbor infers latents from received observations
        def _infer_latents(task: Tuple[int, Agent, torch.Tensor, torch.Tensor]) -> Tuple[int, Agent, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            agent_id, neighbor, created_sampled_o, from_buffer = task
            other_latent = neighbor.infer_latents_from_received(created_sampled_o)
            self_latent = self.agents[agent_id].infer_latents_from_received(created_sampled_o)
            return (agent_id, neighbor, created_sampled_o, other_latent, self_latent, from_buffer)
        
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            inference_results: List[Tuple[int, Agent, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = list(executor.map(_infer_latents, inference_tasks))
        
        # Add to local pools
        for agent_id, neighbor, created_sampled_o, other_latent, self_latent, from_buffer in inference_results:
            agent = self.agents[agent_id]
            
            # Neighbor receives agent's creation
            neighbor.add_to_local_pool(
                created_sampled_o,
                self_latent,
                obs_source_ids=[agent_id]*created_sampled_o.size(0),
                latent_source_ids=[agent_id]*created_sampled_o.size(0),
                from_buffer=from_buffer,
            )
            # Agent receives its own creation with neighbor's interpretation
            agent.add_to_local_pool(
                created_sampled_o,
                other_latent,
                obs_source_ids=[agent_id]*created_sampled_o.size(0),
                latent_source_ids=[neighbor.id]*created_sampled_o.size(0),
                from_buffer=from_buffer,
            )
    
    def mhng(self) -> Tuple[List[float], Tuple[List[torch.Tensor], List[torch.Tensor], List[TensorDict], List[torch.Tensor]]]:
        """Apply MHNG sampling and add accepted samples to buffer.
        
        Returns:
            List of accept counts from others for each agent (excluding self-created)
        """
        accept_ratios = []
        mhng_observations = []
        mhng_latents = []
        mhng_metadata = []
        mhng_accepted_indices = []
        for agent in self.agents:
            observations, other_latents, metadata = agent.local_shared_pool.get_all()
            social_latent, accepted_indices = agent.mhng(observations, other_latents)
            accepted_observations = observations[accepted_indices]
            accepted_latents = social_latent[accepted_indices]
            accepted_metadata = metadata[accepted_indices]
            agent.add_to_mhng_samples(accepted_observations, accepted_latents, **accepted_metadata)
            
            mhng_observations.append(observations)
            mhng_latents.append(social_latent)
            mhng_metadata.append(metadata)
            mhng_accepted_indices.append(accepted_indices)
            
            # Count only accepts from others (exclude self-created)
            self_created = metadata['obs_source_ids'] == agent.id
            num_accepted_from_others = (accepted_indices & ~self_created).sum().item()
            accept_ratio = num_accepted_from_others / (~self_created).sum().item() if (~self_created).sum().item() > 0 else 0.0
            accept_ratios.append(accept_ratio)
        return accept_ratios, (mhng_observations, mhng_latents, mhng_metadata, mhng_accepted_indices)
    
    def memorize(self, social_prior_latents: List[torch.Tensor]) -> List[float]:
        """
        Apply selective memorization for all agents (parallel).

        Observations are selected based on active inference:
        agents accept observations that minimize expected free energy
        (E = α * L_recon - D), prioritizing socially novel yet predictable inputs.
        
        Returns:
            List of memorize counts for each agent
        """
        def _memorize_agent(agent: Agent, social_prior_latent: torch.Tensor) -> int:
            # Get all received observations
            observations, _, metadata = agent.local_shared_pool.get_all()
            
            # Active inference-based selection
            memorized_obs, memorize_ratio, memorize_mask = agent.memorize_observations(observations, social_prior_latent, metadata)
            
            # Add all accepted observations to buffer
            if len(memorized_obs) > 0:
                creator_ids = metadata['obs_source_ids'][memorize_mask]
                agent.add_to_buffer(memorized_obs, creator_ids=creator_ids)
            
            return memorize_ratio, memorize_mask, metadata
        
        # Parallel execution
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            results = list(executor.map(_memorize_agent, self.agents, social_prior_latents))
        memorize_ratios, memorize_masks, metadatas = map(list, zip(*results))
        return memorize_ratios, (memorize_masks, metadatas)
    
    def update_critics_multiple(self, batch_size: int, num_iterations: int) -> List[Dict[str, float]]:
        """Update critics for all agents in parallel, multiple iterations."""
        def _update_agent_critic_multiple(agent: Agent) -> Dict[str, float]:
            accumulated_loss: Dict[str, float] = {}
            for _ in range(num_iterations):
                if agent.mhng_samples.is_empty():
                    continue
                target_observations, target_latents, _ = agent.mhng_samples.get_all()
                self_observations = agent.buffer.sample(batch_size)
                loss = agent.update_critic(self_observations, target_observations, target_latents)
                for key, value in loss.items():
                    if key in accumulated_loss:
                        accumulated_loss[key] += value
                    else:
                        accumulated_loss[key] = value
            # Average
            for key in accumulated_loss:
                accumulated_loss[key] /= num_iterations
            return accumulated_loss
        
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            losses = list(executor.map(_update_agent_critic_multiple, self.agents))
        return losses
    
    def update_models_multiple(self, batch_size: int, num_iterations: int) -> List[Dict[str, float]]:
        """Update models for all agents in parallel, multiple iterations."""
        def _update_agent_model_multiple(agent: Agent) -> Dict[str, float]:
            accumulated_loss: Dict[str, float] = {}
            for _ in range(num_iterations):
                mhng_observations = agent.mhng_samples.get_all()[0]
                buffer_observations = agent.buffer.sample(batch_size)
                observations = torch.cat([mhng_observations, buffer_observations], dim=0)
                loss = agent.update_model(observations)
                for key, value in loss.items():
                    if key in accumulated_loss:
                        accumulated_loss[key] += value
                    else:
                        accumulated_loss[key] = value
            # Average
            for key in accumulated_loss:
                accumulated_loss[key] /= num_iterations
            return accumulated_loss
        
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            losses = list(executor.map(_update_agent_model_multiple, self.agents))
        return losses

    def update_models_initial_multiple(self, batch_size: int, num_iterations: int) -> List[Dict[str, float]]:
        """
        Initial phase: update models using standard VAE loss (parallel, multiple iterations).
        
        Args:
            batch_size: Batch size for sampling from buffer
            num_iterations: Number of update iterations per agent
        
        Returns:
            List of loss dictionaries for each agent (averaged over iterations)
        """
        def _update_agent_model_initial_multiple(agent: Agent) -> Dict[str, float]:
            accumulated_loss: Dict[str, float] = {}
            for _ in range(num_iterations):
                observations = agent.buffer.sample(batch_size)
                loss = agent.update_model_initial(observations)
                for key, value in loss.items():
                    if key in accumulated_loss:
                        accumulated_loss[key] += value
                    else:
                        accumulated_loss[key] = value
            # Average
            for key in accumulated_loss:
                accumulated_loss[key] /= num_iterations
            return accumulated_loss
        
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            losses = list(executor.map(_update_agent_model_initial_multiple, self.agents))
        return losses
    
    def clear_pools(self) -> None:
        """Clear temporary pools in all agents."""
        for agent in self.agents:
            agent.clear_local_pool()
            agent.clear_mhng_samples()
    
    def step(
        self,
        num_creations: int,
        num_creation_iterations: int,
        num_samples_from_buffer: int,
        create: bool = True,
        mhng: bool = True,
        memorize: bool = True,
        update_critics: bool = True,
        update_models: bool = True,
        model_batch_size: int = 64,
        num_model_update_iterations: int = 30,
        num_critic_update_iterations: int = 50,
        action_lr: float = 1e-3,
        creation_std: float = 0.1
    ) -> Tuple[List[Dict[str, float]], Optional[List[torch.Tensor]], Optional[Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict[str, int]], List[int]]], List[Dict[str, int]]]:
        """
        Execute one training step for all agents.
        
        Returns:
            losses: List of loss dictionaries for each agent
            creations: List of created observations for each agent
            mhng_results: Optional tuple containing MHNG results (observations, latents, metadata, accepted indices)
            stats: List of dictionaries with statistics for each agent (mhng_accept_from_others, memorize_accept_from_others)
        """
        timings = {}
        
        # Creation
        if create and self.social_latents is not None and num_creations > 0:
            start = time.time()
            creations, create_losses = self.create(
                num_creations,
                num_creation_iterations,
                self.social_latents,
                action_lr,
                creation_std,
            )
            timings['create'] = time.time() - start
        else:
            creations = None
            create_losses = None
            timings['create'] = 0.0
        
        # Communication
        start = time.time()
        self.communicate(creations, num_samples_from_buffer)
        timings['communicate'] = time.time() - start
        
        # MHNG
        mhng_accept_ratios = [0] * self.num_agents
        if mhng:
            start = time.time()
            mhng_accept_ratios, mhng_results = self.mhng()
            self.social_latents = mhng_results[1]
            timings['mhng'] = time.time() - start
        else:
            timings['mhng'] = 0.0
        
        # Memorization (Phase 4: Active inference-based selection for personal buffer)
        memorize_accept_ratios = [0] * self.num_agents
        if memorize and self.social_latents is not None and num_creations > 0:
            start = time.time()
            memorize_accept_ratios, memorize_results = self.memorize(self.social_latents)
            timings['memorize'] = time.time() - start
        else:
            timings['memorize'] = 0.0
            memorize_results = None
        
        losses = [{} for _ in range(self.num_agents)]

        if create_losses is not None:
            for agent_loss, create_loss in zip(losses, create_losses):
                agent_loss.update(create_loss)
        
        # Update critics
        if update_critics:
            start = time.time()
            critic_losses = self.update_critics_multiple(model_batch_size, num_critic_update_iterations)
            timings['update_critics'] = time.time() - start
            for agent_loss, critic_loss in zip(losses, critic_losses):
                agent_loss.update(critic_loss)
        else:
            timings['update_critics'] = 0.0
        
        # Update models
        if update_models:
            start = time.time()
            model_losses = self.update_models_multiple(model_batch_size, num_model_update_iterations)
            timings['update_models'] = time.time() - start
            for agent_loss, model_loss in zip(losses, model_losses):
                agent_loss.update(model_loss)
        else:
            timings['update_models'] = 0.0
        
        # Clear pools
        start = time.time()
        self.clear_pools()
        timings['clear_pools'] = time.time() - start
        
        # Total time
        timings['total'] = sum(v for k, v in timings.items() if k != 'total')
        
        # Log timing information
        logger.info(f"Step timing breakdown:")
        for phase, elapsed in timings.items():
            if phase != 'total':
                pct = (elapsed / timings['total'] * 100) if timings['total'] > 0 else 0
                logger.info(f"  {phase:20s}: {elapsed:8.3f}s ({pct:5.1f}%)")
            else:
                logger.info(f"  {phase:20s}: {elapsed:8.3f}s")
        
        # Create statistics list (per agent)
        stats = [
            {
                'mhng_accept_from_others_ratio': mhng_accept_ratios[i],
                'memorize_accept_from_others_ratio': memorize_accept_ratios[i]
            }
            for i in range(self.num_agents)
        ]
        
        return losses, creations, mhng_results, memorize_results, stats
    
    def initial_step(
        self,
        batch_size: int = 64,
        num_iterations: int = 30
    ) -> List[Dict[str, float]]:
        """
        Execute initial learning phase step.
        Only updates encoder/decoder using standard VAE loss.
        No creation, communication, MHNG, or critic updates.
        
        Args:
            batch_size: Batch size for model updates
            num_iterations: Number of model update iterations per step
        
        Returns:
            List of loss dictionaries for each agent (averaged over iterations)
        """
        # Update models (encoder/decoder only) in parallel with multiple iterations
        model_losses = self.update_models_initial_multiple(batch_size, num_iterations)
        return model_losses
