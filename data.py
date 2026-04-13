import math
from typing import List
import torch

from config import Config


class DataFactory:
    """Factory for generating agent buffer initialization data from config."""

    @staticmethod
    def create(config: Config) -> List[torch.Tensor]:
        """
        Create buffer data based on config.

        Returns:
            List of tensors, each with shape (buffer_capacity, *obs_shape)
        """
        obs_shape = config.data.obs_shape
        assert len(obs_shape) == 1, "obs_shape must be 1-dimensional for vector data"
        dim = obs_shape[0]

        grid_positions = DataFactory._generate_grid_positions(
            config.num_agents, dim, config.data.vector_separation_scale
        )
        shuffled_positions = grid_positions[torch.randperm(len(grid_positions))]

        buffer_data = []
        for agent_id in range(config.num_agents):
            center = shuffled_positions[agent_id]
            samples = config.data.noise_scale * torch.randn(config.buffer_capacity, dim) + center
            buffer_data.append(samples)

        return buffer_data

    @staticmethod
    def _generate_grid_positions(num_agents: int, dim: int, separation_scale: float) -> torch.Tensor:
        positions = []
        for agent_id in range(num_agents):
            pos = torch.zeros(dim)
            if dim == 1:
                pos[0] = -1.0 + (2.0 * agent_id / max(1, num_agents - 1)) if num_agents > 1 else 0.0
            elif dim == 2:
                grid_1d_size = math.ceil(math.sqrt(num_agents))
                row = agent_id // grid_1d_size
                col = agent_id % grid_1d_size
                pos[0] = -1.0 + (2.0 * col / max(1, grid_1d_size - 1)) if grid_1d_size > 1 else 0.0
                pos[1] = -1.0 + (2.0 * row / max(1, grid_1d_size - 1)) if grid_1d_size > 1 else 0.0
            else:
                pos[0] = -1.0 + (2.0 * agent_id / max(1, num_agents - 1)) if num_agents > 1 else 0.0
            positions.append(pos)
        return torch.stack(positions)
