import torch
import torch.nn as nn
import torch.distributions as td
from typing import Tuple

from .networks import FeedForward


class Decoder(nn.Module):
    def __init__(self,
                 obs_shape: Tuple[int, ...],
                 latent_dim: int,
                 hidden_dim: int,
                 num_layers: int):
        super().__init__()
        assert len(obs_shape) == 1, "obs_shape must be 1-dimensional"

        self.decoder = MLPDecoder(
            obs_shape[0],
            latent_dim,
            hidden_dim,
            num_layers
        )

    def forward(self, z: torch.Tensor):
        return self.decoder(z)


class MLPDecoder(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 latent_dim: int,
                 hidden_dim: int,
                 num_layers: int):
        super().__init__()

        self.network = FeedForward(
            latent_dim,
            obs_dim * 2,
            hidden_dim,
            num_layers,
            norm=True
        )

    def forward(self, z: torch.Tensor):
        stats = self.network(z)
        mu, logvar = torch.chunk(stats, 2, dim=-1)
        std = torch.exp(0.5 * logvar)
        return td.Independent(td.Normal(mu, std), 1)
