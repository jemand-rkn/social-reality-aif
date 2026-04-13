import torch
import torch.nn as nn
import torch.distributions as td
from typing import Tuple

from .networks import FeedForward


class Encoder(nn.Module):
    def __init__(self,
                 obs_shape: Tuple[int, ...],
                 latent_dim: int,
                 hidden_dim: int,
                 num_layers: int):
        super().__init__()
        assert len(obs_shape) == 1, "obs_shape must be 1-dimensional"

        self.encoder = MLPEncoder(
            obs_shape[0],
            latent_dim,
            hidden_dim,
            num_layers
        )

    def forward(self, o: torch.Tensor):
        return self.encoder(o)


class MLPEncoder(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 latent_dim: int,
                 hidden_dim: int,
                 num_layers: int):
        super().__init__()

        self.random_dim = latent_dim // 2

        self.network = FeedForward(
            obs_dim + self.random_dim,
            latent_dim * 2,
            hidden_dim,
            num_layers,
            norm=True
        )

    def forward(self, o: torch.Tensor):
        random = torch.randn(o.shape[0], self.random_dim, device=o.device)
        o = torch.concat([o, random], dim=-1)
        stats = self.network(o)
        mu, logvar = torch.chunk(stats, 2, dim=-1)
        std = torch.exp(0.5 * torch.clamp(logvar, min=-20.0, max=20.0))
        return td.Independent(td.Normal(mu, std), 1)
