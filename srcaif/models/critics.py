import torch
import torch.nn as nn
from typing import Tuple


class SpectralNormFeedForward(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 num_layers: int):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())
        for _ in range(num_layers - 2):
            layers.append(nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)))
            layers.append(nn.SiLU())
        layers.append(nn.utils.spectral_norm(nn.Linear(hidden_dim, output_dim)))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class Critic(nn.Module):
    def __init__(self,
                 obs_shape: Tuple[int, ...],
                 latent_dim: int,
                 hidden_dim: int,
                 num_layers: int):
        super().__init__()
        assert len(obs_shape) == 1, "obs_shape must be 1-dimensional"

        self.obs_critic = SpectralNormFeedForward(obs_shape[0], hidden_dim, hidden_dim, num_layers)
        self.latent_critic = SpectralNormFeedForward(latent_dim, hidden_dim, hidden_dim, num_layers)
        self.unconditinal_head = nn.utils.spectral_norm(nn.Linear(hidden_dim, 1))

    def forward(self, o: torch.Tensor, z: torch.Tensor):
        v_o = self.obs_critic(o)
        h_z = self.latent_critic(z)
        phi_z = self.unconditinal_head(h_z)
        proj = torch.sum(v_o * h_z, dim=-1, keepdim=True)
        return phi_z + proj
