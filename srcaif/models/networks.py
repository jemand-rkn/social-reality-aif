import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, norm=True):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = self._create_layers(norm)

    def _create_layers(self, norm):
        if norm:
            layers = [
                nn.RMSNorm(self.input_dim),
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.SiLU(),
            ]
        else:
            layers = [
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.SiLU(),
            ]
        for _ in range(self.num_layers - 2):
            if norm:
                layers.append(nn.RMSNorm(self.hidden_dim))
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.SiLU())
        if norm:
            layers.append(nn.RMSNorm(self.hidden_dim))
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
