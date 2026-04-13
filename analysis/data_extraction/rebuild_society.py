from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from omegaconf import OmegaConf

from data import DataFactory
from srcaif import Agent, Society


def parse_step(path: Path) -> int:
    name = path.name
    if not name.startswith("step_"):
        return -1
    try:
        token = name.split("_", 1)[1]
        token = token.split(".", 1)[0]
        return int(token)
    except ValueError:
        return -1


def load_config(input_dir: Path):
    config_path = input_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return OmegaConf.load(config_path)


def build_society(cfg, device: torch.device) -> Society:
    agents: List[Agent] = [
        Agent(
            agent_id=agent_id,
            obs_shape=tuple(cfg.data.obs_shape),
            latent_dim=int(cfg.latent_dim),
            hidden_dim=int(cfg.hidden_dim),
            num_layers=int(cfg.num_layers),
            lambda_creative=float(cfg.lambda_creative),
            beta_social=float(cfg.beta_social),
            beta_individual=float(cfg.beta_individual),
            beta_initial=float(cfg.beta_initial),
            model_lr=float(cfg.model_lr),
            critic_lr=float(cfg.critic_lr),
            model_grad_clip=float(cfg.model_grad_clip),
            critic_grad_clip=float(cfg.critic_grad_clip),
            buffer_capacity=int(cfg.buffer_capacity),
            memorization_alpha=float(cfg.memorization_alpha),
            memorization_temperature=float(cfg.memorization_temperature),
            memorization_num_references=int(cfg.memorization_num_references),
            device=device,
        )
        for agent_id in range(int(cfg.num_agents))
    ]
    network_params = OmegaConf.to_container(cfg.network.network_params, resolve=True)
    society = Society(agents, cfg.network.network_type, network_params)

    init_data = DataFactory.create(cfg)
    society.set_data_to_agents(init_data)
    return society


def load_checkpoint(society: Society, checkpoint_dir: Path, num_agents: int) -> None:
    for agent_id in range(num_agents):
        model_path = checkpoint_dir / f"agent_{agent_id}.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {model_path}")
        society.agents[agent_id].load_model(str(model_path))
