from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class NetworkParamsConfig:
    k: int = 2
    p: float = 0.1
    m: int = 2
    num_clusters: int = 3
    cluster_size: int | None = None


@dataclass
class NetworkConfig:
    network_type: str = "small_world"
    network_params: NetworkParamsConfig = NetworkParamsConfig()


@dataclass
class DataConfig:
    obs_shape: Tuple[int, ...] = field(default_factory=lambda: (2,))
    vector_separation_scale: float = 2.0
    noise_scale: float = 0.1


@dataclass
class Config:
    num_steps: int = 1000
    num_creations: int = 10
    num_creation_iterations: int = 50
    creation_std: float = 0.1
    batch_size: int = 64
    num_model_update_iterations: int = 30
    num_critic_update_iterations: int = 50
    
    num_initial_steps: int = 1000
    initial_batch_size: int = 64
    initial_num_iterations: int = 30
    beta_initial: float = 1.0
    
    model_lr: float = 1e-3
    critic_lr: float = 5e-4
    action_lr: float = 1e-3
    model_grad_clip: float = 1.0
    critic_grad_clip: float = 1.0

    latent_dim: int = 32
    hidden_dim: int = 128
    num_layers: int = 4
    lambda_creative: float = 0.1
    beta_social: float = 0.5
    beta_individual: float = 0.1
    buffer_capacity: int = 1000
    
    num_samples_from_buffer: int = 10
    
    memorization_alpha: float = 0.1
    memorization_temperature: float = 0.1
    memorization_num_references: int = 10
    
    num_agents: int = 5
    network: NetworkConfig = NetworkConfig()
    
    data: DataConfig = DataConfig()
    
    seed: int = 42
    
    device: str = "cuda:0"
    
    enable_wandb: bool = False
    project: str = "scena"
    group: str = "default"
    run_name: str = "default"
    
    # Model checkpoint settings
    model_save_interval: int = 500  # Save model every N steps (0 = only at end)
