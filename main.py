import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
import hydra
import wandb
from logging import getLogger

from srcaif import Agent, Society, compute_rsa_within_clusters
from data import DataFactory
from config import Config
from utils import fix_seed


logger = getLogger(__name__)


@hydra.main(config_path=str(Path(__file__).parent.resolve() / "config"), config_name="config", version_base=None)
def main(cfg: Config):
    cfg_yaml = OmegaConf.to_yaml(cfg)
    print(cfg_yaml)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fix_seed(cfg.seed)
    device = torch.device(cfg.device)

    init_data = DataFactory.create(cfg)

    agents = [
        Agent(
            agent_id=agent_id,
            obs_shape=cfg.data.obs_shape,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            lambda_creative=cfg.lambda_creative,
            beta_social=cfg.beta_social,
            beta_individual=cfg.beta_individual,
            beta_initial=cfg.beta_initial,
            model_lr=cfg.model_lr,
            critic_lr=cfg.critic_lr,
            model_grad_clip=cfg.model_grad_clip,
            critic_grad_clip=cfg.critic_grad_clip,
            buffer_capacity=cfg.buffer_capacity,
            memorization_alpha=cfg.memorization_alpha,
            memorization_temperature=cfg.memorization_temperature,
            memorization_num_references=cfg.memorization_num_references,
            device=device,
        )
        for agent_id in range(cfg.num_agents)
    ]
    society = Society(agents, cfg.network.network_type, cfg.network.network_params)
    society.set_data_to_agents(init_data)

    if cfg.enable_wandb:
        wandb.init(
            project=cfg.project,
            group=cfg.group,
            name=cfg.run_name,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            dir=output_dir,
        )
        run_id_path = output_dir / "wandb_run_id.txt"
        run_id_path.write_text(wandb.run.id)
        wandb.define_metric("initial_step")
        wandb.define_metric("social_step")
        wandb.define_metric("initial_agent_*", step_metric="initial_step")
        wandb.define_metric("agent_*", step_metric="social_step")

        fig = society.network_visualization()
        wandb.log({"visualizations/network_structure": wandb.Image(fig)} | {"social_step": 0})
        plt.close(fig)

    for step in tqdm(range(cfg.num_initial_steps)):
        losses = society.initial_step(
            batch_size=cfg.initial_batch_size,
            num_iterations=cfg.initial_num_iterations,
        )
        for agent_id in range(cfg.num_agents):
            agent_loss_dict = losses[agent_id]
            log_str = f"Initial step {step} | Agent {agent_id} | "
            log_str += " | ".join([f"{key}: {value:.4f}" for key, value in agent_loss_dict.items()])
            logger.info(log_str)
            if cfg.enable_wandb:
                wandb.log(
                    {f"initial_agent_{agent_id}/{key}": value for key, value in agent_loss_dict.items()}
                    | {"initial_step": step}
                )

    for step in tqdm(range(cfg.num_steps)):
        losses, creations, mhng_results, memorize_results, stats = society.step(
            num_creations=cfg.num_creations,
            num_creation_iterations=cfg.num_creation_iterations,
            num_samples_from_buffer=cfg.num_samples_from_buffer,
            num_model_update_iterations=cfg.num_model_update_iterations,
            num_critic_update_iterations=cfg.num_critic_update_iterations,
            model_batch_size=cfg.batch_size,
            action_lr=cfg.action_lr,
            creation_std=cfg.creation_std,
        )

        data_save_path = output_dir / "data" / f"step_{step + 1}.pt"
        data_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                losses=losses,
                creations=creations,
                mhng_results=mhng_results,
                memorize_results=memorize_results,
                stats=stats,
            ),
            data_save_path,
        )

        rsa_results = compute_rsa_within_clusters(society)

        if rsa_results is not None:
            log_str = f"Step {step} | RSA Results | "
            for cluster_idx, rsa_value in enumerate(rsa_results["cluster_rsa"]):
                log_str += f"Cluster {cluster_idx} RSA: {rsa_value:.4f} | "
            for agent_idx, rsa_value in enumerate(rsa_results["agent_rsa"]):
                log_str += f"Agent {agent_idx} RSA: {rsa_value:.4f} | "
            logger.info(log_str)
            if cfg.enable_wandb:
                wandb.log(
                    {
                        f"cluster_rsa/cluster_{cluster_idx}": rsa_value
                        for cluster_idx, rsa_value in enumerate(rsa_results["cluster_rsa"])
                    }
                    | {
                        f"agent_rsa/agent_{agent_idx}": rsa_value
                        for agent_idx, rsa_value in enumerate(rsa_results["agent_rsa"])
                    }
                    | {"social_step": step}
                )

        if cfg.model_save_interval > 0 and (step + 1) % cfg.model_save_interval == 0:
            checkpoint_dir = output_dir / "checkpoints" / f"step_{step + 1}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            for agent_id, agent in enumerate(society.agents):
                model_path = checkpoint_dir / f"agent_{agent_id}.pt"
                agent.save_model(str(model_path))
            logger.info(f"Saved model checkpoints at step {step + 1} to {checkpoint_dir}")

        for agent_id in range(cfg.num_agents):
            agent_loss_dict = losses[agent_id]
            agent_stats = stats[agent_id]
            log_str = f"Step {step} | Agent {agent_id} | "
            log_str += " | ".join([f"{key}: {value:.4f}" for key, value in agent_loss_dict.items()])
            log_str += " | " + " | ".join([f"{key}: {value}" for key, value in agent_stats.items()])
            logger.info(log_str)
            if cfg.enable_wandb:
                loss_log_dict = {f"agent_{agent_id}/{key}": value for key, value in agent_loss_dict.items()}
                stats_log_dict = {f"agent_{agent_id}/{key}": value for key, value in agent_stats.items()}
                wandb.log(loss_log_dict | stats_log_dict | {"social_step": step})

    final_checkpoint_dir = output_dir / "checkpoints" / "final"
    final_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for agent_id, agent in enumerate(society.agents):
        model_path = final_checkpoint_dir / f"agent_{agent_id}.pt"
        agent.save_model(str(model_path))
    logger.info(f"Saved final model checkpoints to {final_checkpoint_dir}")


if __name__ == "__main__":
    main()
