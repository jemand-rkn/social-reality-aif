import torch
import torch.distributions as td
from tensordict import TensorDict
from typing import Tuple, List, Dict

from .models import Encoder, Decoder, Critic
from .buffer import ObservationBuffer, TemporaryPool
from .utils import get_parameters


class Agent:
    def __init__(self,
                 agent_id: int,
                 obs_shape: Tuple[int, ...],
                 latent_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 lambda_creative: float,
                 beta_social: float,
                 beta_individual: float,
                 beta_initial: float,
                 model_lr: float,
                 critic_lr: float,
                 model_grad_clip: float,
                 critic_grad_clip: float,
                 buffer_capacity: int,
                 memorization_alpha: float,
                 memorization_temperature: float,
                 memorization_num_references: int,
                 device: torch.device):
        self.id = agent_id
        self.obs_shape = obs_shape
        self.latent_dim = latent_dim
        self.device = device
        self.lambda_creative = lambda_creative
        self.beta_social = beta_social
        self.beta_individual = beta_individual
        self.beta_initial = beta_initial
        self.model_grad_clip = model_grad_clip
        self.critic_grad_clip = critic_grad_clip
        self.memorization_alpha = memorization_alpha
        self.memorization_temperature = memorization_temperature
        self.memorization_num_references = memorization_num_references
        self.memorization_min_buffer_size = 50  # Minimum buffer size to enable memorization filtering

        self.encoder = Encoder(
            obs_shape,
            latent_dim,
            hidden_dim,
            num_layers
        ).to(device)
        self.decoder = Decoder(
            obs_shape,
            latent_dim,
            hidden_dim,
            num_layers
        ).to(device)
        self.critic = Critic(
            obs_shape,
            latent_dim,
            hidden_dim,
            num_layers
        ).to(device)

        self.optim_model = torch.optim.Adam(get_parameters([self.encoder, self.decoder]), lr=model_lr)
        self.optim_critic = torch.optim.Adam(get_parameters([self.critic]), lr=critic_lr)
        
        self.buffer = ObservationBuffer(capacity=buffer_capacity, obs_shape=obs_shape, device=device)
        self.local_shared_pool = TemporaryPool(obs_shape=obs_shape, latent_dim=latent_dim, device=device)
        self.mhng_samples = TemporaryPool(obs_shape=obs_shape, latent_dim=latent_dim, device=device)
    
    # =========================================================================
    # Data Management
    # =========================================================================
    def add_to_local_pool(self, observations: torch.Tensor, latents: torch.Tensor, **metadata):
        self.local_shared_pool.add(observations, latents, **metadata)

    def add_to_mhng_samples(self, observations: torch.Tensor, latents: torch.Tensor, **metadata):
        self.mhng_samples.add(observations, latents, **metadata)
    
    def add_to_buffer(self, observations: torch.Tensor, **metadata):
        self.buffer.add(observations, **metadata)
    
    def clear_local_pool(self):
        self.local_shared_pool.clear()
    
    def clear_mhng_samples(self):
        self.mhng_samples.clear()
    
    # =========================================================================
    # Communication with neighbors
    # =========================================================================
    def infer_latents_from_received(self, observations: torch.Tensor) -> torch.Tensor:
        """Infer latent representations from received observations."""
        with torch.no_grad():
            observations = observations.to(self.device)
            latents_dist = self.encoder(observations)
            latents = latents_dist.rsample()
        return latents
    
    # =========================================================================
    # MHNG Sampling
    # =========================================================================
    def mhng(
        self,
        observations: torch.Tensor,
        other_latents: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Metropolis-Hastings sampling using network guidance.
        
        Returns:
            new_latents: Accepted latent representations
            accept: Boolean tensor of acceptance decisions
        """
        with torch.no_grad():
            observations = observations.to(self.device)
            other_latents = other_latents.to(self.device)
            
            self_latents_dist = self.encoder(observations)
            self_latents = self_latents_dist.sample()
            
            others_recon_dist = self.decoder(other_latents)
            self_recon_dist = self.decoder(self_latents)
            log_r = others_recon_dist.log_prob(observations) - self_recon_dist.log_prob(observations)
            r = torch.exp(log_r)
            r = torch.clamp(r, max=1.0)
            accept = torch.rand(r.shape, device=self.device) < r
            new_latents = torch.where(accept.unsqueeze(-1), other_latents, self_latents)
        return new_latents, accept
    
    # =========================================================================
    # Model updates
    # =========================================================================
    def update_critic(self, self_observations: torch.Tensor, target_observations: torch.Tensor, target_latents: torch.Tensor) -> Dict[str, float]:
        """Update critic with self and target observations."""
        self_observations = self_observations.to(self.device)
        target_observations = target_observations.to(self.device)
        target_latents = target_latents.to(self.device)
        
        self.optim_critic.zero_grad()
        
        self_latents_dist = self.encoder(self_observations)
        self_latents = self_latents_dist.rsample()
        d_self = self.critic.forward(self_observations, self_latents)
        self_loss = -d_self.mean()
        
        d_target = self.critic.forward(target_observations, target_latents)
        target_loss = torch.exp(d_target).mean()
        
        loss = self_loss + target_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_grad_clip)
        self.optim_critic.step()
        
        return dict(
            critic_loss=loss.item(),
            self_loss=self_loss.item(),
            target_loss=target_loss.item(),
            critic_grad_norm=grad_norm.item(),
        )
    
    def update_model(self, observations: torch.Tensor) -> Dict[str, float]:
        """Update encoder and decoder with critic guidance."""
        observations = observations.to(self.device)
        
        self.optim_model.zero_grad()
        
        latents_dist = self.encoder(observations)
        latents = latents_dist.rsample()
        recon_dist = self.decoder(latents)
        
        recon_loss = -recon_dist.log_prob(observations).mean()
        d_loss = self.critic.forward(observations, latents).mean()
        individual_prior = td.Independent(td.Normal(torch.zeros_like(latents), torch.ones_like(latents)), 1)
        kld_loss = td.kl_divergence(latents_dist, individual_prior).mean()
        loss = recon_loss + self.beta_social * d_loss + self.beta_individual * kld_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters()), self.model_grad_clip)
        self.optim_model.step()
        
        return dict(
            model_loss=loss.item(),
            recon_loss=recon_loss.item(),
            d_loss=d_loss.item(),
            kld_loss=kld_loss.item(),
            model_grad_norm=grad_norm.item(),
        )
    
    def update_model_initial(self, observations: torch.Tensor) -> Dict[str, float]:
        """
        Initial phase model update using standard VAE loss.
        Updates encoder/decoder only, without using the critic.
        
        Loss: E[-log p(o|z)] + β * KL(q(z|o) || N(0,I))
        
        Args:
            observations: Batch of observations from buffer
        
        Returns:
            Dictionary containing loss components
        """
        observations = observations.to(self.device)
        
        self.optim_model.zero_grad()
        
        # Encode: q(z|o)
        latents_dist = self.encoder(observations)
        latents = latents_dist.rsample()
        
        # Decode: p(o|z)
        recon_dist = self.decoder(latents)
        
        # Reconstruction loss: -E[log p(o|z)]
        recon_loss = -recon_dist.log_prob(observations).mean()
        
        # KL divergence: KL(q(z|o) || N(0,I))
        prior = torch.distributions.Normal(
            torch.zeros_like(latents_dist.mean),
            torch.ones_like(latents_dist.stddev)
        )
        # Match the event shape of latents_dist
        if hasattr(latents_dist, 'event_shape') and len(latents_dist.event_shape) > 0:
            prior = torch.distributions.Independent(prior, len(latents_dist.event_shape))
        
        kl_loss = torch.distributions.kl_divergence(latents_dist, prior).mean()
        
        # Total loss
        loss = recon_loss + self.beta_initial * kl_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters()), self.model_grad_clip)
        self.optim_model.step()
        
        return dict(
            model_loss=loss.item(),
            recon_loss=recon_loss.item(),
            kl_loss=kl_loss.item(),
            model_grad_norm=grad_norm.item()
        )
    
    # =========================================================================
    # Observation creation
    # =========================================================================
    def create(self, init_observations: torch.Tensor, num_iterations: int, social_prior_latent: torch.Tensor, obs_lr: float, creation_std: float) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Create new observations by directly optimizing in observation space.
        
        Args:
            init_observations: Initial observations as starting point
            num_iterations: Number of optimization steps
            social_prior_latent: Social prior latent tensor
            obs_lr: Learning rate for observation optimization
            creation_std: Standard deviation of noise added to final observation
        
        Returns:
            new_observations: Created observations
            creation_loss_dict: Final iteration metrics with keys {efe, epistemic, extrinsic}
        """
        init_observations = init_observations.to(self.device)
        social_prior_latent = social_prior_latent.to(self.device)
        
        # Initialize observations as optimization variables
        o = init_observations.clone().detach().requires_grad_(True)
        o_optim = torch.optim.Adam([o], lr=obs_lr)

        final_efe = 0.0
        final_epistemic = 0.0
        final_extrinsic = 0.0
        
        for _ in range(num_iterations):
            o_optim.zero_grad()
            
            # Encode to latent space
            latents_dist = self.encoder(o)
            latents = latents_dist.rsample()
            
            # Decode from own latent for homeostasis
            preference_dist = self.decoder(latents)
            calc_homeostasis = lambda o: preference_dist.log_prob(o)
            
            # Compute curiosity (social divergence) and homeostasis (personal coherence)
            curiosity = self.critic.forward(o, latents)
            homeostasis = calc_homeostasis(o)
            
            # Maximize curiosity while maintaining homeostasis
            loss = -curiosity.mean() - self.lambda_creative * homeostasis.mean()
            final_efe = loss.item()
            final_epistemic = curiosity.mean().item()
            final_extrinsic = homeostasis.mean().item()
            loss.backward()
            o_optim.step()
        
        # Final observation with clamp constraint
        # new_observations = torch.clamp(o, -1.0, 1.0).detach()
        new_observations = o.detach() + torch.randn_like(o) * creation_std

        creation_loss_dict = dict(
            efe=final_efe,
            epistemic=final_epistemic,
            extrinsic=final_extrinsic,
        )

        return new_observations, creation_loss_dict
    
    # =========================================================================
    # Observation memorization
    # =========================================================================
    def compute_memorization_energy(
        self,
        observations: torch.Tensor,
        self_latent: torch.Tensor,
        social_prior_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute memorization energy: E(o) = α * L_recon - D(o, z_self)
        
        This is unified with creation energy.
        Note: D(o,z) already contains KL divergence w.r.t. social prior.
        
        Args:
            observations: Candidate observations
            self_latent: Latent encodings from own model
            social_prior_latent: Latent representation from social prior
        
        Returns:
            energy: Lower energy = more likely to memorize [N]
        """
        with torch.no_grad():
            observations = observations.to(self.device)
            self_latent = self_latent.to(self.device)
            social_prior_latent = social_prior_latent.to(self.device)
            
            # preference
            preference_dist = self.decoder(self_latent)
            preference_error = -preference_dist.log_prob(observations)
            
            # Social novelty: D(o, z_self) from discriminator
            # D(o,z) = log[q_self(z|o) / q_social(z|o)]
            social_novelty = self.critic(observations, self_latent).squeeze()
            
            # Energy: unified form with creation
            # E(o) = alpha * L_recon - D
            energy = self.memorization_alpha * preference_error - social_novelty
        
        return energy
    
    def memorize_observations(
        self,
        observations: torch.Tensor,
        social_prior_latent: torch.Tensor,
        metadata: TensorDict
    ) -> Tuple[torch.Tensor, float]:
        """
        Select observations to memorize based on active inference.

        Acceptance is driven by expected free energy: E(o) = α * L_recon - D,
        where lower energy corresponds to observations that are both predictable
        (low reconstruction error) and socially novel (high critic score).

        Args:
            observations: All received observations from local pool
            social_prior_latent: Latent representation from social prior

        Returns:
            memorized_obs: Observations selected for memorization (may be empty)
            accept_ratio: Ratio of observations accepted from others (excluding self-created)
        """
        N = len(observations)
        
        if N == 0 or self.buffer.size < self.memorization_min_buffer_size:
            return observations, 0
        
        with torch.no_grad():
            observations = observations.to(self.device)
            
            # Compute energy for candidates
            z_candidate = self.encoder(observations).sample()
            E_candidate = self.compute_memorization_energy(observations, z_candidate, social_prior_latent)
            
            # Sample references from buffer
            reference_obs = self.buffer.sample(N * self.memorization_num_references)
            z_reference = self.encoder(reference_obs).sample()
            E_reference = self.compute_memorization_energy(reference_obs, z_reference, social_prior_latent)
            E_reference = E_reference.view(N, self.memorization_num_references).mean(dim=1)
            
            # Active inference acceptance: prefer lower energy (higher EFE)
            delta_E = E_candidate - E_reference
            accept_prob = torch.minimum(
                torch.ones_like(delta_E),
                torch.exp(-delta_E / self.memorization_temperature)
            )
            accept_mask = torch.rand(N, device=self.device) < accept_prob
            self_created = metadata['obs_source_ids'] == self.id
            from_buffer = metadata['from_buffer']
            
            # Count accepts from others (before adding self-created)
            num_accepted_from_others = (accept_mask & ~self_created & ~from_buffer).sum().item()
            accept_ratio = num_accepted_from_others / (~self_created & ~from_buffer).sum().item() if (~self_created & ~from_buffer).sum().item() > 0 else 0.0
            
            accept_mask = (accept_mask | self_created) & ~from_buffer  # Always accept self-created, not accept from buffer
            
            # Return all accepted observations and acceptance ratio
            return observations[accept_mask], accept_ratio, accept_mask

    # =========================================================================
    # Utilities
    # =========================================================================
    def save_model(self, path: str) -> None:
        """Save model, optimizer, buffer, and pool states to file."""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optim_model_state_dict': self.optim_model.state_dict(),
            'optim_critic_state_dict': self.optim_critic.state_dict(),
            'buffer_state_dict': self.buffer.state_dict(),
            'local_shared_pool_state_dict': self.local_shared_pool.state_dict(),
            'mhng_samples_state_dict': self.mhng_samples.state_dict(),
        }, path)
    
    def load_model(self, path: str) -> None:
        """Load model, optimizer, buffer, and pool states from file."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optim_model.load_state_dict(checkpoint['optim_model_state_dict'])
        self.optim_critic.load_state_dict(checkpoint['optim_critic_state_dict'])
        
        # Load buffer and pools if present in checkpoint
        if 'buffer_state_dict' in checkpoint:
            self.buffer.load_state_dict(checkpoint['buffer_state_dict'])
        if 'local_shared_pool_state_dict' in checkpoint:
            self.local_shared_pool.load_state_dict(checkpoint['local_shared_pool_state_dict'])
        if 'mhng_samples_state_dict' in checkpoint:
            self.mhng_samples.load_state_dict(checkpoint['mhng_samples_state_dict'])
