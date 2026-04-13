import torch
from tensordict.tensordict import TensorDict
import numpy as np
from typing import Optional, Tuple, Dict


class ObservationBuffer:
    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, ...],
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32
    ):
        """
        Args:
            capacity: maximum number of observations
            obs_shape: shape of single observation (e.g., (784,) or (1, 28, 28))
            device: 'cpu' or 'cuda'
            dtype: data type for storage
        """
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.device = device
        self.dtype = dtype
        
        # Pre-allocate storage tensor
        self.data = torch.zeros(
            (capacity, *obs_shape),
            dtype=dtype,
            device=device
        )
        self.metadata = TensorDict({}, batch_size=capacity, device=device)
        
        # Current size and write position
        self.size = 0
        self.position = 0
    
    def add(self, observations: torch.Tensor, **metadata):
        """
        Add batch of observations to buffer

        Args:
            observations: batch of observations [batch, *obs_shape]
            **metadata: metadata tensors aligned to the batch (e.g., creator_ids)
        """
        batch_size = observations.size(0)
        observations = observations.to(device=self.device, dtype=self.dtype)
        prepared_meta = {}
        for key, value in metadata.items():
            value = torch.as_tensor(value, device=self.device)
            if value.ndim == 0:
                value = value.expand(batch_size)
            if value.shape[0] != batch_size:
                raise ValueError(f"Metadata '{key}' has batch {value.shape[0]}, expected {batch_size}.")
            prepared_meta[key] = value
            if key not in self.metadata.keys():
                storage_shape = (self.capacity, *value.shape[1:])
                self.metadata[key] = torch.zeros(storage_shape, dtype=value.dtype, device=self.device)
        for key in self.metadata.keys():
            if key not in prepared_meta:
                default_shape = (batch_size, *self.metadata[key].shape[1:])
                prepared_meta[key] = torch.zeros(default_shape, dtype=self.metadata[key].dtype, device=self.device)
        
        if self.position + batch_size <= self.capacity:
            # No wrap-around
            self.data[self.position:self.position + batch_size] = observations
            for key, value in prepared_meta.items():
                self.metadata[key][self.position:self.position + batch_size] = value
            self.position = (self.position + batch_size) % self.capacity
            self.size = min(self.size + batch_size, self.capacity)
        else:
            # Wrap-around
            first_part_size = self.capacity - self.position
            self.data[self.position:self.capacity] = observations[:first_part_size]
            for key, value in prepared_meta.items():
                self.metadata[key][self.position:self.capacity] = value[:first_part_size]
            second_part_size = batch_size - first_part_size
            self.data[0:second_part_size] = observations[first_part_size:]
            for key, value in prepared_meta.items():
                self.metadata[key][0:second_part_size] = value[first_part_size:]
            self.position = second_part_size
            self.size = self.capacity  # Buffer is full

    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Sample random batch from buffer
        
        Args:
            batch_size: number of samples
        Returns:
            batch: sampled observations [batch_size, *obs_shape]
        """
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        # Random indices
        indices = torch.randint(0, self.size, (batch_size,))
        
        return self.data[indices]
    
    def get_all(self) -> torch.Tensor:
        """
        Get all stored observations
        
        Returns:
            observations: all data [size, *obs_shape]
        """
        return self.data[:self.size]

    def get_all_with_metadata(self) -> Tuple[torch.Tensor, TensorDict]:
        """Get all stored observations and metadata."""
        return self.data[:self.size], self.metadata[:self.size]
    
    def clear(self):
        """Clear buffer"""
        self.size = 0
        self.position = 0
    
    def __len__(self) -> int:
        return self.size
    
    def is_empty(self) -> bool:
        return self.size == 0
    
    def is_full(self) -> bool:
        return self.size >= self.capacity
    
    def state_dict(self) -> Dict:
        """Return state dictionary for saving"""
        metadata_dict = {}
        if not self.metadata.is_empty():
            for key in self.metadata.keys():
                metadata_dict[key] = self.metadata[key].cpu()
        return {
            'data': self.data.cpu(),
            'metadata': metadata_dict,
            'size': self.size,
            'position': self.position,
            'capacity': self.capacity,
            'obs_shape': self.obs_shape,
            'dtype': self.dtype
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load state from dictionary"""
        self.capacity = state_dict['capacity']
        self.obs_shape = state_dict['obs_shape']
        self.dtype = state_dict['dtype']
        self.size = state_dict['size']
        self.position = state_dict['position']
        self.data = state_dict['data'].to(device=self.device, dtype=self.dtype)
        metadata_dict = state_dict.get('metadata', {})
        if metadata_dict:
            metadata_on_device = {k: v.to(device=self.device) for k, v in metadata_dict.items()}
            self.metadata = TensorDict(metadata_on_device, batch_size=self.capacity, device=self.device)
        else:
            self.metadata = TensorDict({}, batch_size=self.capacity, device=self.device)


class TemporaryPool:
    """
    Temporary storage pool for observations with metadata
    
    Used for creative_pool, local_shared_pool, mhng_samples
    Stores observations with associated metadata (source_id, timestamp, etc.)
    """
    
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        latent_dim: int,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32
    ):
        """
        Args:
            obs_shape: shape of single observation
            latent_dim: dimension of latent variable
            device: 'cpu' or 'cuda'
            dtype: data type
        """
        self.obs_shape = obs_shape
        self.latent_dim = latent_dim
        self.device = device
        self.dtype = dtype
        
        # Dynamic storage (grows as needed)
        self.observations = torch.empty(0, *obs_shape, device=device, dtype=dtype)
        self.latents = torch.empty(0, latent_dim, device=device, dtype=dtype)
        self.metadata = TensorDict({}, batch_size=0, device=device)
    
    def add(
        self,
        observations: torch.Tensor,
        latents: torch.Tensor,
        **metadata
    ):
        """
        add batch of observations and latents with metadata
        
        Args:
            observations: [batch_size, *obs_shape]
            latents: [batch_size, latent_dim]
            **metadata: Each value is converted to torch.Tensor internally (requires_grad=False)
        """
        observations = observations.to(device=self.device, dtype=self.dtype)
        latents = latents.to(device=self.device, dtype=self.dtype)
        
        batch_size = observations.size(0)
        
        # Append observations and latents
        self.observations = torch.cat([self.observations, observations], dim=0)
        self.latents = torch.cat([self.latents, latents], dim=0)
        
        # Append metadata using TensorDict
        meta_td = TensorDict(metadata, batch_size=batch_size, device=self.device)
        # If metadata is empty, initialize directly to avoid key mismatch on concat
        if self.metadata.is_empty():
            self.metadata = meta_td
        else:
            self.metadata = torch.cat([self.metadata, meta_td], dim=0)
    
    def get_batch(
        self,
        indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        """
        Get batch of observations
        
        Args:
            indices: specific indices to retrieve (None = all)
        Returns:
            observations: [batch, *obs_shape]
            latents: [batch, latent_dim]
            metadata: TensorDict with metadata values
        """
        if len(self.observations) == 0:
            return (
                torch.zeros((0, *self.obs_shape), device=self.device, dtype=self.dtype),
                torch.zeros((0, self.latent_dim), device=self.device, dtype=self.dtype),
                TensorDict({}, batch_size=0, device=self.device)
            )
        
        if indices is None:
            # Return all
            obs_batch = self.observations
            latent_batch = self.latents
            meta_batch = self.metadata
        else:
            # Return specific indices using tensor indexing
            obs_batch = self.observations[indices]
            latent_batch = self.latents[indices]
            # Extract metadata for specific indices
            meta_batch = self.metadata[indices]
        
        return obs_batch, latent_batch, meta_batch
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        """
        Sample random batch
        
        Args:
            batch_size: number of samples
        Returns:
            observations: [batch_size, *obs_shape]
            latents: [batch_size, latent_dim]
            metadata: TensorDict with metadata values
        """
        if len(self.observations) == 0:
            raise ValueError("Cannot sample from empty pool")
        
        n = len(self.observations)
        batch_size = min(batch_size, n)
        
        indices = np.random.choice(n, batch_size, replace=False)
        return self.get_batch(indices)
    
    def get_all(self) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
        """Get all observations and metadata"""
        return self.get_batch()
    
    def clear(self):
        """Clear pool"""
        self.observations = torch.empty(0, *self.obs_shape, device=self.device, dtype=self.dtype)
        self.latents = torch.empty(0, self.latent_dim, device=self.device, dtype=self.dtype)
        self.metadata = TensorDict({}, batch_size=0, device=self.device)
    
    def __len__(self) -> int:
        return len(self.observations)
    
    def is_empty(self) -> bool:
        return len(self.observations) == 0
    
    def state_dict(self) -> Dict:
        """Return state dictionary for saving"""
        # Convert TensorDict to regular dict for serialization
        metadata_dict = {}
        if not self.metadata.is_empty():
            for key in self.metadata.keys():
                metadata_dict[key] = self.metadata[key].cpu()
        
        return {
            'observations': self.observations.cpu(),
            'latents': self.latents.cpu(),
            'metadata': metadata_dict,
            'obs_shape': self.obs_shape,
            'latent_dim': self.latent_dim,
            'dtype': self.dtype
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load state from dictionary"""
        self.obs_shape = state_dict['obs_shape']
        self.latent_dim = state_dict['latent_dim']
        self.dtype = state_dict['dtype']
        
        self.observations = state_dict['observations'].to(device=self.device, dtype=self.dtype)
        self.latents = state_dict['latents'].to(device=self.device, dtype=self.dtype)
        
        # Reconstruct TensorDict from regular dict
        metadata_dict = state_dict['metadata']
        if metadata_dict:
            # Move each metadata tensor to device
            metadata_on_device = {k: v.to(device=self.device) for k, v in metadata_dict.items()}
            batch_size = len(self.observations)
            self.metadata = TensorDict(metadata_on_device, batch_size=batch_size, device=self.device)
        else:
            self.metadata = TensorDict({}, batch_size=0, device=self.device)
