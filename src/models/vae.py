"""
Variational Autoencoder (VAE) implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    """VAE Encoder: maps input to latent distribution."""
    
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int, dropout: float = 0.0):
        """
        Initialize encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions (e.g., [128, 64, 32])
            latent_dim: Latent space dimension
            dropout: Dropout probability
        """
        super(Encoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Latent distribution parameters
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        logger.info(f"Encoder: {input_dim} → {hidden_dims} → μ,σ({latent_dim})")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Tuple of (mu, logvar)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """VAE Decoder: reconstructs input from latent code."""
    
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int):
        """
        Initialize decoder.
        
        Args:
            latent_dim: Latent space dimension
            hidden_dims: List of hidden layer dimensions (e.g., [64, 128])
            output_dim: Output feature dimension
        """
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Build decoder layers
        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*layers)
        self.fc_out = nn.Linear(prev_dim, output_dim)
        
        logger.info(f"Decoder: {latent_dim} → {hidden_dims} → {output_dim}")
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent code to reconstruction.
        
        Args:
            z: Latent tensor [batch_size, latent_dim]
            
        Returns:
            Reconstructed output [batch_size, output_dim]
        """
        h = self.decoder(z)
        x_recon = torch.sigmoid(self.fc_out(h))  # Bounded to [0, 1]
        return x_recon


class VAE(nn.Module):
    """Complete Variational Autoencoder."""
    
    def __init__(self, input_dim: int, config: dict):
        """
        Initialize VAE.
        
        Args:
            input_dim: Input feature dimension
            config: VAE configuration from hyperparameters.yaml
        """
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = config['latent_dim']
        
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=config['encoder']['layers'],
            latent_dim=self.latent_dim,
            dropout=config['encoder'].get('dropout', 0.0)
        )
        
        self.decoder = Decoder(
            latent_dim=self.latent_dim,
            hidden_dims=config['decoder']['layers'],
            output_dim=input_dim
        )
        
        logger.info(f"Initialized VAE: {input_dim}D → {self.latent_dim}D latent")
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0,1)
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent code
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent code (deterministic, using mean)."""
        mu, _ = self.encoder(x)
        return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent code to reconstruction."""
        return self.decoder(z)
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Sample from prior p(z) ~ N(0, I) and decode.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate on
            
        Returns:
            Generated samples [num_samples, input_dim]
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples
    
    def reconstruction_loss(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """
        Calculate reconstruction loss (MSE).
        
        Args:
            x: Original input
            x_recon: Reconstructed input
            
        Returns:
            MSE loss
        """
        return F.mse_loss(x_recon, x, reduction='mean')
    
    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Calculate KL divergence KL(q(z|x) || p(z)).
        
        Args:
            mu: Mean of q(z|x)
            logvar: Log variance of q(z|x)
            
        Returns:
            KL divergence
        """
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)


if __name__ == "__main__":
    # Test VAE
    import yaml
    logging.basicConfig(level=logging.INFO)
    
    with open("config/hyperparameters.yaml") as f:
        config = yaml.safe_load(f)
    
    # Create VAE
    input_dim = 11  # Insurance dataset
    vae = VAE(input_dim, config['vae'])
    
    # Test forward pass
    batch = torch.randn(32, input_dim)
    x_recon, mu, logvar = vae(batch)
    
    print(f"\nInput shape: {batch.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    
    # Test sampling
    samples = vae.sample(10, torch.device('cpu'))
    print(f"Generated samples shape: {samples.shape}")