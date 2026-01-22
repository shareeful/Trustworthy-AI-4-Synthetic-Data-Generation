"""
Hybrid VAE-GAN architecture combining strengths of both models.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict
import logging

from .vae import VAE
from .gan import Discriminator

logger = logging.getLogger(__name__)


class HybridVAEGAN(nn.Module):
    """
    Hybrid VAE-GAN architecture.
    
    Combines:
    - VAE's structured latent space (enabling T-AI constraint enforcement)
    - GAN's adversarial training (ensuring data quality)
    """
    
    def __init__(self, input_dim: int, vae_config: dict, gan_config: dict):
        """
        Initialize hybrid model.
        
        Args:
            input_dim: Input feature dimension
            vae_config: VAE configuration
            gan_config: GAN configuration
        """
        super(HybridVAEGAN, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = vae_config['latent_dim']
        
        # VAE components (encoder + decoder/generator)
        self.vae = VAE(input_dim, vae_config)
        
        # GAN discriminator
        self.discriminator = Discriminator(
            input_dim=input_dim,
            hidden_dims=gan_config['discriminator']['layers'],
            dropout=gan_config['discriminator'].get('dropout', 0.3)
        )
        
        logger.info(f"Initialized Hybrid VAE-GAN: {input_dim}D input, {self.latent_dim}D latent")
        logger.info("Architecture combines VAE reconstruction + GAN adversarial training")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hybrid model.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Dictionary with keys: x_recon, mu, logvar, real_validity, fake_validity
        """
        # VAE forward pass
        x_recon, mu, logvar = self.vae(x)
        
        # Discriminator evaluations
        real_validity = self.discriminator(x)
        fake_validity = self.discriminator(x_recon.detach())
        
        return {
            'x_recon': x_recon,
            'mu': mu,
            'logvar': logvar,
            'real_validity': real_validity,
            'fake_validity': fake_validity
        }
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.vae.encode(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent code to reconstruction."""
        return self.vae.decode(z)
    
    def generate(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate synthetic samples from prior.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate on
            
        Returns:
            Generated samples [num_samples, input_dim]
        """
        return self.vae.sample(num_samples, device)
    
    def compute_losses(
        self, 
        x: torch.Tensor, 
        outputs: Dict[str, torch.Tensor],
        loss_weights: Dict[str, float]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components.
        
        Args:
            x: Original input
            outputs: Forward pass outputs
            loss_weights: Dictionary of loss weights (λ values)
            
        Returns:
            Dictionary of individual loss components
        """
        x_recon = outputs['x_recon']
        mu = outputs['mu']
        logvar = outputs['logvar']
        real_validity = outputs['real_validity']
        fake_validity = outputs['fake_validity']
        
        # VAE losses
        recon_loss = self.vae.reconstruction_loss(x, x_recon)
        kl_loss = self.vae.kl_divergence(mu, logvar)
        
        # GAN losses
        criterion = nn.BCELoss()
        real_labels = torch.ones_like(real_validity)
        fake_labels = torch.zeros_like(fake_validity)
        
        d_real_loss = criterion(real_validity, real_labels)
        d_fake_loss = criterion(fake_validity, fake_labels)
        gan_loss = (d_real_loss + d_fake_loss) / 2
        
        # Generator adversarial loss (wants discriminator to classify recon as real)
        fake_validity_gen = self.discriminator(x_recon)
        gen_loss = criterion(fake_validity_gen, real_labels)
        
        # L1 sparsity loss on latent codes (for explainability)
        l1_loss = torch.mean(torch.abs(mu))
        
        losses = {
            'recon': recon_loss,
            'kl': kl_loss,
            'gan': gan_loss,
            'gen': gen_loss,
            'l1': l1_loss
        }
        
        return losses


if __name__ == "__main__":
    # Test Hybrid VAE-GAN
    import yaml
    logging.basicConfig(level=logging.INFO)
    
    with open("config/hyperparameters.yaml") as f:
        config = yaml.safe_load(f)
    
    # Create hybrid model
    input_dim = 11
    model = HybridVAEGAN(input_dim, config['vae'], config['gan'])
    
    # Test forward pass
    batch = torch.randn(32, input_dim)
    outputs = model(batch)
    
    print("\nForward pass outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    # Test loss computation
    loss_weights = {
        'recon': 1.0,
        'kl': 0.1,
        'gan': 2.0,
        'l1': 1.0
    }
    losses = model.compute_losses(batch, outputs, loss_weights)
    
    print("\nLoss components:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    # Test generation
    samples = model.generate(10, torch.device('cpu'))
    print(f"\nGenerated samples shape: {samples.shape}")