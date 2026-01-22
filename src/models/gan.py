"""
Generative Adversarial Network (GAN) implementation.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class Generator(nn.Module):
    """GAN Generator: creates synthetic data from latent noise."""
    
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int):
        """
        Initialize generator.
        
        Args:
            latent_dim: Input noise dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output feature dimension
        """
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Build generator layers
        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.generator = nn.Sequential(*layers)
        self.fc_out = nn.Linear(prev_dim, output_dim)
        
        logger.info(f"Generator: {latent_dim} → {hidden_dims} → {output_dim}")
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate synthetic data from noise.
        
        Args:
            z: Latent noise [batch_size, latent_dim]
            
        Returns:
            Generated data [batch_size, output_dim]
        """
        h = self.generator(z)
        x_fake = torch.sigmoid(self.fc_out(h))  # Bounded to [0, 1]
        return x_fake


class Discriminator(nn.Module):
    """GAN Discriminator: distinguishes real from fake data."""
    
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.3):
        """
        Initialize discriminator.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super(Discriminator, self).__init__()
        
        self.input_dim = input_dim
        
        # Build discriminator layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.discriminator = nn.Sequential(*layers)
        self.fc_out = nn.Linear(prev_dim, 1)
        
        logger.info(f"Discriminator: {input_dim} → {hidden_dims} → 1")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify data as real or fake.
        
        Args:
            x: Input data [batch_size, input_dim]
            
        Returns:
            Probability of being real [batch_size, 1]
        """
        h = self.discriminator(x)
        validity = torch.sigmoid(self.fc_out(h))
        return validity


class GAN(nn.Module):
    """Complete Generative Adversarial Network."""
    
    def __init__(self, input_dim: int, config: dict):
        """
        Initialize GAN.
        
        Args:
            input_dim: Input feature dimension
            config: GAN configuration from hyperparameters.yaml
        """
        super(GAN, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = config.get('latent_dim', 16)
        
        self.generator = Generator(
            latent_dim=self.latent_dim,
            hidden_dims=config['generator']['layers'],
            output_dim=input_dim
        )
        
        self.discriminator = Discriminator(
            input_dim=input_dim,
            hidden_dims=config['discriminator']['layers'],
            dropout=config['discriminator'].get('dropout', 0.3)
        )
        
        logger.info(f"Initialized GAN: {self.latent_dim}D latent → {input_dim}D output")
    
    def generate(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate synthetic samples.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate on
            
        Returns:
            Generated samples [num_samples, input_dim]
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        with torch.no_grad():
            samples = self.generator(z)
        return samples
    
    def discriminator_loss(self, real_validity: torch.Tensor, fake_validity: torch.Tensor) -> torch.Tensor:
        """
        Calculate discriminator loss: maximize log(D(x)) + log(1 - D(G(z)))
        
        Args:
            real_validity: D(x_real)
            fake_validity: D(G(z))
            
        Returns:
            Discriminator loss
        """
        criterion = nn.BCELoss()
        real_labels = torch.ones_like(real_validity)
        fake_labels = torch.zeros_like(fake_validity)
        
        real_loss = criterion(real_validity, real_labels)
        fake_loss = criterion(fake_validity, fake_labels)
        
        return (real_loss + fake_loss) / 2
    
    def generator_loss(self, fake_validity: torch.Tensor) -> torch.Tensor:
        """
        Calculate generator loss: maximize log(D(G(z)))
        
        Args:
            fake_validity: D(G(z))
            
        Returns:
            Generator loss
        """
        criterion = nn.BCELoss()
        real_labels = torch.ones_like(fake_validity)  # Generator wants D to think fake is real
        return criterion(fake_validity, real_labels)


if __name__ == "__main__":
    # Test GAN
    import yaml
    logging.basicConfig(level=logging.INFO)
    
    with open("config/hyperparameters.yaml") as f:
        config = yaml.safe_load(f)
    
    # Create GAN
    input_dim = 11
    gan_config = config['gan']
    gan_config['latent_dim'] = 16
    gan = GAN(input_dim, gan_config)
    
    # Test generator
    z = torch.randn(32, 16)
    x_fake = gan.generator(z)
    print(f"Generated data shape: {x_fake.shape}")
    
    # Test discriminator
    validity = gan.discriminator(x_fake)
    print(f"Discriminator output shape: {validity.shape}")
    
    # Test sampling
    samples = gan.generate(10, torch.device('cpu'))
    print(f"Sampled data shape: {samples.shape}")