import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
        )
        self.mu_layer  = nn.Linear(32, latent_dim)
        self.log_var_layer = nn.Linear(32, latent_dim)

    def forward(self, x):
        h = self.net(x)
        mu = self.mu_layer(h)
        log_var = self.log_var_layer(h)
        return mu, log_var


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 16, output_dim: int = 11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class VAEGAN(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder     = Encoder(input_dim, latent_dim)
        self.generator   = Generator(latent_dim, input_dim)
        self.discriminator = Discriminator(input_dim)
        self.latent_dim  = latent_dim

    def reparameterise(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterise(mu, log_var)
        x_hat = self.generator(z)
        return x_hat, mu, log_var, z

    def sample(self, n: int, device: torch.device):
        z = torch.randn(n, self.latent_dim).to(device)
        return self.generator(z)
