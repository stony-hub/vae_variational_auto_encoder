import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE_network(nn.Module):
    """
    In this module logv means 2 * log(sigma) for convenience.
    """
    def __init__(self, x_dim, h1_dim, h2_dim, z_dim):
        super(VAE_network, self).__init__()

        self.Q1 = nn.Sequential(
            nn.Linear(x_dim, h1_dim),
            nn.ReLU(),
            nn.Linear(h1_dim, h2_dim),
            nn.ReLU(),
        )
        self.Q_mu = nn.Linear(h2_dim, z_dim)
        self.Q_logv = nn.Linear(h2_dim, z_dim)
        self.P = nn.Sequential(
            nn.Linear(z_dim, h2_dim),
            nn.ReLU(),
            nn.Linear(h2_dim, h1_dim),
            nn.ReLU(),
            nn.Linear(h1_dim, x_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.Q1(x)
        return self.Q_mu(h), self.Q_logv(h)

    def sample(self, mu, logv):
        std = torch.exp(0.5 * logv)
        noise = torch.randn_like(mu).to(std.device)
        return mu + noise * std

    def decode(self, z):
        return self.P(z)

    def forward(self, x):
        mu, logv = self.encode(x)
        z = self.sample(mu, logv)
        re_x = self.decode(z)
        return re_x, (mu, logv)

    def loss(self, x, re_x, mu, logv):
        recon_err = F.binary_cross_entropy(re_x, x)
        kl_div = 0.5 * (logv + 1 - mu.pow(2) - logv.exp()).mean()
        return recon_err + kl_div
