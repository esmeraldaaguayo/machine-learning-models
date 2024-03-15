import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any


class VariationalEncoder(nn.Module):
    def __init__(
            self,
            input_dims: int,
            hidden_dims: int,
            latent_dims: int,
    ) -> None:
        super(VariationalEncoder, self).__init__()
        self.fc_linear1 = nn.Linear(input_dims, hidden_dims)
        self.mu_linear2 = nn.Linear(hidden_dims, latent_dims)
        # this log variance, used for stability during training
        self.log_sigma_linear2 = nn.Linear(hidden_dims, latent_dims)

    def forward(self, x: Any) -> torch.Tensor:
        h = F.relu(self.fc_linear1(x))
        mu = self.mu_linear2(h)
        sigma = torch.exp(self.log_sigma_linear2(h))
        return mu, sigma


class Decoder(nn.Module):
    def __init__(
            self,
            latent_dims: int,
            hidden_dims: int,
            input_dims: int,
    ) -> None:
        super(Decoder, self).__init__()
        self.fc_linear1 = nn.Linear(latent_dims, hidden_dims)
        self.fc_linear2 = nn.Linear(hidden_dims, input_dims)

    def forward(self, z: torch.Tensor) -> Any:
        h = F.relu(self.fc_linear1(z))
        x = torch.sigmoid(self.fc_linear2(h))
        return x


class VariationalAutoencoder(nn.Module):
    def __init__(
            self,
            input_dims: int,
            hidden_dims: int,
            latent_dims: int,
    ) -> None:
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(input_dims, hidden_dims, latent_dims)
        self.decoder = Decoder(latent_dims, hidden_dims, input_dims)

    def forward(self, x):
        # encoder part
        mu, sigma = self.encoder(x)

        # sampling with reparameterization trick
        learned_dist = torch.distributions.Normal(mu, sigma)
        z = learned_dist.rsample()

        # decoder part
        x_hat = self.decoder(z)

        return mu, sigma, x_hat
