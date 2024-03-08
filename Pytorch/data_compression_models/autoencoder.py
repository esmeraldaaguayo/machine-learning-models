import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any


class Encoder(nn.Module):
    def __init__(
            self,
            input_dims: int,
            hidden_dims: int,
            latent_dims: int,
    ) -> None:
        super(Encoder, self).__init__()
        self.fc_linear1 = nn.Linear(input_dims, hidden_dims)
        self.fc_linear2 = nn.Linear(hidden_dims, latent_dims)

    def forward(self, x: Any) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        h = F.relu(self.fc_linear1(x))
        z = self.fc_linear2(h)
        return z


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
        return x.reshape((-1, 1, 28, 28))


class Autoencoder(nn.Module):
    def __init__(
            self,
            input_dims: int,
            hidden_dims: int,
            latent_dims: int,
    ) -> None:
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dims, hidden_dims, latent_dims)
        self.decoder = Decoder(latent_dims, hidden_dims, input_dims)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
