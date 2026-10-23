import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict


class VariationalInformationBottleneck(nn.Module):

    def __init__(self, input_dim, bottleneck_dim, beta=0.001):
        super(VariationalInformationBottleneck, self).__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.beta = beta
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )
        self.fc_mu = nn.Linear(256, bottleneck_dim)
        self.fc_logvar = nn.Linear(256, bottleneck_dim)
        self.task_projector = nn.Linear(bottleneck_dim, input_dim)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        original_shape = x.shape
        x_flat = x.view(-1, self.input_dim)
        h = self.encoder(x_flat)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        new_shape = list(original_shape[:-1]) + [self.bottleneck_dim]
        z = z.view(new_shape)
        mu = mu.view(new_shape)
        logvar = logvar.view(new_shape)
        return (z, mu, logvar)

    def project_for_task(self, z: Tensor) -> Tensor:
        original_shape = z.shape
        z_flat = z.view(-1, self.bottleneck_dim)
        projected = self.task_projector(z_flat)
        return projected.view(original_shape[:-1] + (self.input_dim,))

    def compute_vib_loss(self, mu: Tensor, logvar: Tensor) -> Dict[str, Tensor]:
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_div = kl_div / mu.size(0)
        vib_loss = self.beta * kl_div
        return {'total_loss': vib_loss, 'kl_loss': kl_div, 'vib_loss': vib_loss}


class GridVIB(nn.Module):

    def __init__(self, input_dim, bottleneck_dim, beta=0.001, spatial_pooling='mean'):
        super(GridVIB, self).__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.beta = beta
        self.spatial_pooling = spatial_pooling
        self.vib = VariationalInformationBottleneck(input_dim, bottleneck_dim, beta)
        self.projection = nn.Linear(bottleneck_dim, input_dim)

    def forward(self, grid_features: Tensor, grid_mask: Tensor = None) -> Tuple[Tensor, dict]:
        batch_size, seq_len, _, feat_dim = grid_features.shape
        z, mu, logvar = self.vib(grid_features)
        loss_dict = self.vib.compute_vib_loss(mu, logvar)
        compressed_features = self.projection(z)
        if grid_mask is not None:
            compressed_features = compressed_features * grid_mask.unsqueeze(-1)
        return (compressed_features, loss_dict)
