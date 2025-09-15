import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import yaml
from gymnasium import spaces
import os

# Wczytaj konfigurację
base_dir = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, ch, _, _ = x.shape
        y = self.squeeze(x).view(batch, ch)
        y = self.excitation(y).view(batch, ch, 1, 1)
        return x * y.expand_as(x)

class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim=512):
        super().__init__(observation_space, features_dim)
        in_channels = 4 * 1  # 4 ramki × 1 kanał (mapa)
        dropout_rate = config['model'].get('dropout_rate', 0.2)
        leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        class ResidualBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
                self.bn1 = nn.BatchNorm2d(channels)
                self.act = nn.LeakyReLU(0.01)
                self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
                self.bn2 = nn.BatchNorm2d(channels)
            def forward(self, x):
                identity = x
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.act(out)
                out = self.conv2(out)
                out = self.bn2(out)
                out += identity
                out = self.act(out)
                return out

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            leaky_relu,
            SEBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            leaky_relu,
            ResidualBlock(64),
            SEBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            leaky_relu,
            SEBlock(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            leaky_relu,
            SEBlock(256),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        scalar_dim = 4  # direction, grid_size, dx_head, dy_head
        self.scalar_linear = nn.Sequential(
            nn.Linear(scalar_dim, 32),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate)
        )

        cnn_dim = 256 * 4 * 4
        total_dim = cnn_dim + 32
        self.final_linear = nn.Sequential(
            nn.Linear(total_dim, features_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate)
        )

    def forward(self, observations):
        image = observations['image']
        # Upewnij się, że obraz jest w formacie [batch, channels, height, width]
        if image.shape[1] != 4:  # Sprawdzenie, czy kanały są pierwsze
            image = image.permute(0, 3, 1, 2)  # [batch, H, W, C] -> [batch, C, H, W]
        scalars = torch.cat([
            observations['direction'],
            observations['grid_size'],
            observations['dx_head'],
            observations['dy_head']
        ], dim=-1)

        image_features = self.cnn(image)
        scalar_features = self.scalar_linear(scalars)
        features = torch.cat([image_features, scalar_features], dim=-1)
        return self.final_linear(features)