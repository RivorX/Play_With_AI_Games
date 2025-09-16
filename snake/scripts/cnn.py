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

class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim=512):
        super().__init__(observation_space, features_dim)
        in_channels = 4  # 4 ramki × 1 kanał (mapa)
        dropout_rate = config['model'].get('dropout_rate', 0.2)
        leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            leaky_relu,
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            leaky_relu,
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten()
        )

        scalar_dim = 7  # direction, grid_size, dx_head, dy_head, front_coll, left_coll, right_coll
        self.scalar_linear = nn.Sequential(
            nn.Linear(scalar_dim, 128),  # Zwiększono z 64 do 128
            nn.LeakyReLU(0.01),
            nn.Linear(128, 256),  # Zwiększono z 128 do 256
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate)
        )

        cnn_dim = 64 * 2 * 2  # 256 cech
        total_dim = cnn_dim + 256  # 256 (CNN) + 256 (scalars) = 512
        self.final_linear = nn.Sequential(
            nn.Linear(total_dim, features_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate)
        )

    def forward(self, observations):
        image = observations['image']
        if image.shape[1] != 4:  # Sprawdzenie, czy kanały są pierwsze
            image = image.permute(0, 3, 1, 2)  # [batch, H, W, C] -> [batch, C, H, W]
        if image.requires_grad:
            image.retain_grad()
        scalars = torch.cat([
            observations['direction'],
            observations['grid_size'],
            observations['dx_head'],
            observations['dy_head'],
            observations['front_coll'],
            observations['left_coll'],
            observations['right_coll']
        ], dim=-1)

        image_features = self.cnn(image)
        scalar_features = self.scalar_linear(scalars)
        features = torch.cat([image_features, scalar_features], dim=-1)
        return self.final_linear(features)