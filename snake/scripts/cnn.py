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
    """
    Features Extractor dla RecurrentPPO
    RecurrentPPO używa LSTM do zapamiętywania historii
    """
    def __init__(self, observation_space: spaces.Dict, features_dim=512):
        super().__init__(observation_space, features_dim)
        in_channels = 1
        dropout_rate = config['model'].get('dropout_rate', 0.2)
        leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # CNN dla obrazu (16x16x1)
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

        # Sieć dla skalarów: direction (2D sin/cos), dx_head, dy_head, front_coll, left_coll, right_coll
        scalar_dim = 2 + 1 + 1 + 1 + 1 + 1  # 7 wartości
        self.scalar_linear = nn.Sequential(
            nn.Linear(scalar_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 192),  # Zmieniono z 256 na 192
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate)
        )

        # Łączenie cech CNN i skalarów
        cnn_dim = 64 * 2 * 2  # 256 cech z CNN
        total_dim = cnn_dim + 192  # 256 (CNN) + 192 (scalars) = 448
        self.final_linear = nn.Sequential(
            nn.Linear(total_dim, features_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate)
        )

    def forward(self, observations):
        # Obraz: [batch, H, W, C] -> [batch, C, H, W]
        image = observations['image']
        if image.dim() == 4 and image.shape[-1] == 1:  # [batch, H, W, 1]
            image = image.permute(0, 3, 1, 2)  # [batch, 1, H, W]
        
        if image.requires_grad:
            image.retain_grad()
        
        # Skalary (direction jest 2D)
        scalars = torch.cat([
            observations['direction'],      # 2D (sin, cos)
            observations['dx_head'],        # 1D
            observations['dy_head'],        # 1D
            observations['front_coll'],     # 1D
            observations['left_coll'],      # 1D
            observations['right_coll']      # 1D
        ], dim=-1)

        # Przetwórz przez sieci
        image_features = self.cnn(image)
        scalar_features = self.scalar_linear(scalars)
        
        # Połącz cechy
        features = torch.cat([image_features, scalar_features], dim=-1)
        return self.final_linear(features)