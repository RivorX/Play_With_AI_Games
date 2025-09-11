import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import yaml
import os

# Wczytaj konfigurację
base_dir = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

class CustomCNN(BaseFeaturesExtractor):
    """
    Niestandardowa sieć CNN do ekstrakcji cech z mapy gry o stałym rozmiarze 16x16.
    Wejście: (batch_size, stack_size, 16, 16, channels) - mapa z 4 ramkami x 7 kanałami.
    Wyjście: wektor cech o wymiarze features_dim.
    """
    def __init__(self, observation_space, features_dim=512):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        in_channels = 4 * 6  # stack_size = 4, channels = 6 (mapa, dx, dy, kierunek, grid_size, odległość)
        dropout_rate = config['model'].get('dropout_rate', 0.2)  # Pobierz dropout_rate z configu
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            processed = self._process_sample(sample)
            n_flatten = self.cnn(processed).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def _process_sample(self, observations):
        batch_size = observations.shape[0]
        observations = observations.permute(0, 1, 4, 2, 3)  # (batch_size, stack_size, C, H, W)
        observations = observations.reshape(batch_size, -1, observations.shape[3], observations.shape[4])
        return observations

    def forward(self, observations):
        processed = self._process_sample(observations)
        return self.linear(self.cnn(processed))