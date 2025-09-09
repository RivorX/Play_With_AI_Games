import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    """
    Niestandardowa sieć CNN do ekstrakcji cech z mapy gry o zmiennym rozmiarze.
    Wejście: (batch_size, stack_size, grid_size, grid_size, channels) - mapa z 4 ramkami x 4 kanałami.
    Wyjście: wektor cech o wymiarze features_dim.
    """
    def __init__(self, observation_space, features_dim=512):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # Liczba kanałów po połączeniu ramki x kanały
        in_channels = 4 * 4  # stack_size = 4, channels = 4
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Redukuje do 4x4 niezależnie od rozmiaru wejścia
            nn.Flatten(),
        )
        # Oblicz wymiar po flatten
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            processed = self._process_sample(sample)
            n_flatten = self.cnn(processed).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def _process_sample(self, observations):
        # Transpozycja z (batch_size, stack_size, H, W, C) do (batch_size, stack_size * C, H, W)
        batch_size = observations.shape[0]
        observations = observations.permute(0, 1, 4, 2, 3)  # (batch_size, stack_size, C, H, W)
        observations = observations.reshape(batch_size, -1, observations.shape[3], observations.shape[4])  # (batch_size, stack_size*C, H, W)
        return observations

    def forward(self, observations):
        processed = self._process_sample(observations)
        return self.linear(self.cnn(processed))