from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch

class CustomCNN(BaseFeaturesExtractor):
    """
    Niestandardowa sieć CNN do ekstrakcji cech z mapy gry o dowolnym rozmiarze.
    Wejście: (batch_size, 3, grid_size, grid_size) - mapa z 3 kanałami (gra, kierunek, odległość).
    Wyjście: wektor cech o wymiarze features_dim.
    """
    def __init__(self, observation_space, features_dim=512):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 3 kanały wejściowe
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Redukuje do 4x4
            nn.Flatten(),
        )
        # Wymiar po flatten: 128 * 4 * 4 = 2048
        self.linear = nn.Sequential(
            nn.Linear(128 * 4 * 4, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # Transpozycja z NHWC do NCHW
        observations = observations.permute(0, 3, 1, 2)
        return self.linear(self.cnn(observations))