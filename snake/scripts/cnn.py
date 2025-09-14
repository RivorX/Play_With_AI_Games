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

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block: lekka attention na kanałach.
    Wejście: (batch, channels, H, W)
    Wyjście: to samo, ale z ważonymi kanałami.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)  # Global average pooling: ściska HxW do 1x1
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()  # Wagi między 0-1
        )

    def forward(self, x):
        batch, ch, _, _ = x.shape
        y = self.squeeze(x).view(batch, ch)  # Squeeze: średnia po przestrzennych wymiarach
        y = self.excitation(y).view(batch, ch, 1, 1)  # Excitation: oblicz wagi
        return x * y.expand_as(x)  # Mnożymy wagi przez oryginalne feature maps

class CustomCNN(BaseFeaturesExtractor):
    """
    Niestandardowa sieć CNN do ekstrakcji cech z mapy gry o stałym rozmiarze 16x16.
    Wejście: (batch_size, stack_size, 16, 16, channels) - mapa z 4 ramkami x 7 kanałami.
    Wyjście: wektor cech o wymiarze features_dim.
    """
    def __init__(self, observation_space, features_dim=512):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # stack_size = 4, channels = 10 (6 podstawowych + 4 historia kierunków)
        in_channels = 4 * 10
        dropout_rate = config['model'].get('dropout_rate', 0.2)  # Pobierz dropout_rate z configu
        leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # Blok residualny (ResNet-like) po 2 warstwie konwolucyjnej
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
            SEBlock(32),  # Dodano SEBlock po pierwszej warstwie konwolucyjnej
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            leaky_relu,
            ResidualBlock(64),
            SEBlock(64),  # Dodano SEBlock po ResidualBlock
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            leaky_relu,
            SEBlock(128),  # Dodano SEBlock po trzeciej warstwie konwolucyjnej
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            leaky_relu,
            SEBlock(256),  # Dodano SEBlock po czwartej warstwie konwolucyjnej
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            processed = self._process_sample(sample)
            n_flatten = self.cnn(processed).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(features_dim, features_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate)
        )

    def _process_sample(self, observations):
        batch_size = observations.shape[0]
        observations = observations.permute(0, 1, 4, 2, 3)  # (batch_size, stack_size, C, H, W)
        observations = observations.reshape(batch_size, -1, observations.shape[3], observations.shape[4])
        return observations

    def forward(self, observations):
        # Użyj AMP (bfloat16 jeśli dostępne), ale zawsze zwracaj float32
        use_amp = torch.is_autocast_enabled() or (torch.cuda.is_available() and observations.is_cuda)
        # Sprawdź czy karta obsługuje bfloat16
        use_bfloat16 = False
        if use_amp and torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            # Ampere (8.0+) i nowsze obsługują bfloat16
            if cap[0] >= 8:
                use_bfloat16 = True
        if use_amp:
            dtype = torch.bfloat16 if use_bfloat16 else torch.float16
            with torch.amp.autocast(device_type='cuda', dtype=dtype):
                processed = self._process_sample(observations)
                out = self.linear(self.cnn(processed))
                return out.to(torch.float32)
        else:
            processed = self._process_sample(observations)
            out = self.linear(self.cnn(processed))
            return out.to(torch.float32)