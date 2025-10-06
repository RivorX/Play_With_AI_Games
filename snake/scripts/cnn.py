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
    Features Extractor dla RecurrentPPO z dynamiczną architekturą
    Liczba warstw CNN i scalar MLP jest konfigurowalna przez config.yaml
    """
    def __init__(self, observation_space: spaces.Dict, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        # Pobierz parametry z configu
        cnn_config = config['model'].get('convlstm', {})
        cnn_channels = cnn_config.get('cnn_channels', [32, 64])
        scalar_hidden_dims = cnn_config.get('scalar_hidden_dims', [128, 192])
        dropout_rate = config['model'].get('dropout_rate', 0.2)
        
        in_channels = 1
        leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # === DYNAMICZNE BUDOWANIE CNN ===
        cnn_layers = []
        current_channels = in_channels
        
        for i, out_channels in enumerate(cnn_channels):
            # Pierwsza i ostatnia warstwa zmniejszają rozmiar (stride=2), pozostałe nie
            if i == 0 or i == len(cnn_channels) - 1:
                stride = 2
            else:
                stride = 1
            cnn_layers.extend([
                nn.Conv2d(current_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                leaky_relu
            ])
            current_channels = out_channels
        
        # Adaptive pooling na końcu CNN
        cnn_layers.extend([
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten()
        ])
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Oblicz wymiar wyjściowy CNN (ostatni kanał * 2 * 2)
        self.cnn_output_dim = cnn_channels[-1] * 2 * 2

        # === DYNAMICZNE BUDOWANIE SIECI SKALARÓW ===
        # Skalary: direction (2D sin/cos), dx_head, dy_head, front_coll, left_coll, right_coll
        scalar_input_dim = 2 + 1 + 1 + 1 + 1 + 1  # 7 wartości
        
        scalar_layers = []
        current_dim = scalar_input_dim
        
        for i, hidden_dim in enumerate(scalar_hidden_dims):
            scalar_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LeakyReLU(0.01)
            ])
            # Dropout po każdej warstwie oprócz ostatniej
            if i < len(scalar_hidden_dims) - 1:
                scalar_layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        
        # Dropout na końcu sieci skalarów
        scalar_layers.append(nn.Dropout(dropout_rate))
        
        self.scalar_linear = nn.Sequential(*scalar_layers)
        self.scalar_output_dim = scalar_hidden_dims[-1]

        # === WARSTWA ŁĄCZĄCA ===
        total_dim = self.cnn_output_dim + self.scalar_output_dim
        self.final_linear = nn.Sequential(
            nn.Linear(total_dim, features_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate)
        )
        
        print(f"[CNN] Zainicjalizowano architekturę:")
        print(f"  CNN layers: {cnn_channels} -> output_dim: {self.cnn_output_dim}")
        print(f"  Scalar layers: {scalar_hidden_dims} -> output_dim: {self.scalar_output_dim}")
        print(f"  Total features: {total_dim} -> final: {features_dim}")
        scalar_percent = (self.scalar_output_dim / total_dim) * 100
        print(f"  Skalary mają {scalar_percent:.1f}% wpływu")

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