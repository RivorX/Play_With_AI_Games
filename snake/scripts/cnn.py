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
    ZOPTYMALIZOWANY Features Extractor dla RecurrentPPO
    - Używa kanałów z configu
    - Pierwsza warstwa: kernel=5 dla lepszego receptive field
    - Druga i trzecia: kernel=3, stride=2
    - Zachowuje więcej informacji przestrzennej niż poprzednia wersja
    """
    def __init__(self, observation_space: spaces.Dict, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        # Pobierz konfigurację z pliku
        cnn_channels = config['model']['convlstm']['cnn_channels']
        scalar_hidden_dims = config['model']['convlstm']['scalar_hidden_dims']
        dropout_rate = config['model'].get('dropout_rate', 0.1)
        
        in_channels = 1
        
        # ULEPSZONA ARCHITEKTURA CNN
        # Warstwa 1: kernel=5, stride=1, padding=2 → 16x16 (zachowuje rozmiar)
        # Warstwa 2: kernel=3, stride=2, padding=1 → 8x8
        # Warstwa 3: kernel=3, stride=2, padding=1 → 4x4 (więcej niż poprzednie 2x2!)
        cnn_layers = []
        prev_channels = in_channels
        
        for i, out_channels in enumerate(cnn_channels):
            if i == 0:
                # Pierwsza warstwa: większy kernel, bez stride
                cnn_layers.extend([
                    nn.Conv2d(prev_channels, out_channels, kernel_size=5, stride=1, padding=2),
                    nn.LeakyReLU(0.01, inplace=True),
                    nn.Dropout2d(dropout_rate)
                ])
            else:
                # Pozostałe warstwy: kernel=3, stride=2
                cnn_layers.extend([
                    nn.Conv2d(prev_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(0.01, inplace=True),
                    nn.Dropout2d(dropout_rate) if i < len(cnn_channels) - 1 else nn.Identity()
                ])
            prev_channels = out_channels
        
        cnn_layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Oblicz wymiar wyjściowy CNN
        # Po nowej architekturze: 16 (start) -> 16 (w1) -> 8 (w2) -> 4 (w3)
        cnn_output_size = 4 * 4  # finalna wielkość przestrzenna
        cnn_dim = cnn_channels[-1] * cnn_output_size
        
        # ZOPTYMALIZOWANA sieć dla skalarów
        scalar_dim = 2 + 1 + 1 + 1 + 1 + 1  # 7 wartości
        scalar_layers = []
        prev_dim = scalar_dim
        
        for hidden_dim in scalar_hidden_dims:
            scalar_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.scalar_linear = nn.Sequential(*scalar_layers)
        scalar_output_dim = scalar_hidden_dims[-1]
        
        # Łączenie cech - prosta warstwa bez nadmiernej komplikacji
        total_dim = cnn_dim + scalar_output_dim
        self.final_linear = nn.Sequential(
            nn.Linear(total_dim, features_dim),
            nn.LeakyReLU(0.01, inplace=True)
        )
        
        print(f"[CNN] Inicjalizacja: CNN channels={cnn_channels}, CNN output dim={cnn_dim}")
        print(f"[CNN] Scalar hidden dims={scalar_hidden_dims}, Scalar output dim={scalar_output_dim}")
        print(f"[CNN] Total features dim={total_dim} -> {features_dim}")
        print(f"[CNN] Scalery stanowią {scalar_dim/total_dim*100:.1f}% wejścia")

    def forward(self, observations):
        # Obraz: [batch, H, W, C] -> [batch, C, H, W]
        image = observations['image']
        if image.dim() == 4 and image.shape[-1] == 1:
            image = image.permute(0, 3, 1, 2)
        
        # Skalary (direction jest 2D)
        scalars = torch.cat([
            observations['direction'],
            observations['dx_head'],
            observations['dy_head'],
            observations['front_coll'],
            observations['left_coll'],
            observations['right_coll']
        ], dim=-1)

        # Przetwórz przez sieci
        image_features = self.cnn(image)
        scalar_features = self.scalar_linear(scalars)
        
        # Połącz cechy i zwróć
        features = torch.cat([image_features, scalar_features], dim=-1)
        return self.final_linear(features)