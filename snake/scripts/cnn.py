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
        
        # Wczytaj parametry z configu
        convlstm_config = config['model']['convlstm']
        in_channels = 1  # Pojedyncza mapa (viewport zawsze 16x16)
        dropout_rate = config['model'].get('dropout_rate', 0.2)
        leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        
        # CNN do przetwarzania pojedynczych ramek (BEZ LSTM - RecurrentPPO ma własny LSTM)
        cnn_layers = []
        current_channels = in_channels
        
        # Dodaj warstwy CNN
        for layer_channels in convlstm_config['cnn_channels']:
            cnn_layers.extend([
                nn.Conv2d(current_channels, layer_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(layer_channels),
                leaky_relu
            ])
            current_channels = layer_channels
        
        cnn_layers.extend([
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten()
        ])
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Warstwa dla zmiennych skalarnych
        scalar_dim = 6  # direction, dx_head, dy_head, front_coll, left_coll, right_coll
        scalar_layers = []
        current_dim = scalar_dim
        
        for hidden_dim in convlstm_config['scalar_hidden_dims']:
            scalar_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LeakyReLU(0.01),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
        
        self.scalar_linear = nn.Sequential(*scalar_layers)
        
        # Oblicz wymiar po CNN
        cnn_out_channels = convlstm_config['cnn_channels'][-1]
        cnn_dim = cnn_out_channels * 2 * 2
        total_dim = cnn_dim + current_dim
        
        self.final_linear = nn.Sequential(
            nn.Linear(total_dim, features_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, observations):
        image = observations['image']
        
        # Obsługa formatu: [batch, height, width, channels]
        if len(image.shape) == 4:
            # Przekształć na [batch, channels, height, width]
            image = image.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}. Expected 4D tensor [batch, H, W, C]")
        
        # Przetwórz przez CNN (RecurrentPPO ma własny LSTM w policy)
        image_features = self.cnn(image)
        
        # Przetwórz zmienne skalarne
        scalars = torch.cat([
            observations['direction'],
            observations['dx_head'],
            observations['dy_head'],
            observations['front_coll'],
            observations['left_coll'],
            observations['right_coll']
        ], dim=-1)
        
        scalar_features = self.scalar_linear(scalars)
        
        # Połącz cechy
        features = torch.cat([image_features, scalar_features], dim=-1)
        return self.final_linear(features)