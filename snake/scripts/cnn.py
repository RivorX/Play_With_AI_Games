import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import yaml
from gymnasium import spaces
import os

base_dir = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim=128):
        super().__init__(observation_space, features_dim)
        
        convlstm_config = config['model']['convlstm']
        in_channels = 1
        dropout_rate = config['model'].get('dropout_rate', 0.0)
        
        cnn_layers = []
        current_channels = in_channels
        
        # Warstwa 1: 16x16x1 -> 8x8x32
        cnn_layers.extend([
            nn.Conv2d(current_channels, convlstm_config['cnn_channels'][0], 
                     kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(convlstm_config['cnn_channels'][0]),
            nn.ReLU(inplace=True)
        ])
        current_channels = convlstm_config['cnn_channels'][0]
        
        # Warstwa 2: 8x8x32 -> 8x8x48
        cnn_layers.extend([
            nn.Conv2d(current_channels, convlstm_config['cnn_channels'][1], 
                     kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(convlstm_config['cnn_channels'][1]),
            nn.ReLU(inplace=True)
        ])
        current_channels = convlstm_config['cnn_channels'][1]
        
        # Warstwa 3: 8x8x48 -> 4x4x64 (NOWA!)
        cnn_layers.extend([
            nn.Conv2d(current_channels, convlstm_config['cnn_channels'][2], 
                     kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(convlstm_config['cnn_channels'][2]),
            nn.ReLU(inplace=True)
        ])
        current_channels = convlstm_config['cnn_channels'][2]
        
        cnn_layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Skalary
        scalar_dim = 6
        scalar_layers = [
            nn.Linear(scalar_dim, convlstm_config['scalar_hidden_dims'][0]),
            nn.ReLU(inplace=True)
        ]
        self.scalar_linear = nn.Sequential(*scalar_layers)
        
        # Wymiary: 4*4*40 = 640 (zmniejszone z 1024 dla lepszej równowagi)
        cnn_out_channels = convlstm_config['cnn_channels'][-1]
        spatial_size = 4
        cnn_dim = cnn_out_channels * spatial_size * spatial_size
        scalar_out_dim = convlstm_config['scalar_hidden_dims'][0]
        total_dim = cnn_dim + scalar_out_dim
        
        final_layers = [
            nn.Linear(total_dim, features_dim),
            nn.ReLU(inplace=True)
        ]
        if dropout_rate > 0:
            final_layers.append(nn.Dropout(dropout_rate))
        
        self.final_linear = nn.Sequential(*final_layers)
        
        scalar_percent = 100 * scalar_out_dim / total_dim
        print(f"✓ CNN: 16x16 -> 8x8 -> 8x8 -> 4x4 ({cnn_dim} dims + {scalar_out_dim} scalars = {total_dim} → {features_dim})")
        print(f"  Skalary mają {scalar_percent:.1f}% wpływu (direction, dx/dy, collisions)")
        
    def forward(self, observations):
        image = observations['image']
        
        if len(image.shape) == 4:
            image = image.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        image_features = self.cnn(image)
        
        scalars = torch.cat([
            observations['direction'],
            observations['dx_head'],
            observations['dy_head'],
            observations['front_coll'],
            observations['left_coll'],
            observations['right_coll']
        ], dim=-1)
        
        scalar_features = self.scalar_linear(scalars)
        features = torch.cat([image_features, scalar_features], dim=-1)
        return self.final_linear(features)