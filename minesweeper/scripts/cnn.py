import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import os
import yaml

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

class CustomFeaturesExtractor(BaseFeaturesExtractor):
    """
    CNN dla Minesweeper 
    skupiona na rozpoznawaniu wzorców na planszy grid.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim=512):
        # We need to pass the full observation space so SB3 knows what to expect
        super().__init__(observation_space, features_dim)
        
        cnn_config = config['model'].get('cnn_architecture', {})
        cnn_channels = cnn_config.get('cnn_channels', [64, 128, 256])
        cnn_output_dim = cnn_config.get('cnn_output_dim', 512)
        use_layernorm = cnn_config.get('use_layernorm', True)
        dropout = cnn_config.get('dropout', 0.0)
        
        # 3 Channels from environment (Fog, Values, Valid)
        in_channels = 3
        
        layers = []
        for out_channels in cnn_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if use_layernorm:
                layers.append(nn.BatchNorm2d(out_channels)) # BatchNorm is better for CNNs than LayerNorm usually
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            # MaxPool zmniejsza wymiar i zwiększa pole widzenia
            layers.append(nn.MaxPool2d(2)) 
            in_channels = out_channels
            
        self.cnn = nn.Sequential(*layers)
        
        # Calculate Flatten dimension
        # Input: [1, 16, 16] (Max grid size)
        # Layer 1: [64, 8, 8]
        # Layer 2: [128, 4, 4]
        # Layer 3: [256, 2, 2] -> 256 * 4 = 1024 flat
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, config['environment']['max_grid_size'], config['environment']['max_grid_size'])
            cnn_out = self.cnn(dummy_input)
            self.flatten_dim = cnn_out.view(1, -1).shape[1]
            
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # observations['image'] is [Batch, 3, H, W]
        x = observations['image']
        
        # Values are already normalized in environment (0.0 - 1.0)
        # So we just pass them through
        
        features = self.cnn(x)
        return self.linear(features)
