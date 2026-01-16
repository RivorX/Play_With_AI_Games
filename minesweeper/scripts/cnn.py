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
    Prostszy CNN z dekompresją obserwacji (Optimized RAM)
    Input: 2 channels (State, Risk) -> Expanded to 16 channels (OneHot + Numeric + Risk + Needed)
    """
    def __init__(self, observation_space: spaces.Dict, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        # Internal channels: 11 OneHot + 1 Numeric + 4 Extra (Flags, LogicM, LogicS, Needed) = 16
        # Categories: 0=Pad, 1-9=Values, 10=Fog
        # Numeric: Scaled value of the number (magnitude)
        internal_channels = 16
        
        # Pobieranie struktury z configu
        cnn_conf = config['model']['cnn_architecture']
        channel_list = cnn_conf['cnn_channels'] # np. [64, 128, 256]
        use_norm = cnn_conf['use_layernorm']
        dropout_val = cnn_conf['dropout']

        # Dynamiczna budowa warstw
        layers = []
        curr_in = internal_channels
        
        for i, curr_out in enumerate(channel_list):
            # Zmniejszamy wymiar przestrzenny w głębszych warstwach
            # Apply Stride=2 for layers 2 and 3 (0-indexed) -> spatial reduction
            stride = 2 if i >= 2 else 1
            
            layers.append(nn.Conv2d(curr_in, curr_out, kernel_size=3, padding=1, stride=stride))
            
            if use_norm:
                # W CNN "LayerNorm" zazwyczaj realizuje się jako BatchNorm2d
                layers.append(nn.BatchNorm2d(curr_out))
                
            layers.append(nn.ReLU())
            
            if dropout_val > 0:
                layers.append(nn.Dropout2d(p=dropout_val))
                
            curr_in = curr_out
            
        self.cnn = nn.Sequential(*layers)
        
        # Obliczanie wymiaru wyjściowego
        max_grid_size = config['environment']['max_grid_size']
        with torch.no_grad():
            # Symulujemy już przetworzone wejście (12 kanałów)
            dummy_input = torch.zeros(1, internal_channels, max_grid_size, max_grid_size)
            n_flatten = self.cnn(dummy_input).view(1, -1).shape[1]
            
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # Input shape: (B, 4, H, W)
        x = observations['image']
        
        # Split channels
        state_map = x[:, 0, :, :] # (B, H, W)
        extra_maps = x[:, 1:, :, :] # (B, 3, H, W) -> Flags, LogicMine, LogicSafe
        
        # One-Hot Encoding for State Map
        # Values are floats 0.0-10.0, cast to long
        state_long = state_map.long()
        # OneHot: (B, H, W, 11)
        one_hot = torch.nn.functional.one_hot(state_long, num_classes=11)
        # Permute to (B, 11, H, W)
        one_hot = one_hot.permute(0, 3, 1, 2).float()
        
        # --- NEW: Numeric Channel (Magnitude) ---
        # Map Values (1.0..9.0) to (0.0..1.0) representing 0..8 mines
        # Fog (10.0) and Pad (0.0) -> 0.0
        numeric_channel = torch.zeros_like(state_map)
        
        # Mask for digits (1.0 corresponds to 0 mines, 9.0 to 8 mines)
        # obs val: 1.0 (0 mines) -> val-1 = 0
        # obs val: 9.0 (8 mines) -> val-1 = 8
        is_digit = (state_map >= 1.0) & (state_map <= 9.0)
        numeric_channel[is_digit] = (state_map[is_digit] - 1.0) / 8.0
        
        # Add dimension (B, 1, H, W)
        numeric_channel = numeric_channel.unsqueeze(1)
        
        # Concatenate: 11 (OneHot) + 1 (Numeric) + 3 (Extra) = 15 channels
        cnn_in = torch.cat([one_hot, numeric_channel, extra_maps.float()], dim=1)
        
        return self.linear(self.cnn(cnn_in))
