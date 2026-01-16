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

_INFO_PRINTED = False


class SpatialAttention(nn.Module):
    """Spatial attention - fokus na ważnych obszarach (miny, flagi, logika)"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        
    def forward(self, x):
        attention = torch.sigmoid(self.conv(x))  # [B, 1, H, W]
        return x * attention


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    """
    🚀 OPTIMIZED CNN with Global Average Pooling (like Snake)
    Reduces params from ~16M to ~200k
    """
    def __init__(self, observation_space: spaces.Dict, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        global _INFO_PRINTED
        
        # Internal channels: 11 OneHot + 1 Numeric + 4 Extra = 16
        internal_channels = 16
        
        # Pobieranie struktury z configu
        cnn_conf = config['model']['cnn_architecture']
        channel_list = cnn_conf['cnn_channels']
        use_norm = cnn_conf['use_layernorm']
        dropout_val = cnn_conf['dropout']
        use_spatial_attention = cnn_conf.get('use_spatial_attention', True)
        
        viewport_size = config['environment']['viewport_size']

        # ==================== DYNAMIC CNN ====================
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        curr_in = internal_channels
        
        for i, curr_out in enumerate(channel_list):
            stride = 1  # No stride needed with GAP
            
            self.conv_layers.append(
                nn.Conv2d(curr_in, curr_out, kernel_size=3, stride=stride, padding=1)
            )
            self.bn_layers.append(nn.BatchNorm2d(curr_out) if use_norm else nn.Identity())
            
            # Dropout from 2nd layer
            if i > 0 and dropout_val > 0:
                self.dropout_layers.append(nn.Dropout2d(dropout_val))
            
            curr_in = curr_out
        
        cnn_raw_dim = channel_list[-1]
        
        # ==================== SPATIAL ATTENTION ====================
        self.use_spatial_attention = use_spatial_attention
        if use_spatial_attention:
            self.spatial_attention = SpatialAttention(cnn_raw_dim)
        
        # ==================== 🔥 GLOBAL AVERAGE POOLING 🔥 ====================
        # Redukuje (B, C, H, W) → (B, C, 1, 1) → (B, C)
        # Eliminuje 99% parametrów w kolejnej warstwie!
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        
        # ==================== CNN PROJECTION ====================
        cnn_output_dim = cnn_conf['cnn_output_dim']
        self.cnn_projection = nn.Sequential(
            nn.Linear(cnn_raw_dim, cnn_output_dim),
            nn.LayerNorm(cnn_output_dim) if use_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout_val) if dropout_val > 0 else nn.Identity()
        )
        
        # ==================== VECTOR INPUT (4 floats) ====================
        vector_input_dim = 4
        
        # ==================== FUSION ====================
        total_dim = cnn_output_dim + vector_input_dim
        
        self.fusion = nn.Sequential(
            nn.LayerNorm(total_dim) if use_norm else nn.Identity(),
            nn.Linear(total_dim, features_dim),
            nn.GELU()
        )
        
        self._initialize_weights()
        
        # ==================== INFO PRINT ====================
        if not _INFO_PRINTED:
            cnn_params = sum(p.numel() for layer in [self.conv_layers, self.bn_layers] 
                            for module in layer for p in module.parameters())
            projection_params = sum(p.numel() for p in self.cnn_projection.parameters())
            fusion_params = sum(p.numel() for p in self.fusion.parameters())
            attention_params = sum(p.numel() for p in self.spatial_attention.parameters()) if use_spatial_attention else 0
            
            total_params = cnn_params + projection_params + fusion_params + attention_params
            
            print(f"\n{'='*70}")
            print(f"[CNN] 🚀 OPTIMIZED MINESWEEPER CNN (with GAP)")
            print(f"{'='*70}")
            print(f"[CNN] Layers: {len(channel_list)} conv layers")
            print(f"[CNN] Channels: {[internal_channels] + channel_list}")
            print(f"[CNN] Spatial: {viewport_size}×{viewport_size} → ... → 1×1 (GAP)")
            print(f"[CNN] Raw features: {cnn_raw_dim} (after GAP)")
            print(f"[CNN] CNN Output: {cnn_output_dim}")
            print(f"[CNN] Vector Input: {vector_input_dim}")
            print(f"[CNN] Final Output: {features_dim}")
            print(f"[CNN] 📊 CNN params: {cnn_params:,}")
            print(f"[CNN] 📊 Projection params: {projection_params:,}")
            if use_spatial_attention:
                print(f"[CNN] 📊 Attention params: {attention_params:,}")
            print(f"[CNN] 📊 Fusion params: {fusion_params:,}")
            print(f"[CNN] 📊 TOTAL: {total_params:,}")
            print(f"{'='*70}\n")
            
            _INFO_PRINTED = True

    def _initialize_weights(self):
        """He initialization for GELU"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, observations):
        # Image Input: (B, 5, H, W)
        x = observations['image']
        
        # Vector Input: (B, 4)
        v = observations['vector']
        
        # Split channels
        state_map = x[:, 0, :, :]  # (B, H, W)
        extra_maps = x[:, 1:, :, :]  # (B, 4, H, W) -> Flags, LogicMine, LogicSafe, Needed
        
        # One-Hot Encoding for State Map
        state_long = state_map.long()
        one_hot = torch.nn.functional.one_hot(state_long, num_classes=11)
        one_hot = one_hot.permute(0, 3, 1, 2).float()
        
        # Numeric Channel (Magnitude)
        numeric_channel = torch.zeros_like(state_map)
        is_digit = (state_map >= 1.0) & (state_map <= 9.0)
        numeric_channel[is_digit] = (state_map[is_digit] - 1.0) / 8.0
        numeric_channel = numeric_channel.unsqueeze(1)
        
        # Concatenate: 11 (OneHot) + 1 (Numeric) + 4 (Extra) = 16 channels
        cnn_in = torch.cat([one_hot, numeric_channel, extra_maps.float()], dim=1)
        
        # ==================== CNN FORWARD ====================
        x = cnn_in
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            x = self.bn_layers[i](x)
            x = torch.relu(x)
            if i > 0 and i-1 < len(self.dropout_layers):
                x = self.dropout_layers[i-1](x)
        
        # ==================== SPATIAL ATTENTION ====================
        if self.use_spatial_attention:
            x = self.spatial_attention(x)
        
        # ==================== 🔥 GLOBAL AVERAGE POOLING 🔥 ====================
        x = self.global_pool(x)  # (B, C, H, W) → (B, C, 1, 1)
        cnn_raw = self.flatten(x)  # (B, C)
        
        # ==================== CNN PROJECTION ====================
        cnn_features = self.cnn_projection(cnn_raw)
        
        # ==================== FUSION ====================
        fused = torch.cat([cnn_features, v], dim=-1)
        return self.fusion(fused)
