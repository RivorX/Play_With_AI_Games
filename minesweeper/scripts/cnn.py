import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import yaml
from gymnasium import spaces
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

_INFO_PRINTED = False


class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism - focuses on important areas
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        
    def forward(self, x):
        attention = torch.sigmoid(self.conv(x))
        return x * attention


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    """
    CNN Feature Extractor for Minesweeper
    Processes full board (grid_size x grid_size) with CNN
    """
    def __init__(self, observation_space: spaces.Dict, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        global _INFO_PRINTED
        
        # Config
        use_layernorm = config['model']['convlstm'].get('use_layernorm', True)
        use_spatial_attention = config['model']['convlstm'].get('use_spatial_attention', True)
        
        cnn_dropout = config['model'].get('cnn_dropout', 0.0)
        scalar_dropout = config['model'].get('scalar_dropout', 0.0)
        fusion_dropout = config['model'].get('fusion_dropout', 0.0)
        scalar_input_dropout = config['model'].get('scalar_input_dropout', 0.0)
        
        cnn_channels = config['model']['convlstm'].get('cnn_channels', [64, 128, 256])
        cnn_output_dim = config['model']['convlstm'].get('cnn_output_dim', 512)
        scalar_output_dim = config['model']['convlstm'].get('scalar_output_dim', 128)
        
        # BF16 for faster training
        self.use_amp = torch.cuda.is_available()
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        in_channels = 1
        
        # ==================== CNN LAYERS ====================
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for i, out_channels in enumerate(cnn_channels):
            # No stride=2 - we want to preserve spatial information
            stride = 1
            
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            )
            self.bn_layers.append(nn.BatchNorm2d(out_channels))
            
            if i > 0 and cnn_dropout > 0:
                self.dropout_layers.append(nn.Dropout2d(cnn_dropout))
            
            in_channels = out_channels
        
        cnn_raw_dim = cnn_channels[-1]
        
        # ==================== SPATIAL ATTENTION ====================
        self.use_spatial_attention = use_spatial_attention
        if use_spatial_attention:
            self.spatial_attention = SpatialAttention(cnn_raw_dim)
        
        # ==================== GLOBAL POOLING ====================
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        
        # ==================== CNN PROJECTION ====================
        self.cnn_projection = nn.Sequential(
            nn.Linear(cnn_raw_dim, cnn_output_dim),
            nn.LayerNorm(cnn_output_dim) if use_layernorm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(cnn_dropout) if cnn_dropout > 0 else nn.Identity()
        )
        
        # ==================== SCALARS NETWORK ====================
        scalar_dim = 5  # remaining_cells, revealed_ratio, mine_density_norm, steps_per_cell, grid_size_norm
        # Note: action_mask is NOT included here (too large - 256 dims)
        
        self.scalar_input_dropout = nn.Dropout(scalar_input_dropout) if scalar_input_dropout > 0 else nn.Identity()
        
        scalar_hidden_dims = config['model']['convlstm'].get('scalar_hidden_dims', [128, 192])
        
        scalar_layers = []
        in_dim = scalar_dim
        
        for hidden_dim in scalar_hidden_dims:
            scalar_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
                nn.GELU(),
                nn.Dropout(scalar_dropout) if scalar_dropout > 0 else nn.Identity()
            ])
            in_dim = hidden_dim
        
        scalar_layers.extend([
            nn.Linear(in_dim, scalar_output_dim),
            nn.LayerNorm(scalar_output_dim) if use_layernorm else nn.Identity(),
            nn.GELU()
        ])
        
        self.scalar_network = nn.Sequential(*scalar_layers)
        
        # ==================== FUSION ====================
        total_dim = cnn_output_dim + scalar_output_dim
        
        self.fusion = nn.Sequential(
            nn.LayerNorm(total_dim) if use_layernorm else nn.Identity(),
            nn.Linear(total_dim, features_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout) if fusion_dropout > 0 else nn.Identity()
        )
        
        self._initialize_weights()
        
        # ==================== INFO PRINT ====================
        cnn_params = sum(p.numel() for layer in [self.conv_layers, self.bn_layers] 
                        for module in layer for p in module.parameters())
        
        projection_params = sum(p.numel() for p in self.cnn_projection.parameters())
        scalar_params = sum(p.numel() for p in self.scalar_network.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        
        attention_params = 0
        if use_spatial_attention:
            attention_params = sum(p.numel() for p in self.spatial_attention.parameters())
        
        total_params = cnn_params + projection_params + scalar_params + fusion_params + attention_params
        
        if not _INFO_PRINTED:
            print(f"\n{'='*70}")
            print(f"[CNN] 🚀 MINESWEEPER CNN ARCHITECTURE")
            print(f"{'='*70}")
            print(f"[CNN] Layers: {len(cnn_channels)} conv layers")
            print(f"[CNN] Channels: {[1] + cnn_channels}")
            print(f"[CNN] Raw features: {cnn_raw_dim}")
            print(f"[CNN] Output: {cnn_output_dim}")
            print(f"[CNN] 📊 CNN params: {cnn_params:,}")
            print(f"[CNN] 📊 Projection params: {projection_params:,}")
            if use_spatial_attention:
                print(f"[CNN] ✨ Spatial attention: ENABLED ({attention_params:,} params)")
            print(f"[CNN] ⚡ Mixed precision: {self.amp_dtype}")
            print(f"")
            print(f"[SCALARS] Input: {scalar_dim}")
            print(f"[SCALARS] Hidden: {scalar_hidden_dims}")
            print(f"[SCALARS] Output: {scalar_output_dim}")
            print(f"[SCALARS] Features: remaining_cells, revealed_ratio, mine_density, steps_per_cell, grid_size")
            print(f"[SCALARS] Note: Model learns to avoid masked areas (-1.0) through penalties")
            print(f"[SCALARS] 📊 Params: {scalar_params:,}")
            print(f"")
            print(f"[FUSION] Input: {total_dim}")
            print(f"[FUSION] Output: {features_dim}")
            print(f"[FUSION] 📊 Params: {fusion_params:,}")
            print(f"")
            print(f"[TOTAL] 🎯 All parameters: {total_params:,}")
            print(f"{'='*70}\n")
            _INFO_PRINTED = True

    def _initialize_weights(self):
        """He initialization for GELU activation"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, observations):
        image = observations['image']
        if image.dim() == 4 and image.shape[-1] == 1:
            image = image.permute(0, 3, 1, 2)
        
        # ==================== CNN ====================
        with autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            x = image
            
            dropout_idx = 0
            
            for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers)):
                x = conv(x)
                x = bn(x)
                x = F.gelu(x)
                
                if i > 0 and dropout_idx < len(self.dropout_layers):
                    x = self.dropout_layers[dropout_idx](x)
                    dropout_idx += 1
            
            # Spatial attention
            if self.use_spatial_attention:
                x = self.spatial_attention(x)
            
            # Global Average Pooling
            x = self.global_pool(x)
            cnn_raw = self.flatten(x)
        
        cnn_raw = cnn_raw.float()
        
        # ==================== CNN PROJECTION ====================
        cnn_features = self.cnn_projection(cnn_raw)
        
        # ==================== SCALARS ====================
        scalars = torch.cat([
            observations['remaining_cells'],
            observations['revealed_ratio'],
            observations['mine_density_norm'],
            observations['steps_per_cell'],
            observations['grid_size_norm'],
        ], dim=-1)
        
        scalars = self.scalar_input_dropout(scalars)
        scalar_features = self.scalar_network(scalars)
        
        # ==================== FUSION ====================
        features = torch.cat([cnn_features, scalar_features], dim=-1)
        return self.fusion(features)