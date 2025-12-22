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
    Spatial attention mechanism - fokusuje na ważnych obszarach viewport
    (głowa snake, jedzenie, przeszkody)
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        
    def forward(self, x):
        # x: [B, C, H, W]
        attention = torch.sigmoid(self.conv(x))  # [B, 1, H, W]
        return x * attention  # Weighted features


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    """
    ✅ Zoptymalizowane CNN z attention (BEZ residual blocks)
    ✅ BF16 mixed precision dla szybszego treningu
    ✅ ~1.1M parametrów (zamiast 5.16M) - 78% redukcja!
    """
    def __init__(self, observation_space: spaces.Dict, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        global _INFO_PRINTED
        
        # 🔧 Load from config
        viewport_size = config['environment']['viewport_size']
        use_layernorm = config['model']['convlstm'].get('use_layernorm', True)
        use_spatial_attention = config['model']['convlstm'].get('use_spatial_attention', True)
        
        cnn_dropout = config['model'].get('cnn_dropout', 0.0)
        scalar_dropout = config['model'].get('scalar_dropout', 0.0)
        fusion_dropout = config['model'].get('fusion_dropout', 0.0)
        scalar_input_dropout = config['model'].get('scalar_input_dropout', 0.0)
        
        # ✅ CZYTAJ Z CONFIGU
        cnn_channels = config['model']['convlstm'].get('cnn_channels', [96, 160, 224, 320])
        cnn_output_dim = config['model']['convlstm'].get('cnn_output_dim', 768)
        scalar_output_dim = config['model']['convlstm'].get('scalar_output_dim', 192)
        
        # ✅ BF16 dla szybszego treningu
        self.use_amp = torch.cuda.is_available()
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        in_channels = 1
        
        # ==================== DYNAMICZNE CNN (BEZ RESIDUAL BLOCKS) ====================
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for i, out_channels in enumerate(cnn_channels):
            # Stride = 2 tylko dla warstwy 3 (index 2)
            stride = 2 if i == 2 else 1
            
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            )
            self.bn_layers.append(nn.BatchNorm2d(out_channels))
            
            # Dropout od drugiej warstwy
            if i > 0 and cnn_dropout > 0:
                self.dropout_layers.append(nn.Dropout2d(cnn_dropout))
            
            in_channels = out_channels
        
        # Ostatni kanał z configu
        cnn_raw_dim = cnn_channels[-1]
        
        # ✅ SPATIAL ATTENTION (opcjonalne)
        self.use_spatial_attention = use_spatial_attention
        if use_spatial_attention:
            self.spatial_attention = SpatialAttention(cnn_raw_dim)
        
        # ✅ Global Average Pooling (spatial → 1×1)
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
        scalar_dim = 8
        
        self.scalar_input_dropout = nn.Dropout(scalar_input_dropout) if scalar_input_dropout > 0 else nn.Identity()
        
        # ✅ CZYTAJ scalar_hidden_dims Z CONFIGU
        scalar_hidden_dims = config['model']['convlstm'].get('scalar_hidden_dims', [192, 256])
        
        # Dynamiczne budowanie scalar network
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
        
        # Ostatnia warstwa (output)
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
            print(f"[CNN] 🚀 OPTIMIZED CNN ARCHITECTURE (NO RESIDUAL BLOCKS)")
            print(f"{'='*70}")
            print(f"[CNN] Layers: {len(cnn_channels)} conv layers")
            print(f"[CNN] Channels: {[1] + cnn_channels}")
            print(f"[CNN] Spatial: 11×11 → ... → 1×1 (GAP)")
            print(f"[CNN] Raw features: {cnn_raw_dim} ({cnn_channels[-1]} channels after pooling)")
            print(f"[CNN] Output: {cnn_output_dim}")
            print(f"[CNN] Expansion: {cnn_output_dim / cnn_raw_dim:.2f}x")
            print(f"[CNN] 📊 CNN params: {cnn_params:,}")
            print(f"[CNN] 📊 Projection params: {projection_params:,}")
            if use_spatial_attention:
                print(f"[CNN] ✨ Spatial attention: ENABLED ({attention_params:,} params)")
            print(f"[CNN] ⚡ Mixed precision: {self.amp_dtype}")
            print(f"")
            print(f"[SCALARS] Input: {scalar_dim}")
            print(f"[SCALARS] Hidden: {scalar_hidden_dims}")
            print(f"[SCALARS] Output: {scalar_output_dim}")
            print(f"[SCALARS] 📊 Params: {scalar_params:,}")
            print(f"")
            print(f"[FUSION] Input: {total_dim}")
            print(f"[FUSION] Output: {features_dim}")
            print(f"[FUSION] 📊 Params: {fusion_params:,}")
            print(f"")
            print(f"[BALANCE] CNN:Scalar ratio: {cnn_output_dim/scalar_output_dim:.1f}:1")
            print(f"[BALANCE] CNN portion: {cnn_output_dim/total_dim*100:.1f}%")
            print(f"[BALANCE] Scalar portion: {scalar_output_dim/total_dim*100:.1f}%")
            print(f"")
            print(f"[TOTAL] 🎯 All parameters: {total_params:,}")
            print(f"{'='*70}\n")
            _INFO_PRINTED = True

    def _initialize_weights(self):
        """He initialization dla GELU activation"""
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
        
        # ==================== CNN (BEZ RESIDUAL BLOCKS) ====================
        with autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            x = image
            
            # Przejście przez wszystkie warstwy CNN
            dropout_idx = 0
            
            for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers)):
                x = conv(x)
                x = bn(x)
                x = F.gelu(x)
                
                # Dropout od drugiej warstwy
                if i > 0 and dropout_idx < len(self.dropout_layers):
                    x = self.dropout_layers[dropout_idx](x)
                    dropout_idx += 1
            
            # ✅ Spatial attention (opcjonalne)
            if self.use_spatial_attention:
                x = self.spatial_attention(x)
            
            # Global Average Pooling: spatial → 1×1
            x = self.global_pool(x)
            cnn_raw = self.flatten(x)
        
        cnn_raw = cnn_raw.float()
        
        # ==================== CNN PROJECTION ====================
        cnn_features = self.cnn_projection(cnn_raw)
        
        # ==================== SCALARS ====================
        scalars = torch.cat([
            observations['direction'],
            observations['dx_head'],
            observations['dy_head'],
            observations['front_coll'],
            observations['left_coll'],
            observations['right_coll'],
            observations['snake_length']
        ], dim=-1)
        
        scalars = self.scalar_input_dropout(scalars)
        scalar_features = self.scalar_network(scalars)
        
        # ==================== FUSION ====================
        features = torch.cat([cnn_features, scalar_features], dim=-1)
        return self.fusion(features)