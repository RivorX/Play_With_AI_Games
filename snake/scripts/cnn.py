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


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    """
    ✅ Deep Bottleneck Architecture (Pure - NO Skip Connection)
    """
    def __init__(self, observation_space: spaces.Dict, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        global _INFO_PRINTED
        
        # 🔧 Load from config
        cnn_channels = config['model']['convlstm']['cnn_channels'][:2]
        
        # ✅ Multi-stage bottleneck
        cnn_bottleneck_dims = config['model']['convlstm'].get('cnn_bottleneck_dims', [896])
        if isinstance(cnn_bottleneck_dims, int):
            cnn_bottleneck_dims = [cnn_bottleneck_dims]
        
        cnn_output_dim = config['model']['convlstm'].get('cnn_output_dim', 768)
        scalar_hidden_dims = config['model']['convlstm'].get('scalar_hidden_dims', [128, 64])
        
        cnn_dropout = config['model'].get('cnn_dropout', 0.0)
        scalar_dropout = config['model'].get('scalar_dropout', 0.0)
        fusion_dropout = config['model'].get('fusion_dropout', 0.0)
        
        use_layernorm = config['model']['convlstm'].get('use_layernorm', True)
        
        self.use_amp = torch.cuda.is_available()
        in_channels = 1
        
        # ==================== TRADITIONAL CNN ====================
        self.conv1 = nn.Conv2d(in_channels, cnn_channels[0], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(cnn_channels[0])

        self.conv2 = nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(cnn_channels[1])
        self.dropout2 = nn.Dropout2d(cnn_dropout) if cnn_dropout > 0 else nn.Identity()

        self.flatten = nn.Flatten()

        # ==================== DEEP BOTTLENECK ====================
        def compute_spatial_size(input_size, convs):
            size = input_size
            for kernel, stride, padding in convs:
                size = (size + 2*padding - kernel) // stride + 1
            return size

        viewport_size = config['environment']['viewport_size']
        convs = [
            (3, 1, 1),  # conv1
            (3, 2, 1),  # conv2
        ]
        spatial_size = compute_spatial_size(viewport_size, convs)
        cnn_raw_dim = cnn_channels[1] * spatial_size * spatial_size

        # Build multi-stage compression path
        bottleneck_layers = []
        prev_dim = cnn_raw_dim

        for idx, bottleneck_dim in enumerate(cnn_bottleneck_dims):
            bottleneck_layers.append(nn.Linear(prev_dim, bottleneck_dim))
            if use_layernorm:
                bottleneck_layers.append(nn.LayerNorm(bottleneck_dim))
            bottleneck_layers.append(nn.GELU())

            # Dropout between stages (not after last stage)
            if idx < len(cnn_bottleneck_dims) - 1:
                bottleneck_layers.append(nn.Dropout(cnn_dropout))

            prev_dim = bottleneck_dim

        # Final projection to output_dim
        bottleneck_layers.append(nn.Linear(prev_dim, cnn_output_dim))
        if use_layernorm:
            bottleneck_layers.append(nn.LayerNorm(cnn_output_dim))

        self.cnn_bottleneck = nn.Sequential(*bottleneck_layers)
        
        # ==================== SCALARS ====================
        # direction(2) + dx_head(1) + dy_head(1) + front_coll(1) + left_coll(1) + right_coll(1) + snake_length(1) = 8
        scalar_dim = 8
        self.scalar_input_dropout = nn.Dropout(config['model'].get('scalar_input_dropout', 0.0))
        
        scalar_layers = []
        prev_dim = scalar_dim
        for idx, hidden_dim in enumerate(scalar_hidden_dims):
            scalar_layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layernorm:
                scalar_layers.append(nn.LayerNorm(hidden_dim))
            scalar_layers.append(nn.GELU())
            if idx < len(scalar_hidden_dims) - 1:
                scalar_layers.append(nn.Dropout(scalar_dropout))
            prev_dim = hidden_dim
        
        self.scalar_linear = nn.Sequential(*scalar_layers)
        scalar_output_dim = scalar_hidden_dims[-1]
        
        # ==================== FUSION ====================
        total_dim = cnn_output_dim + scalar_output_dim
        
        self.final_linear = nn.Sequential(
            nn.LayerNorm(total_dim) if use_layernorm else nn.Identity(),
            nn.Linear(total_dim, features_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout)
        )
        
        self._initialize_weights()
        
        # ==================== PARAMETER COUNT ====================
        cnn_params = sum(p.numel() for p in self.conv1.parameters()) + \
                     sum(p.numel() for p in self.conv2.parameters())
        bottleneck_params = sum(p.numel() for p in self.cnn_bottleneck.parameters())
        scalar_params = sum(p.numel() for p in self.scalar_linear.parameters())
        fusion_params = sum(p.numel() for p in self.final_linear.parameters())
        total_params = cnn_params + bottleneck_params + scalar_params + fusion_params
        
        # Compression ratio
        if len(cnn_bottleneck_dims) > 0:
            compression_ratio = cnn_raw_dim / cnn_bottleneck_dims[0]
        else:
            compression_ratio = cnn_raw_dim / cnn_output_dim
        
        if not _INFO_PRINTED:
            print(f"\n{'='*70}")
            print(f"[CNN] ⚡ PURE BOTTLENECK CNN (NO SKIP)")
            print(f"{'='*70}")
            
            # Architecture visualization
            arch_str = f"{cnn_raw_dim}"
            for dim in cnn_bottleneck_dims:
                arch_str += f" → {dim}"
            arch_str += f" → {cnn_output_dim}"
            
            print(f"[CNN] Architecture: {arch_str}")
            print(f"[CNN] ✅ Bottleneck stages: {len(cnn_bottleneck_dims)}")
            print(f"[CNN] ✅ Compression ratio: {compression_ratio:.1f}x")
            print(f"[CNN] ✅ LayerNorm: {use_layernorm}")
            print(f"[CNN] 📊 CNN params: {cnn_params:,}")
            print(f"[CNN] 📊 Bottleneck params: {bottleneck_params:,}")
            print(f"[CNN] 📊 Total CNN+Bottleneck: {cnn_params + bottleneck_params:,}")
            print(f"")
            print(f"[SCALARS] Hidden dims: {scalar_hidden_dims}")
            print(f"[SCALARS] Output: {scalar_output_dim}")
            print(f"[SCALARS] 📊 Scalar params: {scalar_params:,}")
            print(f"")
            print(f"[FUSION] Output: {features_dim}")
            print(f"[FUSION] 📊 Fusion params: {fusion_params:,}")
            print(f"")
            print(f"[BALANCE] CNN:Scalar ratio: {cnn_output_dim/scalar_output_dim:.1f}:1")
            print(f"[BALANCE] CNN portion: {cnn_output_dim/total_dim*100:.1f}%")
            print(f"[BALANCE] Scalar portion: {scalar_output_dim/total_dim*100:.1f}%")
            print(f"")
            print(f"[TOTAL] 🎯 All parameters: {total_params:,}")
            print(f"{'='*70}\n")
            _INFO_PRINTED = True

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
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
        with autocast('cuda', enabled=self.use_amp):
            x = image
            
            # Block 1: 12×12 → 12×12
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.gelu(x)
            
            # Block 2: 12×12 → 6×6
            x = self.conv2(x)
            x = self.bn2(x)
            x = F.gelu(x)
            x = self.dropout2(x)
            
            cnn_raw = self.flatten(x)
        
        cnn_raw = cnn_raw.float()
        
        # ==================== PURE BOTTLENECK ====================
        cnn_features = self.cnn_bottleneck(cnn_raw)
        
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
        scalar_features = self.scalar_linear(scalars)
        
        # ==================== FUSION ====================
        features = torch.cat([cnn_features, scalar_features], dim=-1)
        return self.final_linear(features)