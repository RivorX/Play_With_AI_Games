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
    """
    def __init__(self, observation_space: spaces.Dict, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        global _INFO_PRINTED
        
        # 🔧 Load from config
        cnn_channels = config['model']['convlstm']['cnn_channels'][:2]
        cnn_bottleneck_dim = config['model']['convlstm'].get('cnn_bottleneck_dim', 384)
        cnn_output_dim = config['model']['convlstm'].get('cnn_output_dim', 768)
        scalar_hidden_dims = config['model']['convlstm'].get('scalar_hidden_dims', [128, 64])
        
        cnn_dropout = config['model'].get('cnn_dropout', 0.0)
        scalar_dropout = config['model'].get('scalar_dropout', 0.0)
        fusion_dropout = config['model'].get('fusion_dropout', 0.0)
        
        use_layernorm = config['model']['convlstm'].get('use_layernorm', True)
        learnable_alpha = config['model']['convlstm'].get('learnable_alpha', True)
        initial_alpha = config['model']['convlstm'].get('initial_alpha', 0.5)
        
        self.use_amp = torch.cuda.is_available()
        in_channels = 1
        
        # ==================== TRADITIONAL CNN (FAST!) ====================
        self.conv1 = nn.Conv2d(in_channels, cnn_channels[0], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(cnn_channels[0])
        
        self.conv2 = nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(cnn_channels[1])
        self.dropout2 = nn.Dropout2d(cnn_dropout) if cnn_dropout > 0 else nn.Identity()
        
        self.flatten = nn.Flatten()
        
        # ==================== BOTTLENECK FIX ====================
        spatial_size = 8
        cnn_raw_dim = cnn_channels[1] * spatial_size * spatial_size  # 4096
        
        # Main path: 4096 → bottleneck → output
        main_layers = [nn.Linear(cnn_raw_dim, cnn_bottleneck_dim)]
        if use_layernorm:
            main_layers.append(nn.LayerNorm(cnn_bottleneck_dim))
        main_layers.extend([
            nn.GELU(),
            nn.Dropout(cnn_dropout),
            nn.Linear(cnn_bottleneck_dim, cnn_output_dim)
        ])
        if use_layernorm:
            main_layers.append(nn.LayerNorm(cnn_output_dim))
        
        self.cnn_compress = nn.Sequential(*main_layers)
        
        # Skip connection: 4096 → output
        skip_layers = [nn.Linear(cnn_raw_dim, cnn_output_dim, bias=False)]
        if use_layernorm:
            skip_layers.append(nn.LayerNorm(cnn_output_dim))
        
        self.cnn_residual = nn.Sequential(*skip_layers)
        
        # 🆕 Learnable or fixed residual weight
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(initial_alpha))
            self.alpha_mode = 'learnable'
        else:
            self.register_buffer('alpha', torch.tensor(initial_alpha))
            self.alpha_mode = 'fixed'
        
        # ==================== SCALARS ====================
        scalar_dim = 7
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
        
        # Count parameters
        cnn_params = sum(p.numel() for p in self.conv1.parameters()) + \
                     sum(p.numel() for p in self.conv2.parameters())
        bottleneck_params = sum(p.numel() for p in self.cnn_compress.parameters()) + \
                           sum(p.numel() for p in self.cnn_residual.parameters())
        
        if not _INFO_PRINTED:
            print(f"\n{'='*70}")
            print(f"[CNN] ⚡ CNN")
            print(f"{'='*70}")
            print(f"[CNN] Architecture: {cnn_raw_dim} → {cnn_bottleneck_dim} → {cnn_output_dim}")
            print(f"[CNN] ✅ Bottleneck compression: {cnn_raw_dim/cnn_bottleneck_dim:.1f}x")
            print(f"[CNN] ✅ LayerNorm: {use_layernorm}")
            print(f"[CNN] ✅ Alpha mode: {self.alpha_mode} (init={initial_alpha:.2f})")
            print(f"[CNN] 📊 CNN params: {cnn_params:,}")
            print(f"[CNN] 📊 Bottleneck params: {bottleneck_params:,}")
            print(f"[SCALARS] Hidden dims: {scalar_hidden_dims}")
            print(f"[SCALARS] Output: {scalar_output_dim} ({scalar_output_dim/total_dim*100:.1f}%)")
            print(f"[BALANCE] CNN:Scalar ratio: {cnn_output_dim/scalar_output_dim:.1f}:1")
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
        
        # ==================== TRADITIONAL CNN ====================
        with autocast('cuda', enabled=self.use_amp):
            x = image
            
            # Block 1: 16×16 → 16×16
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.gelu(x)
            
            # Block 2: 16×16 → 8×8
            x = self.conv2(x)
            x = self.bn2(x)
            x = F.gelu(x)
            x = self.dropout2(x)
            
            cnn_raw = self.flatten(x)
        
        cnn_raw = cnn_raw.float()
        
        # ==================== BOTTLENECK + RESIDUAL ====================
        main_path = self.cnn_compress(cnn_raw)
        skip_path = self.cnn_residual(cnn_raw)
        
        # Weighted combination (learnable)
        if self.alpha_mode == 'learnable':
            alpha = torch.sigmoid(self.alpha)  # Clamp to [0,1]
        else:
            alpha = self.alpha
        
        cnn_features = alpha * main_path + (1 - alpha) * skip_path
        
        # ==================== SCALARS ====================
        scalars = torch.cat([
            observations['direction'],
            observations['dx_head'],
            observations['dy_head'],
            observations['front_coll'],
            observations['left_coll'],
            observations['right_coll']
        ], dim=-1)
        
        scalars = self.scalar_input_dropout(scalars)
        scalar_features = self.scalar_linear(scalars)
        
        # ==================== FUSION ====================
        features = torch.cat([cnn_features, scalar_features], dim=-1)
        return self.final_linear(features)