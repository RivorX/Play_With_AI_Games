import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import yaml
from gymnasium import spaces
import os

# Wczytaj konfigurację
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

_INFO_PRINTED = False


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    """
    🆕 SIMPLIFIED Pre-Layer Norm CNN
    
    CHANGES:
    ✅ Removed skip connections (Pre-LN handles gradient flow)
    ✅ Only 2 CNN layers (3rd removed)
    ✅ Changed SiLU → GELU (better gradients)
    ✅ Simplified architecture (less oversmoothing)
    
    ARCHITECTURE:
    Input (16x16) → BN → Conv1 (32ch) → BN → GELU → Conv2 (64ch, stride=2) → BN → GELU → 8x8
    """
    def __init__(self, observation_space: spaces.Dict, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        global _INFO_PRINTED
        
        # Pobierz konfigurację (używamy tylko pierwszych 2 kanałów)
        cnn_channels_full = config['model']['convlstm']['cnn_channels']
        cnn_channels = cnn_channels_full[:2]  # [32, 64] - tylko 2 warstwy
        scalar_hidden_dims = config['model']['convlstm']['scalar_hidden_dims']
        
        # Dropouty
        cnn_dropout = config['model'].get('cnn_dropout', 0.0)
        scalar_dropout = config['model'].get('scalar_dropout', 0.0)
        scalar_input_dropout = config['model'].get('scalar_input_dropout', 0.0)
        fusion_dropout = config['model'].get('fusion_dropout', 0.0)
        
        self.use_amp = torch.cuda.is_available()
        in_channels = 1
        
        # ===========================
        # 🆕 SIMPLIFIED CNN (2 LAYERS, NO SKIP)
        # ===========================
        # Input BatchNorm
        self.input_bn = nn.BatchNorm2d(in_channels)
        
        # Warstwa 1: 16x16 → 16x16 (stride=1)
        self.conv1 = nn.Conv2d(in_channels, cnn_channels[0], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(cnn_channels[0])
        
        # Warstwa 2: 16x16 → 8x8 (stride=2)
        # 🆕 Pre-Norm BEFORE Conv2
        self.pre_bn2 = nn.BatchNorm2d(cnn_channels[0])
        self.conv2 = nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(cnn_channels[1])
        self.dropout2 = nn.Dropout2d(cnn_dropout) if cnn_dropout > 0 else nn.Identity()
        
        self.flatten = nn.Flatten()
        
        # Oblicz wymiar po CNN
        spatial_size = 16 // 2  # 16 → 8 (jeden stride=2)
        cnn_output_size = spatial_size * spatial_size  # 64
        cnn_dim = cnn_channels[1] * cnn_output_size  # 64 * 64 = 4096
        
        # Pre-Norm dla CNN output
        self.cnn_pre_norm = nn.LayerNorm(cnn_dim)
        
        # ===========================
        # SCALARS (UNCHANGED)
        # ===========================
        scalar_dim = 7  # 2 (direction) + 5 (inne)
        self.scalar_input_dropout = nn.Dropout(scalar_input_dropout)
        
        # 🆕 Pre-Norm MLP with GELU
        scalar_layers = []
        prev_dim = scalar_dim
        
        for idx, hidden_dim in enumerate(scalar_hidden_dims):
            scalar_layers.extend([
                nn.LayerNorm(prev_dim),
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),  # 🆕 GELU zamiast SiLU
                nn.Dropout(scalar_dropout) if idx < len(scalar_hidden_dims)-1 else nn.Identity()
            ])
            prev_dim = hidden_dim
        
        self.scalar_linear = nn.Sequential(*scalar_layers)
        scalar_output_dim = scalar_hidden_dims[-1]
        self.scalar_pre_norm = nn.LayerNorm(scalar_output_dim)
        
        # ===========================
        # FUSION (UNCHANGED)
        # ===========================
        total_dim = cnn_dim + scalar_output_dim
        
        self.final_linear = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, features_dim),
            nn.GELU(),  # 🆕 GELU zamiast SiLU
            nn.Dropout(fusion_dropout)
        )
        
        # ===========================
        # XAVIER INITIALIZATION
        # ===========================
        self._initialize_weights()
        
        # ===========================
        # LOGGING
        # ===========================
        if not _INFO_PRINTED:
            amp_status = "✅ ENABLED" if self.use_amp else "❌ DISABLED (CPU mode)"
            print(f"\n{'='*70}")
            print(f"[CNN] 🆕 SIMPLIFIED PRE-LAYER NORM (NO SKIP, 2 LAYERS, GELU)")
            print(f"{'='*70}")
            print(f"[CNN] Architektura CNN: {cnn_channels} (3. warstwa USUNIĘTA)")
            print(f"[CNN] ❌ Skip connections: REMOVED (Pre-LN handles gradient flow)")
            print(f"[CNN] ✅ Pre-Layer Norm: ENABLED")
            print(f"[CNN] ✅ Input BatchNorm: ENABLED")
            print(f"[CNN] 🆕 Activation: GELU (better gradients than SiLU)")
            print(f"[CNN] Spatial size po CNN: {spatial_size}x{spatial_size}")
            print(f"[CNN] CNN output dim: {cnn_dim}")
            print(f"[CNN] Scalar hidden dims: {scalar_hidden_dims}")
            print(f"[CNN] Scalar output dim: {scalar_output_dim}")
            print(f"[CNN] Total features dim: {total_dim} -> {features_dim}")
            print(f"[CNN] Scalary stanowią {scalar_output_dim/total_dim*100:.1f}% wejścia")
            print(f"\n[DROPOUT]")
            print(f"  - CNN: {cnn_dropout}")
            print(f"  - Scalar INPUT: {scalar_input_dropout}")
            print(f"  - Scalar hidden: {scalar_dropout}")
            print(f"  - Fusion: {fusion_dropout}")
            
            print(f"\n[WHY THESE CHANGES?]")
            print(f"  🎯 Pre-LN already stabilizes gradients → skip redundant")
            print(f"  🎯 2 CNN layers enough for 16x16 viewport")
            print(f"  🎯 GELU > SiLU for vanishing gradients (smoother)")
            print(f"  🎯 Simpler = less oversmoothing = better features")
            
            if self.use_amp:
                print(f"\n[AMP] 🚀 Expected speedup: 30-50% (RTX series)")

            print(f"{'='*70}\n")
            _INFO_PRINTED = True

    def _initialize_weights(self):
        """Xavier initialization (gain=1.0 dla Pre-LN)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=1.0)
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
        # ===========================
        # 🆕 SIMPLIFIED CNN (NO SKIP)
        # ===========================
        image = observations['image']
        if image.dim() == 4 and image.shape[-1] == 1:
            image = image.permute(0, 3, 1, 2)
        
        with autocast('cuda', enabled=self.use_amp):
            # Input normalization
            x = self.input_bn(image)
            
            # Warstwa 1 (bez pre-norm, bo input_bn już jest)
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.gelu(x)  # 🆕 GELU
            
            # Warstwa 2 z Pre-Norm (BEZ SKIP CONNECTION)
            x = self.pre_bn2(x)  # 🆕 Norm BEFORE Conv2
            x = self.conv2(x)
            x = self.bn2(x)
            x = F.gelu(x)  # 🆕 GELU
            x = self.dropout2(x)
            
            # Flatten
            image_features = self.flatten(x)
        
        # Pre-Norm na CNN output (poza autocast)
        image_features = image_features.float()
        image_features = self.cnn_pre_norm(image_features)
        
        # ===========================
        # SCALARS (UNCHANGED)
        # ===========================
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
        scalar_features = self.scalar_pre_norm(scalar_features)
        
        # ===========================
        # FUSION
        # ===========================
        features = torch.cat([image_features, scalar_features], dim=-1)
        return self.final_linear(features)