import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast  # ✅ Nowy import (bez FutureWarning)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import yaml
from gymnasium import spaces
import os

# Wczytaj konfigurację
base_dir = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# ✅ GLOBAL FLAG - wyświetl info tylko raz
_INFO_PRINTED = False

class CustomFeaturesExtractor(BaseFeaturesExtractor):
    """
    NAPRAWIONY Features Extractor dla RecurrentPPO Z AMP
    """
    def __init__(self, observation_space: spaces.Dict, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        global _INFO_PRINTED
        
        # Pobierz konfigurację
        cnn_channels = config['model']['convlstm']['cnn_channels']
        scalar_hidden_dims = config['model']['convlstm']['scalar_hidden_dims']
        
        # Dropouty - ZMIENIONE dla skalarów
        cnn_dropout = config['model'].get('cnn_dropout', 0.05)
        scalar_dropout = min(0.1, config['model'].get('scalar_dropout', 0.1))
        scalar_input_dropout = min(0.1, config['model'].get('scalar_input_dropout', 0.1))
        fusion_dropout = config['model'].get('fusion_dropout', 0.15)
        
        # Włącz AMP jeśli CUDA dostępne
        self.use_amp = torch.cuda.is_available()
        
        in_channels = 1
        
        # ===========================
        # CNN Z RESIDUAL CONNECTIONS
        # ===========================
        self.conv1 = nn.Conv2d(in_channels, cnn_channels[0], kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(cnn_channels[0])
        self.dropout1 = nn.Dropout2d(cnn_dropout)
        
        # Druga warstwa z residual connection
        self.conv2 = nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(cnn_channels[1])
        self.dropout2 = nn.Dropout2d(cnn_dropout)
        
        # Residual projection (dopasowanie wymiarów dla skip connection)
        self.residual_proj = nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=1, stride=2, padding=0)
        self.residual_bn = nn.BatchNorm2d(cnn_channels[1])
        
        # Trzecia warstwa (jeśli jest więcej kanałów)
        self.conv3 = None
        if len(cnn_channels) > 2:
            self.conv3 = nn.Conv2d(cnn_channels[1], cnn_channels[2], kernel_size=3, stride=2, padding=1)
            self.bn3 = nn.BatchNorm2d(cnn_channels[2])
            self.dropout3 = nn.Dropout2d(cnn_dropout)
            
            # Residual dla warstwy 3
            self.residual_proj2 = nn.Conv2d(cnn_channels[1], cnn_channels[2], kernel_size=1, stride=2, padding=0)
            self.residual_bn2 = nn.BatchNorm2d(cnn_channels[2])
        
        self.flatten = nn.Flatten()
        
        # Oblicz wymiar po CNN
        spatial_size = 16  # viewport size
        for i in range(len(cnn_channels)):
            if i == 0:
                pass  # stride=1
            else:
                spatial_size = spatial_size // 2  # stride=2
        
        cnn_output_size = spatial_size * spatial_size
        cnn_dim = cnn_channels[-1] * cnn_output_size
        
        # LayerNorm dla CNN output (ZAWSZE float32)
        self.cnn_norm = nn.LayerNorm(cnn_dim)
        
        # ===========================
        # SCALARS Z LEPSZYM DROPOUTEM
        # ===========================
        scalar_dim = 7  # 2 (direction) + 5 (inne)
        
        # Zmniejszony dropout na wejściu
        self.scalar_input_dropout = nn.Dropout(scalar_input_dropout)
        
        # Warstwy skalarne
        scalar_layers = []
        prev_dim = scalar_dim
        
        for idx, hidden_dim in enumerate(scalar_hidden_dims):
            scalar_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Dropout(scalar_dropout)
            ])
            prev_dim = hidden_dim
        
        self.scalar_linear = nn.Sequential(*scalar_layers)
        scalar_output_dim = scalar_hidden_dims[-1]
        
        # LayerNorm dla scalar output
        self.scalar_norm = nn.LayerNorm(scalar_output_dim)
        
        # ===========================
        # WEIGHTED FUSION
        # ===========================
        total_dim = cnn_dim + scalar_output_dim
        
        # Learnable weights dla CNN vs Scalar balancing
        self.cnn_weight = nn.Parameter(torch.ones(1) * 2.0)
        self.scalar_weight = nn.Parameter(torch.ones(1))
        
        self.final_linear = nn.Sequential(
            nn.Linear(total_dim, features_dim),
            nn.LayerNorm(features_dim),  # Dodatkowa normalizacja
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(fusion_dropout)
        )
        
        # ===========================
        # HE INITIALIZATION
        # ===========================
        self._initialize_weights()
        
        # ===========================
        # LOGGING (tylko raz)
        # ===========================
        if not _INFO_PRINTED:
            amp_status = "✅ ENABLED" if self.use_amp else "❌ DISABLED (CPU mode)"
            print(f"\n{'='*70}")
            print(f"[CNN] NAPRAWIONY FEATURES EXTRACTOR Z AMP")
            print(f"{'='*70}")
            print(f"[CNN] Architektura CNN: {cnn_channels}")
            print(f"[CNN] ✅ Residual connections: ENABLED")
            print(f"[CNN] ✅ AMP (Mixed Precision): {amp_status}")
            print(f"[CNN] Spatial size po CNN: {spatial_size}x{spatial_size}")
            print(f"[CNN] CNN output dim: {cnn_dim} (BatchNorm + LayerNorm ✓)")
            print(f"[CNN] Scalar hidden dims: {scalar_hidden_dims}")
            print(f"[CNN] Scalar output dim: {scalar_output_dim} (LayerNorm ✓)")
            print(f"[CNN] Total features dim: {total_dim} -> {features_dim}")
            print(f"[CNN] Skalary stanowią {scalar_output_dim/total_dim*100:.1f}% wejścia")
            print(f"\n[DROPOUT] CNN: {cnn_dropout}")
            print(f"[DROPOUT] Scalar INPUT: {scalar_input_dropout}")
            print(f"[DROPOUT] Scalar hidden: {scalar_dropout}")
            print(f"[DROPOUT] Fusion: {fusion_dropout}")
            print(f"\n[NORMALIZATION]")
            print(f"  - CNN: BatchNorm2d + LayerNorm")
            print(f"  - Scalars: LayerNorm")
            print(f"  - Fusion: LayerNorm")
            print(f"\n[FUSION] ✅ Learnable weights: CNN={self.cnn_weight.item():.2f}, Scalar={self.scalar_weight.item():.2f}")
            
            if self.use_amp:
                print(f"\n[AMP] 🚀 Expected speedup: 30-50% (RTX series)")
                print(f"[AMP] 💾 Expected VRAM saving: ~30%")
                print(f"[AMP] ⚠️  Note: Full benefit requires RTX 20xx/30xx/40xx GPUs")
            else:
                print(f"\n[AMP] ⚠️  Running on CPU - AMP disabled")
            
            print(f"{'='*70}\n")
            _INFO_PRINTED = True

    def _initialize_weights(self):
        """He initialization dla stabilności treningu"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, observations):
        # ===========================
        # CNN Z AMP I RESIDUAL CONNECTIONS
        # ===========================
        image = observations['image']
        if image.dim() == 4 and image.shape[-1] == 1:
            image = image.permute(0, 3, 1, 2)
        
        # ✅ AUTOCAST dla CNN (największe przyspieszenie)
        # WAŻNE: Używamy 'cuda' zamiast deprecated autocast()
        with autocast('cuda', enabled=self.use_amp):
            # Warstwa 1
            x = self.conv1(image)
            x = self.bn1(x)
            x = F.leaky_relu(x, 0.01, inplace=True)
            x = self.dropout1(x)
            identity1 = x  # Zachowaj dla skip connection
            
            # Warstwa 2 z residual
            x = self.conv2(x)
            x = self.bn2(x)
            
            # Skip connection (dopasuj wymiary)
            identity1 = self.residual_proj(identity1)
            identity1 = self.residual_bn(identity1)
            
            x = x + identity1  # ✅ RESIDUAL CONNECTION
            x = F.leaky_relu(x, 0.01, inplace=True)
            x = self.dropout2(x)
            
            # Warstwa 3 (jeśli istnieje) z residual
            if self.conv3 is not None:
                identity2 = x
                x = self.conv3(x)
                x = self.bn3(x)
                
                identity2 = self.residual_proj2(identity2)
                identity2 = self.residual_bn2(identity2)
                
                x = x + identity2  # ✅ RESIDUAL CONNECTION
                x = F.leaky_relu(x, 0.01, inplace=True)
                x = self.dropout3(x)
            
            # Flatten (wewnątrz autocast dla spójności)
            image_features = self.flatten(x)
        
        # ✅ KLUCZOWA NAPRAWA: LayerNorm POZA autocast
        # LayerNorm wymaga float32, a autocast daje float16
        # Konwertuj do float32 przed LayerNorm
        image_features = image_features.float()
        image_features = self.cnn_norm(image_features)
        
        # ===========================
        # SCALARS (bez AMP - za mały benefit)
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
        scalar_features = self.scalar_norm(scalar_features)
        
        # ===========================
        # WEIGHTED FUSION
        # ===========================
        # Zastosuj learnable weights (z softplus dla dodatniości)
        cnn_w = F.softplus(self.cnn_weight)
        scalar_w = F.softplus(self.scalar_weight)
        
        image_features_weighted = image_features * cnn_w
        scalar_features_weighted = scalar_features * scalar_w
        
        # Połącz i przetwórz (bez autocast - final linear jest szybki)
        features = torch.cat([image_features_weighted, scalar_features_weighted], dim=-1)
        return self.final_linear(features)