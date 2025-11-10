import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import yaml
from gymnasium import spaces
import os
import numpy as np
import math

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

_INFO_PRINTED = False


class PositionalEncoding2D(nn.Module):
    """
    2D Sinusoidal Positional Encoding dla spatial features
    """
    def __init__(self, channels, height, width):
        super().__init__()
        
        pe = torch.zeros(channels, height, width)
        y_pos = torch.arange(height).unsqueeze(1).float()
        x_pos = torch.arange(width).unsqueeze(0).float()
        
        div_term = torch.exp(torch.arange(0, channels, 2).float() * 
                            (-math.log(10000.0) / channels))
        
        pe[0::2, :, :] = torch.sin(y_pos.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))
        
        if channels > 1:
            pe[1::2, :, :] = torch.cos(x_pos.unsqueeze(0) * div_term.unsqueeze(1).unsqueeze(2))
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe


class MultiQueryCrossAttention(nn.Module):
    """
    🎯 Multi-Query Cross-Attention (FIXED)
    
    FIXES:
    - Output dimension = attended_cnn_dim (configurable, not tied to cnn_dim!)
    - No radial encoding (removed bottleneck)
    - Simpler, more direct path
    """
    def __init__(self, cnn_dim, scalar_dim, output_dim, num_heads=4, num_queries=6, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.head_dim = cnn_dim // num_heads
        assert cnn_dim % num_heads == 0, "cnn_dim must be divisible by num_heads"
        
        # Projections
        self.query_proj = nn.Linear(scalar_dim, cnn_dim * num_queries)
        self.key_proj = nn.Linear(cnn_dim, cnn_dim)
        self.value_proj = nn.Linear(cnn_dim, cnn_dim)
        
        # 🔥 NEW: Output projection directly to attended_cnn_dim
        self.out_proj = nn.Linear(cnn_dim * num_queries, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, cnn_features, scalar_features):
        """
        cnn_features: (B, N, cnn_dim) - pre-normalized!
        scalar_features: (B, scalar_dim)
        returns: (B, output_dim) - attended CNN features
        """
        B, N, D = cnn_features.shape
        
        # Multiple Queries from scalars
        Q = self.query_proj(scalar_features).reshape(B, self.num_queries, D)
        K = self.key_proj(cnn_features)
        V = self.value_proj(cnn_features)
        
        # Multi-head attention
        Q = Q.reshape(B, self.num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = attn @ V
        out = out.transpose(1, 2).reshape(B, self.num_queries, D)
        out = out.reshape(B, self.num_queries * D)
        
        # Project directly to output_dim
        return self.out_proj(out)


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    """
    🔥 FIXED CNN Architecture:
    
    MAJOR FIXES:
    1. ✅ LayerNorm BEFORE attention (normalize CNN features)
    2. ✅ Removed radial encoding (was causing CNN to be ignored)
    3. ✅ Direct output from attention → attended_cnn_dim (no intermediate layers!)
    4. ✅ Skip connection: raw CNN → fusion (bypass attention) - ONLY when scalars enabled
    5. ✅ Better gradient flow to CNN layers
    6. ✅ Support for scalar_hidden_dims: [0] (no scalars mode)
    7. ✅ Dynamic viewport_size support
    8. ✅ CNN-only mode: Direct spatial features → fusion (no compression!)
    
    Architecture:
    
    WITH SCALARS:
    - Conv1 + Conv2 + MaxPool → Positional Encoding
    - Two paths to fusion:
      PATH 1: CNN → LayerNorm → Attention → attended_cnn_dim (448)
      PATH 2: CNN → Direct projection → direct_cnn_dim (128) ✨ Skip connection
    - Fusion: Attended(448) + Direct(128) + Scalars(256) = 832 → 512
    
    WITHOUT SCALARS (CNN-only):
    - Conv1 + Conv2 + MaxPool → Positional Encoding
    - Direct path: Spatial Features (raw) → Fusion
    - Fusion: Raw CNN (1,200) → 512
    """
    def __init__(self, observation_space: spaces.Dict, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        global _INFO_PRINTED
        
        # Configuration
        cnn_channels = config['model']['convlstm']['cnn_channels'][:2]
        
        # 🔥 FIXED: Attention outputs DIRECTLY to attended_cnn_dim
        attended_cnn_dim = config['model']['convlstm'].get('attended_cnn_dim', 448)
        
        # ✨ NEW: Direct CNN path (skip connection) - only used WITH scalars
        direct_cnn_dim = config['model']['convlstm'].get('direct_cnn_dim', 128)
        
        scalar_hidden_dims = config['model']['convlstm'].get('scalar_hidden_dims', [128, 256])
        
        cnn_dropout = config['model'].get('cnn_dropout', 0.01)
        scalar_dropout = config['model'].get('scalar_dropout', 0.1)
        fusion_dropout = config['model'].get('fusion_dropout', 0.1)
        attention_heads = config['model']['convlstm'].get('attention_heads', 4)
        num_queries = config['model']['convlstm'].get('num_queries', 6)
        
        use_layernorm = config['model']['convlstm'].get('use_layernorm', True)
        
        self.use_amp = torch.cuda.is_available()
        in_channels = 1
        
        # ✅ Check if scalars are disabled
        self.use_scalars = len(scalar_hidden_dims) > 0 and scalar_hidden_dims[0] > 0
        
        # 🎯 EXPOSE: Scalars mode flag (for analysis)
        self.scalars_enabled = self.use_scalars
        
        # ==================== CNN: 2 Blocks + MaxPool ====================
        self.conv1 = nn.Conv2d(in_channels, cnn_channels[0], kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(cnn_channels[0])

        self.conv2 = nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(cnn_channels[1])
        self.dropout2 = nn.Dropout2d(cnn_dropout) if cnn_dropout > 0 else nn.Identity()
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ==================== POSITIONAL ENCODING ====================
        # ✅ FIXED: Dynamic viewport_size support
        viewport_size = config['environment']['viewport_size']
        spatial_size = viewport_size // 2  # After maxpool
        self.spatial_size = spatial_size
        
        self.pos_encoding = PositionalEncoding2D(
            channels=cnn_channels[1],
            height=spatial_size,
            width=spatial_size
        )

        # ==================== Spatial Features ====================
        spatial_locations = spatial_size * spatial_size
        cnn_spatial_dim = cnn_channels[1]
        cnn_raw_features = spatial_locations * cnn_spatial_dim
        
        # 🔥 FIX 1: LayerNorm BEFORE attention (only if scalars enabled)
        if self.use_scalars:
            self.cnn_prenorm = nn.LayerNorm(cnn_spatial_dim)
        else:
            self.cnn_prenorm = None
        
        # ==================== SCALARS ====================
        scalar_output_dim = 0
        scalar_params = 0
        
        if self.use_scalars:
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
            
            self.scalar_network = nn.Sequential(*scalar_layers)
            scalar_output_dim = scalar_hidden_dims[-1]
            scalar_params = sum(p.numel() for p in self.scalar_network.parameters())
        else:
            self.scalar_network = None
            scalar_output_dim = 0
        
        # ==================== 🔥 ATTENTION or SKIP-ONLY ====================
        attn_params = 0
        if self.use_scalars:
            # With scalars: use attention
            self.cross_attention = MultiQueryCrossAttention(
                cnn_dim=cnn_spatial_dim,
                scalar_dim=scalar_output_dim,
                output_dim=attended_cnn_dim,
                num_heads=attention_heads,
                num_queries=num_queries,
                dropout=fusion_dropout
            )
            attn_params = sum(p.numel() for p in self.cross_attention.parameters())
            use_attended_path = True
        else:
            # No scalars: skip attention entirely
            self.cross_attention = None
            attended_cnn_dim = 0  # Zero out attended path
            use_attended_path = False
        
        self.use_attended_path = use_attended_path
        
        # ==================== ✨ SKIP CONNECTION (Direct CNN Path) ====================
        # 🔥 NEW: Only used when scalars are ENABLED
        # When scalars disabled → use raw spatial features directly!
        direct_params = 0
        if self.use_scalars:
            # With scalars: compress CNN for skip connection
            self.cnn_direct = nn.Sequential(
                nn.LayerNorm(cnn_raw_features) if use_layernorm else nn.Identity(),
                nn.Linear(cnn_raw_features, direct_cnn_dim),
                nn.LayerNorm(direct_cnn_dim) if use_layernorm else nn.Identity(),
                nn.GELU(),
                nn.Dropout(cnn_dropout)
            )
            direct_params = sum(p.numel() for p in self.cnn_direct.parameters())
        else:
            # Without scalars: no skip connection, use raw features
            self.cnn_direct = None
            direct_cnn_dim = cnn_raw_features  # Use FULL spatial features!
        
        # ==================== FUSION ====================
        # Streams depend on configuration:
        # WITH SCALARS: Attended CNN + Direct CNN + Scalars
        # WITHOUT SCALARS: Raw CNN only (no compression!)
        fusion_input_dim = direct_cnn_dim + attended_cnn_dim + scalar_output_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, features_dim),
            nn.LayerNorm(features_dim) if use_layernorm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(fusion_dropout)
        )
        
        self._initialize_weights()
        
        # ==================== PARAMETER COUNT ====================
        cnn_params = (sum(p.numel() for p in self.conv1.parameters()) + 
                     sum(p.numel() for p in self.conv2.parameters()) +
                     sum(p.numel() for p in self.bn1.parameters()) + 
                     sum(p.numel() for p in self.bn2.parameters()))
        pos_enc_params = sum(p.numel() for p in self.pos_encoding.parameters())
        prenorm_params = sum(p.numel() for p in self.cnn_prenorm.parameters()) if self.cnn_prenorm else 0
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        total_params = (cnn_params + pos_enc_params + prenorm_params + scalar_params + 
                       attn_params + direct_params + fusion_params)
        
        # Percentages
        attended_percentage = (attended_cnn_dim / fusion_input_dim) * 100 if fusion_input_dim > 0 else 0
        direct_percentage = (direct_cnn_dim / fusion_input_dim) * 100 if fusion_input_dim > 0 else 0
        cnn_total_percentage = ((attended_cnn_dim + direct_cnn_dim) / fusion_input_dim) * 100 if fusion_input_dim > 0 else 0
        scalar_percentage = (scalar_output_dim / fusion_input_dim) * 100 if fusion_input_dim > 0 else 0
        
        # Compression ratios
        attended_compression = cnn_raw_features / attended_cnn_dim if attended_cnn_dim > 0 else 0
        direct_compression = cnn_raw_features / direct_cnn_dim if direct_cnn_dim > 0 and self.use_scalars else 1.0
        
        # Receptive field
        rf = 1
        stride_product = 1
        rf = rf + (5 - 1) * stride_product
        stride_product *= 1
        rf = rf + (5 - 1) * stride_product
        stride_product *= 1
        rf = rf + (2 - 1) * stride_product
        stride_product *= 2
        self.receptive_field = rf + (stride_product - 1)
        
        if not _INFO_PRINTED:
            print(f"\n{'='*70}")
            if self.use_scalars:
                print(f"🔥 FIXED CNN: Better Gradient Flow + Skip Connection")
            else:
                print(f"🔥 CNN-ONLY MODE: Direct Spatial Features (No Compression!)")
            print(f"{'='*70}")
            
            print(f"\n[CNN] 🗃️ Architecture:")
            print(f"[CNN] Block 1: Conv(5×5,s=1) → BN → SiLU")
            print(f"[CNN] Block 2: Conv(5×5,s=1) → BN → SiLU → Dropout({cnn_dropout})")
            print(f"[CNN] MaxPool: 2×2 ({viewport_size}×{viewport_size} → {spatial_size}×{spatial_size})")
            print(f"[CNN] Channels: {cnn_channels}")
            print(f"[CNN] ✅ Receptive Field: {self.receptive_field}×{self.receptive_field} pixels")
            print(f"[CNN] Raw features: {cnn_raw_features} ({cnn_spatial_dim}×{spatial_locations})")
            print(f"[CNN] 📊 Params: {cnn_params:,}")
            
            print(f"\n[POS-ENC] 🎯 2D Sinusoidal Encoding")
            print(f"[POS-ENC] Shape: ({cnn_channels[1]}, {spatial_size}, {spatial_size})")
            print(f"[POS-ENC] 📊 Params: {pos_enc_params:,} (frozen)")
            
            if self.use_scalars:
                print(f"\n[PRENORM] 🔥 LayerNorm BEFORE attention")
                print(f"[PRENORM] ✅ Normalizes CNN features → better gradient flow!")
                print(f"[PRENORM] 📊 Params: {prenorm_params:,}")
                
                print(f"\n[SCALARS] 🔢 Network:")
                print(f"[SCALARS] 8 → {scalar_hidden_dims} → {scalar_output_dim}")
                print(f"[SCALARS] 📊 Params: {scalar_params:,}")
                
                print(f"\n[ATTENTION] 🎯 Multi-Query Attention (FIXED)")
                print(f"[ATTENTION] ❌ Removed radial encoding (was causing CNN ignore)")
                print(f"[ATTENTION] ✅ Direct output: {cnn_spatial_dim}×{num_queries} → {attended_cnn_dim}")
                print(f"[ATTENTION] Heads: {attention_heads}, Queries: {num_queries}")
                print(f"[ATTENTION] Compression: {cnn_raw_features}/{attended_cnn_dim} = {attended_compression:.1f}x")
                print(f"[ATTENTION] 📊 Params: {attn_params:,}")
                
                print(f"\n[SKIP-CNN] ✨ Direct CNN Path (Skip Connection)")
                print(f"[SKIP-CNN] Raw CNN → Direct projection → Fusion")
                print(f"[SKIP-CNN] {cnn_raw_features} → {direct_cnn_dim}")
                print(f"[SKIP-CNN] Compression: {direct_compression:.1f}x")
                print(f"[SKIP-CNN] ✅ Ensures CNN features reach fusion!")
                print(f"[SKIP-CNN] 📊 Params: {direct_params:,}")
            else:
                print(f"\n[SCALARS] ❌ DISABLED (scalar_hidden_dims: [0])")
                print(f"[ATTENTION] ❌ SKIPPED (no scalars available)")
                print(f"\n[RAW-CNN] 🔥 Direct Spatial Features Path")
                print(f"[RAW-CNN] Spatial Features → Fusion (NO COMPRESSION!)")
                print(f"[RAW-CNN] {cnn_raw_features} dims → directly to fusion")
                print(f"[RAW-CNN] ✅ Full CNN representation preserved!")
                print(f"[RAW-CNN] 📊 Params: 0 (direct passthrough)")
            
            print(f"\n[FUSION] 🎯 Input Breakdown:")
            if self.use_scalars:
                print(f"[FUSION]   ├─ Attended CNN:   {attended_cnn_dim:4d} dim ({attended_percentage:5.1f}%) [via attention]")
                print(f"[FUSION]   ├─ Direct CNN:     {direct_cnn_dim:4d} dim ({direct_percentage:5.1f}%) [skip connection] ✨")
                print(f"[FUSION]   └─ Scalar features: {scalar_output_dim:4d} dim ({scalar_percentage:5.1f}%)")
            else:
                print(f"[FUSION]   └─ Raw CNN:        {direct_cnn_dim:4d} dim (100.0%) [direct spatial features] 🔥")
            print(f"[FUSION]   ───────────────────────────────────────────────")
            print(f"[FUSION]   CNN Total:  {attended_cnn_dim + direct_cnn_dim:4d} dim ({cnn_total_percentage:5.1f}%) 🔥")
            print(f"[FUSION]   Total: {fusion_input_dim:4d} → {features_dim} dim")
            if self.use_scalars:
                print(f"[FUSION]   ✅ CNN DOMINATES: {cnn_total_percentage:.0f}% vs Scalars {scalar_percentage:.0f}%!")
            else:
                print(f"[FUSION]   ✅ PURE CNN: 100% raw spatial features (no compression)!")
            print(f"[FUSION]   📊 Params: {fusion_params:,}")
            
            print(f"\n[FIXES] ✅")
            if self.use_scalars:
                print(f"  1. ✅ LayerNorm before attention → stable gradients")
                print(f"  2. ✅ No radial encoding → CNN not overwhelmed")
                print(f"  3. ✅ Direct attention output → no bottleneck layers")
                print(f"  4. ✅ Skip connection → guaranteed CNN usage")
                print(f"  5. ✅ CNN dominance: {cnn_total_percentage:.0f}% fusion input")
            else:
                print(f"  1. ✅ Pure CNN mode (no scalars)")
                print(f"  2. ✅ Direct spatial features → fusion (no compression!)")
                print(f"  3. ✅ 100% CNN-based decision making")
                print(f"  4. ✅ Full {cnn_raw_features} features preserved")
            print(f"  6. ✅ Dynamic viewport_size support: {viewport_size}×{viewport_size}")
            
            print(f"\n[TOTAL] 🎯 Parameters: {total_params:,}")
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
        
        B = image.shape[0]
        
        # ==================== CNN ====================
        with autocast('cuda', enabled=self.use_amp):
            x = image
            
            # Block 1
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.silu(x)
            
            # Block 2
            x = self.conv2(x)
            x = self.bn2(x)
            x = F.silu(x)
            x = self.dropout2(x)
            
            # MaxPool
            x = self.maxpool(x)
            
            # Positional Encoding
            x = self.pos_encoding(x)
            
            # Flatten for attention: (B, C, H, W) → (B, H*W, C)
            spatial_features = x.flatten(2).transpose(1, 2)
        
        spatial_features = spatial_features.float()
        
        # ==================== FUSION PATHS ====================
        if self.use_scalars:
            # ========== MODE 1: WITH SCALARS (Attention + Skip) ==========
            
            # PATH 2: SKIP CONNECTION (Direct CNN)
            cnn_raw = spatial_features.flatten(1)
            direct_cnn = self.cnn_direct(cnn_raw)
            
            # SCALARS
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
            
            # PATH 1: ATTENTION
            normalized_features = self.cnn_prenorm(spatial_features)
            attended_cnn = self.cross_attention(normalized_features, scalar_features)
            
            # Fusion: Attended + Direct + Scalars
            fused = torch.cat([attended_cnn, direct_cnn, scalar_features], dim=-1)
            
        else:
            # ========== MODE 2: CNN-ONLY (Direct Spatial Features) ==========
            # 🔥 NEW: Use raw spatial features directly (no compression!)
            cnn_raw = spatial_features.flatten(1)  # (B, spatial_locations * cnn_spatial_dim)
            fused = cnn_raw  # Direct passthrough - no skip connection needed!
        
        return self.fusion(fused)