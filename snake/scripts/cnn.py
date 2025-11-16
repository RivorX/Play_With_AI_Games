import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import yaml
from gymnasium import spaces
import os
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
    def __init__(self, channels, height, width, scale=0.1):
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
        self.scale = scale
    
    def forward(self, x):
        return x + self.scale * self.pe


class MultiQueryCrossAttention(nn.Module):
    """
    🎯 Multi-Query Cross-Attention (FIXED)
    
    FIXES:
    - Output dimension = attended_cnn_dim (configurable, not tied to cnn_dim!)
    - No radial encoding (removed bottleneck)
    - Simpler, more direct path
    - Temperature scaling for sharper attention patterns 🔥
    """
    def __init__(self, cnn_dim, scalar_dim, output_dim, num_heads=4, num_queries=6, dropout=0.1, temperature=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.head_dim = cnn_dim // num_heads
        self.temperature = temperature
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
        # 🔥 Temperature scaling: <1.0 = sharper, >1.0 = softer
        attn = attn / self.temperature
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = attn @ V
        out = out.transpose(1, 2).reshape(B, self.num_queries, D)
        out = out.reshape(B, self.num_queries * D)
        
        # Project directly to output_dim
        return self.out_proj(out)


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    """
    🔥 HYBRID CNN: 5×5 → 3×3 → 3×3 (BEST OF BOTH WORLDS!)
    
    ARCHITECTURE RATIONALE:
    1. ✅ Conv1 (5×5): Fast RF expansion for global context
    2. ✅ Conv2 (3×3): Refine local features
    3. ✅ Conv3 (3×3, stride=2): Learned downsampling + final features
    4. ✅ NO MAXPOOL: Preserves small objects (food!)
    
    WHY THIS IS OPTIMAL:
    - 5×5 first layer: Quick "big picture" understanding (RF: 5×5)
    - 3×3 layers: Fine-grained refinement without losing details
    - Strided 3×3: Learned downsampling (better than MaxPool)
    - RF progression: 5×5 → 7×7 → 9×9 (perfect for 11×11 viewport!)
    
    Architecture:
    
    WITH SCALARS:
    - Conv1 (5×5, s=1) + Conv2 (3×3, s=1) + Conv3 (3×3, s=2) → Positional Encoding
    - Two paths to fusion:
            PATH 1: CNN → LayerNorm → Attention → attended_cnn_dim (448)
            PATH 2: CNN → LayerNorm → Linear → GELU → direct_cnn_dim (128) ✨ Skip connection
        - Fusion: Attended(448) + Direct(128) + Scalars(256) = 832 → 512
    
    WITHOUT SCALARS (CNN-only):
    - Conv1 + Conv2 + Conv3 → Positional Encoding
        - Stabilized path: LayerNorm → Linear (cfg) → GELU → Fusion
        - Fusion: Direct (cfg) → 512
    """
    def __init__(self, observation_space: spaces.Dict, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        global _INFO_PRINTED
        
        # Configuration
        cnn_channels = config['model']['convlstm']['cnn_channels']
        if len(cnn_channels) == 2:
            # Extend to 3 layers for hierarchical
            cnn_channels = [16, 32, 48]  # Default hierarchical
        
        # 🔥 FIXED: Attention outputs DIRECTLY to attended_cnn_dim
        attended_cnn_dim = config['model']['convlstm'].get('attended_cnn_dim', 448)
        
        # ✨ Direct CNN path konfiguracja
        base_direct_cnn_dim = config['model']['convlstm'].get('direct_cnn_dim', 128)
        cnn_only_direct_dim = config['model']['convlstm'].get(
            'cnn_only_direct_dim',
            1024  # 🔥 Default 1024 dla CNN-only (kompromis capacity vs GPU)
        )
        
        scalar_hidden_dims = config['model']['convlstm'].get('scalar_hidden_dims', [128, 256])
        
        cnn_dropout = config['model'].get('cnn_dropout', 0.01)
        scalar_dropout = config['model'].get('scalar_dropout', 0.1)
        fusion_dropout = config['model'].get('fusion_dropout', 0.1)
        attention_heads = config['model']['convlstm'].get('attention_heads', 4)
        num_queries = config['model']['convlstm'].get('num_queries', 6)
        attention_temperature = config['model']['convlstm'].get('attention_temperature', 1.0)
        pos_encoding_scale = config['model']['convlstm'].get('pos_encoding_scale', 0.1)
        
        use_layernorm = config['model']['convlstm'].get('use_layernorm', True)
        
        self.use_amp = torch.cuda.is_available()
        in_channels = 1
        
        # ✅ Check if scalars are disabled
        self.use_scalars = len(scalar_hidden_dims) > 0 and scalar_hidden_dims[0] > 0
        
        # 🎯 EXPOSE: Scalars mode flag (for analysis)
        self.scalars_enabled = self.use_scalars
        
        # ==================== HYBRID CNN: 5×5 → 3×3 → 3×3 ====================
        # Layer 1: Global context with 5×5 (RF: 5×5) 🌟
        # 🔥 FIX: BatchNorm2d dla Conv1 - stabilizacja ujemnych wartości (-1.0 dla ścian)!
        self.conv1 = nn.Conv2d(in_channels, cnn_channels[0], kernel_size=5, stride=1, padding=2)
        self.norm1 = nn.BatchNorm2d(cnn_channels[0])  # ✅ BATCHNORM - lepsze dla ujemnych input!
        self.bn1 = self.norm1  # Alias for debug utilities

        # Layer 2: Local refinement with 3×3 (RF: 7×7)
        # 🔥 FIX: BatchNorm2d - spójna normalizacja z Conv1!
        self.conv2 = nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(cnn_channels[1])  # ✅ BATCHNORM!
        self.bn2 = self.norm2
        
        # Layer 3: Final features + STRIDED DOWNSAMPLE with 3×3 (RF: 9×9) 🔥
        self.conv3 = nn.Conv2d(cnn_channels[1], cnn_channels[2], kernel_size=3, stride=2, padding=1)
        self.norm3 = nn.BatchNorm2d(cnn_channels[2])  # ✅ BATCHNORM!
        self.bn3 = self.norm3
        self.dropout3 = nn.Dropout2d(cnn_dropout) if cnn_dropout > 0 else nn.Identity()

        # ==================== POSITIONAL ENCODING ====================
        # ✅ FIXED: Dynamic viewport_size support
        viewport_size = config['environment']['viewport_size']
        # After stride=2 in conv3: 11×11 → 6×6
        spatial_size = (viewport_size + 1) // 2  # Ceiling division for stride=2
        self.spatial_size = spatial_size
        
        self.pos_encoding = PositionalEncoding2D(
            channels=cnn_channels[2],
            height=spatial_size,
            width=spatial_size,
            scale=pos_encoding_scale
        )

        # ==================== Spatial Features ====================
        spatial_locations = spatial_size * spatial_size
        cnn_spatial_dim = cnn_channels[2]
        cnn_raw_features = spatial_locations * cnn_spatial_dim

        # Stabilizacja bez dużego narzutu na GPU
        self.spatial_norm = nn.LayerNorm(cnn_spatial_dim)
        
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
            
            # 🔥 FIX 1: LayerNorm dla stabilizacji różnych zakresów scalar features
            self.scalar_input_norm = nn.LayerNorm(scalar_dim)
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
            scalar_input_norm_params = sum(p.numel() for p in self.scalar_input_norm.parameters())
            scalar_params += scalar_input_norm_params
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
                dropout=fusion_dropout,
                temperature=attention_temperature
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
        direct_params = 0
        if self.use_scalars:
            direct_cnn_dim = base_direct_cnn_dim
        else:
            direct_cnn_dim = cnn_only_direct_dim

        # 🔥 Calculate compression ratio early for conditional logic
        compression_ratio = cnn_raw_features / direct_cnn_dim
        
        # 🔥 OPTIMIZED: Build direct path with optional compression safeguards
        direct_layers = []
        direct_layers.append(nn.LayerNorm(cnn_raw_features) if use_layernorm else nn.Identity())
        direct_layers.append(nn.Linear(cnn_raw_features, direct_cnn_dim))
        if use_layernorm:
            direct_layers.append(nn.LayerNorm(direct_cnn_dim))
        direct_layers.append(nn.GELU())
        
        # ✅ Apply dropout ONLY if:
        # 1. CNN dropout enabled AND
        # 2. High compression (>2x) to prevent overfitting on bottleneck
        if cnn_dropout > 0 and compression_ratio > 2.0:
            direct_layers.append(nn.Dropout(cnn_dropout))
        
        self.cnn_direct = nn.Sequential(*direct_layers)
        direct_params = sum(p.numel() for p in self.cnn_direct.parameters())
        
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
                     sum(p.numel() for p in self.conv3.parameters()) +
                     sum(p.numel() for p in self.norm1.parameters()) + 
                     sum(p.numel() for p in self.norm2.parameters()) +
                     sum(p.numel() for p in self.norm3.parameters()))
        pos_enc_params = sum(p.numel() for p in self.pos_encoding.parameters())
        prenorm_params = sum(p.numel() for p in self.cnn_prenorm.parameters()) if self.cnn_prenorm else 0
        spatial_norm_params = sum(p.numel() for p in self.spatial_norm.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        total_params = (cnn_params + pos_enc_params + prenorm_params + scalar_params + 
                       attn_params + direct_params + fusion_params + spatial_norm_params)
        
        # Percentages
        attended_percentage = (attended_cnn_dim / fusion_input_dim) * 100 if fusion_input_dim > 0 else 0
        direct_percentage = (direct_cnn_dim / fusion_input_dim) * 100 if fusion_input_dim > 0 else 0
        cnn_total_percentage = ((attended_cnn_dim + direct_cnn_dim) / fusion_input_dim) * 100 if fusion_input_dim > 0 else 0
        scalar_percentage = (scalar_output_dim / fusion_input_dim) * 100 if fusion_input_dim > 0 else 0
        
        # Compression ratios
        attended_compression = cnn_raw_features / attended_cnn_dim if attended_cnn_dim > 0 else 0
        direct_compression = cnn_raw_features / direct_cnn_dim if direct_cnn_dim > 0 else 0
        
        # Receptive field calculation for HYBRID architecture
        # Conv1: 5×5, stride=1 → RF = 5
        # Conv2: 3×3, stride=1 → RF = 5 + (3-1)*1 = 7
        # Conv3: 3×3, stride=2 → RF = 7 + (3-1)*1 = 9
        rf = 1
        stride_product = 1
        # Conv1 (5×5)
        rf = rf + (5 - 1) * stride_product
        stride_product *= 1
        # Conv2 (3×3)
        rf = rf + (3 - 1) * stride_product
        stride_product *= 1
        # Conv3 (3×3)
        rf = rf + (3 - 1) * stride_product
        stride_product *= 2
        self.receptive_field = rf
        
        if not _INFO_PRINTED:
            print(f"\n{'='*70}")
            if self.use_scalars:
                print(f"🔥 HYBRID CNN: 5×5 → 3×3 → 3×3 (BEST OF BOTH!)")
            else:
                print(f"🔥 CNN-ONLY MODE: Hybrid Architecture")
            print(f"{'='*70}")
            
            print(f"\n[CNN] 🗃️ Hybrid Architecture:")
            print(f"[CNN] Block 1: Conv(5×5,s=1) → BatchNorm2d → SiLU  [RF: 5×5] 🔥 Stabilne!")
            print(f"[CNN] Block 2: Conv(3×3,s=1) → BatchNorm2d → SiLU  [RF: 7×7] 🔥 Spójna normalizacja!")
            print(f"[CNN] Block 3: Conv(3×3,s=2) → BatchNorm2d → SiLU  [RF: 9×9] 🔥 No conflict!")
            print(f"[CNN] Dropout: {cnn_dropout}")
            print(f"[CNN] Size: {viewport_size}×{viewport_size} → {spatial_size}×{spatial_size} (via stride=2)")
            print(f"[CNN] Channels: {cnn_channels}")
            print(f"[CNN] ✅ Receptive Field: {self.receptive_field}×{self.receptive_field} pixels")
            print(f"[CNN] 🎯 OPTIMAL for 11×11 viewport (RF fits perfectly!)")
            print(f"[CNN] Raw features: {cnn_raw_features} ({cnn_spatial_dim}×{spatial_locations})")
            print(f"[CNN] 📊 Params: {cnn_params:,}")
            print(f"[CNN] LayerNorm (spatial): {spatial_norm_params:,} params → równy gradient w każdej ścieżce")
            
            print(f"\n[HYBRID] 🌟 WHY 5×5 → 3×3 → 3×3?")
            print(f"[HYBRID] ✅ Layer 1 (5×5): Fast global context (RF=5×5 immediately)")
            print(f"[HYBRID] ✅ Layer 2 (3×3): Refine features without detail loss")
            print(f"[HYBRID] ✅ Layer 3 (3×3): Learned downsampling (no MaxPool!)")
            print(f"[HYBRID] ✅ Best of both: Speed of 5×5 + Precision of 3×3")
            print(f"[HYBRID] ✅ RF progression: 5→7→9 (gradual but fast!)")
            
            print(f"\n[STRIDED] 🔥 WHY NO MAXPOOL?")
            print(f"[STRIDED] ❌ MaxPool: Takes MAX from 2×2 → small objects disappear!")
            print(f"[STRIDED] ✅ Strided Conv: LEARNS what to keep during downsampling")
            print(f"[STRIDED] ✅ Better for 1-pixel food detection")
            print(f"[STRIDED] ✅ Smooth downsampling (not hard max)")
            
            print(f"\n[POS-ENC] 🎯 2D Sinusoidal Encoding")
            print(f"[POS-ENC] Shape: ({cnn_channels[2]}, {spatial_size}, {spatial_size}) * scale={pos_encoding_scale}")
            print(f"[POS-ENC] 📊 Params: {pos_enc_params:,} (frozen)")
            
            if self.use_scalars:
                print(f"\n[PRENORM] 🔥 LayerNorm BEFORE attention")
                print(f"[PRENORM] ✅ Normalizes CNN features → better gradient flow!")
                print(f"[PRENORM] 📊 Params: {prenorm_params:,}")
                
                print(f"\n[SCALARS] 🔢 Network:")
                print(f"[SCALARS] Input: 8 → LayerNorm (🔥 stabilizacja!) → Dropout → Network")
                print(f"[SCALARS] 8 → {scalar_hidden_dims} → {scalar_output_dim}")
                print(f"[SCALARS] 📊 Params: {scalar_params:,} (inkl. input norm: {scalar_input_norm_params})")
                
                print(f"\n[ATTENTION] 🎯 Multi-Query Attention (FIXED)")
                print(f"[ATTENTION] ❌ Removed radial encoding (was causing CNN ignore)")
                print(f"[ATTENTION] ✅ Direct output: {cnn_spatial_dim}×{num_queries} → {attended_cnn_dim}")
                print(f"[ATTENTION] Heads: {attention_heads}, Queries: {num_queries}")
                print(f"[ATTENTION] 🔥 Temperature: {attention_temperature} ({'sharper' if attention_temperature < 1.0 else 'softer' if attention_temperature > 1.0 else 'neutral'})")
                print(f"[ATTENTION] Compression: {cnn_raw_features}/{attended_cnn_dim} = {attended_compression:.1f}x")
                print(f"[ATTENTION] 📊 Params: {attn_params:,}")
                
                print(f"\n[SKIP-CNN] ✨ Direct CNN Path (Skip Connection)")
                print(f"[SKIP-CNN] Raw CNN → LayerNorm → Linear → GELU → Fusion")
                print(f"[SKIP-CNN] {cnn_raw_features} → {direct_cnn_dim}")
                print(f"[SKIP-CNN] Compression: {direct_compression:.1f}x")
                print(f"[SKIP-CNN] Dropout: {'YES (high compression)' if compression_ratio > 2.0 and cnn_dropout > 0 else 'NO (low compression)'}")
                print(f"[SKIP-CNN] ✅ Ensures CNN features reach fusion!")
                print(f"[SKIP-CNN] 📊 Params: {direct_params:,}")
            else:
                print(f"\n[SCALARS] ❌ DISABLED (scalar_hidden_dims: [0])")
                print(f"[ATTENTION] ❌ SKIPPED (no scalars available)")
                print(f"\n[RAW-CNN] 🔥 Stabilized Direct Path")
                print(f"[RAW-CNN] Spatial Features → LayerNorm → Linear ({direct_cnn_dim}) → Fusion")
                print(f"[RAW-CNN] Compression: {direct_compression:.1f}x (konfigurowalne)")
                print(f"[RAW-CNN] Dropout: {'YES (moderate compression)' if compression_ratio > 2.0 and cnn_dropout > 0 else 'NO (preserving capacity)'}")
                print(f"[RAW-CNN] ✅ Better capacity vs GPU tradeoff!")
                print(f"[RAW-CNN] 📊 Params: {direct_params:,}")
            
            print(f"\n[FUSION] 🎯 Input Breakdown:")
            if self.use_scalars:
                print(f"[FUSION]   ├─ Attended CNN:   {attended_cnn_dim:4d} dim ({attended_percentage:5.1f}%) [via attention]")
                print(f"[FUSION]   ├─ Direct CNN:     {direct_cnn_dim:4d} dim ({direct_percentage:5.1f}%) [skip connection] ✨")
                print(f"[FUSION]   └─ Scalar features: {scalar_output_dim:4d} dim ({scalar_percentage:5.1f}%)")
            else:
                print(f"[FUSION]   └─ Raw CNN:        {direct_cnn_dim:4d} dim (100.0%) [ustabilizowany direct path] 🔥")
            print(f"[FUSION]   ────────────────────────────────────────────")
            print(f"[FUSION]   CNN Total:  {attended_cnn_dim + direct_cnn_dim:4d} dim ({cnn_total_percentage:5.1f}%) 🔥")
            print(f"[FUSION]   Total: {fusion_input_dim:4d} → {features_dim} dim")
            if self.use_scalars:
                print(f"[FUSION]   ✅ CNN DOMINATES: {cnn_total_percentage:.0f}% vs Scalars {scalar_percentage:.0f}%!")
            else:
                print(f"[FUSION]   ✅ PURE CNN: 100% raw spatial features (no compression)!")
            print(f"[FUSION]   📊 Params: {fusion_params:,}")
            
            print(f"\n[FIXES] ✅")
            if self.use_scalars:
                print(f"  1. ✅ Hybrid 5×5→3×3→3×3 → best of both worlds!")
                print(f"  2. ✅ Strided Conv (no MaxPool) → better small object detection")
                print(f"  3. ✅ Learned downsampling → preserves food features")
                print(f"  4. ✅ LayerNorm before attention → stable gradients")
                print(f"  5. ✅ Skip connection → guaranteed CNN usage")
                print(f"  6. ✅ CNN dominance: {cnn_total_percentage:.0f}% fusion input")
            else:
                print(f"  1. ✅ Pure CNN mode (no scalars)")
                print(f"  2. ✅ Hybrid architecture (fast + precise)")
                print(f"  3. ✅ Strided Conv → no MaxPool loss")
                print(f"  4. ✅ Direct path z LayerNorm → żadnego biasu w fusion")
                print(f"  5. ✅ Smart compression: {direct_cnn_dim} dims (1728→{direct_cnn_dim} = {direct_compression:.1f}x)")
                print(f"  6. ✅ Conditional dropout: only if compression >2x")
            print(f"  7. ✅ Perfect RF={self.receptive_field}×{self.receptive_field} for viewport {viewport_size}×{viewport_size}")
            
            print(f"\n[TOTAL] 🎯 Parameters: {total_params:,}")
            print(f"{'='*70}\n")
            _INFO_PRINTED = True

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, observations):
        image = observations['image']
        if image.dim() == 4 and image.shape[-1] == 1:
            image = image.permute(0, 3, 1, 2)
        
        B = image.shape[0]
        
        # ==================== HYBRID CNN ====================
        with autocast('cuda', enabled=self.use_amp):
            x = image
            
            # Block 1: Fast global context with 5×5 (RF: 5×5)
            x = self.conv1(x)
            x = self.norm1(x)
            x = F.silu(x)
            
            # Block 2: Refine features with 3×3 (RF: 7×7)
            x = self.conv2(x)
            x = self.norm2(x)
            x = F.silu(x)
            
            # Block 3: Final features + STRIDED DOWNSAMPLE with 3×3 (RF: 9×9)
            x = self.conv3(x)  # 11×11 → 6×6 via stride=2 🔥
            x = self.norm3(x)
            x = F.silu(x)
            x = self.dropout3(x)
            
            # NO MAXPOOL! Downsampling done by strided conv
            
            # Positional Encoding
            x = self.pos_encoding(x)
            
            # Flatten for attention: (B, C, H, W) → (B, H*W, C)
            spatial_features = x.flatten(2).transpose(1, 2)
        
        spatial_features = spatial_features.float()
        spatial_features = self.spatial_norm(spatial_features)
        
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
            
            # 🔥 FIX: Normalize → Dropout → Network (stabilizacja gradientów!)
            scalars = self.scalar_input_norm(scalars)
            scalars = self.scalar_input_dropout(scalars)
            scalar_features = self.scalar_network(scalars)
            
            # PATH 1: ATTENTION
            normalized_features = self.cnn_prenorm(spatial_features) if self.cnn_prenorm else spatial_features
            attended_cnn = self.cross_attention(normalized_features, scalar_features)
            
            # Fusion: Attended + Direct + Scalars
            fused = torch.cat([attended_cnn, direct_cnn, scalar_features], dim=-1)
            
        else:
            # ========== MODE 2: CNN-ONLY (Direct Spatial Features) ==========
            cnn_raw = spatial_features.flatten(1)
            fused = self.cnn_direct(cnn_raw)
        
        return self.fusion(fused)