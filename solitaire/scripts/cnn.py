import torch
import torch.nn as nn
import gymnasium as gym
import os
import yaml
from torch.amp import autocast
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Load config
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

_INFO_PRINTED = False


class AttentionModule(nn.Module):
    """
    Simple attention mechanism dla Solitaire
    Pomaga skupić się na ważnych kartach (top cards, możliwe ruchy)
    """
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [B, D]
        weights = self.attention(x)  # [B, 1]
        return x * weights


class SolitaireFeaturesExtractor(BaseFeaturesExtractor):
    """
    ✅ Ulepszone CNN dla Solitaire z:
    - Attention mechanism
    - Dropout regularization
    - LayerNorm dla stabilności
    - BF16 mixed precision
    - Głębsze sieci oparte na Snake
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        global _INFO_PRINTED
        
        # 🔧 Load from config
        use_layernorm = config['model']['features_extractor'].get('use_layernorm', True)
        use_attention = config['model']['features_extractor'].get('use_attention', True)
        
        tableau_dropout = config['model'].get('tableau_dropout', 0.0)
        foundations_dropout = config['model'].get('foundations_dropout', 0.0)
        waste_dropout = config['model'].get('waste_dropout', 0.0)
        stock_dropout = config['model'].get('stock_dropout', 0.0)
        fusion_dropout = config['model'].get('fusion_dropout', 0.0)
        
        # Architecture from config
        tableau_hidden = config['model']['features_extractor'].get('tableau_hidden', [384, 256])
        foundations_hidden = config['model']['features_extractor'].get('foundations_hidden', 64)
        waste_hidden = config['model']['features_extractor'].get('waste_hidden', 64)
        stock_hidden = config['model']['features_extractor'].get('stock_hidden', [128, 96])
        
        # ✅ BF16 dla szybszego treningu
        self.use_amp = torch.cuda.is_available()
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        # ==================== TABLEAU NETWORK ====================
        # Input: (7, 20, 5) -> Flatten -> 700 (Added Color)
        self.tableau_dim = 7 * 20 * 5
        
        tableau_layers = [nn.Flatten()]
        in_dim = self.tableau_dim
        
        for hidden_dim in tableau_hidden:
            tableau_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
                nn.GELU(),
                nn.Dropout(tableau_dropout) if tableau_dropout > 0 else nn.Identity()
            ])
            in_dim = hidden_dim
        
        self.tableau_net = nn.Sequential(*tableau_layers)
        tableau_output_dim = in_dim
        
        # ==================== FOUNDATIONS NETWORK ====================
        # Input: (4,)
        self.foundations_dim = 4
        self.foundations_net = nn.Sequential(
            nn.Linear(self.foundations_dim, foundations_hidden),
            nn.LayerNorm(foundations_hidden) if use_layernorm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(foundations_dropout) if foundations_dropout > 0 else nn.Identity()
        )
        
        # ==================== WASTE NETWORK ====================
        # Input: (4,)
        self.waste_dim = 4
        self.waste_net = nn.Sequential(
            nn.Linear(self.waste_dim, waste_hidden),
            nn.LayerNorm(waste_hidden) if use_layernorm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(waste_dropout) if waste_dropout > 0 else nn.Identity()
        )
        
        # ==================== STOCK NETWORK ====================
        # Input: (52, 4) -> Flatten -> 208
        self.stock_dim = 52 * 4
        
        stock_layers = [nn.Flatten()]
        in_dim = self.stock_dim
        
        for hidden_dim in stock_hidden:
            stock_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
                nn.GELU(),
                nn.Dropout(stock_dropout) if stock_dropout > 0 else nn.Identity()
            ])
            in_dim = hidden_dim
        
        self.stock_net = nn.Sequential(*stock_layers)
        stock_output_dim = in_dim
        
        # ==================== ATTENTION (opcjonalne) ====================
        self.use_attention = use_attention
        if use_attention:
            self.tableau_attention = AttentionModule(tableau_output_dim)
            self.stock_attention = AttentionModule(stock_output_dim)
        
        # ==================== FUSION ====================
        fusion_input_dim = tableau_output_dim + foundations_hidden + waste_hidden + stock_output_dim
        
        self.fusion_net = nn.Sequential(
            nn.LayerNorm(fusion_input_dim) if use_layernorm else nn.Identity(),
            nn.Linear(fusion_input_dim, features_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout) if fusion_dropout > 0 else nn.Identity()
        )
        
        self._initialize_weights()
        
        # ==================== INFO PRINT ====================
        tableau_params = sum(p.numel() for p in self.tableau_net.parameters())
        foundations_params = sum(p.numel() for p in self.foundations_net.parameters())
        waste_params = sum(p.numel() for p in self.waste_net.parameters())
        stock_params = sum(p.numel() for p in self.stock_net.parameters())
        fusion_params = sum(p.numel() for p in self.fusion_net.parameters())
        
        attention_params = 0
        if use_attention:
            attention_params = sum(p.numel() for p in self.tableau_attention.parameters())
            attention_params += sum(p.numel() for p in self.stock_attention.parameters())
        
        total_params = tableau_params + foundations_params + waste_params + stock_params + fusion_params + attention_params
        
        if not _INFO_PRINTED:
            print(f"\n{'='*70}")
            print(f"[SOLITAIRE FEATURES EXTRACTOR]")
            print(f"{'='*70}")
            print(f"  Tableau Network:     {tableau_params:>10,} params  (hidden: {tableau_hidden})")
            print(f"  Foundations Network: {foundations_params:>10,} params  (hidden: {foundations_hidden})")
            print(f"  Waste Network:       {waste_params:>10,} params  (hidden: {waste_hidden})")
            print(f"  Stock Network:       {stock_params:>10,} params  (hidden: {stock_hidden})")
            if use_attention:
                print(f"  Attention Modules:   {attention_params:>10,} params  ✅ ENABLED")
            print(f"  Fusion Network:      {fusion_params:>10,} params")
            print(f"  {'─'*70}")
            print(f"  TOTAL PARAMETERS:    {total_params:>10,} params")
            print(f"{'='*70}")
            print(f"  LayerNorm:           {'✅ ENABLED' if use_layernorm else '❌ DISABLED'}")
            print(f"  Attention:           {'✅ ENABLED' if use_attention else '❌ DISABLED'}")
            print(f"  Mixed Precision:     {'✅ BF16' if self.use_amp else '❌ FP32'}")
            print(f"  Tableau Dropout:     {tableau_dropout}")
            print(f"  Stock Dropout:       {stock_dropout}")
            print(f"  Fusion Dropout:      {fusion_dropout}")
            print(f"{'='*70}\n")
            _INFO_PRINTED = True

    def _initialize_weights(self):
        """He initialization dla GELU activation"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, observations):
        tableau = observations['tableau']
        foundations = observations['foundations']
        waste = observations['waste']
        stock = observations['stock']
        
        # ==================== PROCESS EACH COMPONENT ====================
        with autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            t_out = self.tableau_net(tableau)
            f_out = self.foundations_net(foundations)
            w_out = self.waste_net(waste)
            s_out = self.stock_net(stock)
        
        # Convert back to FP32 for fusion
        t_out = t_out.float()
        f_out = f_out.float()
        w_out = w_out.float()
        s_out = s_out.float()
        
        # ==================== ATTENTION (optional) ====================
        if self.use_attention:
            t_out = self.tableau_attention(t_out)
            s_out = self.stock_attention(s_out)
        
        # ==================== FUSION ====================
        fusion_in = torch.cat([t_out, f_out, w_out, s_out], dim=1)
        return self.fusion_net(fusion_in)
