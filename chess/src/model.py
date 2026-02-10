import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.data_helpers import ACTION_SIZE


class SE2DBlock(nn.Module):
    """
    🆕 v4.7: TRUE Squeeze-and-Excitation with Channel Attention
    
    FIXED from v3.1: Previous version was just a 1x1 conv bottleneck
    (no global pooling = no actual squeeze = NOT a real SE block!)
    
    Now properly:
    1. SQUEEZE: Global Average Pool -> (B, C, 1, 1)
    2. EXCITE:  FC(C/r) -> ReLU -> FC(C) -> Sigmoid
    3. SCALE:   x * scale (channel-wise attention)
    
    This gives TRUE channel attention ("which channels matter?")
    instead of spatial-dependent scaling.
    
    PAPER: "Squeeze-and-Excitation Networks" (Hu et al. 2018)
    """
    def __init__(self, filters, reduction=16):
        super().__init__()
        mid = max(filters // reduction, 8)  # Ensure at least 8 channels
        self.squeeze = nn.AdaptiveAvgPool2d(1)  # Global Average Pool
        self.fc1 = nn.Conv2d(filters, mid, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(mid, filters, kernel_size=1, bias=False)
    
    def forward(self, x):
        # Squeeze: global average pooling (B, C, H, W) -> (B, C, 1, 1)
        scale = self.squeeze(x)
        # Excitation: bottleneck FC -> channel-wise weights
        scale = F.relu(self.fc1(scale), inplace=True)  # (B, C/r, 1, 1)
        scale = torch.sigmoid(self.fc2(scale))          # (B, C, 1, 1)
        # Scale: channel-wise attention (broadcasts over H, W)
        return x * scale


class CoordConv2d(nn.Module):
    """
    đźš€ OPTIMIZED CoordConv - Position-aware convolution
    No changes - already optimal
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, padding=padding, bias=bias)
        
        # Eager initialization for 8x8 board
        h, w = 8, 8
        xx = torch.linspace(-1, 1, w)
        yy = torch.linspace(-1, 1, h)
        yy, xx = torch.meshgrid(yy, xx, indexing='ij')
        
        self.register_buffer('coord_x', xx.unsqueeze(0).unsqueeze(0))
        self.register_buffer('coord_y', yy.unsqueeze(0).unsqueeze(0))
        
    def forward(self, x):
        batch = x.size(0)
        
        coords_x = self.coord_x.expand(batch, 1, 8, 8)
        coords_y = self.coord_y.expand(batch, 1, 8, 8)
        
        x = torch.cat([x, coords_x, coords_y], dim=1)
        
        return self.conv(x)


class LayerScale(nn.Module):
    """
    đź†• v3.1: LayerScale for better training stability
    
    PAPER: "Going deeper with Image Transformers" (Touvron et al. 2021)
    
    BENEFITS:
    - Enables training of very deep networks (50+ layers)
    - Better gradient flow
    - 0% parameter overhead (just 1 scalar per channel)
    - Typical init: 1e-5 to 1e-4
    
    USAGE:
    - Apply after residual branch, before adding to skip connection
    - Essential for networks with 20+ blocks
    """
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        # Convert to float to handle YAML string inputs like '1e-5'
        init_value = float(init_value)
        self.gamma = nn.Parameter(torch.ones(dim) * init_value)
    
    def forward(self, x):
        # x: (B, C, H, W)
        # gamma: (C,)
        # Broadcasting: (C,) -> (1, C, 1, 1)
        return x * self.gamma.view(1, -1, 1, 1)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    đźš€ OPTIMIZED Stochastic Depth
    """
    if drop_prob == 0. or not training:
        return x
    
    keep_prob = 1 - drop_prob
    
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
    
    return x * mask / keep_prob


class ResidualBlock(nn.Module):
    """
    Pre-activation Residual Block
    
    ARCHITECTURE:
    - POST-ACTIVATION (standard):  x -> Conv -> BN -> ReLU -> Conv -> BN -> (+x) -> ReLU
    - PRE-ACTIVATION (v4.0):       x -> BN -> ReLU -> Conv -> BN -> ReLU -> Conv -> (+x)
    
    BENEFITS OF PRE-ACTIVATION:
    ? Better gradient flow (no ReLU after addition)
    ? More stable training for deep networks
    ? Easier to train 20+ layer networks
    ? Identity mapping is cleaner
    
    PAPER: "Identity Mappings in Deep Residual Networks" (He et al. 2016)
    """
    def __init__(self, filters, use_se2d=False,
                 drop_path_rate=0.0, use_layer_scale=False, layer_scale_init=1e-5):
        super().__init__()
        
        # BN + Conv layers
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        
        self.use_se2d = use_se2d
        if use_se2d:
            self.se = SE2DBlock(filters)
        
        self.drop_path_rate = drop_path_rate
        
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = LayerScale(filters, init_value=layer_scale_init)
    
    def forward(self, x):
        # ?? PRE-ACTIVATION PATH: BN -> ReLU -> Conv -> BN -> ReLU -> Conv
        residual = x
        
        # First conv path: BN -> ReLU -> Conv
        out = self.bn1(x)
        out = F.relu(out, inplace=True)
        out = self.conv1(out)
        
        # Second conv path: BN -> ReLU -> Conv
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        
        # Attention (after convs)
        if self.use_se2d:
            out = self.se(out)
        
        if self.use_layer_scale:
            out = self.layer_scale(out)
        
        if self.drop_path_rate > 0:
            out = drop_path(out, self.drop_path_rate, self.training)
        
        # ? Clean identity mapping (no activation after addition!)
        return residual + out

class ChessNet(nn.Module):
    """
    đź†• Chess Neural Network v4.0 with Pre-activation ResNet
    
    CHANGES from v3.1:
    - đź†• Pre-activation ResNet blocks (BN->ReLU->Conv instead of Conv->BN->ReLU)
    - âś… Better gradient flow for deeper networks
    - âś… More stable training
    - âś… Automatically calculates input_planes from history_positions
    - âś… Formula: input_planes = 16 * (1 + history_positions)
    """
    def __init__(self, config, input_planes=None):
        """
        Args:
            config: Configuration dict
            input_planes: Optional override (auto-calculated if None)
        """
        super().__init__()
        
        model_version = config.get('model', {}).get('version', 'v?.?')
        self.model_version = model_version

        filters = config['model']['filters']
        num_blocks = config['model']['num_residual_blocks']
        dropout = config['model']['dropout']
        
        use_se2d = config['model'].get('use_se2d_blocks', False)
        drop_path_rate = config['model'].get('drop_path_rate', 0.1)
        use_coord_conv = config['model'].get('use_coord_conv', True)
        
        use_layer_scale = config['model'].get('use_layer_scale', True)
        layer_scale_init = config['model'].get('layer_scale_init', 1e-5)
        
        # Multi-Task Learning
        self.use_mtl = config['model'].get('use_multitask_learning', False)
        
        # v4.5: AUTO-CALCULATE input_planes with chess metadata
        # Base: 16 planes (12 pieces + 4 metadata: castling, en passant, halfmove, fullmove)
        # With history: 16 * (1 + history_positions)
        history_positions = config['model']['history_positions']
        if input_planes is None:
            input_planes = 16 * (1 + history_positions)  # 16 instead of 12!
        
        self.input_planes = input_planes
        self.history_positions = history_positions
        
        print_summary = config['model'].get('print_summary', True)
        if print_summary:
            print(f"[MODEL {model_version}] ULTRA-OPTIMIZED (Chess Metadata + Pre-activation ResNet):")
            print(f"  > History positions: {history_positions}")
            print(f"  > Input planes: {input_planes} (16 x {1 + history_positions})")
            print(f"  > Chess metadata: Castling, En Passant, Halfmove, Fullmove")
            
            if use_se2d:
                print(f"  > SE2D-Block: ENABLED (spatial-aware, +5% time)")
            else:
                print(f"  > SE2D-Block: DISABLED")
            
            print(f"  > CoordConv: Only at input")
            print(f"  > Activation: ReLU (5-10x faster than ELU)")
            print(f"  > Stochastic Depth: {drop_path_rate}")
            # Pre-activation ResNet is always enabled
            print(f"  > Pre-activation ResNet: ENABLED (better gradients)")
            
            if use_layer_scale:
                print(f"  > LayerScale: ENABLED (init={layer_scale_init})")
            
            
            print(f"  > Standard 3x3 Conv: ENABLED (preserves spatial info for chess)")
            print(f"  > Policy Head: Standard flatten+FC (spatial preservation for chess)")
            
            if self.use_mtl:
                print(f"  > MTL: ENABLED (Win, Material, Check auxiliary tasks)")
        
        # Input conv with dynamic input_planes
        if use_coord_conv:
            self.conv_block = nn.Sequential(
                CoordConv2d(input_planes, filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(filters),
            )
        else:
            self.conv_block = nn.Sequential(
                nn.Conv2d(input_planes, filters, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(filters),
            )
        
        # Stochastic depth schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        
        # Residual tower
        blocks = []
        for i in range(num_blocks):
            blocks.append(
                ResidualBlock(
                    filters,
                    use_se2d=use_se2d,
                    drop_path_rate=dpr[i],
                    use_layer_scale=use_layer_scale,
                    layer_scale_init=layer_scale_init,
                )
            )
        self.residual_tower = nn.Sequential(*blocks)
        
        # Policy head
        policy_filters = config['model']['policy_head_filters']
        self.policy_conv = nn.Conv2d(filters, policy_filters, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_filters)
        # Policy head keeps full spatial information until the final FC layer.
        # Chess requires spatial information until the very end (position matters!)
        # Use standard flatten + FC for full spatial preservation
        
        self.policy_fc = nn.Linear(policy_filters * 8 * 8, ACTION_SIZE)
        
        self.policy_dropout = nn.Dropout(dropout)
        
        # đź†• Value head - WDL (Win/Draw/Loss) classification
        # AlphaZero-style: Conv 1x1 → BN → ReLU → GlobalAvgPool → FC → WDL
        value_filters = config['model']['value_head_filters']
        value_hidden = config['model']['value_hidden_dim']
        self.value_conv = nn.Conv2d(filters, value_filters, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(value_filters)
        self.value_fc1 = nn.Linear(value_filters, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 3)  # đź†• 3 outputs: [Win, Draw, Loss]
        self.value_dropout = nn.Dropout(dropout)
        
        # Track if we use WDL
        self.use_wdl = config['model'].get('use_wdl_value', True)
        
        # Shared GAP for value + MTL heads
        self.shared_gap = nn.AdaptiveAvgPool2d(1)
        
        # MTL heads (reuse shared_gap)
        if self.use_mtl:
            self.win_fc1 = nn.Linear(filters, 128)
            self.win_fc2 = nn.Linear(128, 1)
            self.win_dropout = nn.Dropout(dropout * 0.5)
            
            self.material_fc1 = nn.Linear(filters, 64)
            self.material_fc2 = nn.Linear(64, 1)
            
            self.check_fc = nn.Linear(filters, 1)
    

    def forward(self, x, return_aux=False):
        """Forward pass"""
        x = x.contiguous(memory_format=torch.channels_last)
        
        x = self.conv_block(x)
        x = F.relu(x, inplace=True)
        
        x = self.residual_tower(x)
        
        # Policy head
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = F.relu(policy, inplace=True)
        policy = policy.flatten(1)  # đź”§ v4.4: Always flatten (no adaptive pool)
        policy = self.policy_dropout(policy)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        
        # 🆕 Value head - AlphaZero-style: Conv → BN → ReLU → GAP → FC → WDL
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = F.relu(value, inplace=True)
        value = self.shared_gap(value)  # GAP: (B, C, 8, 8) → (B, C, 1, 1)
        value = value.flatten(1)        # (B, C, 1, 1) → (B, C)
        value = F.relu(self.value_fc1(value), inplace=True)
        value = self.value_dropout(value)
        value = self.value_fc2(value)  # 🆕 Returns (B, 3) WDL logits
        
        if not return_aux or not self.use_mtl:
            return policy, value
        
        # MTL predictions (shared GAP on trunk features)
        mtl_pooled = self.shared_gap(x).flatten(1)  # (B, filters)
        
        win_pred = F.relu(self.win_fc1(mtl_pooled), inplace=False)
        win_pred = self.win_dropout(win_pred)
        win_pred = self.win_fc2(win_pred)
        
        material_pred = F.relu(self.material_fc1(mtl_pooled), inplace=False)
        material_pred = torch.tanh(self.material_fc2(material_pred))
        
        check_pred = self.check_fc(mtl_pooled)
        
        return policy, value, win_pred, material_pred, check_pred
    
    def predict(self, board_tensor):
        """
        Predict for a single position (used in MCTS)
        đź†• Handles WDL output and converts to scalar
        """
        self.eval()
        with torch.inference_mode():
            if len(board_tensor.shape) == 3:
                board_tensor = board_tensor.unsqueeze(0)
            policy, value_logits = self.forward(board_tensor, return_aux=False)
            
            # Convert WDL logits to scalar value
            if self.use_wdl:
                # value_logits: (1, 3) -> [Win, Draw, Loss]
                wdl_probs = F.softmax(value_logits, dim=1)
                # Scalar: W*1.0 + D*0.0 + L*(-1.0) = W - L
                value_scalar = wdl_probs[0, 0] - wdl_probs[0, 2]
                return torch.exp(policy).cpu().numpy()[0], value_scalar.item()
            else:
                # Legacy scalar output
                return torch.exp(policy).cpu().numpy()[0], value_logits.cpu().item()


def load_model(checkpoint_path, config, device):
    """Load model from checkpoint"""
    model = ChessNet(config).to(device)
    model = model.to(memory_format=torch.channels_last)
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def save_checkpoint(model, optimizer, epoch, loss, path, metadata=None, save_optimizer=False, save_dtype=None):
    """Save model checkpoint
    
    Args:
        save_dtype: Optional dtype to convert state_dict (e.g. torch.bfloat16)
                    Converts the saved copy WITHOUT modifying the live model.
    """
    state_dict = model.state_dict()
    if save_dtype is not None:
        state_dict = {k: v.to(save_dtype) for k, v in state_dict.items()}
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': state_dict,
        'loss': loss,
    }
    
    if save_optimizer and optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if metadata:
        checkpoint.update(metadata)
    
    torch.save(checkpoint, path)
    
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 ** 2)
        opt_status = "with optimizer" if save_optimizer else "without optimizer"
        print(f"Checkpoint saved to {path} ({size_mb:.2f} MB, {opt_status})")
