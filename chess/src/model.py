import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.data_helpers import ACTION_PLANES, ACTION_SIZE, get_policy_index_maps


class SEBlock(nn.Module):
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
    def __init__(self, filters, reduction=16, use_bottleneck=True):
        super().__init__()
        filters = int(filters)
        if filters < 1:
            raise ValueError(f"filters must be >= 1, got {filters}")

        self.use_bottleneck = bool(use_bottleneck)
        self.reduction = int(reduction)
        if self.use_bottleneck and self.reduction < 1:
            raise ValueError(f"se_reduction must be >= 1 when use_se_bottleneck=true, got {self.reduction}")

        self.squeeze = nn.AdaptiveAvgPool2d(1)  # Global Average Pool
        if self.use_bottleneck:
            mid = max(filters // self.reduction, 8)  # Ensure at least 8 channels
            self.mid = mid
            self.fc1 = nn.Conv2d(filters, mid, kernel_size=1, bias=False)
            self.fc2 = nn.Conv2d(mid, filters, kernel_size=1, bias=True)  # 🔧 v4.8: bias=True for better calibration
        else:
            self.mid = filters
            self.fc = nn.Conv2d(filters, filters, kernel_size=1, bias=True)
    
    def forward(self, x):
        # Squeeze: global average pooling (B, C, H, W) -> (B, C, 1, 1)
        scale = self.squeeze(x)
        # Excitation: optional bottleneck FC -> channel-wise weights
        if self.use_bottleneck:
            scale = F.relu(self.fc1(scale), inplace=True)  # (B, C/r, 1, 1)
            scale = torch.sigmoid(self.fc2(scale))          # (B, C, 1, 1)
        else:
            scale = torch.sigmoid(self.fc(scale))           # (B, C, 1, 1)
        # Scale: channel-wise attention (broadcasts over H, W)
        return x * scale


class CoordConv2d(nn.Module):
    """
    đźš€ OPTIMIZED CoordConv - Position-aware convolution
    No changes - already optimal
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, padding=padding, bias=bias)
        
        # Eager initialization for 8x8 board
        h, w = 8, 8
        xx = torch.linspace(-1, 1, w)
        yy = torch.linspace(-1, 1, h)
        yy, xx = torch.meshgrid(yy, xx, indexing='ij')
        
        # Ensure non-overlapping contiguous storage so load_state_dict can copy safely.
        self.register_buffer('coord_x', xx.unsqueeze(0).unsqueeze(0).contiguous())
        self.register_buffer('coord_y', yy.unsqueeze(0).unsqueeze(0).contiguous())
        
    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"CoordConv2d expected 4D input (B,C,H,W), got shape {tuple(x.shape)}")
        if x.shape[-2:] != (8, 8):
            raise ValueError(
                f"CoordConv2d expects board spatial size 8x8, got {x.shape[-2]}x{x.shape[-1]}"
            )

        batch = x.size(0)
        
        coords_x = self.coord_x.expand(batch, 1, 8, 8)
        coords_y = self.coord_y.expand(batch, 1, 8, 8)
        if coords_x.dtype != x.dtype:
            coords_x = coords_x.to(dtype=x.dtype)
            coords_y = coords_y.to(dtype=x.dtype)
        
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
    def __init__(self, filters, use_se=False, use_se_bottleneck=True,
                 drop_path_rate=0.0, use_layer_scale=False, layer_scale_init=1e-5,
                 se_reduction=16):
        super().__init__()
        
        # BN + Conv layers
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(filters, reduction=se_reduction, use_bottleneck=use_se_bottleneck)
        
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
        if self.use_se:
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
        
        use_se = config['model'].get('use_se_blocks', False)
        use_se_bottleneck = config['model'].get('use_se_bottleneck', True)
        se_reduction = int(config['model'].get('se_reduction', 16))
        drop_path_rate = config['model'].get('drop_path_rate', 0.1)
        use_coord_conv = config['model'].get('use_coord_conv', True)
        
        use_layer_scale = config['model'].get('use_layer_scale', True)
        layer_scale_init = config['model'].get('layer_scale_init', 1e-5)

        # Policy head: LC0-style compact policy gathered from 8x8x73 conv planes.
        policy_channels = int(config['model'].get('policy_head_channels', 96))
        if policy_channels < 1:
            raise ValueError(f"policy_head_channels must be >= 1, got {policy_channels}")
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
            
            if use_se:
                if use_se_bottleneck:
                    mid = max(filters // se_reduction, 8)
                    print(f"  > SE blocks: ENABLED (bottleneck=yes, reduction={se_reduction}, mid={mid})")
                else:
                    print(f"  > SE blocks: ENABLED (bottleneck=no, direct channel gate)")
            else:
                print(f"  > SE blocks: DISABLED")
            
            print(f"  > CoordConv: Only at input")
            print(f"  > Activation: ReLU (5-10x faster than ELU)")
            print(f"  > Stochastic Depth: {drop_path_rate}")
            # Pre-activation ResNet is always enabled
            print(f"  > Pre-activation ResNet: ENABLED (better gradients)")
            
            if use_layer_scale:
                print(f"  > LayerScale: ENABLED (init={layer_scale_init})")
            
            
            print(f"  > Policy conv: 3x3 ({policy_channels} channels)")
            print(f"  > Policy Head: LC0-style {ACTION_SIZE} logits gathered from 8x8x73 planes")
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
                    use_se=use_se,
                    use_se_bottleneck=use_se_bottleneck,
                    drop_path_rate=dpr[i],
                    use_layer_scale=use_layer_scale,
                    layer_scale_init=layer_scale_init,
                    se_reduction=se_reduction,
                )
            )
        self.residual_tower = nn.Sequential(*blocks)
        
        # 🔧 v4.8: Final BN+ReLU after pre-activation residual tower
        # CRITICAL: In pre-act ResNet, BN is at the START of each block,
        # so the last block outputs un-normalized features. Without this,
        # all heads receive unstable activations. (He et al. 2016)
        self.final_bn = nn.BatchNorm2d(filters)
        
        policy_to_az, _ = get_policy_index_maps()
        self.register_buffer(
            "policy_index_to_az",
            torch.as_tensor(policy_to_az, dtype=torch.long),
            persistent=False,
        )

        # Policy head - spatial conv planes gathered to compact LC0-style policy.
        # 3x3 conv (C -> policy_channels) + BN + ReLU + 1x1 conv (policy_channels -> 73 planes)
        self.policy_conv = nn.Conv2d(filters, policy_channels, kernel_size=3, padding=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_logits_conv = nn.Conv2d(policy_channels, ACTION_PLANES, kernel_size=1, bias=True)
        
        # đź†• Value head - WDL (Win/Draw/Loss) classification
        # AlphaZero-style: Conv 1x1 → BN → ReLU → GlobalAvgPool → FC → WDL
        value_filters = int(config['model'].get('value_head_filters', 16))
        value_hidden = int(config['model'].get('value_hidden_dim', 384))
        self.value_conv = nn.Conv2d(filters, value_filters, kernel_size=3, padding=1, bias=False)
        self.value_bn = nn.BatchNorm2d(value_filters)
        value_in_dim = value_filters * 8 * 8 + filters
        self.value_ln = nn.LayerNorm(value_in_dim)
        self.value_fc1 = nn.Linear(value_in_dim, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 3)  # đź†• 3 outputs: [Win, Draw, Loss]
        self.value_dropout = nn.Dropout(dropout)
        mlh_hidden = int(config['model'].get('moves_left_hidden_dim', 128))
        self.moves_left_fc1 = nn.Linear(filters, mlh_hidden)
        self.moves_left_ln = nn.LayerNorm(mlh_hidden)
        self.moves_left_fc2 = nn.Linear(mlh_hidden, 1)
        
        # Shared GAP for value head
        self.shared_gap = nn.AdaptiveAvgPool2d(1)
        
        # 🔧 v4.8: Proper weight initialization
        self._initialize_weights()

        if print_summary:
            self._print_parameter_summary(num_blocks)
    
    def _initialize_weights(self):
        """Initialize weights for better training stability and convergence.
        
        - Conv2d: Kaiming normal (fan_out, relu) — standard for ResNets
        - Linear: Kaiming normal — matches ReLU activations
        - BatchNorm: weight=1, bias=0 — standard
        - Final output layers (policy_logits_conv, value_fc2): small init (std=0.01)
          to prevent heads from dominating early training
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Small init for final output layers (prevent overconfident early predictions)
        nn.init.normal_(self.policy_logits_conv.weight, std=0.01)
        if self.policy_logits_conv.bias is not None:
            nn.init.zeros_(self.policy_logits_conv.bias)
        nn.init.normal_(self.value_fc2.weight, std=0.01)
        nn.init.zeros_(self.value_fc2.bias)
        nn.init.normal_(self.moves_left_fc2.weight, std=0.01)
        nn.init.zeros_(self.moves_left_fc2.bias)

    @staticmethod
    def _count_parameters(module, trainable_only=False):
        """Count parameters for a module."""
        if trainable_only:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        return sum(p.numel() for p in module.parameters())

    def _print_parameter_summary(self, num_blocks):
        """Print a compact parameter breakdown for key model parts."""
        stem_params = self._count_parameters(self.conv_block)
        tower_params = self._count_parameters(self.residual_tower)
        final_bn_params = self._count_parameters(self.final_bn)

        policy_params = (
            self._count_parameters(self.policy_conv) +
            self._count_parameters(self.policy_bn) +
            self._count_parameters(self.policy_logits_conv)
        )
        value_params = (
            self._count_parameters(self.value_conv) +
            self._count_parameters(self.value_bn) +
            self._count_parameters(self.value_ln) +
            self._count_parameters(self.value_fc1) +
            self._count_parameters(self.value_fc2)
        )
        moves_left_params = (
            self._count_parameters(self.moves_left_fc1) +
            self._count_parameters(self.moves_left_ln) +
            self._count_parameters(self.moves_left_fc2)
        )

        total_params = self._count_parameters(self)
        trainable_params = self._count_parameters(self, trainable_only=True)
        frozen_params = total_params - trainable_params
        per_block = tower_params // max(1, num_blocks)

        print("  > Parameter breakdown:")
        print(f"    - Stem (input conv_block): {stem_params:,}")
        print(f"    - Residual tower ({num_blocks} blocks): {tower_params:,} (~{per_block:,}/block)")
        print(f"    - Final BN: {final_bn_params:,}")
        print(f"    - Policy head: {policy_params:,}")
        print(f"    - Value head: {value_params:,}")
        print(f"    - Moves-left head: {moves_left_params:,}")
        print(f"    - Trainable params: {trainable_params:,}")
        if frozen_params > 0:
            print(f"    - Frozen params: {frozen_params:,}")
        print(f"    - Total params: {total_params:,}")

    def forward(self, x, apply_log_softmax=True, policy_only=False, return_moves_left=False):
        """Forward pass (policy as log-probs by default, raw logits when apply_log_softmax=False)."""
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)

        x = self.conv_block(x)
        x = F.relu(x, inplace=True)

        x = self.residual_tower(x)

        # Final BN+ReLU after pre-activation residual tower
        x = self.final_bn(x)
        x = F.relu(x, inplace=True)

        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = F.relu(policy, inplace=True)
        policy_logits_planes = self.policy_logits_conv(policy)  # (B, 73, 8, 8)
        # ACTION index layout is from_square-major then plane:
        # index = (row*8 + col) * 73 + plane.
        az_policy_logits = (
            policy_logits_planes.permute(0, 2, 3, 1).contiguous().view(policy_logits_planes.size(0), -1)
        )
        policy_logits = az_policy_logits.index_select(1, self.policy_index_to_az)
        if apply_log_softmax:
            policy = F.log_softmax(policy_logits, dim=1)
        else:
            policy = policy_logits
        if policy_only:
            return (policy, None, None) if return_moves_left else (policy, None)

        trunk_global = self.shared_gap(x).flatten(1)

        # Value head: spatial conv features + global trunk mean -> WDL
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = F.silu(value, inplace=True)
        value = value.flatten(1)
        value = torch.cat([value, trunk_global], dim=1)
        value = self.value_ln(value)
        value_hidden = F.silu(self.value_fc1(value), inplace=True)
        value_hidden = self.value_dropout(value_hidden)
        value = self.value_fc2(value_hidden)  # (B, 3) WDL logits

        if not return_moves_left:
            return policy, value

        moves_left = F.relu(self.moves_left_fc1(trunk_global), inplace=True)
        moves_left = self.moves_left_ln(moves_left)
        moves_left = self.moves_left_fc2(moves_left)
        return policy, value, moves_left

    def predict(self, board_tensor):
        """
        Predict for a single position (used in MCTS)
        đź†• Handles WDL output and converts to scalar
        """
        was_training = self.training
        if was_training:
            self.eval()
        try:
            with torch.inference_mode():
                if len(board_tensor.shape) == 3:
                    board_tensor = board_tensor.unsqueeze(0)
                policy_logits, value_logits = self.forward(
                    board_tensor,
                    apply_log_softmax=False,
                )
                
                # value_logits: (1, 3) -> [Win, Draw, Loss]
                wdl_probs = F.softmax(value_logits, dim=1)
                # Scalar: W*1.0 + D*0.0 + L*(-1.0) = W - L
                value_scalar = wdl_probs[0, 0] - wdl_probs[0, 2]
                policy_probs = F.softmax(policy_logits, dim=1)
                return policy_probs.float().cpu().numpy()[0], value_scalar.item()
        finally:
            if was_training:
                self.train()


def load_checkpoint_file(checkpoint_path, device):
    """Load checkpoint with PyTorch 2.6+ compatibility."""
    try:
        try:
            from torch.serialization import safe_globals
            try:
                import numpy._core.multiarray
                scalar_class = numpy._core.multiarray.scalar
            except (ImportError, AttributeError):
                import numpy.core.multiarray
                scalar_class = numpy.core.multiarray.scalar

            with safe_globals([scalar_class]):
                return torch.load(checkpoint_path, map_location=device, weights_only=True)
        except Exception:
            return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        # Trusted local file fallback for legacy checkpoints.
        return torch.load(checkpoint_path, map_location=device, weights_only=False)


def normalize_state_dict_keys(source_state, target_keys=None):
    """Normalize common wrapper prefixes in state_dict keys.

    Handles keys created by DataParallel/DistributedDataParallel (`module.`)
    and torch.compile (`_orig_mod.`). If `target_keys` is provided, only
    strips wrappers when it helps match target names.
    """
    if source_state is None:
        return {}

    wrapper_prefixes = ("module", "_orig_mod")
    normalized = {}

    for key, tensor in source_state.items():
        norm_key = key
        parts = key.split(".")

        if target_keys is not None and norm_key in target_keys:
            pass
        else:
            while len(parts) > 1 and parts[0] in wrapper_prefixes:
                parts = parts[1:]
                candidate = ".".join(parts)
                if target_keys is None:
                    norm_key = candidate
                    continue
                if candidate in target_keys:
                    norm_key = candidate
                    break

        if norm_key not in normalized:
            normalized[norm_key] = tensor

    return normalized


def transfer_matching_weights(model, checkpoint_or_state):
    """Transfer only matching tensors from checkpoint to model.

    Matches by tensor name and exact shape.
    Useful when architecture changed (e.g. different depth), so strict load fails.
    """
    if isinstance(checkpoint_or_state, dict) and 'model_state_dict' in checkpoint_or_state:
        source_state = checkpoint_or_state['model_state_dict']
    else:
        source_state = checkpoint_or_state

    target_state = model.state_dict()
    target_keys = set(target_state.keys())
    normalized_source = normalize_state_dict_keys(source_state, target_keys=target_keys)

    matched_keys = []
    missing_keys = []
    unexpected_keys = []
    shape_mismatch = []

    for key, tensor in normalized_source.items():
        if key not in target_state:
            unexpected_keys.append(key)
            continue
        if target_state[key].shape != tensor.shape:
            shape_mismatch.append((key, tuple(tensor.shape), tuple(target_state[key].shape)))
            continue

        target_state[key] = tensor.to(dtype=target_state[key].dtype, device=target_state[key].device)
        matched_keys.append(key)

    matched_set = set(matched_keys)
    for key in target_state.keys():
        if key not in matched_set:
            missing_keys.append(key)

    model.load_state_dict(target_state, strict=False)

    matched_elements = sum(target_state[k].numel() for k in matched_keys)
    total_elements = sum(v.numel() for v in target_state.values())

    return {
        'matched_keys': matched_keys,
        'missing_keys': missing_keys,
        'unexpected_keys': unexpected_keys,
        'shape_mismatch': shape_mismatch,
        'matched_tensors': len(matched_keys),
        'total_tensors': len(target_state),
        'matched_elements': matched_elements,
        'total_elements': total_elements,
        'match_ratio': (matched_elements / total_elements) if total_elements else 0.0,
    }


def load_model(checkpoint_path, config, device, strict=True):
    """Load model from checkpoint."""
    model = ChessNet(config).to(device)
    model = model.to(memory_format=torch.channels_last)

    if checkpoint_path:
        checkpoint = load_checkpoint_file(checkpoint_path, device)
        source_state = checkpoint['model_state_dict']
        target_keys = set(model.state_dict().keys())
        normalized_state = normalize_state_dict_keys(source_state, target_keys=target_keys)
        if strict:
            model.load_state_dict(normalized_state)
        else:
            transfer_matching_weights(model, normalized_state)

    return model


def save_checkpoint(model, optimizer, epoch, loss, path, metadata=None,
                    save_optimizer=False, save_dtype=None, extra_state=None):
    """Save model checkpoint
    
    Args:
        save_dtype: Optional dtype to convert state_dict (e.g. torch.bfloat16)
                    Converts the saved copy WITHOUT modifying the live model.
        extra_state: Optional dict with additional runtime state
                    (e.g. scheduler/scaler/best metrics).
    """
    # Save canonical keys (without wrapper prefixes like `_orig_mod.`/`module.`)
    state_dict = normalize_state_dict_keys(model.state_dict())
    if save_dtype is not None:
        state_dict = {
            k: (v.to(save_dtype) if v.is_floating_point() else v.clone())
            for k, v in state_dict.items()
        }
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': state_dict,
        'loss': loss,
    }
    
    if save_optimizer and optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if metadata:
        checkpoint.update(metadata)

    if extra_state:
        checkpoint.update(extra_state)
    
    torch.save(checkpoint, path)
    
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 ** 2)
        opt_status = "with optimizer" if save_optimizer else "without optimizer"
        print(f"Checkpoint saved to {path} ({size_mb:.2f} MB, {opt_status})")
