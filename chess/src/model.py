import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, filters, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(filters, filters // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(filters // reduction, filters, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = self.squeeze(x).view(batch, channels)
        y = self.excitation(y).view(batch, channels, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial attention - learns WHERE to look on the board"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(1)
    
    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        attention = torch.cat([max_pool, avg_pool], dim=1)
        attention = self.conv(attention)
        attention = self.bn(attention)
        attention = torch.sigmoid(attention)
        return x * attention


class CoordConv2d(nn.Module):
    """Convolution that 'knows' where it is on the board"""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, padding=padding, bias=bias)
        
    def forward(self, x):
        batch, _, h, w = x.size()
        
        xx_channel = torch.arange(w, dtype=x.dtype, device=x.device).repeat(1, h, 1)
        yy_channel = torch.arange(h, dtype=x.dtype, device=x.device).repeat(1, w, 1).transpose(1, 2)
        
        xx_channel = xx_channel.float() / (w - 1) * 2 - 1
        yy_channel = yy_channel.float() / (h - 1) * 2 - 1
        
        xx_channel = xx_channel.repeat(batch, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch, 1, 1, 1)
        
        x = torch.cat([x, xx_channel, yy_channel], dim=1)
        
        return self.conv(x)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Stochastic Depth - randomly skip residual blocks"""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class ResidualBlock(nn.Module):
    """Residual block with SE-Block, Spatial Attention, and Stochastic Depth"""
    def __init__(self, filters, use_se=True, use_spatial=True, drop_path_rate=0.0, use_elu=True):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)
        
        self.use_elu = use_elu
        self.activation = F.elu if use_elu else F.relu
        
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(filters)
        
        self.use_spatial = use_spatial
        if use_spatial:
            self.spatial = SpatialAttention()
        
        self.drop_path_rate = drop_path_rate
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_se:
            out = self.se(out)
        
        if self.use_spatial:
            out = self.spatial(out)
        
        if self.drop_path_rate > 0:
            out = drop_path(out, self.drop_path_rate, self.training)
        
        out += residual
        out = self.activation(out)
        
        return out


class ChessNet(nn.Module):
    """Enhanced Chess neural network"""
    def __init__(self, config):
        super().__init__()
        
        filters = config['model']['filters']
        num_blocks = config['model']['num_residual_blocks']
        dropout = config['model']['dropout']
        
        use_se = config['model'].get('use_se_blocks', True)
        use_spatial = config['model'].get('use_spatial_attention', True)
        drop_path_rate = config['model'].get('drop_path_rate', 0.1)
        use_elu = config['model'].get('activation', 'elu') == 'elu'
        use_coord_conv = config['model'].get('use_coord_conv', True)
        
        print(f"🧠 Model Features:")
        print(f"  • SE-Blocks (channel attention): {use_se}")
        print(f"  • Spatial Attention: {use_spatial}")
        print(f"  • Stochastic Depth: {drop_path_rate}")
        print(f"  • Activation: {'ELU' if use_elu else 'ReLU'}")
        print(f"  • CoordConv: {use_coord_conv}")
        print(f"  • Memory Format: Channels Last (GPU optimized)")
        
        self.use_elu = use_elu
        self.activation = F.elu if use_elu else F.relu
        
        if use_coord_conv:
            self.conv_block = nn.Sequential(
                CoordConv2d(12, filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(filters),
            )
        else:
            self.conv_block = nn.Sequential(
                nn.Conv2d(12, filters, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(filters),
            )
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(
                filters, 
                use_se=use_se, 
                use_spatial=use_spatial,
                drop_path_rate=dpr[i],
                use_elu=use_elu
            ) for i in range(num_blocks)
        ])
        
        policy_filters = config['model']['policy_head_filters']
        self.policy_conv = nn.Conv2d(filters, policy_filters, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_filters)
        self.policy_fc = nn.Linear(policy_filters * 8 * 8, 4096)
        self.policy_dropout = nn.Dropout(dropout)
        
        value_filters = config['model']['value_head_filters']
        value_hidden = config['model']['value_hidden_dim']
        self.value_conv = nn.Conv2d(filters, value_filters, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(value_filters)
        self.value_fc1 = nn.Linear(value_filters * 8 * 8, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)
        self.value_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x.contiguous(memory_format=torch.channels_last)
        
        x = self.conv_block(x)
        x = self.activation(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        policy = self.activation(self.policy_bn(self.policy_conv(x)))
        policy = policy.reshape(policy.size(0), -1)
        policy = self.policy_dropout(policy)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        
        value = self.activation(self.value_bn(self.value_conv(x)))
        value = value.reshape(value.size(0), -1)
        value = self.activation(self.value_fc1(value))
        value = self.value_dropout(value)
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def predict(self, board_tensor):
        """Predict for a single position (used in MCTS)"""
        self.eval()
        with torch.no_grad():
            if len(board_tensor.shape) == 3:
                board_tensor = board_tensor.unsqueeze(0)
            policy, value = self.forward(board_tensor)
            return torch.exp(policy).cpu().numpy()[0], value.cpu().item()


def load_model(checkpoint_path, config, device):
    """Load model from checkpoint"""
    model = ChessNet(config).to(device)
    model = model.to(memory_format=torch.channels_last)
    
    if checkpoint_path and torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


# ============================================================================
# SAVE CHECKPOINT - ZMIENIONA FUNKCJA
# ============================================================================

def save_checkpoint(model, optimizer, epoch, loss, path, metadata=None, save_optimizer=False):
    """
    Save model checkpoint with OPTIONAL optimizer state
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer (can be None)
        epoch: Current epoch
        loss: Current loss
        path: Save path
        metadata: Optional metadata dict
        save_optimizer: If True, save optimizer state (DEFAULT: False)
    
    File sizes:
        save_optimizer=True:  ~100 MB (for resume training)
        save_optimizer=False: ~2-4 MB (for inference/deployment)
    
    Examples:
        # Best model (minimal size for Git/deployment)
        save_checkpoint(model, None, epoch, loss, 'best.pt', save_optimizer=False)
        
        # Checkpoint (for resume training)
        save_checkpoint(model, optimizer, epoch, loss, 'ckpt.pt', save_optimizer=True)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }
    
    # ⭐ KLUCZOWA ZMIANA: Optimizer jest opcjonalny
    if save_optimizer and optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if metadata:
        checkpoint.update(metadata)
    
    torch.save(checkpoint, path)
    
    # Print file size for verification
    import os
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 ** 2)
        opt_status = "with optimizer" if save_optimizer else "without optimizer"
        print(f"Checkpoint saved to {path} ({size_mb:.2f} MB, {opt_status})")