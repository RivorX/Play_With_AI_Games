import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers"""
    def __init__(self, filters):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class ChessNet(nn.Module):
    """
    Chess neural network with policy and value heads
    Input: (batch, 12, 8, 8) - 12 piece types on 8x8 board
    Output: policy (4096,) and value (1,)
    """
    def __init__(self, config):
        super().__init__()
        
        filters = config['model']['filters']
        num_blocks = config['model']['num_residual_blocks']
        dropout = config['model']['dropout']
        
        # Initial convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv2d(12, filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU()
        )
        
        # Residual tower
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(filters) for _ in range(num_blocks)]
        )
        
        # Policy head
        policy_filters = config['model']['policy_head_filters']
        self.policy_conv = nn.Conv2d(filters, policy_filters, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(policy_filters)
        self.policy_fc = nn.Linear(policy_filters * 8 * 8, 4096)
        self.policy_dropout = nn.Dropout(dropout)
        
        # Value head
        value_filters = config['model']['value_head_filters']
        value_hidden = config['model']['value_hidden_dim']
        self.value_conv = nn.Conv2d(filters, value_filters, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(value_filters)
        self.value_fc1 = nn.Linear(value_filters * 8 * 8, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)
        self.value_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch, 12, 8, 8) board representation
        Returns:
            policy: (batch, 4096) move probabilities
            value: (batch, 1) position evaluation
        """
        # Shared trunk
        x = self.conv_block(x)
        x = self.residual_blocks(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_dropout(policy)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
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
    if checkpoint_path and torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    return model


def save_checkpoint(model, optimizer, epoch, loss, path, metadata=None):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if metadata:
        checkpoint.update(metadata)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")