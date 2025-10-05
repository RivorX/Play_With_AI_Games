import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import yaml
from gymnasium import spaces
import os

# Wczytaj konfigurację
base_dir = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(base_dir, 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        
    def forward(self, x, hidden_state):
        h, c = hidden_state
        combined = torch.cat([x, h], dim=1)
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, (h_next, c_next)


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        cell_list = []
        for i in range(num_layers):
            cur_input_channels = input_channels if i == 0 else hidden_channels
            cell_list.append(ConvLSTMCell(cur_input_channels, hidden_channels, kernel_size))
        
        self.cell_list = nn.ModuleList(cell_list)
        
    def forward(self, x, hidden_state=None):
        # x: [batch, seq_len, channels, height, width]
        batch_size, seq_len, _, height, width = x.size()
        
        # Inicjalizuj stan jeśli None
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, height, width, x.device)
        
        layer_output_list = []
        layer_hidden_list = []
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            for t in range(seq_len):
                h, (h, c) = self.cell_list[layer_idx](x[:, t, :, :, :], (h, c))
                output_inner.append(h)
            
            layer_output = torch.stack(output_inner, dim=1)
            x = layer_output
            
            layer_output_list.append(layer_output)
            layer_hidden_list.append((h, c))
        
        return layer_output_list[-1], layer_hidden_list
    
    def _init_hidden(self, batch_size, height, width, device):
        hidden_state_list = []
        for _ in range(self.num_layers):
            h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
            c = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
            hidden_state_list.append((h, c))
        return hidden_state_list


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        # Wczytaj parametry z configu
        convlstm_config = config['model']['convlstm']
        in_channels = 1  # Pojedyncza mapa (viewport zawsze 16x16)
        dropout_rate = config['model'].get('dropout_rate', 0.2)
        leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        
        # ConvLSTM do przetwarzania sekwencji
        self.convlstm = ConvLSTM(
            input_channels=in_channels,
            hidden_channels=convlstm_config['hidden_channels'],
            kernel_size=convlstm_config['kernel_size'],
            num_layers=convlstm_config['num_layers']
        )
        
        # CNN po ConvLSTM
        cnn_layers = []
        current_channels = convlstm_config['hidden_channels']
        
        for layer_channels in convlstm_config['cnn_channels']:
            cnn_layers.extend([
                nn.Conv2d(current_channels, layer_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(layer_channels),
                leaky_relu
            ])
            current_channels = layer_channels
        
        cnn_layers.extend([
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten()
        ])
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Warstwa dla zmiennych skalarnych
        scalar_dim = 6  # direction, dx_head, dy_head, front_coll, left_coll, right_coll (usunięto grid_size)
        scalar_layers = []
        current_dim = scalar_dim
        
        for hidden_dim in convlstm_config['scalar_hidden_dims']:
            scalar_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LeakyReLU(0.01),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
        
        self.scalar_linear = nn.Sequential(*scalar_layers)
        
        # Oblicz wymiar po CNN
        cnn_out_channels = convlstm_config['cnn_channels'][-1]
        cnn_dim = cnn_out_channels * 2 * 2
        total_dim = cnn_dim + current_dim
        
        self.final_linear = nn.Sequential(
            nn.Linear(total_dim, features_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, observations):
        image = observations['image']
        
        # Obsługa formatu z SequenceWrapper: [batch, seq_len, height, width, channels]
        if len(image.shape) == 5:
            # Przekształć na [batch, seq_len, channels, height, width]
            image = image.permute(0, 1, 4, 2, 3)
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}. Expected 5D tensor [batch, seq_len, H, W, C]")
        
        # Przetwórz przez ConvLSTM (bez zachowywania stanu między batch'ami)
        # PPO zbiera dane z wielu środowisk w jednym batch'u, więc nie ma sensu
        # zachowywać stan między forward pass'ami
        lstm_out, _ = self.convlstm(image, hidden_state=None)
        
        # Weź ostatnią ramkę z sekwencji
        last_frame = lstm_out[:, -1, :, :, :]
        
        # Przetwórz przez CNN
        image_features = self.cnn(last_frame)
        
        # Przetwórz zmienne skalarne
        scalars = torch.cat([
            observations['direction'],
            observations['dx_head'],
            observations['dy_head'],
            observations['front_coll'],
            observations['left_coll'],
            observations['right_coll']
        ], dim=-1)
        
        scalar_features = self.scalar_linear(scalars)
        
        # Połącz cechy
        features = torch.cat([image_features, scalar_features], dim=-1)
        return self.final_linear(features)