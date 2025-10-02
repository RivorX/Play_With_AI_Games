import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if kernel_size % 2 == 1:
            padding = kernel_size // 2
        else:
            raise ValueError("Kernel size must be odd")

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # N x (input+hidden) x H x W
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, kernel_size=3, bias=True, **conv_layers):
        super(ConvLSTM, self).__init__()
        self.input_dim = input_dim[0]  # Channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
        self.lstm_layers = nn.ModuleList()
        
        # Dodaj Conv2D layers przed LSTM (z config)
        current_dim = self.input_dim
        for layer in conv_layers.get('conv_layers', [[32, (3,3), 1]]):
            out_ch, ksize, stride = layer
            self.conv_layers.append(nn.Conv2d(current_dim, out_ch, ksize, stride=stride, padding=ksize[0]//2))
            current_dim = out_ch
        
        for i in range(num_layers):
            cur_input_dim = current_dim if i == 0 else hidden_dim
            self.lstm_layers.append(ConvLSTMCell(cur_input_dim, hidden_dim, kernel_size, bias))

    def forward(self, input_tensor):
        # input_tensor: B x T x C x H x W -> B*T x C x H x W
        b, t, c, h, w = input_tensor.size()
        input_tensor = input_tensor.view(b * t, c, h, w)
        
        # Conv preprocessing
        for conv in self.conv_layers:
            input_tensor = torch.relu(conv(input_tensor))
        
        # LSTM forward
        layer_output_list = []
        last_state_list = []
        
        cur_layer_input = input_tensor.view(b, t, *input_tensor.shape[1:])
        
        for layer_idx in range(self.num_layers):
            h, c = self.lstm_layers[layer_idx].init_hidden(batch_size=b, image_size=(h, w))
            output_inner = []
            for time_step in range(t):
                h, c = self.lstm_layers[layer_idx](input_tensor=cur_layer_input[:, time_step, :], cur_state=[h, c])
                output_inner.append(h)
            
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
        
        return layer_output_list[-1]  # Last layer outputs