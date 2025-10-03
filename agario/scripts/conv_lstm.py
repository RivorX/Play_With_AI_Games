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

    def init_hidden(self, batch_size, image_size, device):
        # Poprawka: Użyj device z inputu (bezpieczniej niż self.conv.weight.device dla multi-layer)
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, kernel_size=3, bias=True, conv_layers=None, **kwargs):
        # Poprawka: **kwargs zamiast **conv_layers; conv_layers jako bezpośredni arg (lista z configu)
        super(ConvLSTM, self).__init__()
        self.input_dim = input_dim[0]  # Channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
        self.lstm_layers = nn.ModuleList()
        
        # Poprawka: conv_layers to lista z configu (np. [[32, (8,8), 4], ...])
        current_dim = self.input_dim
        default_conv = [[32, (3,3), 1]]  # Domyślne z oryginalnego kodu
        cfg = conv_layers if conv_layers is not None else default_conv
        for layer in cfg:
            # Poprawka: Indeksowanie zamiast unpackingu (odporne na błędy parsowania YAML)
            if len(layer) != 3:
                raise ValueError(f"conv_layers layer musi mieć dokładnie 3 elementy: [out_ch, ksize, stride]. Błąd w: {layer}")
            out_ch = layer[0]
            ksize = layer[1]  # Może być int lub tuple/lista (z YAML)
            stride = layer[2]
            padding = ksize[0] // 2 if isinstance(ksize, (tuple, list)) else ksize // 2
            self.conv_layers.append(nn.Conv2d(current_dim, out_ch, ksize, stride=stride, padding=padding))
            current_dim = out_ch
        
        for i in range(num_layers):
            cur_input_dim = current_dim if i == 0 else hidden_dim
            self.lstm_layers.append(ConvLSTMCell(cur_input_dim, hidden_dim, kernel_size, bias))

    def forward(self, input_tensor):
        # input_tensor: (B, T, C, H, W)
        b, t, c, h, w = input_tensor.size()
        
        # Poprawka: Conv preprocessing na (B*T, C, H, W) -> convs -> view back (B, T, C', H', W')
        input_tensor = input_tensor.view(b * t, c, h, w)  # (B*T, C, H, W)
        for conv in self.conv_layers:
            input_tensor = torch.relu(conv(input_tensor))
        new_c = input_tensor.shape[1]  # Nowe kanały po convs
        new_h = input_tensor.shape[2]  # Nowa wysokość po convs (stride!)
        new_w = input_tensor.shape[3]  # Nowa szerokość po convs (stride!)
        input_tensor = input_tensor.view(b, t, new_c, new_h, new_w)  # (B, T, C', H', W')
        
        # LSTM forward
        layer_output_list = []
        last_state_list = []
        
        cur_layer_input = input_tensor  # Już (B, T, C', H', W')
        device = input_tensor.device
        
        for layer_idx in range(self.num_layers):
            # Poprawka: Przekaż device do init_hidden oraz nowe h,w po convs
            h_state, c_state = self.lstm_layers[layer_idx].init_hidden(batch_size=b, image_size=(new_h, new_w), device=device)
            output_inner = []
            for time_step in range(t):
                # cur_layer_input[:, time_step, :] -> (B, C', H', W')
                h_state, c_state = self.lstm_layers[layer_idx](
                    input_tensor=cur_layer_input[:, time_step], 
                    cur_state=[h_state, c_state]
                )
                output_inner.append(h_state)
            
            layer_output = torch.stack(output_inner, dim=1)  # (B, T, hidden, H', W')
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append([h_state, c_state])
        
        return layer_output_list[-1]  # (B, T, hidden, H', W') – last layer