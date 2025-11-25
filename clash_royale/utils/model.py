import torch
import torch.nn as nn
import torch.nn.functional as F

class ClashRoyaleAgent(nn.Module):
    def __init__(self, input_channels=3, num_cards=4):
        super(ClashRoyaleAgent, self).__init__()
        
        # CNN Backbone (Widzenie planszy)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Obliczanie wymiaru wejścia do FC dynamicznie lub dla ustalonego rozmiaru (C, H, W)
        # Dla wejścia 270x480 (9:16)
        self.fc_input_dim = self._get_conv_output((input_channels, 480, 270)) 
        
        # +1 dla wejścia eliksiru (skalar)
        self.fc1 = nn.Linear(self.fc_input_dim + 1, 512)
        
        # Głowa decyzyjna: Którą kartę zagrać? (0-3 to karty w ręce, 4 to "czekaj")
        self.actor_card = nn.Linear(512, num_cards + 1)
        
        # Głowa pozycyjna: Gdzie zagrać? (X, Y) - znormalizowane 0-1
        self.actor_pos = nn.Linear(512, 2)
        
        # Critic (dla algorytmów RL, opcjonalne przy Imitation Learning, ale przydatne)
        self.critic = nn.Linear(512, 1)

    def _get_conv_output(self, shape):
        bs = 1
        input = torch.autograd.Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x, elixir):
        # x shape: (batch, channels, height, width)
        # elixir shape: (batch, 1)
        
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        
        # Połącz cechy obrazu z informacją o eliksirze
        x = torch.cat((x, elixir), dim=1)
        
        x = F.relu(self.fc1(x))
        
        card_logits = self.actor_card(x)
        pos_coords = torch.sigmoid(self.actor_pos(x)) # Sigmoid by wymusić 0-1
        value = self.critic(x)
        
        return card_logits, pos_coords, value
