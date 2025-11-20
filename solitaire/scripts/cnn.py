import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SolitaireFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Tableau: (7, 20, 4) -> Flatten -> 560
        self.tableau_dim = 7 * 20 * 4
        self.tableau_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.tableau_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # Foundations: (4,)
        self.foundations_dim = 4
        self.foundations_net = nn.Sequential(
            nn.Linear(self.foundations_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )
        
        # Waste: (3,)
        self.waste_dim = 3
        self.waste_net = nn.Sequential(
            nn.Linear(self.waste_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )
        
        # Stock: (52, 3) -> Flatten -> 156
        self.stock_dim = 52 * 3
        self.stock_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.stock_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        
        # Fusion
        fusion_input_dim = 128 + 32 + 32 + 64
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_input_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        tableau = observations['tableau']
        foundations = observations['foundations']
        waste = observations['waste']
        stock = observations['stock']
        
        t_out = self.tableau_net(tableau)
        f_out = self.foundations_net(foundations)
        w_out = self.waste_net(waste)
        s_out = self.stock_net(stock)
        
        fusion_in = torch.cat([t_out, f_out, w_out, s_out], dim=1)
        return self.fusion_net(fusion_in)
