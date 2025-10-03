import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces

from conv_lstm import ConvLSTM

class CnnLstmExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512, **conv_lstm_kwargs):
        super().__init__(observation_space, features_dim)
        # Poprawka: n_input_channels = 3 (RGB), bo shape=(T, 3, H, W), nie stacked channels
        n_input_channels = observation_space.shape[1]  # shape[1] = 3 (channels)
        frame_history = conv_lstm_kwargs.get('frame_history', 4)  # Z configu, do walidacji
        assert observation_space.shape[0] == frame_history, f"Frame history mismatch: {observation_space.shape[0]} != {frame_history}"
        # Wyciągnij conv_layers z kwargs (lista z configu)
        conv_layers_cfg = conv_lstm_kwargs.pop('conv_layers', None)
        if conv_layers_cfg is None:
            raise ValueError("conv_layers musi być podany w configu (conv_lstm_kwargs)")
        # Poprawka: Pop() dla wszystkich używanych args, by uniknąć duplikatów w **
        hidden_dim = conv_lstm_kwargs.pop('hidden_dim', 128)  # Z configu
        num_layers = conv_lstm_kwargs.pop('lstm_layers', 1)   # Z configu (lstm_layers -> num_layers)
        kernel_size = conv_lstm_kwargs.pop('kernel_size', 3)
        bias = conv_lstm_kwargs.pop('bias', True)
        dropout_rate = conv_lstm_kwargs.pop('dropout_rate', 0.2)  # Z configu jeśli jest
        # Reszta kwargs (np. inne custom, jeśli dodasz)
        self.conv_lstm = ConvLSTM(
            input_dim=(n_input_channels, *observation_space.shape[-2:]),  # (3, H, W)
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            bias=bias,
            conv_layers=conv_layers_cfg,  # Lista z configu
            **conv_lstm_kwargs  # Tylko reszta bez duplikatów
        )
        # POPRAWKA: Oblicz faktyczny rozmiar po conv_layers dynamicznie
        # Symuluj forward pass z dummy input aby uzyskać prawdziwy output shape
        with th.no_grad():
            dummy_input = th.zeros(1, frame_history, n_input_channels, *observation_space.shape[-2:])
            dummy_lstm_out = self.conv_lstm(dummy_input)  # (1, T, hidden, H_out, W_out)
            dummy_last = dummy_lstm_out[:, -1]  # (1, hidden, H_out, W_out)
            self.flatten_dim = dummy_last.view(1, -1).shape[1]  # Faktyczny rozmiar po flatten
        
        print(f"CnnLstmExtractor: flatten_dim obliczony dynamicznie = {self.flatten_dim}")
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.flatten_dim, features_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size, seq_len, c, h, w = observations.shape  # (B, T, 3, H, W)
        # Poprawka: Przekazuj bezpośrednio (B, T, C, H, W), nie view (usuń view w extractor)
        lstm_out = self.conv_lstm(observations)  # (B, T, hidden, H_out, W_out)
        # Poprawka: Last timestep of last layer
        last_timestep = lstm_out[:, -1]  # (B, hidden, H_out, W_out)
        features = self.flatten(last_timestep)  # (B, hidden * H_out * W_out)
        return self.dropout(self.linear(features)).view(batch_size, -1)

class CnnLstmPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        net_arch=None,
        activation_fn=nn.ReLU,
        *args,
        **kwargs,
    ):
        # Poprawka: Nie nadpisuj features_extractor_kwargs – pozwól na **kwargs z PPO
        # Usuń pop i ręczne ustawianie, bo config jest przekazywany poprawnie z train.py
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,  # Przekaż wszystko, w tym features_extractor_class i _kwargs
        )
    
    # USUNIĘTE: _get_action_dist_from_latent - SB3 ma już wbudowaną obsługę dla Box action space