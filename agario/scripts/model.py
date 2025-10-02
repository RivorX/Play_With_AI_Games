import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces
from torch.distributions import Categorical, Normal

from conv_lstm import ConvLSTM

class CnnLstmExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512, **conv_lstm_kwargs):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0] // conv_lstm_kwargs.get('frame_history', 4)  # Zakładamy stacked
        self.conv_lstm = ConvLSTM(
            input_dim=(n_input_channels, *observation_space.shape[-2:]),
            hidden_dim=conv_lstm_kwargs.get('hidden_dim', 128),
            num_layers=conv_lstm_kwargs.get('lstm_layers', 1),
            **conv_lstm_kwargs.get('conv_layers', [])
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512, features_dim)  # Dostosuj
        self.dropout = nn.Dropout(0.2)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size, seq_len, c, h, w = observations.shape
        # Przetwarzaj sekwencyjnie
        lstm_out = self.conv_lstm(observations.view(batch_size * seq_len, c, h, w))
        features = self.flatten(lstm_out[-1])  # Last timestep
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
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            features_extractor_class=CnnLstmExtractor,
            features_extractor_kwargs={"conv_lstm_kwargs": kwargs.get('conv_lstm_kwargs', {})},
            **kwargs,
        )

    def _get_action_dist_from_latent(self, latent_vec: th.Tensor):
        # Dla Box - Normal; dostosuj
        mean = self.action_net(latent_vec)
        log_std = self.log_std
        return Normal(mean, th.exp(log_std))