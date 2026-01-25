"""
Utilities module for chess AI training

Exports:
- TrainingLogger: Unified logger for IL and RL
- LabelSmoothingNLLLoss: Custom loss function
- ReplayBuffer: Standard replay buffer
- PrioritizedReplayBuffer: Prioritized experience replay
- TemperatureSchedule: Temperature decay schedule
- MetricsCalculator: Comprehensive metrics tracking
- Training functions: train_epoch_il, evaluate_il, train_on_batch_rl, evaluate_models
"""

from .logger import TrainingLogger
from .loss import LabelSmoothingNLLLoss
from .replay import ReplayBuffer, PrioritizedReplayBuffer
from .temperature import TemperatureSchedule
from .metrics import MetricsCalculator, compute_batch_metrics
from .training import (
    train_epoch_il,
    evaluate_il,
    train_on_batch_rl,
    evaluate_models
)

__all__ = [
    'TrainingLogger',
    'LabelSmoothingNLLLoss',
    'ReplayBuffer',
    'PrioritizedReplayBuffer',
    'TemperatureSchedule',
    'MetricsCalculator',
    'compute_batch_metrics',
    'train_epoch_il',
    'evaluate_il',
    'train_on_batch_rl',
    'evaluate_models',
]