"""Named compatibility view for the positional replay batch format."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class ReplayBatch:
    boards: torch.Tensor
    policy_indices: torch.Tensor
    policy_values: torch.Tensor
    policy_mask: torch.Tensor
    value_targets: torch.Tensor
    policy_sample_weights: torch.Tensor
    value_sample_weights: torch.Tensor
    moves_left_targets: torch.Tensor
    legal_indices: torch.Tensor
    legal_mask: torch.Tensor
    root_q_targets: torch.Tensor
    search_changed_top: torch.Tensor
    search_q_deltas: torch.Tensor
    best_q_targets: torch.Tensor
    played_q_targets: torch.Tensor
    orig_q_targets: torch.Tensor
    policy_kld_targets: torch.Tensor
    search_visits: torch.Tensor
    search_budget_assigned: torch.Tensor
    search_simulations_effective: torch.Tensor
    search_simulations_fresh: torch.Tensor
    search_visits_inherited_credit: torch.Tensor

    @classmethod
    def from_tuple(cls, batch: Sequence[torch.Tensor]) -> "ReplayBatch":
        """Decode supported legacy layouts and supply their historical defaults."""
        if isinstance(batch, cls):
            return batch
        if len(batch) >= 18:
            fields = list(batch[:18])
            fields.extend(list(batch[18:22]))
            fields.extend([None] * (22 - len(fields)))
        elif len(batch) >= 13:
            fields = list(batch[:13]) + [None] * 5
        elif len(batch) >= 11:
            fields = list(batch[:11]) + [None] * 7
        elif len(batch) >= 10:
            fields = list(batch[:10]) + [None] * 8
        elif len(batch) >= 8:
            # Older batches use sparse policy support as their legal support.
            fields = list(batch[:8]) + [None] * 10
            fields[8:10] = fields[1], fields[3]
        elif len(batch) >= 7:
            fields = list(batch[:7]) + [None] * 11
            fields[7] = torch.full_like(fields[4], -1.0, dtype=torch.float32)
            fields[8:10] = fields[1], fields[3]
        else:
            # Preserve the original six-field fallback, including its unpacking
            # failure for malformed shorter batches.
            boards, policy_indices, policy_values, policy_mask, value_targets, policy_sample_weights = batch
            value_sample_weights = torch.ones_like(policy_sample_weights, dtype=torch.float32)
            moves_left_targets = torch.full_like(value_targets, -1.0, dtype=torch.float32)
            fields = [
                boards, policy_indices, policy_values, policy_mask, value_targets,
                policy_sample_weights, value_sample_weights, moves_left_targets,
                policy_indices, policy_mask,
            ] + [None] * 8

        if len(fields) < 22:
            fields.extend([None] * (22 - len(fields)))

        value_targets = fields[4]
        policy_sample_weights = fields[5]
        defaults = (
            (10, value_targets, float("nan"), torch.float32),
            (11, policy_sample_weights, 0, torch.bool),
            (12, policy_sample_weights, float("nan"), torch.float32),
            (13, value_targets, float("nan"), torch.float32),
            (14, value_targets, float("nan"), torch.float32),
            (15, value_targets, float("nan"), torch.float32),
            (16, policy_sample_weights, float("nan"), torch.float32),
            (17, policy_sample_weights, 0, torch.int32),
            (18, policy_sample_weights, -1, torch.int32),
            (19, policy_sample_weights, -1, torch.int32),
            (20, policy_sample_weights, -1, torch.int32),
            (21, policy_sample_weights, -1, torch.int32),
        )
        for index, template, fill_value, dtype in defaults:
            if fields[index] is None:
                fields[index] = torch.full_like(
                    template,
                    fill_value,
                    dtype=dtype,
                )
        return cls(*fields)
