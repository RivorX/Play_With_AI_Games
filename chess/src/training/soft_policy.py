"""Shared soft-policy training objective used by IL and RL.

The soft-policy auxiliary target is derived only from moves already present in
the sparse empirical target. It never invents probability mass for moves that
were not observed. The legal mask, when supplied, also limits the softmax
denominator so the objective can be used by both IL and MCTS training.
"""

from collections.abc import Mapping
import math
from typing import NamedTuple

import torch
import torch.nn.functional as F


class PreparedLegalMaskedSoftPolicyTarget(NamedTuple):
    """Target-side tensors shared by soft-policy heads for one training batch."""

    softened_target: torch.Tensor
    safe_legal: torch.Tensor
    valid_rows: torch.Tensor
    effective_weights: torch.Tensor


def _unpack_policy_target(target_policy):
    if isinstance(target_policy, Mapping):
        indices = target_policy.get("indices", target_policy.get("policy_indices"))
        values = target_policy.get("values", target_policy.get("policy_values"))
        if indices is not None or values is not None:
            if indices is None or values is None:
                raise ValueError("Sparse policy targets require both indices and values")
            return "sparse", indices, values
        dense = target_policy.get("dense", target_policy.get("policy"))
        if dense is not None:
            return "dense", dense, None
        raise ValueError("Policy target mapping must contain dense targets or indices and values")

    if isinstance(target_policy, (tuple, list)) and len(target_policy) == 2:
        return "sparse", target_policy[0], target_policy[1]
    if torch.is_tensor(target_policy):
        return "dense", target_policy, None
    raise TypeError("target_policy must be a dense tensor, (indices, values), or mapping")


def prepare_legal_masked_soft_policy_target(
    logits_reference,
    target_policy,
    legal_mask=None,
    row_weights=None,
    temperature=4.0,
    *,
    row_mask=None,
    legal_indices=None,
    min_support=2,
):
    """Prepare target-only tensors once for one or more policy heads.

    Args:
        logits_reference: Raw logits used to define expected shape and device.
        target_policy: Dense ``(batch, actions)`` probabilities, sparse
            ``(indices, values)`` tensors of shape ``(batch, support)``, or a
            mapping containing ``indices``/``values`` (or ``dense``).
        legal_mask: Optional bool mask ``(batch, actions)``. Illegal logits are
            removed from the softmax denominator and illegal target mass is
            discarded before the target is normalized.
        row_weights: Optional non-negative per-row weights.
        temperature: Positive exponent temperature; targets are transformed
            as ``p ** (1 / temperature)`` and renormalized.
        row_mask: Optional bool mask for rows participating in the objective.
        legal_indices: Optional padded legal-action indices ``(batch, count)``;
            negative/out-of-range entries are ignored. Combines with legal_mask.
        min_support: Minimum number of positive legal target actions required.
            The default skips one-hot targets so the auxiliary loss only learns
            from genuinely soft empirical targets.

    Returns:
        A detached immutable bundle of the softened target, legal mask,
        valid rows and effective sample weights. This bundle can be reused
        across heads that share the same targets and weighting.
    """
    if logits_reference.ndim != 2:
        raise ValueError(
            "logits_reference must have shape (batch, actions), "
            f"got {tuple(logits_reference.shape)}"
        )
    batch_size, action_count = logits_reference.shape
    if batch_size <= 0 or action_count <= 0:
        raise ValueError("logits must have a non-empty batch and action dimension")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be finite and positive, got {temperature!r}")
    min_support = max(1, int(min_support))

    device = logits_reference.device
    with torch.no_grad():
        legal = torch.ones((batch_size, action_count), dtype=torch.bool, device=device)
        if legal_mask is not None:
            legal_mask = torch.as_tensor(legal_mask, device=device, dtype=torch.bool)
            if tuple(legal_mask.shape) != (batch_size, action_count):
                raise ValueError(
                    f"legal_mask must have shape {(batch_size, action_count)}, "
                    f"got {tuple(legal_mask.shape)}"
                )
            legal &= legal_mask
        if legal_indices is not None:
            legal_indices = torch.as_tensor(legal_indices, device=device, dtype=torch.long)
            if legal_indices.ndim != 2 or legal_indices.shape[0] != batch_size:
                raise ValueError("legal_indices must have shape (batch, legal_actions)")
            legal_idx_valid = (legal_indices >= 0) & (legal_indices < action_count)
            legal_from_indices = torch.zeros(
                (batch_size, action_count), dtype=torch.int32, device=device
            )
            legal_from_indices.scatter_add_(
                1,
                legal_indices.clamp(0, action_count - 1),
                legal_idx_valid.to(dtype=legal_from_indices.dtype),
            )
            legal &= legal_from_indices > 0
        legal_row_present = legal.any(dim=1)

        target_kind, target_a, target_b = _unpack_policy_target(target_policy)
        target = torch.zeros((batch_size, action_count), dtype=torch.float32, device=device)
        if target_kind == "dense":
            dense = torch.as_tensor(target_a, device=device, dtype=torch.float32)
            if tuple(dense.shape) != (batch_size, action_count):
                raise ValueError(
                    f"Dense target must have shape {(batch_size, action_count)}, "
                    f"got {tuple(dense.shape)}"
                )
            target = torch.nan_to_num(
                dense.detach(), nan=0.0, posinf=0.0, neginf=0.0
            ).clamp_min_(0.0)
        else:
            indices = torch.as_tensor(target_a, device=device, dtype=torch.long)
            values = torch.as_tensor(target_b, device=device, dtype=torch.float32)
            if indices.ndim != 2 or tuple(indices.shape) != tuple(values.shape):
                raise ValueError(
                    "Sparse target indices and values must share shape (batch, support)"
                )
            if indices.shape[0] != batch_size:
                raise ValueError("Sparse target batch dimension must match logits")
            values = torch.nan_to_num(values.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            valid = (indices >= 0) & (indices < action_count) & (values > 0.0)
            safe_indices = indices.clamp(0, action_count - 1)
            if legal_mask is not None or legal_indices is not None:
                valid &= legal.gather(1, safe_indices)
            target.scatter_add_(
                1, safe_indices, values.clamp_min(0.0) * valid.to(values.dtype)
            )

        target = target * legal.to(dtype=target.dtype)
        support = (target > 0.0).sum(dim=1)
        target_mass = target.sum(dim=1, keepdim=True)
        valid_rows = (support >= min_support) & (target_mass[:, 0] > 0.0) & legal_row_present
        if row_mask is not None:
            row_mask = torch.as_tensor(row_mask, device=device, dtype=torch.bool).reshape(-1)
            if row_mask.numel() != batch_size:
                raise ValueError("row_mask must contain one value per batch row")
            valid_rows &= row_mask

        if row_weights is None:
            weights = torch.ones((batch_size,), dtype=torch.float32, device=device)
        else:
            weights = torch.as_tensor(row_weights, device=device, dtype=torch.float32).reshape(-1)
            if weights.numel() != batch_size:
                raise ValueError("row_weights must contain one value per batch row")
            weights = torch.nan_to_num(
                weights.detach(), nan=0.0, posinf=0.0, neginf=0.0
            ).clamp_min_(0.0)
        valid_rows &= weights > 0.0

        softened = target.pow(1.0 / temperature)
        softened = softened / softened.sum(dim=1, keepdim=True).clamp_min(1.0e-12)

        # Avoid log_softmax over an all -inf row. Such rows are excluded below.
        safe_legal = legal | (~legal_row_present).unsqueeze(1)
        effective_weights = weights * valid_rows.to(dtype=weights.dtype)

    return PreparedLegalMaskedSoftPolicyTarget(
        softened,
        safe_legal,
        valid_rows,
        effective_weights,
    )


def legal_masked_soft_policy_loss_from_prepared(logits, prepared_target):
    """Apply one head's logits to target-side tensors prepared for this batch."""
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape (batch, actions), got {tuple(logits.shape)}")
    expected_shape = tuple(prepared_target.softened_target.shape)
    if tuple(logits.shape) != expected_shape:
        raise ValueError(f"logits must have shape {expected_shape}, got {tuple(logits.shape)}")
    if logits.device != prepared_target.softened_target.device:
        raise ValueError(
            f"logits must be on {prepared_target.softened_target.device}, got {logits.device}"
        )

    fp32_logits = logits.float()
    masked_logits = fp32_logits.masked_fill(~prepared_target.safe_legal, -torch.inf)
    log_probs = F.log_softmax(masked_logits, dim=1)
    log_probs = torch.where(torch.isfinite(log_probs), log_probs, torch.zeros_like(log_probs))
    per_row_loss = -(prepared_target.softened_target * log_probs).sum(dim=1)
    per_row_loss = torch.where(
        prepared_target.valid_rows, per_row_loss, torch.zeros_like(per_row_loss)
    )

    effective_weights = prepared_target.effective_weights
    loss = (per_row_loss * effective_weights).sum() / effective_weights.sum().clamp_min(1.0e-12)
    return loss, per_row_loss, prepared_target.valid_rows



