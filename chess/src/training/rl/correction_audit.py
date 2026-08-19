"""Controlled before/after audit for useful MCTS policy corrections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


CORRECTION_AUDIT_MAX_ROWS = 2048
CORRECTION_AUDIT_BATCH_SIZE = 1024


@dataclass(frozen=True)
class CorrectionAuditSnapshot:
    """Per-position policy diagnostics for one immutable correction cohort."""

    top1: torch.Tensor
    rank: torch.Tensor
    target_probability: torch.Tensor
    logit_margin: torch.Tensor
    target_kl: torch.Tensor
    value_wdl_ce: torch.Tensor
    value_brier: torch.Tensor
    value_mae: torch.Tensor

    @property
    def rows(self) -> int:
        return int(self.top1.numel())


def freeze_replay_batch(batch: Iterable[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Detach a replay batch from reusable replay scratch buffers."""

    return tuple(tensor.detach().cpu().clone() for tensor in batch)


def _model_outputs(model, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    output = model(
        boards,
        apply_log_softmax=False,
        return_moves_left=True,
        return_search_aux=True,
    )
    if isinstance(output, (tuple, list)):
        value = output[1] if len(output) > 1 else None
        return output[0], value
    return output, None


def evaluate_correction_cohort(
    model,
    batch: tuple[torch.Tensor, ...],
    device,
    *,
    use_amp: bool,
    use_bfloat16: bool,
    batch_size: int = CORRECTION_AUDIT_BATCH_SIZE,
) -> CorrectionAuditSnapshot:
    """Evaluate exactly the stored MCTS target moves without augmentation."""

    if len(batch) < 10:
        raise ValueError("Correction audit requires policy and legal-move replay fields.")

    boards, policy_indices, policy_values, policy_mask, value_targets = batch[:5]
    legal_indices, legal_mask = batch[8:10]
    row_count = int(boards.shape[0])
    if row_count <= 0:
        empty_bool = torch.empty(0, dtype=torch.bool)
        empty_float = torch.empty(0, dtype=torch.float32)
        return CorrectionAuditSnapshot(
            empty_bool, empty_float, empty_float, empty_float, empty_float,
            empty_float, empty_float, empty_float,
        )

    device = torch.device(device)
    chunk_size = max(1, int(batch_size))
    amp_enabled = bool(use_amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    was_training = bool(model.training)
    model.eval()

    top1_parts = []
    rank_parts = []
    probability_parts = []
    margin_parts = []
    target_kl_parts = []
    value_wdl_ce_parts = []
    value_brier_parts = []
    value_mae_parts = []
    try:
        with torch.inference_mode():
            for start in range(0, row_count, chunk_size):
                end = min(row_count, start + chunk_size)
                board_chunk = boards[start:end].to(
                    device,
                    memory_format=torch.channels_last,
                    non_blocking=True,
                )
                policy_index_chunk = policy_indices[start:end].to(device, non_blocking=True).long()
                policy_value_chunk = policy_values[start:end].to(device, non_blocking=True).float()
                policy_mask_chunk = policy_mask[start:end].to(device, non_blocking=True).bool()
                legal_index_chunk = legal_indices[start:end].to(device, non_blocking=True).long()
                legal_mask_chunk = legal_mask[start:end].to(device, non_blocking=True).bool()
                value_target_chunk = value_targets[start:end].to(device, non_blocking=True).float().view(-1)

                with torch.amp.autocast(
                    "cuda",
                    enabled=amp_enabled,
                    dtype=amp_dtype,
                ):
                    policy_logits, value_logits = _model_outputs(model, board_chunk)
                    policy_logits = policy_logits.float()
                    if value_logits is not None:
                        value_logits = value_logits.float()

                policy_size = int(policy_logits.shape[1])
                valid_policy = (
                    policy_mask_chunk
                    & (policy_index_chunk >= 0)
                    & (policy_index_chunk < policy_size)
                )
                valid_legal = (
                    legal_mask_chunk
                    & (legal_index_chunk >= 0)
                    & (legal_index_chunk < policy_size)
                )
                target_slots = policy_value_chunk.masked_fill(
                    ~valid_policy,
                    -torch.inf,
                ).argmax(dim=1, keepdim=True)
                safe_policy_indices = policy_index_chunk.clamp(0, max(0, policy_size - 1))
                target_moves = torch.gather(safe_policy_indices, 1, target_slots).squeeze(1)

                safe_legal_indices = legal_index_chunk.clamp(0, max(0, policy_size - 1))
                legal_logits = torch.gather(policy_logits, 1, safe_legal_indices).masked_fill(
                    ~valid_legal,
                    -torch.inf,
                )
                dense_targets = torch.zeros_like(policy_logits)
                dense_targets.scatter_add_(
                    1,
                    safe_policy_indices,
                    torch.where(
                        valid_policy,
                        torch.clamp(policy_value_chunk, min=0.0),
                        torch.zeros_like(policy_value_chunk),
                    ),
                )
                legal_targets = torch.gather(dense_targets, 1, safe_legal_indices)
                legal_targets = torch.where(
                    valid_legal, legal_targets, torch.zeros_like(legal_targets)
                )
                legal_targets = legal_targets / legal_targets.sum(
                    dim=1, keepdim=True
                ).clamp_min(1e-8)
                legal_log_probs = torch.log_softmax(legal_logits, dim=1)
                # Padded legal slots have log-probability -inf. Multiplying
                # those by a zero target yields 0*inf -> NaN, which poisoned
                # every logged KL despite otherwise valid rows.
                legal_log_probs = torch.where(
                    valid_legal,
                    legal_log_probs,
                    torch.zeros_like(legal_log_probs),
                )
                positive_targets = legal_targets > 0.0
                safe_target_logs = torch.where(
                    positive_targets,
                    torch.log(legal_targets.clamp_min(1e-12)),
                    torch.zeros_like(legal_targets),
                )
                target_kl = (
                    legal_targets * (safe_target_logs - legal_log_probs)
                ).sum(dim=1)
                if (
                    value_logits is not None
                    and value_logits.ndim == 2
                    and int(value_logits.shape[1]) == 3
                ):
                    target_scalar = torch.where(
                        value_target_chunk.abs() <= 1e-6,
                        torch.zeros_like(value_target_chunk),
                        torch.sign(value_target_chunk),
                    )
                    target_class = torch.where(
                        target_scalar > 0.0,
                        torch.zeros_like(target_scalar, dtype=torch.long),
                        torch.where(
                            target_scalar < 0.0,
                            torch.full_like(target_scalar, 2, dtype=torch.long),
                            torch.ones_like(target_scalar, dtype=torch.long),
                        ),
                    )
                    value_log_probs = torch.log_softmax(value_logits, dim=1)
                    value_probs = torch.softmax(value_logits, dim=1)
                    value_wdl_ce = -torch.gather(
                        value_log_probs, 1, target_class.unsqueeze(1)
                    ).squeeze(1)
                    value_target_wdl = torch.nn.functional.one_hot(
                        target_class, num_classes=3
                    ).float()
                    value_brier = torch.square(value_probs - value_target_wdl).sum(dim=1)
                    value_scalar = value_probs[:, 0] - value_probs[:, 2]
                    value_mae = torch.abs(value_scalar - target_scalar)
                else:
                    value_wdl_ce = torch.full(
                        (end - start,), float("nan"), device=device
                    )
                    value_brier = torch.full_like(value_wdl_ce, float("nan"))
                    value_mae = torch.full_like(value_wdl_ce, float("nan"))
                target_is_legal = (
                    valid_legal & (safe_legal_indices == target_moves.unsqueeze(1))
                ).any(dim=1)
                valid_rows = valid_policy.any(dim=1) & valid_legal.any(dim=1) & target_is_legal
                if not bool(valid_rows.any()):
                    continue

                predicted_slots = legal_logits.argmax(dim=1, keepdim=True)
                predicted_moves = torch.gather(
                    safe_legal_indices,
                    1,
                    predicted_slots,
                ).squeeze(1)
                target_logits = torch.gather(
                    policy_logits,
                    1,
                    target_moves.clamp(0, max(0, policy_size - 1)).unsqueeze(1),
                ).squeeze(1)
                target_probabilities = torch.exp(
                    target_logits - torch.logsumexp(legal_logits, dim=1)
                )
                ranks = 1.0 + (
                    valid_legal & (legal_logits > target_logits.unsqueeze(1))
                ).sum(dim=1).float()
                alternative_logits = legal_logits.masked_fill(
                    valid_legal & (safe_legal_indices == target_moves.unsqueeze(1)),
                    -torch.inf,
                )
                best_alternative = alternative_logits.max(dim=1).values
                margins = target_logits - best_alternative
                margins = torch.where(torch.isfinite(margins), margins, torch.zeros_like(margins))

                top1_parts.append((predicted_moves == target_moves)[valid_rows].cpu())
                rank_parts.append(ranks[valid_rows].cpu())
                probability_parts.append(target_probabilities[valid_rows].cpu())
                margin_parts.append(margins[valid_rows].cpu())
                target_kl_parts.append(target_kl[valid_rows].cpu())
                value_wdl_ce_parts.append(value_wdl_ce[valid_rows].cpu())
                value_brier_parts.append(value_brier[valid_rows].cpu())
                value_mae_parts.append(value_mae[valid_rows].cpu())
    finally:
        model.train(was_training)

    if not top1_parts:
        empty_bool = torch.empty(0, dtype=torch.bool)
        empty_float = torch.empty(0, dtype=torch.float32)
        return CorrectionAuditSnapshot(
            empty_bool, empty_float, empty_float, empty_float, empty_float,
            empty_float, empty_float, empty_float,
        )
    return CorrectionAuditSnapshot(
        torch.cat(top1_parts),
        torch.cat(rank_parts),
        torch.cat(probability_parts),
        torch.cat(margin_parts),
        torch.cat(target_kl_parts),
        torch.cat(value_wdl_ce_parts),
        torch.cat(value_brier_parts),
        torch.cat(value_mae_parts),
    )


def holdout_metrics(
    before: CorrectionAuditSnapshot | None,
    after: CorrectionAuditSnapshot | None,
) -> dict[str, float | int]:
    """Measure generalization on fresh positions excluded from optimizer replay."""
    metrics: dict[str, float | int] = {
        "holdout_rows": 0,
        "holdout_policy_kl_before": float("nan"),
        "holdout_policy_kl_after": float("nan"),
        "holdout_policy_kl_reduction": float("nan"),
        "holdout_value_wdl_ce_before": float("nan"),
        "holdout_value_wdl_ce_after": float("nan"),
        "holdout_value_wdl_ce_reduction": float("nan"),
        "holdout_value_brier_before": float("nan"),
        "holdout_value_brier_after": float("nan"),
        "holdout_value_brier_reduction": float("nan"),
        "holdout_value_mae_before": float("nan"),
        "holdout_value_mae_after": float("nan"),
        "holdout_value_mae_reduction": float("nan"),
    }
    if before is None or after is None or before.rows != after.rows or after.rows <= 0:
        return metrics

    def _mean(tensor):
        finite = tensor[torch.isfinite(tensor)]
        return float(finite.mean().item()) if finite.numel() else float("nan")

    policy_before = _mean(before.target_kl)
    policy_after = _mean(after.target_kl)
    value_ce_before = _mean(before.value_wdl_ce)
    value_ce_after = _mean(after.value_wdl_ce)
    value_brier_before = _mean(before.value_brier)
    value_brier_after = _mean(after.value_brier)
    value_mae_before = _mean(before.value_mae)
    value_mae_after = _mean(after.value_mae)
    metrics.update({
        "holdout_rows": int(after.rows),
        "holdout_policy_kl_before": policy_before,
        "holdout_policy_kl_after": policy_after,
        "holdout_policy_kl_reduction": policy_before - policy_after,
        "holdout_value_wdl_ce_before": value_ce_before,
        "holdout_value_wdl_ce_after": value_ce_after,
        "holdout_value_wdl_ce_reduction": value_ce_before - value_ce_after,
        "holdout_value_brier_before": value_brier_before,
        "holdout_value_brier_after": value_brier_after,
        "holdout_value_brier_reduction": value_brier_before - value_brier_after,
        "holdout_value_mae_before": value_mae_before,
        "holdout_value_mae_after": value_mae_after,
        "holdout_value_mae_reduction": value_mae_before - value_mae_after,
    })
    return metrics


def correction_audit_metrics(
    before: CorrectionAuditSnapshot | None,
    after: CorrectionAuditSnapshot | None,
    *,
    previous_after: CorrectionAuditSnapshot | None = None,
    previous_now: CorrectionAuditSnapshot | None = None,
) -> dict[str, float | int]:
    """Summarize direct uptake and one-iteration retention."""

    metrics: dict[str, float | int] = {
        "correction_audit_rows": 0,
        "correction_audit_top1_before": float("nan"),
        "correction_audit_top1_after": float("nan"),
        "correction_audit_top1_gain": float("nan"),
        "correction_audit_rank_before": float("nan"),
        "correction_audit_rank_after": float("nan"),
        "correction_audit_rank_gain": float("nan"),
        "correction_audit_target_probability_before": float("nan"),
        "correction_audit_target_probability_after": float("nan"),
        "correction_audit_target_probability_gain": float("nan"),
        "correction_audit_logit_margin_before": float("nan"),
        "correction_audit_logit_margin_after": float("nan"),
        "correction_audit_logit_margin_gain": float("nan"),
        "correction_audit_target_kl_before": float("nan"),
        "correction_audit_target_kl_after": float("nan"),
        "correction_audit_target_kl_reduction": float("nan"),
        "correction_audit_target_kl_closed_fraction": float("nan"),
        "correction_audit_retention": float("nan"),
        "correction_audit_retention_rows": 0,
    }
    if before is not None and after is not None and before.rows == after.rows and after.rows > 0:
        top1_before = float(before.top1.float().mean().item())
        top1_after = float(after.top1.float().mean().item())
        rank_before = float(before.rank.mean().item())
        rank_after = float(after.rank.mean().item())
        probability_before = float(before.target_probability.mean().item())
        probability_after = float(after.target_probability.mean().item())
        margin_before = float(before.logit_margin.mean().item())
        margin_after = float(after.logit_margin.mean().item())
        target_kl_before = float(before.target_kl.mean().item())
        target_kl_after = float(after.target_kl.mean().item())
        target_kl_reduction = target_kl_before - target_kl_after
        metrics.update({
            "correction_audit_rows": after.rows,
            "correction_audit_top1_before": top1_before,
            "correction_audit_top1_after": top1_after,
            "correction_audit_top1_gain": top1_after - top1_before,
            "correction_audit_rank_before": rank_before,
            "correction_audit_rank_after": rank_after,
            "correction_audit_rank_gain": rank_before - rank_after,
            "correction_audit_target_probability_before": probability_before,
            "correction_audit_target_probability_after": probability_after,
            "correction_audit_target_probability_gain": probability_after - probability_before,
            "correction_audit_logit_margin_before": margin_before,
            "correction_audit_logit_margin_after": margin_after,
            "correction_audit_logit_margin_gain": margin_after - margin_before,
            "correction_audit_target_kl_before": target_kl_before,
            "correction_audit_target_kl_after": target_kl_after,
            "correction_audit_target_kl_reduction": target_kl_reduction,
            "correction_audit_target_kl_closed_fraction": (
                target_kl_reduction / max(1e-8, target_kl_before)
            ),
        })

    if (
        previous_after is not None
        and previous_now is not None
        and previous_after.rows == previous_now.rows
        and previous_now.rows > 0
    ):
        resolved = previous_after.top1.bool()
        resolved_count = int(resolved.sum().item())
        metrics["correction_audit_retention_rows"] = resolved_count
        if resolved_count > 0:
            metrics["correction_audit_retention"] = float(
                previous_now.top1[resolved].float().mean().item()
            )
    return metrics
