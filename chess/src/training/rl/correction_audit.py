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

    @property
    def rows(self) -> int:
        return int(self.top1.numel())


def freeze_replay_batch(batch: Iterable[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Detach a replay batch from reusable replay scratch buffers."""

    return tuple(tensor.detach().cpu().clone() for tensor in batch)


def _model_policy_logits(model, boards: torch.Tensor) -> torch.Tensor:
    output = model(
        boards,
        apply_log_softmax=False,
        return_moves_left=True,
        return_search_aux=True,
    )
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


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

    boards, policy_indices, policy_values, policy_mask = batch[:4]
    legal_indices, legal_mask = batch[8:10]
    row_count = int(boards.shape[0])
    if row_count <= 0:
        empty_bool = torch.empty(0, dtype=torch.bool)
        empty_float = torch.empty(0, dtype=torch.float32)
        return CorrectionAuditSnapshot(empty_bool, empty_float, empty_float, empty_float)

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

                with torch.amp.autocast(
                    "cuda",
                    enabled=amp_enabled,
                    dtype=amp_dtype,
                ):
                    policy_logits = _model_policy_logits(model, board_chunk).float()

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
    finally:
        model.train(was_training)

    if not top1_parts:
        empty_bool = torch.empty(0, dtype=torch.bool)
        empty_float = torch.empty(0, dtype=torch.float32)
        return CorrectionAuditSnapshot(empty_bool, empty_float, empty_float, empty_float)
    return CorrectionAuditSnapshot(
        torch.cat(top1_parts),
        torch.cat(rank_parts),
        torch.cat(probability_parts),
        torch.cat(margin_parts),
    )


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
