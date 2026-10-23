"""Small raw-prediction helpers shared by local and remote inference callers.

``predict_raw`` deliberately stops at policy/value tensors. Callers retain
ownership of input encoding, batching, legal-move gathering, output transfer,
and any chess-specific interpretation. An optional graph runner is only a
dispatch target; graph capture and lifetime remain its owner's responsibility.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from contextlib import nullcontext
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F


_PREDICTION_PROVENANCE_SCHEMA_VERSION = 1
_PREDICTION_ROW_SOURCES = frozenset(
    {"fresh_nn", "cache_hit", "deduplicated", "unknown"}
)


def _prediction_provenance_for_sources(
    row_sources: Any, batch_size: int | None = None
) -> dict[str, Any]:
    """Build the versioned row-attribution contract used by local and remote inference."""
    if not isinstance(row_sources, (list, tuple)):
        sources = []
    else:
        sources = [
            source
            if isinstance(source, str) and source in _PREDICTION_ROW_SOURCES
            else "unknown"
            for source in row_sources
        ]
    aligned = True
    if batch_size is not None:
        expected = max(0, int(batch_size))
        aligned = len(sources) == expected
        if not aligned:
            sources = ["unknown"] * expected
    complete = aligned and all(source != "unknown" for source in sources)
    counts = {"fresh_nn": 0, "cache_hit": 0, "deduplicated": 0}
    for source in sources:
        if source in counts:
            counts[source] += 1
    return {
        "schema_version": _PREDICTION_PROVENANCE_SCHEMA_VERSION,
        "row_sources": sources,
        "nn_evaluated_positions": counts["fresh_nn"] if complete else None,
        "cache_hits": counts["cache_hit"] if complete else None,
        "dedup_hits": counts["deduplicated"] if complete else None,
        "provenance_complete": complete,
    }


def _copy_prediction_provenance(
    target: MutableMapping[str, Any], value: Any, batch_size: int
) -> None:
    """Copy a producer's row attribution, failing closed for absent/legacy data."""
    if (
        isinstance(value, Mapping)
        and type(value.get("schema_version")) is int
        and value.get("schema_version") == _PREDICTION_PROVENANCE_SCHEMA_VERSION
    ):
        result = _prediction_provenance_for_sources(value.get("row_sources"), batch_size)
    else:
        result = _prediction_provenance_for_sources(None, batch_size)
    target.clear()
    target.update(result)


def _raw_output_pair(outputs: Any, inputs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(outputs, (tuple, list)) or len(outputs) < 2:
        raise TypeError("prediction callable must return (policy_logits, value_logits)")
    policy, value = outputs[0], outputs[1]
    if not isinstance(policy, torch.Tensor) or not isinstance(value, torch.Tensor):
        raise TypeError("prediction policy and value outputs must be tensors")
    if not policy.is_floating_point():
        raise TypeError("prediction policy logits must have a floating-point dtype")
    if not value.is_floating_point():
        raise TypeError("prediction values must have a floating-point dtype")
    if policy.ndim != 2:
        raise ValueError("prediction policy logits must have shape (B, policy_width)")
    if value.ndim == 1:
        pass
    elif value.ndim == 2 and int(value.shape[1]) in (1, 3):
        pass
    else:
        raise ValueError(
            "prediction values must have shape (B,), (B, 1), or (B, 3); "
            f"got {tuple(value.shape)}"
        )
    if int(policy.shape[0]) != int(value.shape[0]):
        raise ValueError("prediction policy and value output batches must match")
    if policy.device != value.device:
        raise ValueError("prediction policy and value outputs must be on the same device")

    input_shape = getattr(inputs, "shape", None)
    if input_shape is not None and len(input_shape) > 0:
        batch_size = int(input_shape[0])
        if int(policy.shape[0]) != batch_size or int(value.shape[0]) != batch_size:
            raise ValueError("prediction output batch does not match input batch")
    return policy, value


def value_logits_to_scalar(values: torch.Tensor) -> torch.Tensor:
    """Return one scalar value per batch row without changing dtype/device.

    Three-column inputs are W/D/L logits and become ``P(W) - P(L)``. One-
    dimensional inputs and ``(B, 1)`` inputs are already-scalar values and are
    returned as-is (apart from removing the singleton column in the latter).
    """
    if not isinstance(values, torch.Tensor):
        raise TypeError("values must be a torch.Tensor")
    if not values.is_floating_point():
        raise TypeError("values must have a floating-point dtype")
    if values.ndim == 1:
        return values
    if values.ndim == 2 and int(values.shape[1]) == 1:
        return values.squeeze(1)
    if values.ndim == 2 and int(values.shape[1]) == 3:
        wdl_probs = F.softmax(values, dim=1)
        return wdl_probs[:, 0] - wdl_probs[:, 2]
    raise ValueError(
        "values must have shape (B,), (B, 1), or W/D/L logits shaped (B, 3); "
        f"got {tuple(values.shape)}"
    )


def predict_raw(
    model: Any,
    inputs: Any,
    *,
    use_amp: bool = False,
    amp_dtype: torch.dtype | None = None,
    amp_cache_enabled: bool = True,
    graph_runner: Any = None,
    model_kwargs: Mapping[str, Any] | None = None,
    provenance: MutableMapping[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one raw policy/value prediction and preserve producer-owned outputs.

    The eager local path runs under ``inference_mode`` and uses CUDA autocast
    only when requested for a CUDA tensor. A graph runner handles its own
    inference context and precision. Any callable proxy, including the central
    remote client, uses the same raw-output contract; ``model_kwargs`` carries
    existing transport arguments such as ``legal_index_matrix`` and
    ``legal_counts`` through unchanged.

    This helper never moves or converts inputs or outputs. In particular, a
    remote proxy keeps ownership of its queue/shared-memory protocol. This
    helper adds no output copies: eager outputs are passed through, and any
    graph-runner output ownership guarantees (such as cloned replay tensors)
    remain the runner's responsibility. Policy logits must be ``(B, width)``;
    the width is intentionally unrestricted for compact remote policies.
    Values may be scalar ``(B,)``/``(B, 1)`` or W/D/L logits ``(B, 3)``.

    When ``provenance`` is supplied, it is replaced with versioned per-row
    attribution. Local module and graph rows are ``fresh_nn``; remote rows use
    server attribution, with missing legacy metadata reported as ``unknown``.
    """
    if provenance is not None and not isinstance(provenance, MutableMapping):
        raise TypeError("provenance must be a mutable mapping when provided")

    kwargs = dict(model_kwargs or {})
    kwargs.setdefault("apply_log_softmax", False)
    if kwargs.get("apply_log_softmax") is not False:
        raise ValueError("predict_raw only supports raw policy logits")

    if graph_runner is not None and kwargs == {"apply_log_softmax": False}:
        outputs = _raw_output_pair(graph_runner.run_forward(inputs), inputs)
        if provenance is not None:
            _copy_prediction_provenance(
                provenance,
                _prediction_provenance_for_sources(
                    ["fresh_nn"] * int(outputs[0].shape[0])
                ),
                int(outputs[0].shape[0]),
            )
        return outputs

    amp_context = nullcontext()
    if (
        use_amp
        and isinstance(inputs, torch.Tensor)
        and inputs.device.type == "cuda"
    ):
        amp_context = torch.autocast(
            device_type="cuda",
            dtype=amp_dtype or torch.float16,
            enabled=True,
            cache_enabled=bool(amp_cache_enabled),
        )

    with torch.inference_mode(), amp_context:
        outputs = model(inputs, **kwargs)
    outputs = _raw_output_pair(outputs, inputs)
    if provenance is not None:
        batch_size = int(outputs[0].shape[0])
        if isinstance(model, torch.nn.Module):
            source_info = _prediction_provenance_for_sources(["fresh_nn"] * batch_size)
        elif bool(getattr(model, "_is_remote_inference_client", False)):
            _copy_prediction_provenance(
                provenance,
                getattr(model, "last_prediction_provenance", None),
                batch_size,
            )
            return outputs
        else:
            source_info = _prediction_provenance_for_sources(
                ["unknown"] * batch_size
            )
        _copy_prediction_provenance(provenance, source_info, batch_size)
    return outputs


def copy_compact_policy_value_to_cpu(
    legal_policy_logits: torch.Tensor,
    values: torch.Tensor,
    *,
    pinned_policy: torch.Tensor | None = None,
    pinned_values: torch.Tensor | None = None,
    transfer_event: torch.cuda.Event | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read back compact policy logits and scalar values with one CUDA wait.

    Policy results are rounded to FP16 and values to FP32, matching the local
    MCTS contract. CUDA callers supply reusable pinned CPU buffers and one
    reusable event. Both nonblocking copies are queued on the current stream,
    then a single event wait completes them before independent NumPy arrays are
    returned. CPU tensors retain the existing direct-conversion behavior.

    ``legal_policy_logits`` has shape ``(B, legal_width)``; ``values`` has
    shape ``(B,)`` or ``(B, 1)`` and should already contain the caller's final
    scalar interpretation (for example WDL-to-scalar conversion).
    """
    if not isinstance(legal_policy_logits, torch.Tensor):
        raise TypeError("legal_policy_logits must be a torch.Tensor")
    if not isinstance(values, torch.Tensor):
        raise TypeError("values must be a torch.Tensor")
    if legal_policy_logits.ndim != 2:
        raise ValueError("legal_policy_logits must have shape (B, legal_width)")

    batch_size, legal_width = map(int, legal_policy_logits.shape)
    if values.ndim == 1:
        value_batch_size = int(values.shape[0])
    elif values.ndim == 2 and int(values.shape[1]) == 1:
        value_batch_size = int(values.shape[0])
    else:
        raise ValueError("values must have shape (B,) or (B, 1)")
    if value_batch_size != batch_size:
        raise ValueError("policy and value output batches must match")
    if values.device != legal_policy_logits.device:
        raise ValueError("policy and value outputs must be on the same device")

    if legal_policy_logits.device.type == "cpu":
        policy_cpu = legal_policy_logits.detach().to(dtype=torch.float16).cpu().numpy()
        values_cpu = values.detach().reshape(batch_size).to(dtype=torch.float32).cpu().numpy()
        return policy_cpu, values_cpu
    if legal_policy_logits.device.type != "cuda":
        raise ValueError("asynchronous compact output transfer requires CUDA tensors")
    if pinned_policy is None or pinned_values is None or transfer_event is None:
        raise ValueError("CUDA compact output transfer requires reusable pinned buffers and event")
    if (
        pinned_policy.device.type != "cpu"
        or pinned_policy.dtype != torch.float16
        or not pinned_policy.is_pinned()
        or pinned_policy.ndim != 2
        or int(pinned_policy.shape[0]) < batch_size
        or int(pinned_policy.shape[1]) < legal_width
    ):
        raise ValueError("pinned_policy must be a sufficiently large pinned FP16 CPU tensor")
    if (
        pinned_values.device.type != "cpu"
        or pinned_values.dtype != torch.float32
        or not pinned_values.is_pinned()
        or pinned_values.ndim != 1
        or int(pinned_values.shape[0]) < batch_size
    ):
        raise ValueError("pinned_values must be a sufficiently large pinned FP32 CPU tensor")

    policy_dst = pinned_policy[:batch_size, :legal_width]
    values_dst = pinned_values[:batch_size]
    policy_fp16 = legal_policy_logits.detach().to(dtype=torch.float16)
    values_fp32 = values.detach().reshape(batch_size).to(dtype=torch.float32)
    policy_dst.copy_(policy_fp16, non_blocking=True)
    values_dst.copy_(values_fp32, non_blocking=True)
    transfer_event.record(torch.cuda.current_stream(legal_policy_logits.device))
    transfer_event.synchronize()
    return policy_dst.numpy().copy(), values_dst.numpy().copy()


def predict_single_position(
    model: Any,
    board_tensor: torch.Tensor,
    *,
    use_amp: bool = False,
    amp_dtype: torch.dtype | None = None,
) -> tuple[Any, float]:
    """Return one position's policy probabilities and WDL scalar value.

    This preserves the legacy architecture ``predict`` return shape: a NumPy
    policy vector and ``W - L`` scalar. Batched input is accepted and its first
    position is returned, matching the existing model convenience methods.
    """
    if not isinstance(board_tensor, torch.Tensor):
        raise TypeError("board_tensor must be a torch.Tensor")
    if board_tensor.ndim == 3:
        board_tensor = board_tensor.unsqueeze(0)
    elif board_tensor.ndim != 4:
        raise ValueError(
            "board_tensor must have shape (C, 8, 8) or (B, C, 8, 8), "
            f"got {tuple(board_tensor.shape)}"
        )

    was_training = bool(getattr(model, "training", False))
    module_modes: list[tuple[Any, bool]] = []
    if was_training:
        modules = getattr(model, "modules", None)
        if callable(modules):
            module_modes = [(module, bool(module.training)) for module in modules()]
        else:
            module_modes = [(model, was_training)]
    try:
        if was_training:
            model.eval()
        with torch.inference_mode():
            policy_logits, value_logits = predict_raw(
                model,
                board_tensor,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            if value_logits.ndim != 2 or value_logits.shape[1] != 3:
                raise ValueError(
                    "single-position prediction expects value logits shaped (B, 3)"
                )
            policy_probs = F.softmax(policy_logits, dim=1)
            value_scalar = value_logits_to_scalar(value_logits)[0]
            return policy_probs[0].float().cpu().numpy(), float(value_scalar.item())
    finally:
        for module, training in module_modes:
            module.training = training

