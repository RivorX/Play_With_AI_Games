"""Checkpoint I/O and explicit weight-transfer operations."""

from __future__ import annotations

import os
import time

import torch


_DEPRECATED_MODEL_STATE_PREFIXES = (
    "search_error_fc.",
    "search_value_ready",
)


def load_checkpoint_file(checkpoint_path, device):
    """Load a tensor-only checkpoint without unsafe pickle fallback."""
    return torch.load(checkpoint_path, map_location=device, weights_only=True)


def normalize_state_dict_keys(source_state, target_keys=None):
    """Strip DataParallel/torch.compile wrapper prefixes."""
    if source_state is None:
        return {}

    wrapper_prefixes = ("module", "_orig_mod")
    normalized = {}
    for key, tensor in source_state.items():
        canonical_parts = key.split(".")
        while len(canonical_parts) > 1 and canonical_parts[0] in wrapper_prefixes:
            canonical_parts = canonical_parts[1:]
        canonical_key = ".".join(canonical_parts)
        if canonical_key.startswith(_DEPRECATED_MODEL_STATE_PREFIXES):
            continue

        norm_key = key
        parts = key.split(".")
        if target_keys is None or norm_key not in target_keys:
            while len(parts) > 1 and parts[0] in wrapper_prefixes:
                parts = parts[1:]
                candidate = ".".join(parts)
                if target_keys is None:
                    norm_key = candidate
                    continue
                if candidate in target_keys:
                    norm_key = candidate
                    break
        normalized.setdefault(norm_key, tensor)
    return normalized


def transfer_matching_weights(model, checkpoint_or_state):
    """Explicitly transfer tensors whose name and shape match."""
    source_state = (
        checkpoint_or_state["model_state_dict"]
        if isinstance(checkpoint_or_state, dict)
        and "model_state_dict" in checkpoint_or_state
        else checkpoint_or_state
    )
    target_state = model.state_dict()
    normalized = normalize_state_dict_keys(
        source_state, target_keys=set(target_state)
    )

    matched_keys = []
    unexpected_keys = []
    shape_mismatch = []
    for key, tensor in normalized.items():
        if key not in target_state:
            unexpected_keys.append(key)
        elif target_state[key].shape != tensor.shape:
            shape_mismatch.append(
                (key, tuple(tensor.shape), tuple(target_state[key].shape))
            )
        else:
            target_state[key] = tensor.to(
                dtype=target_state[key].dtype,
                device=target_state[key].device,
            )
            matched_keys.append(key)

    matched_set = set(matched_keys)
    missing_keys = [key for key in target_state if key not in matched_set]
    model.load_state_dict(target_state, strict=False)
    matched_elements = sum(target_state[key].numel() for key in matched_keys)
    total_elements = sum(value.numel() for value in target_state.values())
    return {
        "matched_keys": matched_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatch": shape_mismatch,
        "matched_tensors": len(matched_keys),
        "total_tensors": len(target_state),
        "matched_elements": matched_elements,
        "total_elements": total_elements,
        "match_ratio": matched_elements / total_elements if total_elements else 0.0,
    }


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    path,
    metadata=None,
    save_optimizer=False,
    save_dtype=None,
    extra_state=None,
):
    """Atomically save a checkpoint containing its mandatory model_spec."""
    state_dict = normalize_state_dict_keys(model.state_dict())
    if save_dtype is not None:
        state_dict = {
            key: (
                value.to(save_dtype)
                if value.is_floating_point()
                else value.clone()
            )
            for key, value in state_dict.items()
        }

    unwrapped = getattr(model, "_orig_mod", getattr(model, "module", model))
    model_spec = getattr(unwrapped, "model_spec", None)
    if not isinstance(model_spec, dict):
        raise ValueError("Cannot save model without a model_spec")

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": state_dict,
        "model_spec": model_spec,
        "loss": loss,
    }
    try:
        checkpoint["rng_state"] = {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
        }
    except Exception:
        pass
    if save_optimizer and optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if metadata:
        checkpoint.update(metadata)
    if extra_state:
        checkpoint.update(extra_state)

    path = os.fspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp-{os.getpid()}-{time.time_ns()}"
    torch.save(checkpoint, tmp_path)

    last_error = None
    for attempt in range(6):
        try:
            os.replace(tmp_path, path)
            last_error = None
            break
        except OSError as exc:
            last_error = exc
            time.sleep(0.15 * (attempt + 1))
    if last_error is not None:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise last_error

    size_mb = os.path.getsize(path) / (1024 ** 2)
    opt_status = "with optimizer" if save_optimizer else "without optimizer"
    print(f"Checkpoint saved to {path} ({size_mb:.2f} MB, {opt_status})")
    return path
