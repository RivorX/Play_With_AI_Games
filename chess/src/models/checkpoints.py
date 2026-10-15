"""Checkpoint I/O and explicit weight-transfer operations."""

from __future__ import annotations

import os
import random
import time

import numpy as np
import torch


_DEPRECATED_MODEL_STATE_PREFIXES = (
    "search_error_fc.",
    "search_value_ready",
)


def load_checkpoint_file(checkpoint_path, device):
    """Load a tensor-only checkpoint without unsafe pickle fallback."""
    return torch.load(checkpoint_path, map_location=device, weights_only=True)


def capture_rng_state():
    """Capture all RNGs used by RL sampling in a weights-only-safe format."""
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": [],
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    """Restore a state from ``capture_rng_state``; return restored domains."""
    if not isinstance(state, dict):
        return ()
    restored = []
    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)
        restored.append("python")
    numpy_state = state.get("numpy")
    if isinstance(numpy_state, dict) and torch.is_tensor(numpy_state.get("keys")):
        np.random.set_state((
            str(numpy_state.get("bit_generator", "MT19937")),
            numpy_state["keys"].cpu().numpy().astype(np.uint32, copy=False),
            int(numpy_state.get("position", 0) or 0),
            int(numpy_state.get("has_gauss", 0) or 0),
            float(numpy_state.get("cached_gaussian", 0.0) or 0.0),
        ))
        restored.append("numpy")
    torch_cpu = state.get("torch_cpu")
    if torch.is_tensor(torch_cpu):
        torch.set_rng_state(torch_cpu.cpu())
        restored.append("torch_cpu")
    torch_cuda = state.get("torch_cuda")
    if torch.cuda.is_available() and isinstance(torch_cuda, (list, tuple)) and torch_cuda:
        torch.cuda.set_rng_state_all([item.cpu() for item in torch_cuda])
        restored.append("torch_cuda")
    return tuple(restored)


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
        checkpoint["rng_state"] = capture_rng_state()
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
