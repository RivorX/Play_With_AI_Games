"""Canonical public API for constructing and loading chess models."""

import torch

from src.models.contracts import (
    legacy_model_spec,
    model_spec_from_config,
    validate_model_spec,
)
from src.models.registry import (
    MODEL_REGISTRY,
    config_for_model_spec,
    create_model,
    register_model,
    resolve_model_spec,
)
from src.models.architecture.se_cnn_v9 import SECNNV9
from src.models.checkpoints import (
    load_checkpoint_file,
    normalize_state_dict_keys,
    save_checkpoint,
    transfer_matching_weights,
)

def load_model(checkpoint_path, config, device, strict=True):
    checkpoint = (
        load_checkpoint_file(checkpoint_path, device) if checkpoint_path else None
    )
    spec = resolve_model_spec(config, checkpoint)
    model = create_model(config, model_spec=spec).to(device)
    model = model.to(memory_format=torch.channels_last)
    if checkpoint:
        normalized = normalize_state_dict_keys(
            checkpoint["model_state_dict"], target_keys=set(model.state_dict())
        )
        if strict:
            incompatible = model.load_state_dict(normalized, strict=False)
            optional_prefixes = (
                "search_q_fc.",
                "moves_left_fc",
                "moves_left_ln.",
            )
            missing = [
                key
                for key in incompatible.missing_keys
                if not key.startswith(optional_prefixes)
            ]
            if missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "Checkpoint state does not match model_spec: "
                    f"missing={missing[:5]}, "
                    f"unexpected={incompatible.unexpected_keys[:5]}"
                )
        else:
            transfer_matching_weights(model, normalized)
    return model

__all__ = [
    "SECNNV9",
    "MODEL_REGISTRY",
    "config_for_model_spec",
    "create_model",
    "legacy_model_spec",
    "load_checkpoint_file",
    "load_model",
    "model_spec_from_config",
    "normalize_state_dict_keys",
    "register_model",
    "resolve_model_spec",
    "save_checkpoint",
    "transfer_matching_weights",
    "validate_model_spec",
]
