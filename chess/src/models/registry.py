"""Architecture registry and checkpoint-driven model factory."""

from __future__ import annotations

from copy import deepcopy

from src.models.contracts import (
    legacy_model_spec,
    model_spec_from_config,
    validate_model_spec,
)
from src.models.architecture.se_cnn_v9 import SECNNV9
from src.models.data import validate_data_contract


MODEL_REGISTRY = {
    "se_cnn_v9": SECNNV9,
}


def register_model(architecture_id: str, model_class) -> None:
    architecture_id = str(architecture_id).strip()
    if not architecture_id:
        raise ValueError("architecture_id cannot be empty")
    if architecture_id in MODEL_REGISTRY:
        raise ValueError(f"Architecture already registered: {architecture_id}")
    MODEL_REGISTRY[architecture_id] = model_class


def resolve_model_spec(
    config: dict,
    checkpoint: dict | None = None,
) -> dict:
    raw_spec = (checkpoint or {}).get("model_spec")
    spec = deepcopy(raw_spec) if isinstance(raw_spec, dict) else legacy_model_spec(
        checkpoint, config
    )
    validate_model_spec(spec)
    validate_data_contract(spec)
    return spec


def config_for_model_spec(config: dict, spec: dict) -> dict:
    """Apply checkpoint architecture while retaining runtime-only settings."""
    validate_model_spec(spec)
    resolved = deepcopy(config)
    model_cfg = dict(spec["model_kwargs"])
    model_cfg["architecture_id"] = spec["architecture_id"]
    model_cfg["input_encoder_id"] = spec["input_spec"]["encoder_id"]
    model_cfg["history_positions"] = int(
        spec["input_spec"].get(
            "history_positions", model_cfg.get("history_positions", 0)
        )
    )
    model_cfg["policy_codec_id"] = spec["policy_spec"]["codec_id"]
    model_cfg["value_head_id"] = spec.get("value_spec", {}).get(
        "head_id", "wdl_search_aux_v1"
    )
    if "print_summary" in config.get("model", {}):
        model_cfg["print_summary"] = bool(config["model"]["print_summary"])
    resolved["model"] = model_cfg
    return resolved


def create_model(config: dict, *, model_spec: dict | None = None):
    spec = deepcopy(model_spec) if model_spec is not None else model_spec_from_config(
        config
    )
    validate_model_spec(spec)
    validate_data_contract(spec)
    architecture_id = spec["architecture_id"]
    try:
        model_class = MODEL_REGISTRY[architecture_id]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown architecture {architecture_id!r}; registered: {known}"
        ) from exc
    model = model_class(config_for_model_spec(config, spec))
    model.model_spec = spec
    model.input_encoder_id = spec["input_spec"]["encoder_id"]
    model.policy_codec_id = spec["policy_spec"]["codec_id"]
    model.action_size = int(spec["policy_spec"]["action_size"])
    return model
