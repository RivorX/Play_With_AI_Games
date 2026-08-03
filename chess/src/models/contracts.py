"""Stable identifiers connecting a checkpoint, model, encoder and policy."""

from __future__ import annotations

from copy import deepcopy


MODEL_SPEC_SCHEMA_VERSION = 1
DEFAULT_ARCHITECTURE_ID = "se_cnn_v9"
DEFAULT_INPUT_ENCODER_ID = "planes_history_v1"
DEFAULT_POLICY_CODEC_ID = "lc0_1858_v1"
DEFAULT_VALUE_HEAD_ID = "wdl_search_aux_v1"

SE_CNN_V9_MODEL_KEYS = (
    "version",
    "num_residual_blocks",
    "filters",
    "policy_head_channels",
    "value_head_filters",
    "value_hidden_dim",
    "moves_left_hidden_dim",
    "dropout",
    "history_positions",
    "use_se_blocks",
    "use_se_bottleneck",
    "se_reduction",
    "drop_path_rate",
    "use_coord_conv",
    "use_layer_scale",
    "layer_scale_init",
)


def model_spec_from_config(config: dict) -> dict:
    model_cfg = config.get("model", {}) or {}
    architecture_id = str(
        model_cfg.get("architecture_id", DEFAULT_ARCHITECTURE_ID)
    )
    if architecture_id != "se_cnn_v9":
        kwargs = {
            key: deepcopy(value)
            for key, value in model_cfg.items()
            if key not in {
                "architecture_id",
                "input_encoder_id",
                "policy_codec_id",
                "value_head_id",
            }
        }
    else:
        kwargs = {
            key: deepcopy(model_cfg[key])
            for key in SE_CNN_V9_MODEL_KEYS
            if key in model_cfg
        }
    history_positions = int(model_cfg.get("history_positions", 0))
    return {
        "schema_version": MODEL_SPEC_SCHEMA_VERSION,
        "architecture_id": architecture_id,
        "model_kwargs": kwargs,
        "input_spec": {
            "encoder_id": str(
                model_cfg.get("input_encoder_id", DEFAULT_INPUT_ENCODER_ID)
            ),
            "history_positions": history_positions,
            "input_planes": 16 * (1 + history_positions),
        },
        "policy_spec": {
            "codec_id": str(
                model_cfg.get("policy_codec_id", DEFAULT_POLICY_CODEC_ID)
            ),
            "action_size": 1858,
            "policy_planes": 73,
        },
        "value_spec": {
            "head_id": str(
                model_cfg.get("value_head_id", DEFAULT_VALUE_HEAD_ID)
            ),
            "wdl_size": 3,
        },
    }


def legacy_model_spec(checkpoint: dict | None, runtime_config: dict) -> dict:
    """Infer the current SE-CNN contract for checkpoints predating model_spec."""
    config = deepcopy(runtime_config)
    checkpoint = checkpoint or {}
    legacy = checkpoint.get("model_architecture")
    if isinstance(legacy, dict):
        config.setdefault("model", {}).update(legacy)
    state = checkpoint.get("model_state_dict")
    if isinstance(state, dict):
        config.setdefault("model", {}).update(_infer_se_cnn_v9_kwargs(state))
    config.setdefault("model", {}).setdefault(
        "architecture_id", DEFAULT_ARCHITECTURE_ID
    )
    return model_spec_from_config(config)


def _infer_se_cnn_v9_kwargs(state: dict) -> dict:
    """Best-effort bridge for old checkpoints that saved no architecture metadata."""
    inferred = {}
    stem_coord = state.get("conv_block.0.conv.weight")
    stem_plain = state.get("conv_block.0.weight")
    stem = stem_coord if stem_coord is not None else stem_plain
    if stem is not None and getattr(stem, "ndim", 0) == 4:
        inferred["filters"] = int(stem.shape[0])
        input_planes = int(stem.shape[1]) - (2 if stem_coord is not None else 0)
        inferred["use_coord_conv"] = stem_coord is not None
        if input_planes > 0 and input_planes % 16 == 0:
            inferred["history_positions"] = input_planes // 16 - 1

    block_ids = set()
    for key in state:
        if key.startswith("residual_tower."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                block_ids.add(int(parts[1]))
    if block_ids:
        inferred["num_residual_blocks"] = max(block_ids) + 1

    shape_fields = (
        ("policy_conv.weight", "policy_head_channels", 0),
        ("value_conv.weight", "value_head_filters", 0),
        ("value_fc1.weight", "value_hidden_dim", 0),
        ("moves_left_fc1.weight", "moves_left_hidden_dim", 0),
    )
    for tensor_key, config_key, axis in shape_fields:
        tensor = state.get(tensor_key)
        if tensor is not None:
            inferred[config_key] = int(tensor.shape[axis])

    se_bottleneck = [key for key in state if ".se.fc1.weight" in key]
    se_direct = [key for key in state if ".se.fc.weight" in key]
    inferred["use_se_blocks"] = bool(se_bottleneck or se_direct)
    if inferred["use_se_blocks"]:
        inferred["use_se_bottleneck"] = bool(se_bottleneck)
        if se_bottleneck:
            tensor = state[se_bottleneck[0]]
            inferred["se_reduction"] = max(
                1, int(tensor.shape[1]) // max(1, int(tensor.shape[0]))
            )
    inferred["use_layer_scale"] = any(
        ".layer_scale.gamma" in key for key in state
    )
    return inferred


def validate_model_spec(spec: dict) -> None:
    required = ("architecture_id", "model_kwargs", "input_spec", "policy_spec")
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(f"Invalid model_spec; missing: {', '.join(missing)}")
    if int(spec.get("schema_version", 0)) != MODEL_SPEC_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported model_spec schema: {spec.get('schema_version')!r}"
        )
