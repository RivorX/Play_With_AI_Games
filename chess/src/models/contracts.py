"""Stable identifiers connecting a checkpoint, model, encoder and policy."""

from __future__ import annotations

from copy import deepcopy


MODEL_SPEC_SCHEMA_VERSION = 1
DEFAULT_ARCHITECTURE_ID = "se_cnn_v9"
DEFAULT_INPUT_ENCODER_ID = "planes_history_v1"
DEFAULT_POLICY_CODEC_ID = "lc0_1858_v1"
DEFAULT_VALUE_HEAD_ID = "wdl_search_aux_v1"
V10_ARCHITECTURE_ID = "nested_cnn_v10"
V10_INPUT_ENCODER_ID = "planes_history_v2"

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

NESTED_CNN_V10_MODEL_KEYS = (
    "version",
    "history_positions",
    "input_planes",
    "trunk_channels",
    "bottleneck_channels",
    "outer_blocks",
    "inner_residual_units",
    "global_pool_blocks",
    "drop_path_rate",
    "policy_head_channels",
    "value_head_filters",
    "value_hidden_dim",
    "moves_left_hidden_dim",
    "dropout",
)


def model_spec_from_config(config: dict) -> dict:
    model_cfg = config.get("model", {}) or {}
    architecture_id = str(
        model_cfg.get("architecture_id", DEFAULT_ARCHITECTURE_ID)
    )
    if architecture_id == DEFAULT_ARCHITECTURE_ID:
        kwargs = {
            key: deepcopy(model_cfg[key])
            for key in SE_CNN_V9_MODEL_KEYS
            if key in model_cfg
        }
    elif architecture_id == V10_ARCHITECTURE_ID:
        kwargs = {
            key: deepcopy(model_cfg[key])
            for key in NESTED_CNN_V10_MODEL_KEYS
            if key in model_cfg
        }
    else:
        raise ValueError(f"No model-spec contract registered for {architecture_id!r}")
    history_positions = int(model_cfg.get("history_positions", 0))
    encoder_id = str(
        model_cfg.get(
            "input_encoder_id",
            V10_INPUT_ENCODER_ID if architecture_id == V10_ARCHITECTURE_ID else DEFAULT_INPUT_ENCODER_ID,
        )
    )
    from src.models.data.registry import get_input_encoder

    encoder = get_input_encoder(encoder_id)
    expected_planes = int(encoder.input_planes(history_positions))
    configured_planes = model_cfg.get("input_planes")
    if configured_planes is not None and int(configured_planes) != expected_planes:
        raise ValueError(
            f"Configured input_planes={configured_planes} does not match "
            f"{encoder_id} contract ({expected_planes})"
        )
    return {
        "schema_version": MODEL_SPEC_SCHEMA_VERSION,
        "architecture_id": architecture_id,
        "model_kwargs": kwargs,
        "input_spec": {
            "encoder_id": encoder_id,
            "history_positions": history_positions,
            "input_planes": expected_planes,
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
    legacy_architecture_id = (
        str(legacy.get("architecture_id"))
        if isinstance(legacy, dict) and legacy.get("architecture_id")
        else None
    )
    # Old checkpoints without an explicit architecture identifier are v9.
    # Never let a newly selected runtime profile reinterpret their state dict.
    model_cfg = config.setdefault("model", {})
    architecture_id = legacy_architecture_id or DEFAULT_ARCHITECTURE_ID
    if architecture_id not in {DEFAULT_ARCHITECTURE_ID, V10_ARCHITECTURE_ID}:
        raise ValueError(f"No legacy model-spec contract for {architecture_id!r}")
    model_cfg["architecture_id"] = architecture_id
    if isinstance(legacy, dict):
        model_cfg.update(legacy)
    model_cfg["architecture_id"] = architecture_id
    expected_encoder = (
        V10_INPUT_ENCODER_ID
        if architecture_id == V10_ARCHITECTURE_ID
        else DEFAULT_INPUT_ENCODER_ID
    )
    if not isinstance(legacy, dict) or "input_encoder_id" not in legacy:
        model_cfg["input_encoder_id"] = expected_encoder
    if not isinstance(legacy, dict) or "policy_codec_id" not in legacy:
        model_cfg["policy_codec_id"] = DEFAULT_POLICY_CODEC_ID
    if not isinstance(legacy, dict) or "value_head_id" not in legacy:
        model_cfg["value_head_id"] = DEFAULT_VALUE_HEAD_ID
    if architecture_id == DEFAULT_ARCHITECTURE_ID:
        # `input_planes` was introduced by v10. Discard a value inherited from
        # the active v10 profile before inferring this older checkpoint's shape.
        model_cfg.pop("input_planes", None)
    state = checkpoint.get("model_state_dict")
    if isinstance(state, dict) and model_cfg["architecture_id"] == DEFAULT_ARCHITECTURE_ID:
        model_cfg.update(_infer_se_cnn_v9_kwargs(state))
        model_cfg["architecture_id"] = DEFAULT_ARCHITECTURE_ID
        model_cfg["input_encoder_id"] = DEFAULT_INPUT_ENCODER_ID
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
    architecture_id = str(spec["architecture_id"])
    encoder_id = str((spec.get("input_spec") or {}).get("encoder_id", ""))
    expected_encoder = {
        DEFAULT_ARCHITECTURE_ID: DEFAULT_INPUT_ENCODER_ID,
        V10_ARCHITECTURE_ID: V10_INPUT_ENCODER_ID,
    }.get(architecture_id)
    if expected_encoder is None:
        raise ValueError(f"No model-spec contract registered for {architecture_id!r}")
    if encoder_id != expected_encoder:
        raise ValueError(
            f"Architecture {architecture_id!r} requires encoder {expected_encoder!r}, "
            f"got {encoder_id!r}"
        )
