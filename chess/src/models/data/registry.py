"""Versioned model-data contract registry."""

from __future__ import annotations

from src.models.data.se_cnn_v9 import encoder, pipeline, policy


INPUT_ENCODERS = {
    encoder.ENCODER_ID: encoder,
}
POLICY_CODECS = {
    policy.CODEC_ID: policy,
}
IL_DATA_PIPELINES = {
    (encoder.ENCODER_ID, policy.CODEC_ID): pipeline,
}


def get_input_encoder(encoder_id: str):
    try:
        return INPUT_ENCODERS[encoder_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown input encoder {encoder_id!r}; known: {sorted(INPUT_ENCODERS)}"
        ) from exc


def get_policy_codec(codec_id: str):
    try:
        return POLICY_CODECS[codec_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown policy codec {codec_id!r}; known: {sorted(POLICY_CODECS)}"
        ) from exc


def get_il_data_pipeline(encoder_id: str, codec_id: str):
    key = (str(encoder_id), str(codec_id))
    try:
        return IL_DATA_PIPELINES[key]
    except KeyError as exc:
        raise ValueError(
            "No IL data pipeline registered for "
            f"encoder={key[0]!r}, policy={key[1]!r}; "
            f"known: {sorted(IL_DATA_PIPELINES)}"
        ) from exc


def get_il_data_pipeline_for_config(config: dict):
    model_cfg = config.get("model", {}) or {}
    return get_il_data_pipeline(
        model_cfg.get("input_encoder_id", "planes_history_v1"),
        model_cfg.get("policy_codec_id", "lc0_1858_v1"),
    )


def validate_data_contract(model_spec: dict) -> None:
    encoder = get_input_encoder(model_spec["input_spec"]["encoder_id"])
    codec = get_policy_codec(model_spec["policy_spec"]["codec_id"])
    expected_planes = encoder.input_planes(
        model_spec["input_spec"]["history_positions"]
    )
    actual_planes = int(model_spec["input_spec"]["input_planes"])
    if expected_planes != actual_planes:
        raise ValueError(
            f"Encoder expects {expected_planes} input planes, spec has {actual_planes}"
        )
    if int(model_spec["policy_spec"]["action_size"]) != codec.ACTION_SIZE:
        raise ValueError("Policy action size does not match its codec")
