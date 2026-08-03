"""SE-CNN v9 IL data pipeline.

The current SE-CNN pipeline remains the implementation behind the v1 contract.
A future token transformer can register another pair without branching the IL
entrypoint or pretending its cache is compatible with CNN planes.
"""

from __future__ import annotations


IL_PIPELINES = {
    ("planes_history_v1", "lc0_1858_v1"): "compact_planes_v1",
}


def _pipeline_key(config: dict) -> tuple[str, str]:
    model_cfg = config.get("model", {}) or {}
    return (
        str(model_cfg.get("input_encoder_id", "planes_history_v1")),
        str(model_cfg.get("policy_codec_id", "lc0_1858_v1")),
    )


def _require_current_pipeline(config: dict) -> None:
    key = _pipeline_key(config)
    if key not in IL_PIPELINES:
        raise ValueError(
            "No IL data pipeline registered for "
            f"encoder={key[0]!r}, policy={key[1]!r}. "
            f"Known contracts: {sorted(IL_PIPELINES)}"
        )


def process_pgn_files(pgn_files, config):
    _require_current_pipeline(config)
    from src.models.data.se_cnn_v9.preprocessing import (
        process_pgn_files as implementation,
    )

    return implementation(pgn_files, config)


def create_dataloaders(metadata, config):
    _require_current_pipeline(config)
    from src.models.data.se_cnn_v9.dataset import create_dataloaders as implementation

    return implementation(metadata, config)
