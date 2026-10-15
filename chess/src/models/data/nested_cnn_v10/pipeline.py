"""IL pipeline adapter for the v10 encoder and shared compact chess data."""

from __future__ import annotations

IL_PIPELINES = {
    ("planes_history_v2", "lc0_1858_v1"): "compact_planes_v2",
}


def _require_current_pipeline(config: dict) -> None:
    model_cfg = config.get("model", {}) or {}
    key = (
        str(model_cfg.get("input_encoder_id", "planes_history_v2")),
        str(model_cfg.get("policy_codec_id", "lc0_1858_v1")),
    )
    if key not in IL_PIPELINES:
        raise ValueError(
            "No v10 IL data pipeline registered for "
            f"encoder={key[0]!r}, policy={key[1]!r}; known: {sorted(IL_PIPELINES)}"
        )


def process_pgn_files(pgn_files, config):
    """Reuse the encoder-independent compact position preprocessing format."""
    _require_current_pipeline(config)
    from src.models.data.se_cnn_v9.preprocessing import process_pgn_files as implementation

    return implementation(pgn_files, config)


def create_dataloaders(metadata, config):
    """Delegate to the shared IL data loader after v10 routing is integrated."""
    _require_current_pipeline(config)
    from src.models.data.se_cnn_v9.dataset import create_dataloaders as implementation

    return implementation(metadata, config)

