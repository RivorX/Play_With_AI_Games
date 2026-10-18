"""IL pipeline adapter for the v10 encoder and shared compact chess data."""

from __future__ import annotations

IL_PIPELINES = {
    ("planes_history_v2", "lc0_1858_v1"): "compact_planes_v2",
}


def _require_current_pipeline(config: dict) -> None:
    model_cfg = config.get("model", {}) or {}
    from src.models.data.nested_cnn_v10.encoder import resolve_repetition_plane_mode

    # Exact mode is supported by the shared PGN preprocessor's aligned sidecar.
    # Existing checkpoint specs still override this profile when the model loads.
    resolve_repetition_plane_mode(model_config=model_cfg)
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
    """Preprocess PGNs through the shared format, including exact v10 counts when requested."""
    _require_current_pipeline(config)
    from src.models.data.se_cnn_v9.preprocessing import process_pgn_files as implementation

    return implementation(pgn_files, config)


def create_dataloaders(metadata, config):
    """Use shared IL selection and load the exact v10 repetition sidecar when configured."""
    _require_current_pipeline(config)
    from src.models.data.se_cnn_v9.dataset import create_dataloaders as implementation

    return implementation(metadata, config)

