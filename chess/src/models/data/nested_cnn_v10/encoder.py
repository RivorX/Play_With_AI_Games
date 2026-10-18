"""Input contract for the nested CNN v10 chess model."""

from __future__ import annotations

ENCODER_ID = "planes_history_v2"
POLICY_CODEC_ID = "lc0_1858_v1"
PLANES_PER_POSITION = 13
CURRENT_METADATA_PLANES = 7
REPETITION_PLANE_MODE_UNKNOWN = "unknown"
REPETITION_PLANE_MODE_EXACT = "exact"
REPETITION_PLANE_MODES = frozenset(
    {REPETITION_PLANE_MODE_UNKNOWN, REPETITION_PLANE_MODE_EXACT}
)


def normalize_repetition_plane_mode(mode=None) -> str:
    """Normalize the v10 repetition feature contract, defaulting to legacy-safe unknown."""
    normalized = str(mode or REPETITION_PLANE_MODE_UNKNOWN).strip().lower()
    if normalized not in REPETITION_PLANE_MODES:
        raise ValueError(
            "repetition_plane_mode must be 'unknown' or 'exact', "
            f"got {mode!r}"
        )
    return normalized


def resolve_repetition_plane_mode(model_spec=None, model_config=None) -> str:
    """Resolve how a checkpoint was trained to interpret repetition planes.

    A checkpoint model_spec takes precedence over the active runtime config.
    Missing metadata in historical v10 specs means unknown, matching the old
    compact-PGN training pipeline which did not persist repetition provenance.
    """
    input_spec = (
        model_spec.get("input_spec", {})
        if isinstance(model_spec, dict)
        else {}
    ) or {}
    if isinstance(model_spec, dict) and isinstance(input_spec, dict):
        return normalize_repetition_plane_mode(
            input_spec.get("repetition_plane_mode", REPETITION_PLANE_MODE_UNKNOWN)
        )
    model_config = model_config if isinstance(model_config, dict) else {}
    return normalize_repetition_plane_mode(
        model_config.get("repetition_plane_mode", REPETITION_PLANE_MODE_UNKNOWN)
    )


def repetition_count_for_mode(mode, exact_count):
    """Return a count only when the checkpoint was explicitly trained on it."""
    mode = normalize_repetition_plane_mode(mode)
    return exact_count if mode == REPETITION_PLANE_MODE_EXACT else None


def input_planes(history_positions: int) -> int:
    history_positions = int(history_positions)
    if history_positions < 0:
        raise ValueError("history_positions must be nonnegative")
    return PLANES_PER_POSITION * (1 + history_positions) + CURRENT_METADATA_PLANES


def encode(
    board,
    flip_perspective=None,
    dtype=None,
    *,
    out=None,
    repetition_count=None,
):
    """Encode one board as 13 state planes followed by 7 current metadata planes.

    A repetition count is the number of *previous* occurrences of this exact
    position, clipped to [0, 2]. Pass ``None`` when its provenance is unknown;
    the repetition plane is then filled with -1.
    """
    from src.models.data.nested_cnn_v10.helpers import board_to_tensor

    kwargs = {"flip_perspective": flip_perspective, "repetition_count": repetition_count}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if out is not None:
        kwargs["out"] = out
    return board_to_tensor(board, **kwargs)


def encode_pair(board, dtype=None, *, repetition_count=None):
    """Encode one board from both player POVs."""
    from src.models.data.nested_cnn_v10.helpers import board_to_tensor_pair

    kwargs = {"repetition_count": repetition_count}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return board_to_tensor_pair(board, **kwargs)


def encode_history(
    current_board,
    history_boards=(),
    history_positions=0,
    *,
    repetition_counts=None,
    flip_perspective=None,
    dtype=None,
):
    """Encode chronological history and current board into one model input.

    ``repetition_counts`` corresponds to ``history_boards`` followed by the
    current board. Unknown values must be ``None`` and are represented by -1.
    History frames are oldest first and contain only their 12 piece planes and
    repetition plane. The current frame is followed by its seven metadata
    planes.
    """
    from src.models.data.nested_cnn_v10.helpers import encode_history

    kwargs = {
        "repetition_counts": repetition_counts,
        "flip_perspective": flip_perspective,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    return encode_history(
        current_board,
        history_boards=history_boards,
        history_positions=history_positions,
        **kwargs,
    )


def stack_history(current_tensor, history_tensors, history_positions, dtype=None):
    """Stack POV-aligned encoded v10 frames into the 13/20-plane layout."""
    from src.models.data.nested_cnn_v10.helpers import stack_encoded_history

    return stack_encoded_history(
        current_tensor,
        history_tensors,
        history_positions,
        dtype=dtype,
    )

