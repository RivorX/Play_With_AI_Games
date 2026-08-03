"""UI helpers for local play/debug.

Symbols are loaded lazily to avoid importing pygame-heavy modules unless needed.
"""

from importlib import import_module

_EXPORT_MAP = {
    "build_pgn_game": ".gui_helpers",
    "create_piece_surfaces": ".gui_helpers",
    "format_recent_moves": ".gui_helpers",
    "get_game_mode_labels": ".gui_helpers",
    "get_result_message": ".gui_helpers",
    "get_turn_color": ".gui_helpers",
    "load_model_from_checkpoint": ".game_setup",
    "resolve_games_dir": ".gui_helpers",
    "save_game_to_pgn": ".gui_helpers",
    "select_models": ".game_setup",
}

__all__ = sorted(_EXPORT_MAP)


def __getattr__(name):
    module_path = _EXPORT_MAP.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_path, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
