"""Composable project configuration.

Stable data/evaluation settings and frequently tuned training settings live in
separate YAML files. Architecture-specific settings live in
``models/<architecture_id>.yaml``.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
DEFAULT_ARCHITECTURE_ID = "se_cnn_v9"
PROJECT_CONFIG_FILES = (
    CONFIG_DIR / "data.yaml",
    CONFIG_DIR / "evaluation.yaml",
    CONFIG_DIR / "default.yaml",
)


def _deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return value


def _load_project_parts(config_path: str | Path | None) -> dict:
    """Merge standard project parts and an optional external override."""
    runtime: dict = {}
    for path in PROJECT_CONFIG_FILES:
        runtime = _deep_merge(runtime, _read_yaml(path))

    if config_path is None:
        return runtime

    override_path = Path(config_path).resolve()
    standard_paths = {path.resolve() for path in PROJECT_CONFIG_FILES}
    if override_path not in standard_paths:
        runtime = _deep_merge(runtime, _read_yaml(override_path))
    return runtime


def load_project_config(
    architecture_id: str | None = None,
    *,
    config_path: str | Path | None = None,
) -> dict:
    """Load all project parts, an optional override and an architecture profile."""
    runtime = _load_project_parts(config_path)
    selected = str(
        architecture_id
        or runtime.get("model", {}).get("architecture_id")
        or runtime.get("project", {}).get("active_architecture")
        or DEFAULT_ARCHITECTURE_ID
    ).strip()
    profile_path = CONFIG_DIR / "models" / f"{selected}.yaml"
    if not profile_path.is_file():
        raise FileNotFoundError(
            f"Unknown architecture profile {selected!r}: {profile_path}"
        )
    config = _deep_merge(_read_yaml(profile_path), runtime)

    from src.config.normalization import normalize_config

    return normalize_config(config)


def default_config_path() -> Path:
    """Return the main user-facing config path used by CLI defaults."""
    return CONFIG_DIR / "default.yaml"
