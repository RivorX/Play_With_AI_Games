"""Single absolute cache policy for every torch.compile call in the project."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TORCH_COMPILE_CACHE_ROOT = PROJECT_ROOT / ".cache" / "torch_compile"


def _token(value) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown")).strip("._")


def configure_default_torch_cache_environment() -> Path:
    """Force even unclassified compile calls into the project cache tree."""
    cache_dir = (TORCH_COMPILE_CACHE_ROOT / "default").resolve()
    inductor_dir = cache_dir / "inductor"
    triton_dir = cache_dir / "triton"
    inductor_dir.mkdir(parents=True, exist_ok=True)
    triton_dir.mkdir(parents=True, exist_ok=True)
    # Deliberately overwrite stale relative/user values. A relative value is
    # exactly what previously created chess/.torch_compile_cache depending on
    # the process working directory.
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_dir)
    os.environ["TRITON_CACHE_DIR"] = str(triton_dir)
    return cache_dir


def torch_compile_cache_paths(torch_module, device, role: str):
    """Return a versioned absolute cache root and deterministic role folder."""
    if getattr(device, "type", None) != "cuda":
        return None, None
    try:
        device_index = int(
            device.index
            if device.index is not None
            else torch_module.cuda.current_device()
        )
    except Exception:
        device_index = 0
    try:
        major, minor = torch_module.cuda.get_device_capability(device_index)
        compute_capability = f"sm{int(major)}{int(minor)}"
    except Exception:
        compute_capability = "sm_unknown"
    environment = "-".join(
        (
            # v3 invalidates v2 central graphs that were compiled while the
            # legacy relative .torch_compile_cache path was active. Some
            # autotune artifacts retain that absolute path and otherwise
            # recreate the old directory even after the environment is fixed.
            "v3",
            f"py{sys.version_info.major}{sys.version_info.minor}",
            f"torch{_token(torch_module.__version__)}",
            f"cuda{_token(torch_module.version.cuda)}",
            compute_capability,
            f"gpu{device_index}",
        )
    )
    root = (TORCH_COMPILE_CACHE_ROOT / environment).resolve()
    role_name = _token(role) or "default"
    return root, root / role_name


def configure_torch_compile_cache(torch_module, device, role: str) -> Path | None:
    """Select and create the canonical cache namespace for one compile role."""
    root, role_dir = torch_compile_cache_paths(torch_module, device, role)
    if root is None or role_dir is None:
        return None
    inductor_dir = role_dir / "inductor"
    triton_dir = role_dir / "triton"
    inductor_dir.mkdir(parents=True, exist_ok=True)
    triton_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_dir.resolve())
    os.environ["TRITON_CACHE_DIR"] = str(triton_dir.resolve())
    return root
