"""Importable chess training, search and evaluation library."""

from src.common.torch_cache import configure_default_torch_cache_environment


# PyTorch reads these variables lazily when compilation starts. Configure them
# at package import so every entrypoint, spawned worker and future compile call
# stays below chess/.cache even if it has no role-specific setup yet.
configure_default_torch_cache_environment()
