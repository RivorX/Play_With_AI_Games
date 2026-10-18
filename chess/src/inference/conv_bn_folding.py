"""Evaluation-only Conv2d/BatchNorm2d folding for the v9 chess model.

The returned model is a disposable deep copy. Never save its state dict as a
training checkpoint: folding changes the parameterization and removes the BN
modules. Only the directly adjacent inference pairs listed below are folded.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval


def _fold_pair(conv: nn.Module, bn: nn.Module) -> None:
    """Fold BN into a Conv2d, including the Conv2d inside CoordConv wrappers."""
    if isinstance(conv, nn.Conv2d):
        folded = fuse_conv_bn_eval(conv, bn)
        conv.weight = folded.weight
        conv.bias = folded.bias
        return

    inner = getattr(conv, "conv", None)
    if isinstance(inner, nn.Conv2d):
        folded = fuse_conv_bn_eval(inner, bn)
        inner.weight = folded.weight
        inner.bias = folded.bias
        return

    raise TypeError(f"Expected Conv2d or CoordConv2d, got {type(conv).__name__}")


def fold_eval_conv_bn_copy(model: nn.Module) -> tuple[nn.Module, int]:
    """Return a folded eval copy and the number of folded adjacent pairs.

    The v10 model has its own fixed-variance inference-copy preparation and is
    intentionally not handled here. BatchNorm must use frozen running
    statistics. Unknown architectures/topology drift fail closed, leaving the
    caller's original model untouched.
    """
    if model.training:
        raise ValueError("Conv-BN folding requires an eval-mode model")

    model_spec: dict[str, Any] = getattr(model, "model_spec", {}) or {}
    architecture_id = str(model_spec.get("architecture_id", ""))
    if not architecture_id:
        class_name = type(model).__name__
        architecture_id = {
            "SECNNV9": "se_cnn_v9",
        }.get(class_name, "")

    if architecture_id != "se_cnn_v9":
        raise ValueError(f"Unsupported Conv-BN folding architecture: {architecture_id!r}")

    folded = copy.deepcopy(model)

    pairs: list[tuple[nn.Module, str, str]] = []

    def add(parent: nn.Module, conv_name: str, bn_name: str) -> None:
        conv = getattr(parent, conv_name, None)
        bn = getattr(parent, bn_name, None)
        if not isinstance(bn, nn.BatchNorm2d):
            raise TypeError(f"{type(parent).__name__}.{bn_name} is not BatchNorm2d")
        if not isinstance(conv, nn.Conv2d) and not isinstance(
            getattr(conv, "conv", None), nn.Conv2d
        ):
            raise TypeError(f"{type(parent).__name__}.{conv_name} is not a supported convolution")
        pairs.append((parent, conv_name, bn_name))

    if not isinstance(folded.conv_block, nn.Sequential) or len(folded.conv_block) < 2:
        raise TypeError("v9 stem no longer has the validated conv/BN sequence")
    stem_bn = folded.conv_block[1]
    if not isinstance(stem_bn, nn.BatchNorm2d):
        raise TypeError("v9 stem batch norm topology changed")
    pairs.append((folded.conv_block, "0", "1"))
    for block in folded.residual_tower:
        add(block, "conv1", "bn2")
    add(folded, "policy_conv", "policy_bn")
    add(folded, "value_conv", "value_bn")

    # Validate every BN before changing the disposable copy. This prevents
    # silently folding a train-mode/statistics-free normalization.
    for parent, conv_name, bn_name in pairs:
        bn = parent[int(bn_name)] if isinstance(parent, nn.Sequential) else getattr(parent, bn_name)
        if bn.training or not bn.track_running_stats:
            raise ValueError(f"{architecture_id}.{bn_name} has no fixed eval statistics")
        if bn.running_mean is None or bn.running_var is None:
            raise ValueError(f"{architecture_id}.{bn_name} lacks running statistics")

    for parent, conv_name, bn_name in pairs:
        conv = parent[int(conv_name)] if isinstance(parent, nn.Sequential) else getattr(parent, conv_name)
        bn = parent[int(bn_name)] if isinstance(parent, nn.Sequential) else getattr(parent, bn_name)
        _fold_pair(conv, bn)
        if isinstance(parent, nn.Sequential):
            parent[int(bn_name)] = nn.Identity()
        else:
            setattr(parent, bn_name, nn.Identity())

    folded.eval()
    try:
        folded.to(memory_format=torch.channels_last)
    except (RuntimeError, TypeError):
        # Some custom modules may reject a blanket memory-format conversion;
        # folding itself remains valid and the caller's model is still intact.
        pass
    return folded, len(pairs)
