"""Nested residual CNN for the planes_history_v2 chess input contract."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.inference.prediction import predict_single_position
from src.models.data.se_cnn_v9.policy import policy_index_maps
from src.models.data.nested_cnn_v10.encoder import (
    input_planes as encoder_input_planes,
    normalize_repetition_plane_mode,
)


class CoordConv2d(nn.Module):
    """3x3 stem convolution with fixed normalized board coordinates."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, 3, padding=1, bias=False)
        axis = torch.linspace(-1.0, 1.0, 8)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("coord_x", xx[None, None].contiguous(), persistent=False)
        self.register_buffer("coord_y", yy[None, None].contiguous(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        coords = torch.cat((self.coord_x, self.coord_y), dim=1).to(dtype=x.dtype)
        coords = coords.expand(batch, -1, -1, -1)
        return self.conv(torch.cat((x, coords), dim=1))


class GlobalPoolBias(nn.Module):
    """Project global channel means and maxima into an additive feature bias."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Linear(2 * in_channels, out_channels)
        # Start as a neutral residual/context path; the projection learns its bias.
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        x: torch.Tensor,
        pooled: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pooled is None:
            pooled = torch.cat((x.mean(dim=(-2, -1)), x.amax(dim=(-2, -1))), dim=1)
        return self.projection(pooled)[:, :, None, None]


class GatedGlobalPoolAffine(nn.Module):
    """Identity-initialized global mean/max affine modulation of a trunk map.

    The projection emits per-channel gate logits followed by an additive
    shift. ``2 * sigmoid(0)`` is exactly one, so zero initialization preserves
    the original residual path while allowing context to learn both
    multiplicative and additive corrections.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.projection = nn.Linear(2 * self.channels, 2 * self.channels)
        self.reset_identity()

    def reset_identity(self) -> None:
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def add_shift_offset(self, offset: torch.Tensor) -> None:
        """Fold an FVI threshold into the shift half, never into the gate."""
        if offset.numel() != self.channels:
            raise ValueError(
                f"Expected {self.channels} FVI offsets, got {offset.numel()}"
            )
        with torch.no_grad():
            self.projection.bias[self.channels :].add_(
                offset.to(dtype=self.projection.bias.dtype)
            )

    def forward(
        self,
        x: torch.Tensor,
        pooled: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pooled is None:
            pooled = torch.cat(
                (x.mean(dim=(-2, -1)), x.amax(dim=(-2, -1))), dim=1
            )
        gate_logits, shift = self.projection(pooled).split(self.channels, dim=1)
        scale = 2.0 * torch.sigmoid(gate_logits)
        return x * scale[:, :, None, None] + shift[:, :, None, None]


class FixedVarianceScale(nn.Module):
    """Fixed FVI scale followed by a learned per-channel offset."""

    def __init__(self, scale: float, channels: int) -> None:
        super().__init__()
        self.scale_factor = float(scale)
        if int(channels) < 1:
            raise ValueError("FVI scale requires a positive channel count")
        self.beta = nn.Parameter(torch.zeros(int(channels)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale_factor == 1.0:
            scaled = x
        else:
            scaled = x * self.scale_factor
        beta = self.beta.to(dtype=x.dtype).view(1, -1, 1, 1)
        return scaled + beta


class _FVIInferenceScale(nn.Module):
    """Parameter-free scalar FVI operation retained only in an eval copy."""

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale_factor = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.scale_factor == 1.0 else x * self.scale_factor


class _FVIInferenceShift(nn.Module):
    """Frozen per-channel threshold retained in a disposable eval copy."""

    def __init__(self, offset: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("offset", offset.detach().clone())
        # Central eager inference commonly autocasts activations to BF16. Keep
        # that exact cast with the inference copy so each forward does not
        # launch a tiny dtype-conversion kernel for every retained shift.
        self.register_buffer(
            "_offset_bfloat16",
            offset.detach().to(dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "_offset_float16",
            offset.detach().to(dtype=torch.float16),
            persistent=False,
        )

    def _refresh_dtype_caches(self) -> None:
        # Rebuild from the current source buffer after dtype/device transforms
        # or state loading; cached casts must never outlive their source value.
        self._buffers["_offset_bfloat16"] = self.offset.detach().to(
            dtype=torch.bfloat16
        )
        self._buffers["_offset_float16"] = self.offset.detach().to(
            dtype=torch.float16
        )

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        self._refresh_dtype_caches()
        return result

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._refresh_dtype_caches()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.bfloat16 and self._offset_bfloat16.dtype == x.dtype:
            offset = self._offset_bfloat16
        elif x.dtype == torch.float16 and self._offset_float16.dtype == x.dtype:
            offset = self._offset_float16
        else:
            offset = self.offset.to(dtype=x.dtype)
        return x + offset.view(1, -1, 1, 1)


class NestedResidualUnit(nn.Module):
    """Two pre-activation 3x3 convolutions with an inner identity skip."""

    def __init__(
        self,
        channels: int,
        inner_index: int,
    ) -> None:
        super().__init__()
        # The inner residual stream starts at variance 1. Each previous inner
        # unit adds one approximately independent unit-variance residual.
        self.norm1 = FixedVarianceScale(
            1.0 / math.sqrt(inner_index + 1.0),
            channels
        )
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = FixedVarianceScale(1.0, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(F.relu(self.norm1(x), inplace=False))
        residual = self.conv2(F.relu(self.norm2(residual), inplace=False))
        return x + residual


class NestedResidualBlock(nn.Module):
    """Wide outer residual around a narrower nested residual tower."""

    def __init__(
        self,
        trunk_channels: int,
        bottleneck_channels: int,
        inner_units: int,
        use_global_pool: bool,
        drop_path_rate: float,
        outer_index: int,
    ) -> None:
        super().__init__()
        # At outer block i, the trunk contains the unit-variance stem plus i
        # residual updates. Reset that accumulated scale before reducing.
        self.reduce_norm = FixedVarianceScale(
            1.0 / math.sqrt(outer_index + 1.0),
            trunk_channels
        )
        self.reduce = nn.Conv2d(trunk_channels, bottleneck_channels, 1, bias=False)
        self.inner_units = nn.Sequential(
            *(
                NestedResidualUnit(
                    bottleneck_channels,
                    i,
                )
                for i in range(inner_units)
            )
        )
        self.global_pool = (
            GatedGlobalPoolAffine(bottleneck_channels)
            if use_global_pool
            else None
        )
        # Each nested residual unit adds one unit-variance residual.
        self.expand_norm = FixedVarianceScale(
            1.0 / math.sqrt(inner_units + 1.0),
            bottleneck_channels
        )
        self.expand = nn.Conv2d(bottleneck_channels, trunk_channels, 1, bias=False)
        self.output_norm = FixedVarianceScale(1.0, trunk_channels)
        self.drop_path_rate = float(drop_path_rate)

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        rate = self.drop_path_rate
        if rate <= 0.0 or not self.training:
            return x
        keep = 1.0 - rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x * mask / keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.reduce(F.relu(self.reduce_norm(x), inplace=False))
        residual = self.inner_units(residual)
        if self.global_pool is not None:
            residual = self.global_pool(residual)
        residual = self.expand(F.relu(self.expand_norm(residual), inplace=False))
        residual = self.output_norm(residual)
        return x + self._drop_path(residual)


class NestedCNNV10(nn.Module):
    """Nested v10 chess CNN with FVI, One-BN training and WDL heads."""

    ACTION_PLANES = 73
    ACTION_SIZE = 1858
    REQUIRED_VERSION = "v10"
    supports_training_aux = True

    def __init__(self, config: dict, input_planes: int | None = None) -> None:
        super().__init__()
        model_cfg = dict(config.get("model", {}) or {})
        self.model_version = str(model_cfg.get("version", "v10"))
        self.architecture_id = "nested_cnn_v10"
        self.model_config = model_cfg
        if self.model_version != self.REQUIRED_VERSION:
            raise ValueError(
                "nested_cnn_v10 requires model.version='v10'. "
                f"Received version={self.model_version!r}; start a fresh v10 run "
                "instead of resuming an older checkpoint."
            )
        self.normalization_mode = str(model_cfg.get("normalization_mode", ""))
        if self.normalization_mode != "fixed_variance_one_bn":
            raise ValueError(
                "nested_cnn_v10 requires normalization_mode="
                "'fixed_variance_one_bn'"
            )
        if not bool(model_cfg.get("auxiliary_soft_policy", False)):
            raise ValueError("nested_cnn_v10 requires auxiliary_soft_policy=true")
        self._is_fvi_inference_copy = False
        self.bn_head_loss_weight = float(model_cfg.get("bn_head_loss_weight", 0.8))
        self.value_hidden_init = str(
            model_cfg.get("value_hidden_init", "fan_in_he")
        ).strip().lower()
        if self.value_hidden_init != "fan_in_he":
            raise ValueError(
                "nested_cnn_v10 requires value_hidden_init='fan_in_he'"
            )
        self.soft_policy_temperature = float(
            model_cfg.get("soft_policy_temperature", 4.0)
        )
        self.soft_policy_loss_weight = float(
            model_cfg.get("soft_policy_loss_weight", 0.25)
        )
        if not 0.0 <= self.bn_head_loss_weight <= 1.0:
            raise ValueError("bn_head_loss_weight must be in [0, 1]")
        if self.soft_policy_temperature <= 1.0:
            raise ValueError("soft_policy_temperature must be greater than 1")
        if self.soft_policy_loss_weight < 0.0:
            raise ValueError("soft_policy_loss_weight must be non-negative")
        self.repetition_plane_mode = normalize_repetition_plane_mode(
            model_cfg.get("repetition_plane_mode", "unknown")
        )

        history_positions = int(model_cfg.get("history_positions", 4))
        if history_positions < 0:
            raise ValueError("history_positions must be >= 0")
        expected_input_planes = encoder_input_planes(history_positions)
        configured_input_planes = model_cfg.get("input_planes")
        if configured_input_planes is not None and int(configured_input_planes) != expected_input_planes:
            raise ValueError(
                "nested_cnn_v10 input_planes mismatch: encoder contract expects "
                f"{expected_input_planes}, config has {configured_input_planes}"
            )
        if input_planes is not None and int(input_planes) != expected_input_planes:
            raise ValueError(
                "nested_cnn_v10 input_planes mismatch: encoder contract expects "
                f"{expected_input_planes}, caller supplied {input_planes}"
            )
        self.input_planes = expected_input_planes
        self.history_positions = history_positions

        trunk_channels = int(model_cfg.get("trunk_channels", 192))
        bottleneck_channels = int(model_cfg.get("bottleneck_channels", 96))
        outer_blocks = int(model_cfg.get("outer_blocks", 6))
        inner_units = int(model_cfg.get("inner_residual_units", 2))
        if min(trunk_channels, bottleneck_channels, outer_blocks, inner_units) < 1:
            raise ValueError("v10 channel widths and residual block counts must be positive")
        if bottleneck_channels > trunk_channels:
            raise ValueError("bottleneck_channels must be <= trunk_channels")

        drop_path_rate = float(model_cfg.get("drop_path_rate", 0.05))
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError("drop_path_rate must be in [0, 1)")
        self.policy_head_channels = int(model_cfg.get("policy_head_channels", 96))
        self.value_head_filters = int(model_cfg.get("value_head_filters", 32))
        value_hidden_dim = int(model_cfg.get("value_hidden_dim", 192))
        moves_left_hidden_dim = int(model_cfg.get("moves_left_hidden_dim", 128))
        if min(self.policy_head_channels, self.value_head_filters, value_hidden_dim, moves_left_hidden_dim) < 1:
            raise ValueError("v10 head widths must be positive")
        dropout = float(model_cfg.get("dropout", 0.08))

        self.trunk_channels = trunk_channels
        self.bottleneck_channels = bottleneck_channels
        self.outer_blocks = outer_blocks

        self.stem_conv = CoordConv2d(self.input_planes, trunk_channels)
        self.stem_norm = FixedVarianceScale(1.0, trunk_channels)
        context_blocks = {int(i) for i in model_cfg.get("global_pool_blocks", [2, 4, 6])}
        if any(i < 1 or i > outer_blocks for i in context_blocks):
            raise ValueError(
                f"global_pool_blocks must use 1-based ordinals in 1..{outer_blocks}"
            )
        rates = torch.linspace(0.0, drop_path_rate, outer_blocks).tolist()
        self.residual_tower = nn.ModuleList(
            NestedResidualBlock(
                trunk_channels=trunk_channels,
                bottleneck_channels=bottleneck_channels,
                inner_units=inner_units,
                use_global_pool=(i + 1) in context_blocks,
                drop_path_rate=rates[i],
                outer_index=i,
            )
            for i in range(outer_blocks)
        )
        # The single BN is training-only. The deployment heads use raw trunk
        # features through the fixed scale derived from the outer residuals.
        self.training_trunk_bn = nn.BatchNorm2d(trunk_channels)
        self.deploy_trunk_scale = FixedVarianceScale(
            1.0 / math.sqrt(outer_blocks + 1.0),
            trunk_channels
        )

        # Deployment policy head. The fixed scale following each convolution
        # replaces its old BN while preserving the activation/conv topology.
        self.policy_conv = nn.Conv2d(
            trunk_channels, self.policy_head_channels, 3, padding=1, bias=False
        )
        self.policy_norm = FixedVarianceScale(1.0, self.policy_head_channels)
        self.policy_global_bias = GlobalPoolBias(
            trunk_channels, self.policy_head_channels
        )
        self.policy_logits_conv = nn.Conv2d(
            self.policy_head_channels, self.ACTION_PLANES, 3, padding=1, bias=True
        )
        policy_to_az, _ = policy_index_maps()
        self.register_buffer(
            "policy_index_to_az",
            torch.as_tensor(policy_to_az, dtype=torch.long),
            persistent=False,
        )
        self.soft_policy_logits_conv = nn.Conv2d(
            self.policy_head_channels, self.ACTION_PLANES, 3, padding=1, bias=True
        )

        # Preserve explicit square-wise value features plus global mean/max context.
        self.value_conv = nn.Conv2d(trunk_channels, self.value_head_filters, 1, bias=False)
        self.value_norm = FixedVarianceScale(1.0, self.value_head_filters)
        self.value_global_dim = 2 * trunk_channels
        value_input_dim = self.value_head_filters * 8 * 8 + self.value_global_dim
        self.value_norm_flat = nn.LayerNorm(value_input_dim)
        self.value_fc1 = nn.Linear(value_input_dim, value_hidden_dim)
        self.value_dropout = nn.Dropout(dropout)
        self.value_fc2 = nn.Linear(value_hidden_dim, 3)
        self.search_q_fc = nn.Linear(value_hidden_dim, 1)

        self.moves_left_fc1 = nn.Linear(self.value_global_dim, moves_left_hidden_dim)
        self.moves_left_norm = nn.LayerNorm(moves_left_hidden_dim)
        self.moves_left_fc2 = nn.Linear(moves_left_hidden_dim, 1)

        # Independently learned heads for the normalized training branch.
        # They are never evaluated by the ordinary inference forward.
        self.bn_policy_conv = nn.Conv2d(
            trunk_channels, self.policy_head_channels, 3, padding=1, bias=False
        )
        self.bn_policy_norm = FixedVarianceScale(1.0, self.policy_head_channels)
        self.bn_policy_global_bias = GlobalPoolBias(
            trunk_channels, self.policy_head_channels
        )
        self.bn_policy_logits_conv = nn.Conv2d(
            self.policy_head_channels, self.ACTION_PLANES, 3, padding=1, bias=True
        )
        self.bn_soft_policy_logits_conv = nn.Conv2d(
            self.policy_head_channels, self.ACTION_PLANES, 3, padding=1, bias=True
        )

        self.bn_value_conv = nn.Conv2d(
            trunk_channels, self.value_head_filters, 1, bias=False
        )
        self.bn_value_norm = FixedVarianceScale(1.0, self.value_head_filters)
        self.bn_value_norm_flat = nn.LayerNorm(value_input_dim)
        self.bn_value_fc1 = nn.Linear(value_input_dim, value_hidden_dim)
        self.bn_value_dropout = nn.Dropout(dropout)
        self.bn_value_fc2 = nn.Linear(value_hidden_dim, 3)
        self.bn_search_q_fc = nn.Linear(value_hidden_dim, 1)
        self.bn_moves_left_fc1 = nn.Linear(
            self.value_global_dim, moves_left_hidden_dim
        )
        self.bn_moves_left_norm = nn.LayerNorm(moves_left_hidden_dim)
        self.bn_moves_left_fc2 = nn.Linear(moves_left_hidden_dim, 1)
        self._initialize_weights()

        if bool(model_cfg.get("print_summary", True)):
            print(
                f"[MODEL {self.model_version}] nested_cnn_v10: "
                f"input={self.input_planes}, trunk={outer_blocks}x"
                f"({trunk_channels}/{bottleneck_channels}), "
                f"policy={self.policy_head_channels}, value={self.value_head_filters}"
            )
            self._print_parameter_summary()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Fan-in He initialization keeps the second moment of each
                # ReLU/conv pair near one, which is the premise used by FVI.
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                # The WDL hidden layers receive LayerNorm output. Using fan_out
                # made their pre-activation variance scale with 2432 / 192,
                # despite unit-scale inputs. Keep fan-in He for these two
                # SiLU layers; output projections below retain their explicit
                # small/zero initialization, and the MLH path keeps its
                # fan-out init before LayerNorm (which removes that amplitude).
                fan_mode = (
                    "fan_in"
                    if module is self.value_fc1 or module is self.bn_value_fc1
                    else "fan_out"
                )
                nn.init.kaiming_normal_(
                    module.weight, mode=fan_mode, nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for module in self.modules():
            if isinstance(module, GlobalPoolBias):
                nn.init.zeros_(module.projection.weight)
                nn.init.zeros_(module.projection.bias)
            elif isinstance(module, GatedGlobalPoolAffine):
                module.reset_identity()

        nn.init.normal_(self.policy_logits_conv.weight, std=0.01)
        nn.init.zeros_(self.policy_logits_conv.bias)
        nn.init.normal_(self.soft_policy_logits_conv.weight, std=0.01)
        nn.init.zeros_(self.soft_policy_logits_conv.bias)
        nn.init.normal_(self.bn_policy_logits_conv.weight, std=0.01)
        nn.init.zeros_(self.bn_policy_logits_conv.bias)
        nn.init.normal_(self.bn_soft_policy_logits_conv.weight, std=0.01)
        nn.init.zeros_(self.bn_soft_policy_logits_conv.bias)
        nn.init.normal_(self.value_fc2.weight, std=0.01)
        nn.init.zeros_(self.value_fc2.bias)
        nn.init.normal_(self.bn_value_fc2.weight, std=0.01)
        nn.init.zeros_(self.bn_value_fc2.bias)
        nn.init.zeros_(self.search_q_fc.weight)
        nn.init.zeros_(self.search_q_fc.bias)
        nn.init.zeros_(self.bn_search_q_fc.weight)
        nn.init.zeros_(self.bn_search_q_fc.bias)
        nn.init.normal_(self.moves_left_fc2.weight, std=0.01)
        nn.init.zeros_(self.moves_left_fc2.bias)
        nn.init.normal_(self.bn_moves_left_fc2.weight, std=0.01)
        nn.init.zeros_(self.bn_moves_left_fc2.bias)

    def _print_parameter_summary(self) -> None:
        core_inference_modules = (
            self.stem_conv,
            self.stem_norm,
            self.residual_tower,
            self.deploy_trunk_scale,
            self.policy_conv,
            self.policy_norm,
            self.policy_global_bias,
            self.policy_logits_conv,
            self.value_conv,
            self.value_norm,
            self.value_norm_flat,
            self.value_fc1,
            self.value_fc2,
        )
        optional_inference_modules = (
            self.search_q_fc,
            self.moves_left_fc1,
            self.moves_left_norm,
            self.moves_left_fc2,
        )
        training_only_modules = (
            self.training_trunk_bn,
            self.bn_policy_norm,
            self.soft_policy_logits_conv,
            self.bn_policy_conv,
            self.bn_policy_global_bias,
            self.bn_policy_logits_conv,
            self.bn_soft_policy_logits_conv,
            self.bn_value_conv,
            self.bn_value_norm,
            self.bn_value_norm_flat,
            self.bn_value_fc1,
            self.bn_value_fc2,
            self.bn_search_q_fc,
            self.bn_moves_left_fc1,
            self.bn_moves_left_norm,
            self.bn_moves_left_fc2,
        )
        core_count = sum(
            parameter.numel()
            for module in core_inference_modules
            for parameter in module.parameters()
        )
        optional_count = sum(
            parameter.numel()
            for module in optional_inference_modules
            for parameter in module.parameters()
        )
        training_only_count = sum(
            parameter.numel()
            for module in training_only_modules
            for parameter in module.parameters()
        )
        inference_count = core_count + optional_count
        total_count = sum(parameter.numel() for parameter in self.parameters())
        print(f"  > Core policy/value inference parameters: {core_count:,}")
        print(f"  > Optional search/moves-left parameters: {optional_count:,}")
        print(f"  > Training-only BN/soft-policy parameters: {training_only_count:,}")
        print(f"  > Active inference parameters: {inference_count:,}")
        print(f"  > Total training parameters: {total_count:,}")

    @staticmethod
    def _global_mean_max(x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x.mean(dim=(-2, -1)), x.amax(dim=(-2, -1))), dim=1)

    def _forward_trunk(self, x: torch.Tensor) -> torch.Tensor:
        trunk = F.relu(self.stem_norm(self.stem_conv(x)), inplace=False)
        for block in self.residual_tower:
            trunk = block(trunk)
        return trunk

    def _policy_branch(
        self,
        trunk: torch.Tensor,
        *,
        pooled: torch.Tensor | None = None,
        bn_branch: bool = False,
        include_soft_policy: bool = False,
        apply_log_softmax: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        trunk_global = pooled if pooled is not None else self._global_mean_max(trunk)
        if bn_branch:
            policy_conv = self.bn_policy_conv
            policy_norm = self.bn_policy_norm
            policy_global_bias = self.bn_policy_global_bias
            policy_logits_conv = self.bn_policy_logits_conv
            soft_logits_conv = self.bn_soft_policy_logits_conv
        else:
            policy_conv = self.policy_conv
            policy_norm = self.policy_norm
            policy_global_bias = self.policy_global_bias
            policy_logits_conv = self.policy_logits_conv
            soft_logits_conv = self.soft_policy_logits_conv

        features = policy_norm(policy_conv(trunk))
        features = features + policy_global_bias(trunk, pooled=trunk_global)
        features = F.relu(features, inplace=False)

        def legal_logits(conv: nn.Module) -> torch.Tensor:
            planes = conv(features)
            az_logits = (
                planes.permute(0, 2, 3, 1)
                .contiguous()
                .view(planes.shape[0], -1)
            )
            return az_logits.index_select(1, self.policy_index_to_az)

        policy_logits = legal_logits(policy_logits_conv)
        policy = F.log_softmax(policy_logits, dim=1) if apply_log_softmax else policy_logits
        soft_logits = legal_logits(soft_logits_conv) if include_soft_policy else None
        return policy, soft_logits

    def _value_branch(
        self,
        trunk: torch.Tensor,
        *,
        pooled: torch.Tensor | None = None,
        bn_branch: bool = False,
        return_moves_left: bool = False,
        return_search_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if bn_branch:
            value_conv = self.bn_value_conv
            value_norm = self.bn_value_norm
            value_norm_flat = self.bn_value_norm_flat
            value_fc1 = self.bn_value_fc1
            value_dropout = self.bn_value_dropout
            value_fc2 = self.bn_value_fc2
            moves_left_fc1 = self.bn_moves_left_fc1
            moves_left_norm = self.bn_moves_left_norm
            moves_left_fc2 = self.bn_moves_left_fc2
            search_q_fc = self.bn_search_q_fc
        else:
            value_conv = self.value_conv
            value_norm = self.value_norm
            value_norm_flat = self.value_norm_flat
            value_fc1 = self.value_fc1
            value_dropout = self.value_dropout
            value_fc2 = self.value_fc2
            moves_left_fc1 = self.moves_left_fc1
            moves_left_norm = self.moves_left_norm
            moves_left_fc2 = self.moves_left_fc2
            search_q_fc = self.search_q_fc

        trunk_global = pooled if pooled is not None else self._global_mean_max(trunk)
        value_spatial = F.silu(value_norm(value_conv(trunk)), inplace=False)
        value_features = torch.cat((value_spatial.flatten(1), trunk_global), dim=1)
        value_features = value_norm_flat(value_features)
        value_hidden = F.silu(value_fc1(value_features), inplace=False)
        value = value_fc2(value_dropout(value_hidden))

        moves_left = None
        if return_moves_left:
            moves_hidden = F.relu(moves_left_fc1(trunk_global), inplace=False)
            moves_hidden = moves_left_norm(moves_hidden)
            moves_left = moves_left_fc2(moves_hidden)
        search_q = torch.tanh(search_q_fc(value_hidden)) if return_search_aux else None
        return value, moves_left, search_q

    def _forward_training_aux(
        self,
        trunk_raw: torch.Tensor,
        *,
        return_moves_left: bool,
        return_search_aux: bool,
    ) -> dict[str, tuple | torch.Tensor]:
        if not self.training:
            raise RuntimeError(
                "return_training_aux=True is a training-only path; call model.train()"
            )
        deployed_trunk = F.relu(self.deploy_trunk_scale(trunk_raw), inplace=False)
        bn_trunk = F.relu(self.training_trunk_bn(trunk_raw), inplace=False)
        deployed_global = self._global_mean_max(deployed_trunk)
        bn_global = self._global_mean_max(bn_trunk)

        main_policy, main_soft = self._policy_branch(
            deployed_trunk, pooled=deployed_global, include_soft_policy=True
        )
        main_value, main_moves, main_search = self._value_branch(
            deployed_trunk,
            pooled=deployed_global,
            return_moves_left=return_moves_left,
            return_search_aux=return_search_aux,
        )
        bn_policy, bn_soft = self._policy_branch(
            bn_trunk, pooled=bn_global, bn_branch=True, include_soft_policy=True
        )
        bn_value, bn_moves, bn_search = self._value_branch(
            bn_trunk,
            pooled=bn_global,
            bn_branch=True,
            return_moves_left=return_moves_left,
            return_search_aux=return_search_aux,
        )
        return {
            "main": (main_policy, main_value, main_moves, main_search),
            "bn": (bn_policy, bn_value, bn_moves, bn_search),
            "soft_policy": main_soft,
            "bn_soft_policy": bn_soft,
        }

    @staticmethod
    def _add_beta_to_conv_bias(conv: nn.Conv2d, beta: torch.Tensor) -> None:
        if conv.bias is None:
            conv.bias = nn.Parameter(
                torch.zeros(
                    conv.out_channels,
                    dtype=conv.weight.dtype,
                    device=conv.weight.device,
                )
            )
        with torch.no_grad():
            conv.bias.add_(beta.detach().reshape(-1).to(dtype=conv.bias.dtype))

    @staticmethod
    def _fold_affine_post_conv(
        scale_module: FixedVarianceScale,
        conv: nn.Conv2d,
    ) -> None:
        """Fold ``scale * conv(x) + beta`` into an eval-only convolution."""
        scale = float(scale_module.scale_factor)
        if scale <= 0.0:
            raise ValueError("Only positive FVI scales can be folded across ReLU")
        if conv.bias is None:
            conv.bias = nn.Parameter(
                torch.zeros(
                    conv.out_channels,
                    dtype=conv.weight.dtype,
                    device=conv.weight.device,
                )
            )
        with torch.no_grad():
            conv.weight.mul_(scale)
            conv.bias.mul_(scale)
            conv.bias.add_(scale_module.beta.detach().to(dtype=conv.bias.dtype))

    @staticmethod
    def _fold_pre_activation_scale(
        scale_module: FixedVarianceScale,
        following_conv: nn.Conv2d,
    ) -> None:
        """Move a positive scalar through ReLU into the following convolution."""
        scale = float(scale_module.scale_factor)
        if scale <= 0.0:
            raise ValueError("Only positive FVI scales can be folded across ReLU")
        if scale != 1.0:
            with torch.no_grad():
                following_conv.weight.mul_(scale)

    @staticmethod
    def _replace_with_inference_shift(
        parent: nn.Module,
        name: str,
        scale_module: FixedVarianceScale,
    ) -> None:
        """Keep a residual/ReLU threshold as a frozen, activation-dtype shift."""
        beta = scale_module.beta.detach()
        if not bool(torch.count_nonzero(beta).item()):
            setattr(parent, name, nn.Identity())
            return
        offset = beta / float(scale_module.scale_factor)
        setattr(parent, name, _FVIInferenceShift(offset))

    def _make_fvi_inference_copy(self) -> "NestedCNNV10":
        """Fold learned FVI offsets into a disposable eval copy where exact."""
        model = copy.deepcopy(self)
        model.eval()
        # Post-convolution affine FVI: scale the weights and set the output
        # bias to s*b + beta. These locations have no residual identity input.
        self._fold_affine_post_conv(model.stem_norm, model.stem_conv.conv)
        model.stem_norm = nn.Identity()

        for block in model.residual_tower:
            # Before ReLU, reduce_norm and inner norm1 shift only a residual
            # branch while its input also follows an identity skip. Keep beta
            # there; the positive scalar commutes through ReLU into the next
            # convolution's weights.
            self._fold_pre_activation_scale(block.reduce_norm, block.reduce)
            self._replace_with_inference_shift(
                block, "reduce_norm", block.reduce_norm
            )
            for unit in block.inner_units:
                self._fold_pre_activation_scale(unit.norm1, unit.conv1)
                self._replace_with_inference_shift(unit, "norm1", unit.norm1)

                # norm2 follows conv1 and precedes its residual-branch ReLU.
                # Its offset can be absorbed into conv1's bias while its scale
                # moves through ReLU into conv2.
                self._add_beta_to_conv_bias(
                    unit.conv1,
                    unit.norm2.beta.detach() / unit.norm2.scale_factor,
                )
                self._fold_pre_activation_scale(unit.norm2, unit.conv2)
                unit.norm2 = nn.Identity()

            # Shift this single-use branch output via the pool projection or
            # final inner-conv bias, then fold its positive scale into expand.
            if block.global_pool is None:
                self._add_beta_to_conv_bias(
                    block.inner_units[-1].conv2,
                    block.expand_norm.beta.detach() / block.expand_norm.scale_factor,
                )
            else:
                block.global_pool.add_shift_offset(
                    block.expand_norm.beta.detach()
                    / block.expand_norm.scale_factor
                )
            self._fold_pre_activation_scale(block.expand_norm, block.expand)
            block.expand_norm = nn.Identity()

            # output_norm is directly after expand, before the outer skip.
            self._fold_affine_post_conv(block.output_norm, block.expand)
            block.output_norm = nn.Identity()

        # The final trunk offset follows a sum with an identity path. Add it
        # to the last residual projection before retaining the shared scale.
        trunk_scale = model.deploy_trunk_scale
        self._add_beta_to_conv_bias(
            model.residual_tower[-1].expand,
            trunk_scale.beta.detach() / trunk_scale.scale_factor,
        )
        model.deploy_trunk_scale = _FVIInferenceScale(trunk_scale.scale_factor)

        for prefix in ("policy", "value", "bn_policy", "bn_value"):
            norm = getattr(model, f"{prefix}_norm")
            conv = getattr(model, f"{prefix}_conv")
            self._fold_affine_post_conv(norm, conv)
            setattr(model, f"{prefix}_norm", nn.Identity())

        model._is_fvi_inference_copy = True
        scale_count_before = sum(
            isinstance(module, FixedVarianceScale) for module in self.modules()
        )
        runtime_scales = sum(
            isinstance(module, _FVIInferenceScale) for module in model.modules()
        )
        model._fvi_folded_scale_count = scale_count_before - runtime_scales
        model._fvi_runtime_scale_count = runtime_scales
        return model

    def make_inference_copy(self) -> "NestedCNNV10":
        """Return an eval-only copy with learned FVI terms folded where exact."""
        if self.training:
            raise RuntimeError("make_inference_copy requires model.eval()")
        if self._is_fvi_inference_copy:
            return self
        return self._make_fvi_inference_copy()

    def train(self, mode: bool = True):
        if mode and getattr(self, "_is_fvi_inference_copy", False):
            raise RuntimeError(
                "This FVI-folded v10 copy is inference-only; train the original model."
            )
        return super().train(mode)

    def forward(
        self,
        x: torch.Tensor,
        apply_log_softmax: bool = True,
        policy_only: bool = False,
        return_moves_left: bool = False,
        return_search_aux: bool = False,
        return_training_aux: bool = False,
    ):
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B,C,8,8), got {tuple(x.shape)}")
        if x.shape[1] != self.input_planes or x.shape[-2:] != (8, 8):
            raise ValueError(
                f"nested_cnn_v10 expects (B,{self.input_planes},8,8), got {tuple(x.shape)}"
            )
        if not x.is_floating_point():
            raise TypeError(f"Model input must be floating point, got {x.dtype}")
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        if return_training_aux:
            if policy_only:
                raise ValueError("policy_only cannot be combined with return_training_aux")
            if apply_log_softmax:
                raise ValueError(
                    "return_training_aux returns raw policy logits; set apply_log_softmax=False"
                )
            return self._forward_training_aux(
                self._forward_trunk(x),
                return_moves_left=return_moves_left,
                return_search_aux=return_search_aux,
            )

        trunk_raw = self._forward_trunk(x)
        trunk = F.relu(self.deploy_trunk_scale(trunk_raw), inplace=False)
        trunk_global = self._global_mean_max(trunk)
        policy, _ = self._policy_branch(
            trunk, pooled=trunk_global, apply_log_softmax=apply_log_softmax
        )

        if policy_only:
            if return_search_aux and return_moves_left:
                return policy, None, None, None
            if return_search_aux:
                return policy, None, None
            return (policy, None, None) if return_moves_left else (policy, None)

        value, moves_left, search_q = self._value_branch(
            trunk,
            pooled=trunk_global,
            return_moves_left=return_moves_left,
            return_search_aux=return_search_aux,
        )
        if not return_moves_left and not return_search_aux:
            return policy, value

        if return_search_aux and return_moves_left:
            return policy, value, moves_left, search_q
        if return_search_aux:
            return policy, value, search_q
        return policy, value, moves_left

    def predict(self, board_tensor):
        """Return single-position policy probabilities and WDL value scalar."""
        return predict_single_position(self, board_tensor)
