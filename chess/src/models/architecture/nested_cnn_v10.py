"""Nested residual CNN for the planes_history_v2 chess input contract."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.data.se_cnn_v9.policy import policy_index_maps
from src.models.data.nested_cnn_v10.encoder import input_planes as encoder_input_planes


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


class NestedResidualUnit(nn.Module):
    """Two pre-activation 3x3 convolutions with an inner identity skip."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)
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
    ) -> None:
        super().__init__()
        self.reduce_norm = nn.BatchNorm2d(trunk_channels)
        self.reduce = nn.Conv2d(trunk_channels, bottleneck_channels, 1, bias=False)
        self.inner_units = nn.Sequential(
            *(NestedResidualUnit(bottleneck_channels) for _ in range(inner_units))
        )
        self.global_pool = (
            GlobalPoolBias(bottleneck_channels, bottleneck_channels)
            if use_global_pool
            else None
        )
        self.expand_norm = nn.BatchNorm2d(bottleneck_channels)
        self.expand = nn.Conv2d(bottleneck_channels, trunk_channels, 1, bias=False)
        self.output_norm = nn.BatchNorm2d(trunk_channels)
        # A small residual scale keeps the deeper new tower close to identity at init.
        nn.init.constant_(self.output_norm.weight, 0.1)
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
            residual = residual + self.global_pool(residual)
        residual = self.expand(F.relu(self.expand_norm(residual), inplace=False))
        residual = self.output_norm(residual)
        return x + self._drop_path(residual)


class NestedCNNV10(nn.Module):
    """8x8 chess CNN with nested bottlenecks, KataGo-style global pool bias and WDL heads."""

    ACTION_PLANES = 73
    ACTION_SIZE = 1858

    def __init__(self, config: dict, input_planes: int | None = None) -> None:
        super().__init__()
        model_cfg = dict(config.get("model", {}) or {})
        self.model_version = str(model_cfg.get("version", "v10"))
        self.architecture_id = "nested_cnn_v10"
        self.model_config = model_cfg

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
        self.stem_norm = nn.BatchNorm2d(trunk_channels)
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
            )
            for i in range(outer_blocks)
        )
        self.final_norm = nn.BatchNorm2d(trunk_channels)

        # Spatial policy features receive their own whole-board mean/max context.
        self.policy_conv = nn.Conv2d(
            trunk_channels, self.policy_head_channels, 3, padding=1, bias=False
        )
        self.policy_norm = nn.BatchNorm2d(self.policy_head_channels)
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

        # Preserve explicit square-wise value features plus global mean/max context.
        self.value_conv = nn.Conv2d(trunk_channels, self.value_head_filters, 1, bias=False)
        self.value_norm = nn.BatchNorm2d(self.value_head_filters)
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
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for module in self.modules():
            if isinstance(module, GlobalPoolBias):
                nn.init.zeros_(module.projection.weight)
                nn.init.zeros_(module.projection.bias)
            elif isinstance(module, NestedResidualBlock):
                nn.init.constant_(module.output_norm.weight, 0.1)

        nn.init.normal_(self.policy_logits_conv.weight, std=0.01)
        nn.init.zeros_(self.policy_logits_conv.bias)
        nn.init.normal_(self.value_fc2.weight, std=0.01)
        nn.init.zeros_(self.value_fc2.bias)
        nn.init.zeros_(self.search_q_fc.weight)
        nn.init.zeros_(self.search_q_fc.bias)
        nn.init.normal_(self.moves_left_fc2.weight, std=0.01)
        nn.init.zeros_(self.moves_left_fc2.bias)

    def _print_parameter_summary(self) -> None:
        groups = {
            "stem": (self.stem_conv, self.stem_norm),
            "trunk": (self.residual_tower, self.final_norm),
            "policy": (
                self.policy_conv,
                self.policy_norm,
                self.policy_global_bias,
                self.policy_logits_conv,
            ),
            "value": (
                self.value_conv,
                self.value_norm,
                self.value_norm_flat,
                self.value_fc1,
                self.value_fc2,
                self.value_dropout,
            ),
            "search_q": (self.search_q_fc,),
            "moves_left": (self.moves_left_fc1, self.moves_left_norm, self.moves_left_fc2),
        }
        counts = {
            name: sum(parameter.numel() for module in modules for parameter in module.parameters())
            for name, modules in groups.items()
        }
        print("  > Parameters: " + ", ".join(f"{name}={count:,}" for name, count in counts.items()))
        print(f"  > Total parameters: {sum(counts.values()):,}")

    @staticmethod
    def _global_mean_max(x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x.mean(dim=(-2, -1)), x.amax(dim=(-2, -1))), dim=1)

    def forward(
        self,
        x: torch.Tensor,
        apply_log_softmax: bool = True,
        policy_only: bool = False,
        return_moves_left: bool = False,
        return_search_aux: bool = False,
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

        trunk = F.relu(self.stem_norm(self.stem_conv(x)), inplace=False)
        for block in self.residual_tower:
            trunk = block(trunk)
        trunk = F.relu(self.final_norm(trunk), inplace=False)
        trunk_global = self._global_mean_max(trunk)

        policy_features = self.policy_norm(self.policy_conv(trunk))
        policy_features = policy_features + self.policy_global_bias(
            trunk, pooled=trunk_global
        )
        policy_features = F.relu(policy_features, inplace=False)
        policy_planes = self.policy_logits_conv(policy_features)
        az_logits = (
            policy_planes.permute(0, 2, 3, 1)
            .contiguous()
            .view(policy_planes.shape[0], -1)
        )
        policy_logits = az_logits.index_select(1, self.policy_index_to_az)
        policy = (
            F.log_softmax(policy_logits, dim=1) if apply_log_softmax else policy_logits
        )

        if policy_only:
            if return_search_aux and return_moves_left:
                return policy, None, None, None
            if return_search_aux:
                return policy, None, None
            return (policy, None, None) if return_moves_left else (policy, None)

        value_spatial = F.silu(self.value_norm(self.value_conv(trunk)), inplace=False)
        value_features = torch.cat((value_spatial.flatten(1), trunk_global), dim=1)
        value_features = self.value_norm_flat(value_features)
        value_hidden = F.silu(self.value_fc1(value_features), inplace=False)
        value = self.value_fc2(self.value_dropout(value_hidden))

        if not return_moves_left and not return_search_aux:
            return policy, value

        moves_left = None
        if return_moves_left:
            moves_hidden = F.relu(self.moves_left_fc1(trunk_global), inplace=False)
            moves_hidden = self.moves_left_norm(moves_hidden)
            moves_left = self.moves_left_fc2(moves_hidden)
        search_q = torch.tanh(self.search_q_fc(value_hidden)) if return_search_aux else None

        if return_search_aux and return_moves_left:
            return policy, value, moves_left, search_q
        if return_search_aux:
            return policy, value, search_q
        return policy, value, moves_left

    def predict(self, board_tensor):
        was_training = self.training
        if was_training:
            self.eval()
        try:
            with torch.inference_mode():
                if board_tensor.ndim == 3:
                    board_tensor = board_tensor.unsqueeze(0)
                policy_logits, value_logits = self.forward(
                    board_tensor, apply_log_softmax=False
                )
                wdl_probs = F.softmax(value_logits, dim=1)
                value_scalar = wdl_probs[0, 0] - wdl_probs[0, 2]
                policy_probs = F.softmax(policy_logits, dim=1)
                return policy_probs.float().cpu().numpy()[0], value_scalar.item()
        finally:
            if was_training:
                self.train()
