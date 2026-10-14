"""Optional Triton kernel for compact-policy inference output staging.

The fast path gathers the legal policy columns, casts them to FP16, and writes
the FP32 value head into the existing packed output buffer in one GPU launch.
Callers must pass legal indices validated by the normal inference request
contract; this module deliberately does not read them back to the CPU.
"""

from __future__ import annotations

try:
    import triton
    import triton.language as tl
except Exception:  # Triton is optional for CPU-only and unsupported installs.
    triton = None
    tl = None


_BLOCK_N = 128
_WARMUP_LEGAL_WIDTH = 64


def packed_output_width_bytes(policy_width: int) -> int:
    """Byte width of FP16 policy (even padded) plus three FP32 values."""
    width = int(policy_width)
    if width < 0:
        raise ValueError("policy_width must be non-negative")
    padded_width = width + (width & 1)
    return padded_width * 2 + 3 * 4


def packed_output_views(packed, rows: int, policy_width: int):
    """Return typed views into the packed byte buffer used by inference."""
    import torch

    rows = int(rows)
    width = int(policy_width)
    packed_bytes = int(packed.shape[1])
    policy_bytes = width * 2
    if (
        rows < 0
        or width < 0
        or int(packed.shape[0]) < rows
        or packed_bytes < packed_output_width_bytes(width)
        or packed_bytes < 12
        or (packed_bytes - 12) % 4 != 0
    ):
        raise ValueError("packed output buffer is too small or misaligned")
    # Buffers are retained at their largest capacity. Match the finalizer's
    # value offset (capacity width), even when this request has fewer columns.
    value_offset = packed_bytes - 3 * 4
    policy = packed[:rows, :policy_bytes].view(torch.float16)
    values = packed[
        :rows,
        value_offset:value_offset + 3 * 4,
    ].view(torch.float32)
    return policy, values


if triton is not None:
    @triton.jit(
        do_not_specialize=[
            "rows",
            "legal_width",
            "policy_stride0",
            "policy_stride1",
            "legal_stride0",
            "legal_stride1",
            "value_stride0",
            "value_stride1",
            "policy_out_stride0",
            "value_out_stride0",
        ],
        do_not_specialize_on_alignment=[
            "policy_ptr",
            "legal_ptr",
            "value_ptr",
            "policy_out_ptr",
            "value_out_ptr",
        ],
    )
    def _gather_cast_pack_kernel(
        policy_ptr,
        legal_ptr,
        value_ptr,
        policy_out_ptr,
        value_out_ptr,
        rows,
        legal_width,
        policy_stride0,
        policy_stride1,
        legal_stride0,
        legal_stride1,
        value_stride0,
        value_stride1,
        policy_out_stride0,
        value_out_stride0,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        col = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (row < rows) & (col < legal_width)
        legal_index = tl.load(
            legal_ptr + row * legal_stride0 + col * legal_stride1,
            mask=mask,
            other=0,
        ).to(tl.int32)
        logits = tl.load(
            policy_ptr + row * policy_stride0 + legal_index * policy_stride1,
            mask=mask,
            other=0,
        )
        tl.store(
            policy_out_ptr + row * policy_out_stride0 + col,
            logits.to(tl.float16),
            mask=mask,
        )

        if tl.program_id(1) == 0:
            value_col = tl.arange(0, 4)
            value_mask = (row < rows) & (value_col < 3)
            values = tl.load(
                value_ptr + row * value_stride0 + value_col * value_stride1,
                mask=value_mask,
                other=0,
            )
            tl.store(
                value_out_ptr + row * value_out_stride0 + value_col,
                values.to(tl.float32),
                mask=value_mask,
            )


class FusedLegalOutput:
    """Tracks warmed Triton dtype variants and launches the packed output op."""

    def __init__(self):
        self.available = triton is not None
        self._warmed = set()
        self._disabled = set()

    @staticmethod
    def _key(policy, legal_indices, values):
        device = policy.device
        return (
            str(device),
            policy.dtype,
            legal_indices.dtype,
            values.dtype,
        )

    @staticmethod
    def _supported(policy, legal_indices, values, packed, width):
        import torch

        if triton is None or policy.device.type != "cuda":
            return False
        if legal_indices.device != policy.device or values.device != policy.device:
            return False
        if packed.device != policy.device or packed.dtype != torch.uint8:
            return False
        if policy.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False
        if values.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False
        if legal_indices.dtype not in (torch.int16, torch.int64):
            return False
        if policy.ndim != 2 or legal_indices.ndim != 2 or values.ndim != 2:
            return False
        rows = int(policy.shape[0])
        width = int(width)
        if rows <= 0 or width <= 0:
            return False
        if int(legal_indices.shape[0]) < rows or int(legal_indices.shape[1]) < width:
            return False
        if int(values.shape[0]) < rows or int(values.shape[1]) != 3:
            return False
        if int(policy.shape[1]) <= 0 or int(packed.shape[0]) < rows:
            return False
        if int(packed.shape[1]) < packed_output_width_bytes(width):
            return False
        return True

    def is_warmed_for(self, policy, legal_indices, values, packed, width):
        if not self.available or not self._supported(
            policy, legal_indices, values, packed, width
        ):
            return False
        return self.is_warmed_types(policy, legal_indices, values)

    def is_warmed_types(self, policy, legal_indices, values):
        """Fast readiness gate that avoids allocating output slots on fallback."""
        if not self.available:
            return False
        if (
            policy.device.type != "cuda"
            or legal_indices.device != policy.device
            or values.device != policy.device
        ):
            return False
        key = self._key(policy, legal_indices, values)
        return key in self._warmed and key not in self._disabled

    def _launch(self, policy, legal_indices, values, packed, width):
        policy_out, value_out = packed_output_views(
            packed, int(policy.shape[0]), int(width)
        )
        rows = int(policy.shape[0])
        import torch

        with torch.cuda.device(policy.device):
            _gather_cast_pack_kernel[(rows, triton.cdiv(int(width), _BLOCK_N))](
                policy,
                legal_indices,
                values,
                policy_out,
                value_out,
                rows,
                int(width),
                int(policy.stride(0)),
                int(policy.stride(1)),
                int(legal_indices.stride(0)),
                int(legal_indices.stride(1)),
                int(values.stride(0)),
                int(values.stride(1)),
                int(policy_out.stride(0)),
                int(value_out.stride(0)),
                BLOCK_N=_BLOCK_N,
                num_warps=4,
            )

    def warmup(self, policy, legal_indices, values):
        """Compile and validate one dtype variant before enabling its fast path."""
        import torch

        if not self.available or policy.device.type != "cuda":
            return False
        key = self._key(policy, legal_indices, values)
        if key in self._warmed:
            return True
        if key in self._disabled:
            return False
        try:
            warm_rows = min(1, int(policy.shape[0]), int(values.shape[0]))
            if warm_rows <= 0 or int(policy.shape[1]) <= 0:
                self._disabled.add(key)
                return False
            dummy_legal = torch.zeros(
                (warm_rows, _WARMUP_LEGAL_WIDTH),
                device=policy.device,
                dtype=legal_indices.dtype,
            )
            dummy_packed = torch.empty(
                (warm_rows, packed_output_width_bytes(_WARMUP_LEGAL_WIDTH)),
                device=policy.device,
                dtype=torch.uint8,
            )
            self._launch(
                policy[:warm_rows],
                dummy_legal,
                values[:warm_rows],
                dummy_packed,
                _WARMUP_LEGAL_WIDTH,
            )
            torch.cuda.synchronize(policy.device)
            self._warmed.add(key)
            return True
        except Exception:
            self._disabled.add(key)
            return False

    def warmup_types(self, device, policy_dtype, value_dtype, legal_dtype=None):
        """Warm the expected model output dtype before requests are served."""
        import torch

        if not self.available or getattr(device, "type", None) != "cuda":
            return False
        if legal_dtype is None:
            legal_dtype = torch.int16
        try:
            policy = torch.empty((1, 1), device=device, dtype=policy_dtype)
            legal = torch.zeros(
                (1, _WARMUP_LEGAL_WIDTH), device=device, dtype=legal_dtype
            )
            values = torch.empty((1, 3), device=device, dtype=value_dtype)
        except Exception:
            return False
        return self.warmup(policy, legal, values)

    def launch(self, policy, legal_indices, values, packed, width):
        """Launch only an already-warmed, supported dtype/layout variant."""
        if not self.is_warmed_for(policy, legal_indices, values, packed, width):
            return False
        try:
            self._launch(policy, legal_indices, values, packed, int(width))
            return True
        except Exception:
            key = self._key(policy, legal_indices, values)
            self._disabled.add(key)
            return False
