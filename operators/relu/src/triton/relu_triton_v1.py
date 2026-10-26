"""GPU-only Triton relu kernel and wrapper (float32, contiguous 1D tensors)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    value = tl.where(x > 0, x, tl.where(x != x, x, 0.0))
    tl.store(out_ptr + offsets, value, mask=mask)


def _check(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cuda":
        raise ValueError(f"{name} must be on CUDA; CPU fallback is disabled")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32")
    if value.ndim != 1 or not value.is_contiguous():
        raise ValueError(f"{name} must be a contiguous 1D tensor")


def run(x: torch.Tensor) -> torch.Tensor:
    """Launch the relu Triton kernel on the input CUDA device."""
    _check(x, "x")
    n_elements = x.numel()
    out = torch.empty_like(x)
    if n_elements == 0:
        return out
    block_size = 256
    _kernel[(triton.cdiv(n_elements, block_size),)](x, out, n_elements, BLOCK=block_size)
    return out
