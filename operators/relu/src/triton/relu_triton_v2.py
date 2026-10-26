"""Tuned single-kernel ReLU for contiguous CUDA float32 vectors."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, tl.where(x < 0.0, 0.0, x), mask=mask)


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
    """Launch one 1024-element Triton program per output tile."""
    _check(x, "x")
    n_elements = x.numel()
    out = torch.empty_like(x)
    if n_elements == 0:
        return out
    block_size = 1024
    _kernel[(triton.cdiv(n_elements, block_size),)](x, out, n_elements, BLOCK=block_size)
    return out
