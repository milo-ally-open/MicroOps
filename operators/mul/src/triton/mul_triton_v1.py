"""GPU-only Triton mul kernel and wrapper (float32, contiguous 1D tensors)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    value = a * b
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


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Launch the mul Triton kernel on the input CUDA device."""
    _check(a, "a")
    _check(b, "b")
    if a.shape != b.shape:
        raise ValueError("a and b must have identical shapes; broadcasting is not supported")
    if a.device != b.device:
        raise ValueError("a and b must be on the same CUDA device")
    n_elements = a.numel()
    out = torch.empty_like(a)
    if n_elements == 0:
        return out
    block_size = 256
    _kernel[(triton.cdiv(n_elements, block_size),)](a, b, out, n_elements, BLOCK=block_size)
    return out
