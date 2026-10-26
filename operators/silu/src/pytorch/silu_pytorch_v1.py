"""GPU-only PyTorch silu implementation (float32, contiguous 1D tensors)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


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
    """Run the silu operation on the input CUDA device."""
    _check(x, "x")
    return F.silu(x)
