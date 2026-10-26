"""GPU-only PyTorch mul implementation (float32, contiguous 1D tensors)."""

from __future__ import annotations

import torch


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
    """Run the mul operation on the input CUDA device."""
    _check(a, "a")
    _check(b, "b")
    if a.shape != b.shape:
        raise ValueError("a and b must have identical shapes; broadcasting is not supported")
    if a.device != b.device:
        raise ValueError("a and b must be on the same CUDA device")
    return torch.mul(a, b)
