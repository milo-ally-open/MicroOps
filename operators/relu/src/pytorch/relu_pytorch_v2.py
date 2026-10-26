"""Single-operation PyTorch ReLU for CUDA float32 contiguous 1D tensors."""

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


def run(x: torch.Tensor) -> torch.Tensor:
    """Use PyTorch's fused ReLU kernel while retaining NaN propagation."""
    _check(x, "x")
    return torch.relu(x)
