"""Enumeration and feature maps for binary visible states."""

from typing import Literal

import torch
from torch import Tensor

PairMode = Literal["cross", "all"]


def all_binary_states(
    n_visible: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Return all binary states in lexicographic integer order.

    Column zero is the most significant bit. The result has shape
    ``(2**n_visible, n_visible)``.
    """
    if n_visible < 1:
        raise ValueError("n_visible must be positive")
    if n_visible > 62:
        raise ValueError("integer-based exact enumeration supports at most 62 visible bits")

    integers = torch.arange(2**n_visible, device=device, dtype=torch.int64)
    shifts = torch.arange(n_visible - 1, -1, -1, device=device, dtype=torch.int64)
    return ((integers[:, None] >> shifts[None, :]) & 1).to(dtype=dtype)


def visible_pair_indices(n_visible: int, mode: PairMode) -> tuple[Tensor, Tensor]:
    """Return index arrays defining allowed visible pairs for a 3RBM."""
    if n_visible < 2:
        raise ValueError("at least two visible units are required")

    if mode == "cross":
        if n_visible % 2:
            raise ValueError("cross-register pairs require an even visible dimension")
        n_ip = n_visible // 2
        left = torch.arange(n_ip, dtype=torch.int64).repeat_interleave(n_ip)
        right = torch.arange(n_ip, n_visible, dtype=torch.int64).repeat(n_ip)
        return left, right

    if mode == "all":
        indices = torch.triu_indices(n_visible, n_visible, offset=1)
        return indices[0], indices[1]

    raise ValueError(f"unknown pair mode: {mode}")


def pair_features(states: Tensor, mode: PairMode) -> Tensor:
    """Evaluate the selected products ``v_i v_j`` for every state."""
    left, right = visible_pair_indices(states.shape[1], mode)
    left = left.to(states.device)
    right = right.to(states.device)
    return states[:, left] * states[:, right]
