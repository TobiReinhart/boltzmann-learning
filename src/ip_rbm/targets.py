"""Exact inner-product-mod-2 target distributions."""

from dataclasses import dataclass

import torch
from torch import Tensor

from ip_rbm.states import all_binary_states


@dataclass(frozen=True)
class ExactTarget:
    """A fully enumerated target distribution."""

    states: Tensor
    ip_sign: Tensor
    log_prob: Tensor
    prob: Tensor
    n_ip: int
    beta: float


def inner_product_sign(states: Tensor) -> Tensor:
    """Return ``(-1)**IP(x,y)`` for states ordered as ``(x,y)``."""
    n_visible = states.shape[1]
    if n_visible % 2:
        raise ValueError("the visible vector must contain equally sized x and y registers")
    n_ip = n_visible // 2
    parity = torch.remainder(
        (states[:, :n_ip] * states[:, n_ip:]).sum(dim=1).to(torch.int64),
        2,
    )
    return 1.0 - 2.0 * parity.to(dtype=states.dtype)


def make_ip_target(
    n_ip: int,
    beta: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> ExactTarget:
    """Enumerate ``p(v) ∝ exp(beta * (-1)**IP(x,y))`` exactly."""
    if n_ip < 1:
        raise ValueError("n_ip must be positive")
    if beta < 0:
        raise ValueError("beta must be nonnegative")

    states = all_binary_states(2 * n_ip, dtype=dtype, device=device)
    sign = inner_product_sign(states)
    logits = beta * sign
    log_prob = logits - torch.logsumexp(logits, dim=0)
    prob = torch.exp(log_prob)
    return ExactTarget(
        states=states,
        ip_sign=sign,
        log_prob=log_prob,
        prob=prob,
        n_ip=n_ip,
        beta=beta,
    )
