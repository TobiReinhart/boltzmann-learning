"""Exact visible log-densities for RBMs and three-body RBMs."""

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor
from torch.nn import functional as functional

from ip_rbm.states import PairMode, pair_features, visible_pair_indices


class ExactModel(Protocol):
    """Protocol required by the exact population objective."""

    @property
    def n_visible(self) -> int: ...

    @property
    def n_hidden(self) -> int: ...

    @property
    def n_parameters(self) -> int: ...

    def log_unnormalized(self, states: Tensor, theta: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class RBM:
    """Bernoulli--Bernoulli restricted Boltzmann machine."""

    n_visible: int
    n_hidden: int

    def __post_init__(self) -> None:
        if self.n_visible < 1 or self.n_hidden < 0:
            raise ValueError("invalid RBM dimensions")

    @property
    def n_parameters(self) -> int:
        return self.n_visible + self.n_hidden + self.n_visible * self.n_hidden

    def unpack(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if theta.numel() != self.n_parameters:
            raise ValueError(f"expected {self.n_parameters} parameters, received {theta.numel()}")
        cursor = 0
        a = theta[cursor : cursor + self.n_visible]
        cursor += self.n_visible
        b = theta[cursor : cursor + self.n_hidden]
        cursor += self.n_hidden
        weights = theta[cursor:].reshape(self.n_visible, self.n_hidden)
        return a, b, weights

    def log_unnormalized(self, states: Tensor, theta: Tensor) -> Tensor:
        """Return the hidden-marginalized negative free energy ``-F(v)``."""
        a, b, weights = self.unpack(theta)
        hidden_fields = b + states @ weights
        return states @ a + functional.softplus(hidden_fields).sum(dim=1)


@dataclass(frozen=True)
class ThreeBodyRBM:
    """RBM with additional interactions ``U_{ikj} v_i v_k h_j``."""

    n_visible: int
    n_hidden: int
    pair_mode: PairMode = "cross"

    def __post_init__(self) -> None:
        if self.n_visible < 2 or self.n_hidden < 0:
            raise ValueError("invalid 3RBM dimensions")
        visible_pair_indices(self.n_visible, self.pair_mode)

    @property
    def n_pairs(self) -> int:
        left, _ = visible_pair_indices(self.n_visible, self.pair_mode)
        return left.numel()

    @property
    def n_parameters(self) -> int:
        return (
            self.n_visible
            + self.n_hidden
            + self.n_visible * self.n_hidden
            + self.n_pairs * self.n_hidden
        )

    def unpack(self, theta: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if theta.numel() != self.n_parameters:
            raise ValueError(f"expected {self.n_parameters} parameters, received {theta.numel()}")
        cursor = 0
        a = theta[cursor : cursor + self.n_visible]
        cursor += self.n_visible
        b = theta[cursor : cursor + self.n_hidden]
        cursor += self.n_hidden
        weights = theta[cursor : cursor + self.n_visible * self.n_hidden].reshape(
            self.n_visible, self.n_hidden
        )
        cursor += self.n_visible * self.n_hidden
        cubic = theta[cursor:].reshape(self.n_pairs, self.n_hidden)
        return a, b, weights, cubic

    def log_unnormalized(self, states: Tensor, theta: Tensor) -> Tensor:
        """Return the exactly hidden-marginalized negative free energy."""
        a, b, weights, cubic = self.unpack(theta)
        hidden_fields = b + states @ weights + pair_features(states, self.pair_mode) @ cubic
        return states @ a + functional.softplus(hidden_fields).sum(dim=1)


def zero_parameters(model: ExactModel, *, dtype: torch.dtype = torch.float64) -> Tensor:
    """Construct a zero parameter vector of the appropriate length."""
    return torch.zeros(model.n_parameters, dtype=dtype)
