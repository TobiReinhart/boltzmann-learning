"""Structured, accelerator-ready RBM and three-body RBM modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from ip_rbm.models import RBM, ExactModel, ThreeBodyRBM
from ip_rbm.states import PairMode, visible_pair_indices

ModelKind = Literal["rbm", "3rbm"]


def _sample_bernoulli(probability: Tensor, generator: torch.Generator | None) -> Tensor:
    """Sample Bernoulli variables without changing the floating-point dtype."""
    uniforms = torch.rand(
        probability.shape,
        dtype=probability.dtype,
        device=probability.device,
        generator=generator,
    )
    return (uniforms < probability).to(dtype=probability.dtype)


class TrainableEnergyModel(nn.Module, ABC):
    """Common interface for sample-based energy-model training."""

    n_visible: int
    n_hidden: int
    kind: ModelKind

    @property
    def n_parameters(self) -> int:
        """Return the number of trainable model parameters."""
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def device(self) -> torch.device:
        """Return the device holding the model parameters."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Return the floating-point dtype of the model parameters."""
        return next(self.parameters()).dtype

    @abstractmethod
    def log_unnormalized(self, visible: Tensor) -> Tensor:
        """Return the hidden-marginalized visible log weight."""

    @abstractmethod
    def joint_log_unnormalized(self, visible: Tensor, hidden: Tensor) -> Tensor:
        """Return the joint visible-hidden log weight."""

    @abstractmethod
    def hidden_probability(self, visible: Tensor, *, inverse_temperature: float = 1.0) -> Tensor:
        """Return conditional hidden activation probabilities."""

    @abstractmethod
    def sample_visible(
        self,
        hidden: Tensor,
        visible: Tensor,
        *,
        inverse_temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample one complete visible update conditional on hidden state."""

    @abstractmethod
    def flat_parameters(self) -> Tensor:
        """Return parameters in the exact-model ordering."""

    @abstractmethod
    def load_flat_parameters(self, theta: Tensor) -> None:
        """Load parameters in the exact-model ordering."""

    @abstractmethod
    def exact_model(self) -> ExactModel:
        """Return the matching stateless exact model."""

    def sample_hidden(
        self,
        visible: Tensor,
        *,
        inverse_temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample hidden units conditional on a visible batch."""
        return _sample_bernoulli(
            self.hidden_probability(visible, inverse_temperature=inverse_temperature),
            generator,
        )

    def gibbs_joint_step(
        self,
        visible: Tensor,
        *,
        inverse_temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Perform one joint Gibbs sweep and return visible and hidden states."""
        hidden = self.sample_hidden(
            visible,
            inverse_temperature=inverse_temperature,
            generator=generator,
        )
        updated_visible = self.sample_visible(
            hidden,
            visible,
            inverse_temperature=inverse_temperature,
            generator=generator,
        )
        return updated_visible, hidden

    @torch.no_grad()
    def gibbs_visible(
        self,
        visible: Tensor,
        steps: int,
        *,
        inverse_temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Advance visible chains by a requested number of Gibbs sweeps."""
        if steps < 1:
            raise ValueError("steps must be positive")
        state = visible
        for _ in range(steps):
            state, _ = self.gibbs_joint_step(
                state,
                inverse_temperature=inverse_temperature,
                generator=generator,
            )
        return state

    @torch.no_grad()
    def project_parameters(self, weight_bound: float | None) -> None:
        """Project every model parameter onto a hard box constraint."""
        if weight_bound is None:
            return
        if weight_bound <= 0:
            raise ValueError("weight_bound must be positive")
        for parameter in self.parameters():
            parameter.clamp_(-weight_bound, weight_bound)


class TrainableRBM(TrainableEnergyModel):
    """Bernoulli--Bernoulli RBM with structured PyTorch parameters."""

    kind: ModelKind = "rbm"

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        *,
        init_std: float = 0.01,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if n_visible < 1 or n_hidden < 1:
            raise ValueError("RBM dimensions must be positive")
        if init_std < 0:
            raise ValueError("init_std must be nonnegative")
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.visible_bias = nn.Parameter(torch.zeros(n_visible, device=device, dtype=dtype))
        self.hidden_bias = nn.Parameter(torch.zeros(n_hidden, device=device, dtype=dtype))
        self.weights = nn.Parameter(torch.empty(n_visible, n_hidden, device=device, dtype=dtype))
        with torch.no_grad():
            self.weights.normal_(0.0, init_std, generator=generator)

    def _hidden_field(self, visible: Tensor) -> Tensor:
        return self.hidden_bias + visible @ self.weights

    def log_unnormalized(self, visible: Tensor) -> Tensor:
        return visible @ self.visible_bias + functional.softplus(self._hidden_field(visible)).sum(
            dim=1
        )

    def joint_log_unnormalized(self, visible: Tensor, hidden: Tensor) -> Tensor:
        return visible @ self.visible_bias + (hidden * self._hidden_field(visible)).sum(dim=1)

    def hidden_probability(self, visible: Tensor, *, inverse_temperature: float = 1.0) -> Tensor:
        return torch.sigmoid(inverse_temperature * self._hidden_field(visible))

    def visible_probability(self, hidden: Tensor, *, inverse_temperature: float = 1.0) -> Tensor:
        field = self.visible_bias + hidden @ self.weights.T
        return torch.sigmoid(inverse_temperature * field)

    def sample_visible(
        self,
        hidden: Tensor,
        visible: Tensor,
        *,
        inverse_temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        del visible
        return _sample_bernoulli(
            self.visible_probability(hidden, inverse_temperature=inverse_temperature),
            generator,
        )

    def flat_parameters(self) -> Tensor:
        return torch.cat(
            (self.visible_bias.flatten(), self.hidden_bias.flatten(), self.weights.flatten())
        )

    @torch.no_grad()
    def load_flat_parameters(self, theta: Tensor) -> None:
        if theta.numel() != self.n_parameters:
            raise ValueError(f"expected {self.n_parameters} parameters, received {theta.numel()}")
        exact = RBM(self.n_visible, self.n_hidden)
        visible_bias, hidden_bias, weights = exact.unpack(theta.to(self.device, self.dtype))
        self.visible_bias.copy_(visible_bias)
        self.hidden_bias.copy_(hidden_bias)
        self.weights.copy_(weights)

    def exact_model(self) -> ExactModel:
        return RBM(self.n_visible, self.n_hidden)


class TrainableThreeBodyRBM(TrainableEnergyModel):
    """Three-body RBM supporting cross-register and all-pairs interactions."""

    kind: ModelKind = "3rbm"
    pair_left: Tensor
    pair_right: Tensor
    coupling_lookup: Tensor

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        *,
        pair_mode: PairMode = "cross",
        init_std: float = 0.01,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if n_visible < 2 or n_hidden < 1:
            raise ValueError("3RBM dimensions are invalid")
        if init_std < 0:
            raise ValueError("init_std must be nonnegative")
        pair_left, pair_right = visible_pair_indices(n_visible, pair_mode)
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.pair_mode = pair_mode
        self.n_pairs = pair_left.numel()
        if device is not None:
            pair_left = pair_left.to(device)
            pair_right = pair_right.to(device)
        self.register_buffer("pair_left", pair_left, persistent=False)
        self.register_buffer("pair_right", pair_right, persistent=False)
        # A sentinel addresses a padded zero column, including the diagonal.
        lookup = torch.full((n_visible, n_visible), self.n_pairs, dtype=torch.long, device=device)
        lookup[pair_left, pair_right] = torch.arange(self.n_pairs, device=device)
        lookup[pair_right, pair_left] = torch.arange(self.n_pairs, device=device)
        self.register_buffer("coupling_lookup", lookup, persistent=False)
        self.all_pairs_sampler = "legacy"
        self.all_pairs_block_size = 4
        self.visible_bias = nn.Parameter(torch.zeros(n_visible, device=device, dtype=dtype))
        self.hidden_bias = nn.Parameter(torch.zeros(n_hidden, device=device, dtype=dtype))
        self.weights = nn.Parameter(torch.empty(n_visible, n_hidden, device=device, dtype=dtype))
        self.cubic = nn.Parameter(torch.empty(self.n_pairs, n_hidden, device=device, dtype=dtype))
        with torch.no_grad():
            self.weights.normal_(0.0, init_std, generator=generator)
            self.cubic.normal_(0.0, init_std, generator=generator)

    @property
    def n_ip(self) -> int:
        if self.pair_mode != "cross":
            raise ValueError("n_ip is defined only for cross-register interactions")
        return self.n_visible // 2

    def _pair_features(self, visible: Tensor) -> Tensor:
        return visible[:, self.pair_left] * visible[:, self.pair_right]

    def _hidden_field(self, visible: Tensor) -> Tensor:
        return self.hidden_bias + visible @ self.weights + self._pair_features(visible) @ self.cubic

    def log_unnormalized(self, visible: Tensor) -> Tensor:
        return visible @ self.visible_bias + functional.softplus(self._hidden_field(visible)).sum(
            dim=1
        )

    def joint_log_unnormalized(self, visible: Tensor, hidden: Tensor) -> Tensor:
        return visible @ self.visible_bias + (hidden * self._hidden_field(visible)).sum(dim=1)

    def hidden_probability(self, visible: Tensor, *, inverse_temperature: float = 1.0) -> Tensor:
        return torch.sigmoid(inverse_temperature * self._hidden_field(visible))

    def _sample_cross_visible(
        self,
        hidden: Tensor,
        visible: Tensor,
        inverse_temperature: float,
        generator: torch.Generator | None,
    ) -> Tensor:
        n_ip = self.n_ip
        left = visible[:, :n_ip]
        right = visible[:, n_ip:]
        cubic = self.cubic.reshape(n_ip, n_ip, self.n_hidden)

        left_field = self.visible_bias[:n_ip] + hidden @ self.weights[:n_ip].T
        left_field = left_field + torch.einsum("bj,bh,ijh->bi", right, hidden, cubic)
        left = _sample_bernoulli(torch.sigmoid(inverse_temperature * left_field), generator)

        right_field = self.visible_bias[n_ip:] + hidden @ self.weights[n_ip:].T
        right_field = right_field + torch.einsum("bi,bh,ijh->bj", left, hidden, cubic)
        right = _sample_bernoulli(torch.sigmoid(inverse_temperature * right_field), generator)
        return torch.cat((left, right), dim=1)

    def _sample_all_pairs_visible(
        self,
        hidden: Tensor,
        visible: Tensor,
        inverse_temperature: float,
        generator: torch.Generator | None,
    ) -> Tensor:
        updated = visible.clone()
        couplings = hidden @ self.cubic.T
        linear_field = self.visible_bias + hidden @ self.weights.T
        for bit in range(self.n_visible):
            left_mask = self.pair_left == bit
            right_mask = self.pair_right == bit
            field = linear_field[:, bit].clone()
            if bool(torch.any(left_mask)):
                field += (couplings[:, left_mask] * updated[:, self.pair_right[left_mask]]).sum(
                    dim=1
                )
            if bool(torch.any(right_mask)):
                field += (couplings[:, right_mask] * updated[:, self.pair_left[right_mask]]).sum(
                    dim=1
                )
            probability = torch.sigmoid(inverse_temperature * field)
            updated[:, bit] = _sample_bernoulli(probability, generator)
        return updated

    def _conditional_couplings(self, hidden: Tensor) -> Tensor:
        """Symmetric J(h), rebuilt on every sweep as hidden states change."""
        packed = hidden @ self.cubic.T
        padded = functional.pad(packed, (0, 1))
        return padded[:, self.coupling_lookup]

    def _sample_cached_visible(
        self,
        hidden: Tensor,
        visible: Tensor,
        inverse_temperature: float,
        generator: torch.Generator | None,
    ) -> Tensor:
        updated = visible.clone()
        couplings = self._conditional_couplings(hidden)
        fields = self.visible_bias + hidden @ self.weights.T
        fields = fields + torch.bmm(couplings, updated.unsqueeze(2)).squeeze(2)
        for bit in range(self.n_visible):
            new = _sample_bernoulli(torch.sigmoid(inverse_temperature * fields[:, bit]), generator)
            delta = new - updated[:, bit]
            updated[:, bit] = new
            fields = fields + delta.unsqueeze(1) * couplings[:, :, bit]
        return updated

    @staticmethod
    def _block_logits(
        updated: Tensor,
        linear: Tensor,
        couplings: Tensor,
        indices: Tensor,
        states: Tensor,
    ) -> Tensor:
        """Exact conditional block scores, counting internal edges once."""
        rows = couplings[:, indices, :]
        internal = rows[:, :, indices]
        outside_field = (
            linear[:, indices]
            + torch.bmm(rows, updated.unsqueeze(2)).squeeze(2)
            - torch.bmm(internal, updated[:, indices].unsqueeze(2)).squeeze(2)
        )
        return outside_field @ states.T + 0.5 * torch.einsum(
            "si,bij,sj->bs", states, internal, states
        )

    def _sample_block_visible(
        self,
        hidden: Tensor,
        visible: Tensor,
        inverse_temperature: float,
        generator: torch.Generator | None,
    ) -> Tensor:
        updated = visible.clone()
        couplings = self._conditional_couplings(hidden)
        linear = self.visible_bias + hidden @ self.weights.T
        # State-independent random partitions avoid encoding the target pairing.
        order = torch.randperm(self.n_visible, device=visible.device, generator=generator)
        for start in range(0, self.n_visible, self.all_pairs_block_size):
            indices = order[start : start + self.all_pairs_block_size]
            width = min(self.all_pairs_block_size, self.n_visible - start)
            states = getattr(self, f"block_states_{width}")
            logits = self._block_logits(updated, linear, couplings, indices, states)
            probability = torch.softmax(inverse_temperature * logits, dim=1)
            uniform = torch.rand(
                (visible.shape[0], 1),
                device=visible.device,
                dtype=visible.dtype,
                generator=generator,
            )
            chosen = (uniform > probability.cumsum(dim=1)).sum(dim=1)
            chosen = chosen.clamp_max(states.shape[0] - 1)
            updated[:, indices] = states[chosen]
        return updated

    def configure_all_pairs_sampler(self, sampler: str, block_size: int = 4) -> None:
        """Configure a kernel without changing the energy or trainable parameters."""
        if sampler not in {"legacy", "cached", "block"}:
            raise ValueError("unknown all-pairs sampler")
        if not 1 <= block_size <= 8:
            raise ValueError("all-pairs block size must be between 1 and 8")
        if self.pair_mode != "all" and sampler != "legacy":
            raise ValueError("all-pairs sampler requires pair_mode=all")
        self.all_pairs_sampler = sampler
        self.all_pairs_block_size = block_size
        if sampler == "block":
            widths = {min(block_size, self.n_visible)}
            if self.n_visible % block_size:
                widths.add(self.n_visible % block_size)
            for width in widths:
                integers = torch.arange(2**width, device=self.device)
                shifts = torch.arange(width, device=self.device)
                states = ((integers[:, None] >> shifts) & 1).to(self.dtype)
                self.register_buffer(f"block_states_{width}", states, persistent=False)

    def sample_visible(
        self,
        hidden: Tensor,
        visible: Tensor,
        *,
        inverse_temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if self.pair_mode == "cross":
            return self._sample_cross_visible(hidden, visible, inverse_temperature, generator)
        if self.all_pairs_sampler == "cached":
            return self._sample_cached_visible(hidden, visible, inverse_temperature, generator)
        if self.all_pairs_sampler == "block":
            return self._sample_block_visible(hidden, visible, inverse_temperature, generator)
        return self._sample_all_pairs_visible(hidden, visible, inverse_temperature, generator)

    def flat_parameters(self) -> Tensor:
        return torch.cat(
            (
                self.visible_bias.flatten(),
                self.hidden_bias.flatten(),
                self.weights.flatten(),
                self.cubic.flatten(),
            )
        )

    @torch.no_grad()
    def load_flat_parameters(self, theta: Tensor) -> None:
        if theta.numel() != self.n_parameters:
            raise ValueError(f"expected {self.n_parameters} parameters, received {theta.numel()}")
        exact = ThreeBodyRBM(self.n_visible, self.n_hidden, self.pair_mode)
        visible_bias, hidden_bias, weights, cubic = exact.unpack(theta.to(self.device, self.dtype))
        self.visible_bias.copy_(visible_bias)
        self.hidden_bias.copy_(hidden_bias)
        self.weights.copy_(weights)
        self.cubic.copy_(cubic)

    def exact_model(self) -> ExactModel:
        return ThreeBodyRBM(self.n_visible, self.n_hidden, self.pair_mode)


def make_trainable_model(
    kind: ModelKind,
    n_visible: int,
    n_hidden: int,
    *,
    pair_mode: PairMode = "cross",
    init_std: float = 0.01,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> TrainableEnergyModel:
    """Construct a structured trainable energy model."""
    if kind == "rbm":
        return TrainableRBM(
            n_visible,
            n_hidden,
            init_std=init_std,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    return TrainableThreeBodyRBM(
        n_visible,
        n_hidden,
        pair_mode=pair_mode,
        init_std=init_std,
        device=device,
        dtype=dtype,
        generator=generator,
    )
