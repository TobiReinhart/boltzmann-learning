"""Sample-based datasets and analytically normalized IP targets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ip_rbm.states import all_binary_states
from ip_rbm.targets import inner_product_sign


class ScalableTarget(Protocol):
    """Interface required by scalable data generation and evaluation."""

    @property
    def n_ip(self) -> int: ...

    @property
    def beta(self) -> float: ...

    @property
    def n_visible(self) -> int: ...

    @property
    def log_partition(self) -> float: ...

    @property
    def entropy(self) -> float: ...

    @property
    def mean_sign(self) -> float: ...

    def sign(self, states: Tensor) -> Tensor: ...

    def log_unnormalized(self, states: Tensor) -> Tensor: ...

    def log_prob(self, states: Tensor) -> Tensor: ...

    def sample(
        self,
        sample_size: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> Tensor: ...


class BinaryTensorDataset(Dataset[Tensor]):
    """A binary visible-state matrix suitable for minibatch training."""

    def __init__(self, states: Tensor) -> None:
        if states.ndim != 2 or states.shape[0] < 1 or states.shape[1] < 1:
            raise ValueError("states must have shape (observations, visible units)")
        if not states.is_floating_point():
            raise ValueError("states must use a floating-point dtype")
        if bool(torch.any((states != 0) & (states != 1))):
            raise ValueError("states must be binary")
        self.states = states

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, index: int) -> Tensor:
        return self.states[index]


class FreshTargetBatches:
    """Independent CPU target draws per update, with an isolated random stream."""

    def __init__(self, target: ScalableTarget, batch_size: int, seed: int) -> None:
        self.target = target
        self.batch_size = batch_size
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.draws = 0

    def __call__(self) -> Tensor:
        batch = self.target.sample(
            self.batch_size, device="cpu", dtype=torch.float32, generator=self.generator
        )
        self.draws += self.batch_size
        return batch


@dataclass(frozen=True)
class IPTarget:
    """Non-enumerating inner-product-mod-2 target distribution."""

    n_ip: int
    beta: float

    def __post_init__(self) -> None:
        if self.n_ip < 1:
            raise ValueError("n_ip must be positive")
        if math.isnan(self.beta) or self.beta < 0:
            raise ValueError("beta must be nonnegative")

    @property
    def n_visible(self) -> int:
        return 2 * self.n_ip

    @property
    def log_partition(self) -> float:
        """Return the analytically known target log partition function."""
        leading = (2 * self.n_ip - 1) * math.log(2.0)
        imbalance = 2.0 ** (-self.n_ip)
        log_even = leading + math.log1p(imbalance)
        log_odd = leading + math.log1p(-imbalance)
        if math.isinf(self.beta):
            # Hard-support convention: log weight zero on even, -inf on odd.
            return log_even
        return float(
            torch.logaddexp(
                torch.tensor(log_even + self.beta, dtype=torch.float64),
                torch.tensor(log_odd - self.beta, dtype=torch.float64),
            )
        )

    @property
    def mean_sign(self) -> float:
        """Return the exact target expectation of ``(-1)**IP``."""
        if math.isinf(self.beta):
            return 1.0
        leading = (2 * self.n_ip - 1) * math.log(2.0)
        imbalance = 2.0 ** (-self.n_ip)
        log_even_weight = leading + math.log1p(imbalance) + self.beta
        log_odd_weight = leading + math.log1p(-imbalance) - self.beta
        normalizer = self.log_partition
        return math.exp(log_even_weight - normalizer) - math.exp(log_odd_weight - normalizer)

    @property
    def entropy(self) -> float:
        """Return the exact Shannon entropy of the target."""
        if math.isinf(self.beta):
            return self.log_partition
        return self.log_partition - self.beta * self.mean_sign

    def sign(self, states: Tensor) -> Tensor:
        """Return the IP sign for an arbitrary state batch."""
        if states.ndim != 2 or states.shape[1] != self.n_visible:
            raise ValueError(f"states must have shape (batch, {self.n_visible})")
        return inner_product_sign(states)

    def log_prob(self, states: Tensor) -> Tensor:
        """Return exact normalized log probabilities without enumeration."""
        return self.log_unnormalized(states) - self.log_partition

    def log_unnormalized(self, states: Tensor) -> Tensor:
        """Return the IP target score before normalization."""
        if math.isinf(self.beta):
            sign = self.sign(states)
            return torch.where(sign > 0, torch.zeros_like(sign), -torch.inf)
        return self.beta * self.sign(states)

    def sample(
        self,
        sample_size: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Draw exact i.i.d. samples by rejection from the uniform distribution."""
        if sample_size < 1:
            raise ValueError("sample_size must be positive")
        if not dtype.is_floating_point:
            raise ValueError("dtype must be floating point")
        accepted: list[Tensor] = []
        remaining = sample_size
        odd_acceptance = math.exp(-2.0 * self.beta)
        while remaining:
            proposal_size = max(32, 2 * remaining)
            proposal = torch.randint(
                0,
                2,
                (proposal_size, self.n_visible),
                device=device,
                dtype=torch.int64,
                generator=generator,
            ).to(dtype=dtype)
            sign = self.sign(proposal)
            acceptance_probability = torch.where(
                sign > 0,
                torch.ones_like(sign),
                torch.full_like(sign, odd_acceptance),
            )
            uniforms = torch.rand(
                proposal_size,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            selected = proposal[uniforms < acceptance_probability]
            if selected.shape[0] == 0:
                continue
            selected = selected[:remaining]
            accepted.append(selected)
            remaining -= selected.shape[0]
        return torch.cat(accepted, dim=0)

    def sample_sector(
        self,
        sample_size: int,
        *,
        even: bool,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Draw uniformly within one IP sector by rejection from uniform states."""
        if sample_size < 1:
            raise ValueError("sample_size must be positive")
        accepted: list[Tensor] = []
        remaining = sample_size
        while remaining:
            proposal = uniform_binary(
                max(32, 2 * remaining),
                self.n_visible,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            sign = self.sign(proposal)
            selected = proposal[sign > 0] if even else proposal[sign < 0]
            if selected.shape[0] == 0:
                continue
            selected = selected[:remaining]
            accepted.append(selected)
            remaining -= selected.shape[0]
        return torch.cat(accepted, dim=0)


@dataclass(frozen=True)
class CountCosineTarget:
    r"""Oscillatory target whose score is ``beta*cos(pi*K/w)``.

    Here ``K=sum_i x_i*y_i``.  Normalization and exact i.i.d. sampling use the
    ``n_ip + 1`` possible count values, including the visible-state
    multiplicity ``binom(n, K) * 3**(n-K)``.  Consequently this target remains
    scalable without enumerating either visible or product configurations.
    """

    n_ip: int
    beta: float
    w: int

    def __post_init__(self) -> None:
        if self.n_ip < 1:
            raise ValueError("n_ip must be positive")
        if self.beta < 0:
            raise ValueError("beta must be nonnegative")
        if self.w < 1:
            raise ValueError("count-cosine w must be a positive integer")

    @property
    def n_visible(self) -> int:
        return 2 * self.n_ip

    @cached_property
    def count_values(self) -> Tensor:
        return torch.arange(self.n_ip + 1, dtype=torch.float64)

    @cached_property
    def count_scores(self) -> Tensor:
        return self.beta * torch.cos(math.pi * self.count_values / self.w)

    @cached_property
    def count_log_multiplicity(self) -> Tensor:
        counts = self.count_values
        n = torch.tensor(float(self.n_ip), dtype=torch.float64)
        log_binomial = (
            torch.lgamma(n + 1.0) - torch.lgamma(counts + 1.0) - torch.lgamma(n - counts + 1.0)
        )
        return log_binomial + (self.n_ip - counts) * math.log(3.0)

    @cached_property
    def count_log_prob(self) -> Tensor:
        logits = self.count_scores + self.count_log_multiplicity
        return logits - torch.logsumexp(logits, dim=0)

    @property
    def log_partition(self) -> float:
        return float(torch.logsumexp(self.count_scores + self.count_log_multiplicity, dim=0))

    @property
    def entropy(self) -> float:
        probabilities = torch.exp(self.count_log_prob)
        mean_score = torch.sum(probabilities * self.count_scores)
        return self.log_partition - float(mean_score)

    @property
    def mean_sign(self) -> float:
        counts = self.count_values.to(torch.int64)
        signs = 1.0 - 2.0 * torch.remainder(counts, 2).to(torch.float64)
        return float(torch.sum(torch.exp(self.count_log_prob) * signs))

    def _counts(self, states: Tensor) -> Tensor:
        if states.ndim != 2 or states.shape[1] != self.n_visible:
            raise ValueError(f"states must have shape (batch, {self.n_visible})")
        products = states[:, : self.n_ip] * states[:, self.n_ip :]
        return products.sum(dim=1)

    def sign(self, states: Tensor) -> Tensor:
        """Return the ordinary IP sign as a secondary diagnostic."""
        return inner_product_sign(states)

    def log_unnormalized(self, states: Tensor) -> Tensor:
        """Return ``beta*cos(pi*K/w)`` for each visible state."""
        return self.beta * torch.cos(math.pi * self._counts(states) / self.w)

    def log_prob(self, states: Tensor) -> Tensor:
        return self.log_unnormalized(states) - self.log_partition

    def sample(
        self,
        sample_size: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Draw exact visible samples by first sampling ``K`` and then its preimage."""
        if sample_size < 1:
            raise ValueError("sample_size must be positive")
        if not dtype.is_floating_point:
            raise ValueError("dtype must be floating point")
        if generator is not None and generator.device.type != "cpu":
            raise ValueError("count-cosine exact sampling requires a CPU generator")

        counts = torch.multinomial(
            torch.exp(self.count_log_prob),
            sample_size,
            replacement=True,
            generator=generator,
        ).to(torch.int64)
        products = torch.zeros((sample_size, self.n_ip), dtype=torch.bool)
        remaining = counts.clone()
        for index in range(self.n_ip):
            probability = remaining.to(torch.float64) / (self.n_ip - index)
            selected = torch.rand(sample_size, generator=generator) < probability
            products[:, index] = selected
            remaining -= selected.to(torch.int64)
        if bool(torch.any(remaining != 0)):
            raise AssertionError("failed to sample the requested product count")

        preimage = torch.randint(
            0,
            3,
            products.shape,
            dtype=torch.int64,
            generator=generator,
        )
        zeros = ~products
        left = torch.where(zeros, preimage == 2, torch.ones_like(zeros))
        right = torch.where(zeros, preimage == 1, torch.ones_like(zeros))
        return torch.cat((left, right), dim=1).to(device=device, dtype=dtype)


@dataclass(frozen=True)
class IPCosineTarget(CountCosineTarget):
    r"""Sign-preserving IP target with an added low-frequency cosine score.

    With ``K=sum_i x_i*y_i``, the visible score is

    ``beta * ((-1)**K + rho*cos(pi*K/w))``.

    Requiring ``0 < rho < 1`` preserves the sign of ordinary IP everywhere,
    with score margin at least ``beta*(1-rho)`` when ``beta > 0``. Exact
    normalization and sampling reuse the count-based machinery of
    ``CountCosineTarget``.
    """

    rho: float

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.w < 2:
            raise ValueError("IP-cosine w must be at least two")
        if not 0.0 < self.rho < 1.0:
            raise ValueError("IP-cosine rho must lie strictly between zero and one")

    @cached_property
    def count_scores(self) -> Tensor:
        counts = self.count_values
        parity = torch.remainder(counts.to(torch.int64), 2)
        signs = 1.0 - 2.0 * parity.to(torch.float64)
        cosine = torch.cos(math.pi * counts / self.w)
        return self.beta * (signs + self.rho * cosine)

    def cosine_component(self, states: Tensor) -> Tensor:
        """Return the unscaled low-frequency cosine component."""
        return torch.cos(math.pi * self._counts(states) / self.w)

    def log_unnormalized(self, states: Tensor) -> Tensor:
        """Return the sign-preserving IP-plus-cosine target score."""
        return self.beta * (self.sign(states) + self.rho * self.cosine_component(states))


@dataclass(frozen=True)
class BlockIPTarget:
    r"""Sum of independent IP signs on contiguous coordinate blocks.

    For a block size ``k`` dividing ``n_ip``, the visible score is

    ``beta * sum_b (-1)**sum_{i in block b}(x_i*y_i)``.

    The target distribution factorizes across blocks. Its normalization,
    entropy, and exact i.i.d. sampler therefore reuse the analytic ``IPTarget``
    formulas on blocks of size ``k`` without enumerating visible states.
    """

    n_ip: int
    beta: float
    block_size: int

    def __post_init__(self) -> None:
        if self.n_ip < 1:
            raise ValueError("n_ip must be positive")
        if self.beta < 0:
            raise ValueError("beta must be nonnegative")
        if self.block_size < 1:
            raise ValueError("block-IP block_size must be positive")
        if self.n_ip % self.block_size != 0:
            raise ValueError("block-IP block_size must divide n_ip")

    @property
    def n_visible(self) -> int:
        return 2 * self.n_ip

    @property
    def n_blocks(self) -> int:
        return self.n_ip // self.block_size

    @cached_property
    def block_target(self) -> IPTarget:
        return IPTarget(self.block_size, self.beta)

    @property
    def log_partition(self) -> float:
        return self.n_blocks * self.block_target.log_partition

    @property
    def entropy(self) -> float:
        return self.n_blocks * self.block_target.entropy

    @property
    def mean_sign(self) -> float:
        """Return the expectation of the global IP sign across all blocks."""
        return self.block_target.mean_sign**self.n_blocks

    def _block_signs(self, states: Tensor) -> Tensor:
        if states.ndim != 2 or states.shape[1] != self.n_visible:
            raise ValueError(f"states must have shape (batch, {self.n_visible})")
        products = states[:, : self.n_ip] * states[:, self.n_ip :]
        block_counts = products.reshape(-1, self.n_blocks, self.block_size).sum(dim=2)
        parity = torch.remainder(block_counts.to(torch.int64), 2)
        return 1.0 - 2.0 * parity.to(dtype=states.dtype)

    def sign(self, states: Tensor) -> Tensor:
        """Return the ordinary global IP sign as a secondary diagnostic."""
        return self._block_signs(states).prod(dim=1)

    def log_unnormalized(self, states: Tensor) -> Tensor:
        """Return the sum of the blockwise IP target scores."""
        return self.beta * self._block_signs(states).sum(dim=1)

    def log_prob(self, states: Tensor) -> Tensor:
        return self.log_unnormalized(states) - self.log_partition

    def sample(
        self,
        sample_size: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Draw exact samples independently from every IP block."""
        if sample_size < 1:
            raise ValueError("sample_size must be positive")
        if not dtype.is_floating_point:
            raise ValueError("dtype must be floating point")
        left_blocks: list[Tensor] = []
        right_blocks: list[Tensor] = []
        for _ in range(self.n_blocks):
            block = self.block_target.sample(
                sample_size,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            left_blocks.append(block[:, : self.block_size])
            right_blocks.append(block[:, self.block_size :])
        return torch.cat((*left_blocks, *right_blocks), dim=1)


@dataclass(frozen=True)
class AnchoredTeacherTarget:
    r"""Dense RBM teacher on products ``z_i=x_i*y_i`` with linear anchors.

    The visible score is

    ``a @ (z-center) + sum_r softplus(b_r + U_r @ (z-center))``.

    Exact i.i.d. sampling enumerates the ``2**n_ip`` product states rather than
    the ``4**n_ip`` visible states.  The factor ``3**(n_ip-sum(z))`` accounts
    for the three visible preimages of every zero product bit.
    """

    n_ip: int
    teacher_hidden: int
    anchor_strength: float
    interaction_scale: float
    bias_span: float
    teacher_seed: int
    center: float = 0.25
    beta: float = 1.0

    def __post_init__(self) -> None:
        if self.n_ip < 1 or self.teacher_hidden < 1:
            raise ValueError("n_ip and teacher_hidden must be positive")
        if self.anchor_strength < 0 or self.interaction_scale < 0 or self.bias_span < 0:
            raise ValueError("teacher strengths and bias_span must be nonnegative")
        if self.teacher_seed < 0:
            raise ValueError("teacher_seed must be nonnegative")
        if not 0.0 <= self.center <= 1.0:
            raise ValueError("center must lie in [0, 1]")
        if self.beta != 1.0:
            raise ValueError(
                "anchored_teacher uses beta=1 so it remains exactly representable "
                "by O(n) 3RBM hidden units"
            )
        if self.n_ip > 20:
            raise ValueError("exact anchored-teacher sampling currently supports n_ip <= 20")

    @property
    def n_visible(self) -> int:
        return 2 * self.n_ip

    @staticmethod
    def _balanced_signs(length: int, generator: torch.Generator) -> Tensor:
        signs = torch.ones(length, dtype=torch.float64)
        signs[: length // 2] = -1.0
        return signs[torch.randperm(length, generator=generator)]

    @cached_property
    def anchor_weights(self) -> Tensor:
        """Return deterministic balanced signed anchor coefficients."""
        generator = torch.Generator().manual_seed(self.teacher_seed)
        return self.anchor_strength * self._balanced_signs(self.n_ip, generator)

    @cached_property
    def teacher_weights(self) -> Tensor:
        """Return deterministic, diverse, dense balanced teacher directions."""
        generator = torch.Generator().manual_seed(self.teacher_seed + 1)
        rows: list[Tensor] = []
        candidates_per_row = max(64, 16 * self.n_ip)
        for _ in range(self.teacher_hidden):
            best: Tensor | None = None
            best_key: tuple[int, float, float] | None = None
            current_rank = int(torch.linalg.matrix_rank(torch.stack(rows))) if rows else 0
            for _ in range(candidates_per_row):
                candidate = self._balanced_signs(self.n_ip, generator)
                if not rows:
                    best = candidate
                    break
                existing = torch.stack(rows)
                correlations = torch.abs(existing @ candidate) / self.n_ip
                candidate_rank = int(
                    torch.linalg.matrix_rank(torch.cat((existing, candidate[None, :]), dim=0))
                )
                key = (
                    0 if candidate_rank > current_rank else 1,
                    float(correlations.max()),
                    float(torch.sum(correlations.square())),
                )
                if best_key is None or key < best_key:
                    best = candidate
                    best_key = key
            if best is None:
                raise AssertionError("failed to generate a teacher direction")
            rows.append(best)
        return self.interaction_scale * torch.stack(rows) / math.sqrt(self.n_ip)

    @cached_property
    def teacher_bias(self) -> Tensor:
        """Return distinct, reproducibly permuted teacher hidden biases."""
        values = torch.linspace(
            -self.bias_span,
            self.bias_span,
            self.teacher_hidden,
            dtype=torch.float64,
        )
        generator = torch.Generator().manual_seed(self.teacher_seed + 2)
        return values[torch.randperm(self.teacher_hidden, generator=generator)]

    @cached_property
    def product_states(self) -> Tensor:
        """Enumerate all product-register states on CPU in float64."""
        return all_binary_states(self.n_ip)

    def _score_products(self, products: Tensor) -> Tensor:
        anchor = self.anchor_weights.to(device=products.device, dtype=products.dtype)
        weights = self.teacher_weights.to(device=products.device, dtype=products.dtype)
        bias = self.teacher_bias.to(device=products.device, dtype=products.dtype)
        centered = products - self.center
        return centered @ anchor + torch.nn.functional.softplus(bias + centered @ weights.T).sum(
            dim=1
        )

    def _products(self, states: Tensor) -> Tensor:
        if states.ndim != 2 or states.shape[1] != self.n_visible:
            raise ValueError(f"states must have shape (batch, {self.n_visible})")
        return states[:, : self.n_ip] * states[:, self.n_ip :]

    def sign(self, states: Tensor) -> Tensor:
        """Return the IP sign as a secondary diagnostic for this target."""
        return inner_product_sign(states)

    def log_unnormalized(self, states: Tensor) -> Tensor:
        """Return the anchored dense-teacher score on visible states."""
        return self._score_products(self._products(states))

    @cached_property
    def product_log_multiplicity(self) -> Tensor:
        zeros = self.n_ip - self.product_states.sum(dim=1)
        return zeros * math.log(3.0)

    @cached_property
    def product_scores(self) -> Tensor:
        return self._score_products(self.product_states)

    @cached_property
    def product_log_prob(self) -> Tensor:
        logits = self.product_scores + self.product_log_multiplicity
        return logits - torch.logsumexp(logits, dim=0)

    @property
    def log_partition(self) -> float:
        return float(torch.logsumexp(self.product_scores + self.product_log_multiplicity, dim=0))

    @property
    def entropy(self) -> float:
        product_probability = torch.exp(self.product_log_prob)
        mean_score = torch.sum(product_probability * self.product_scores)
        return self.log_partition - float(mean_score)

    @property
    def mean_sign(self) -> float:
        parity = torch.remainder(self.product_states.sum(dim=1).to(torch.int64), 2)
        sign = 1.0 - 2.0 * parity.to(torch.float64)
        return float(torch.sum(torch.exp(self.product_log_prob) * sign))

    def log_prob(self, states: Tensor) -> Tensor:
        """Return exactly normalized visible log probabilities."""
        return self.log_unnormalized(states) - self.log_partition

    def sample(
        self,
        sample_size: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Draw exact i.i.d. visible samples through enumerated product states."""
        if sample_size < 1:
            raise ValueError("sample_size must be positive")
        if not dtype.is_floating_point:
            raise ValueError("dtype must be floating point")
        if generator is not None and generator.device.type != "cpu":
            raise ValueError("anchored-teacher exact sampling requires a CPU generator")
        product_indices = torch.multinomial(
            torch.exp(self.product_log_prob),
            sample_size,
            replacement=True,
            generator=generator,
        )
        products = self.product_states[product_indices]
        preimage = torch.randint(
            0,
            3,
            products.shape,
            dtype=torch.int64,
            generator=generator,
        )
        zeros = products == 0
        left = torch.where(zeros, preimage == 2, torch.ones_like(zeros))
        right = torch.where(zeros, preimage == 1, torch.ones_like(zeros))
        return torch.cat((left, right), dim=1).to(device=device, dtype=dtype)


def uniform_binary(
    sample_size: int,
    n_visible: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> Tensor:
    """Draw uniform binary states for initialization or NCE noise."""
    if sample_size < 1 or n_visible < 1:
        raise ValueError("sample_size and n_visible must be positive")
    return torch.randint(
        0,
        2,
        (sample_size, n_visible),
        device=device,
        dtype=torch.int64,
        generator=generator,
    ).to(dtype=dtype)
