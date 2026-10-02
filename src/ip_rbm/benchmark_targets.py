"""Small, fully enumerable targets for the frozen supplementary studies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from ip_rbm.states import all_binary_states
from ip_rbm.targets import inner_product_sign


def four_body_terms(width: int, family: str, seed: int) -> NDArray[np.int64]:
    """Lifted and rewired designs have width/2 terms and degree two at every site."""
    if width < 8 or width % 4:
        raise ValueError("four-body designs require a multiple of four visibles >=8")
    half = width // 2
    if family == "four_disjoint":
        return np.arange(width, dtype=np.int64).reshape(-1, 4)
    if family == "four_lifted":
        return np.array(
            [sorted([i, (i + 1) % half, half + i, half + (i + 1) % half]) for i in range(half)],
            dtype=np.int64,
        )
    if family != "four_rewired":
        raise ValueError(f"unknown four-body design: {family}")
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        terms = np.sort(rng.permutation(np.tile(np.arange(width), 2)).reshape(-1, 4), axis=1)
        if np.any(np.diff(terms, axis=1) == 0) or len(np.unique(terms, axis=0)) != half:
            continue
        incidence = np.array([np.isin(np.arange(width), term) for term in terms]).T
        # A common perfect pairing requires each incidence class to have even size.
        _, counts = np.unique(incidence, axis=0, return_counts=True)
        if np.any(counts % 2):
            return terms.astype(np.int64)
    raise ValueError("failed to draw a rewired design without a common perfect pairing")


@lru_cache(maxsize=32)
def _spec(width: int, family: str, seed: int) -> dict[str, NDArray[np.float64]]:
    if width < 4 or width > 16:
        raise ValueError("benchmark enumeration requires 4..16 visibles")
    rng = np.random.default_rng(seed)
    states = all_binary_states(width).numpy()
    spins = 2 * states - 1
    extras: dict[str, NDArray[np.float64]] = {}
    if family.startswith("four_"):
        terms = four_body_terms(width, family, seed)
        # Same sign sequence for lifted and rewired; topology uses its own RNG.
        weights = rng.choice([-1.0, 1.0], len(terms)).astype(np.float64)
        score = np.prod(spins[:, terms], axis=2) @ weights / math.sqrt(len(terms))
        extras = {"terms": terms.astype(np.float64), "weights": weights}
    elif family == "lattice_ising":
        side = math.isqrt(width)
        if side * side != width:
            raise ValueError("lattice_ising needs a square number of visible spins")
        edges = set()
        for i in range(side):
            for j in range(side):
                a = i * side + j
                edges.add(tuple(sorted((a, i * side + (j + 1) % side))))
                edges.add(tuple(sorted((a, ((i + 1) % side) * side + j))))
        terms = np.array(sorted(edges), dtype=np.int64)
        score = np.prod(spins[:, terms], axis=2).sum(axis=1)
        extras = {"terms": terms.astype(np.float64)}
    elif family == "independent":
        fields = rng.uniform(-0.7, 0.7, width)
        score = spins @ fields
        extras = {"fields": fields}
    elif family == "mixture":
        logits = rng.normal(0, 1.25, (4, width))
        components = states @ logits.T - np.logaddexp(0, logits).sum(axis=1)
        score = np.logaddexp.reduce(components, axis=1) - math.log(4)
        extras = {"component_logits": logits}
    elif family == "rbm_teacher":
        weights = rng.normal(0, 1.5 / math.sqrt(width), (width, 4))
        hidden_bias = rng.normal(0, 0.3, 4) - 0.5 * weights.sum(axis=0)
        visible_bias = -0.5 * weights.sum(axis=1)
        score = states @ visible_bias + np.logaddexp(0, hidden_bias + states @ weights).sum(axis=1)
        extras = {"weights": weights, "hidden_bias": hidden_bias, "visible_bias": visible_bias}
    elif family == "random_pairwise":
        terms = np.array(
            [(i, j) for i in range(width) for j in range(i + 1, width)], dtype=np.int64
        )
        weights = rng.normal(0, 1, len(terms))
        weights /= np.linalg.norm(weights)
        score = np.prod(spins[:, terms], axis=2) @ weights
        extras = {"terms": terms.astype(np.float64), "weights": weights}
    elif family == "random_table":
        score = rng.normal(size=1 << width)
        score = (score - score.mean()) / score.std()
    else:
        raise ValueError(f"unknown benchmark family: {family}")
    return {"raw_score": score, **extras}


@dataclass(frozen=True)
class BenchmarkTarget:
    n_ip: int
    beta: float
    family: str
    instance_seed: int = 914001

    def __post_init__(self) -> None:
        if not math.isfinite(self.beta) or self.beta <= 0 or self.instance_seed < 0:
            raise ValueError("require finite positive beta and nonnegative seed")
        if self.family in {"rbm_teacher", "mixture"} and self.beta != 1:
            raise ValueError("teacher and mixture targets require beta=1 to preserve their family")
        _ = self.spec

    @property
    def n_visible(self) -> int:
        return 2 * self.n_ip

    @cached_property
    def spec(self) -> dict[str, NDArray[np.float64]]:
        return _spec(self.n_visible, self.family, self.instance_seed)

    @cached_property
    def scores(self) -> Tensor:
        return torch.from_numpy(self.beta * self.spec["raw_score"])

    @cached_property
    def probabilities(self) -> Tensor:
        return torch.softmax(self.scores, dim=0)

    @property
    def log_partition(self) -> float:
        return float(torch.logsumexp(self.scores, dim=0))

    @property
    def entropy(self) -> float:
        return self.log_partition - float(self.probabilities @ self.scores)

    @property
    def mean_sign(self) -> float:
        return float(self.probabilities @ self.sign(all_binary_states(self.n_visible)))

    def sign(self, states: Tensor) -> Tensor:
        return inner_product_sign(states)

    def log_unnormalized(self, states: Tensor) -> Tensor:
        if states.ndim != 2 or states.shape[1] != self.n_visible:
            raise ValueError("wrong visible dimension")
        powers = 2 ** torch.arange(self.n_visible - 1, -1, -1, device=states.device)
        indices = (states.long() * powers).sum(dim=1)
        return self.scores.to(device=states.device, dtype=states.dtype)[indices]

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
        if sample_size < 1 or not dtype.is_floating_point:
            raise ValueError("require positive sample_size and floating dtype")
        if generator is not None and generator.device.type != "cpu":
            raise ValueError("exact sampling requires a CPU generator")
        indices = torch.multinomial(self.probabilities, sample_size, True, generator=generator)
        bits = (indices[:, None] >> torch.arange(self.n_visible - 1, -1, -1)) & 1
        return bits.to(device=device, dtype=dtype)

    def save_spec(self, path: Path) -> None:
        payload: dict[str, Any] = dict(self.spec)
        payload.update(
            scores=self.scores.numpy(),
            family=self.family,
            instance_seed=self.instance_seed,
            beta=self.beta,
            log_partition=self.log_partition,
            entropy=self.entropy,
        )
        np.savez_compressed(path, **payload)
