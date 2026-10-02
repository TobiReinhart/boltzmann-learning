"""Exactly tabulated Ising-energy and disorder-ground-state targets.

Bit i is the *least* significant bit of the internal table index. These are
distributions over visible encodings, not supervised energy regressions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from ip_rbm.targets import inner_product_sign


def _bits(width: int) -> NDArray[np.int64]:
    return (np.arange(1 << width, dtype=np.int64)[:, None] >> np.arange(width)) & 1


def ising_graph(vertices: int, graph: str, seed: int) -> NDArray[np.int64]:
    """A simple graph with canonical edge order; seeded chords are reproducible."""
    if vertices < 4:
        raise ValueError("Ising graphs require at least four vertices")
    if graph == "cycles":
        if vertices % 4:
            raise ValueError("cycles requires a multiple of four vertices")
        edges = [(base + i, base + (i + 1) % 4) for base in range(0, vertices, 4) for i in range(4)]
    elif graph == "ladder":
        if vertices % 2:
            raise ValueError("ladder requires an even number of vertices")
        length = vertices // 2
        edges = [
            (row * length + i, row * length + i + 1) for row in range(2) for i in range(length - 1)
        ]
        edges += [(i, length + i) for i in range(length)]
    elif graph in {"ring", "ring_chords"}:
        edges = [(i, (i + 1) % vertices) for i in range(vertices)]
        if graph == "ring_chords":
            existing = {tuple(sorted(edge)) for edge in edges}
            candidates = [
                (i, j)
                for i in range(vertices)
                for j in range(i + 1, vertices)
                if (i, j) not in existing
            ]
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(candidates))
            edges += [candidates[i] for i in order[: min(vertices, len(candidates))]]
    else:
        raise ValueError(f"unknown Ising graph: {graph}")
    return np.array(sorted({tuple(sorted(edge)) for edge in edges}), dtype=np.int64)


def ground_graph(edge_count: int, graph: str, seed: int) -> NDArray[np.int64]:
    """Ground-state inputs encode exactly edge_count coupling signs."""
    if graph == "cycles":
        return ising_graph(edge_count, graph, seed)
    if graph != "ring_chords":
        raise ValueError("ground-state graphs support cycles or ring_chords")
    vertices = edge_count // 2 + 1
    # Preserve the connected ring, then choose enough random chords.
    ring = {tuple(sorted((i, (i + 1) % vertices))) for i in range(vertices)}
    candidates = [
        (i, j) for i in range(vertices) for j in range(i + 1, vertices) if (i, j) not in ring
    ]
    needed = edge_count - len(ring)
    if needed < 0 or needed > len(candidates):
        raise ValueError("too few coupling bits for an overlapping-cycle graph")
    order = np.random.default_rng(seed).permutation(len(candidates))
    return np.array(sorted(ring | {candidates[i] for i in order[:needed]}), dtype=np.int64)


def frustration_table(
    edges: NDArray[np.int64], weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Exact weighted distance to the cut space via a hypercube distance transform.

    For each disorder J, min_sigma sum_e |J_e| [sign(J_e) != sigma_a sigma_b].
    This uses O(M 2**M) work and O(2**M) storage, rather than enumerating every
    spin configuration separately for every disorder. It remains exponential.
    """
    width = len(edges)
    if not 1 <= width <= 20 or len(weights) != width or np.any(weights <= 0):
        raise ValueError("require 1..20 edges and positive edge weights")
    vertices = int(edges.max()) + 1
    if vertices > 20:
        raise ValueError("exact cut-space enumeration supports at most 20 vertices")
    spins = _bits(vertices)
    satisfied = spins[:, edges[:, 0]] == spins[:, edges[:, 1]]
    codes = satisfied.astype(np.int64) @ (1 << np.arange(width))
    distance = np.full(1 << width, np.inf)
    distance[codes] = 0.0
    for bit, weight in enumerate(weights):
        view = distance.reshape(-1, 2, 1 << bit)
        low, high = view[:, 0].copy(), view[:, 1].copy()
        view[:, 0] = np.minimum(low, high + weight)
        view[:, 1] = np.minimum(high, low + weight)
    return distance


@lru_cache(maxsize=32)
def _landscape(
    width: int, kind: str, graph: str, couplings: str, seed: int
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    if not 4 <= width <= 16:
        raise ValueError("exact Ising target tables currently require 4..16 encoded variables")
    rng = np.random.default_rng(seed)
    if kind == "ising_energy":
        edges = ising_graph(width, graph, seed)
        if couplings == "ferro":
            weights = np.ones(len(edges))
        elif couplings == "mixed":
            weights = rng.choice([-1.0, 1.0], len(edges)).astype(np.float64)
        else:
            raise ValueError("energy couplings must be ferro or mixed")
        spins = 2 * _bits(width) - 1
        energy = -(spins[:, edges[:, 0]] * spins[:, edges[:, 1]]) @ weights
    elif kind == "ising_ground":
        edges = ground_graph(width, graph, seed)
        if couplings == "unit":
            weights = np.ones(len(edges))
        elif couplings == "weighted":
            weights = rng.choice([0.5, 1.5], len(edges)).astype(np.float64)
        else:
            raise ValueError("ground couplings must be unit or weighted; signs are input bits")
        energy = 2 * frustration_table(edges, weights) - weights.sum()
    else:
        raise ValueError(f"unknown Ising target kind: {kind}")
    return edges, weights, energy


@dataclass(frozen=True)
class IsingTarget:
    """p(v) proportional to exp(-beta * standardized energy), or a threshold tilt.

    For energy targets the encoded signs are physical spins of one fixed instance.
    For ground targets they are disorder signs; physical spins are minimized out.
    Signed-product encoding is z_i=(2x_i-1)(2y_i-1), NOT x_i*y_i.
    """

    n_ip: int
    beta: float
    kind: str
    graph: str = "ring_chords"
    couplings: str = "mixed"
    graph_seed: int = 11
    encoding: str = "signed_product"
    objective: str = "smooth"

    def __post_init__(self) -> None:
        if self.encoding not in {"direct", "signed_product"}:
            raise ValueError("Ising encoding must be direct or signed_product")
        if self.objective not in {"smooth", "threshold"}:
            raise ValueError("Ising objective must be smooth or threshold")
        if not math.isfinite(self.beta) or self.beta < 0 or self.graph_seed < 0:
            raise ValueError("Ising beta must be finite/nonnegative and seed nonnegative")
        # Fail before training, including invalid graph/coupling combinations.
        _ = self.landscape

    @property
    def n_visible(self) -> int:
        return 2 * self.n_ip

    @property
    def encoded_variables(self) -> int:
        return self.n_ip if self.encoding == "signed_product" else self.n_visible

    @cached_property
    def landscape(self) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
        return _landscape(
            self.encoded_variables, self.kind, self.graph, self.couplings, self.graph_seed
        )

    @cached_property
    def threshold(self) -> float:
        values, counts = np.unique(self.landscape[2], return_counts=True)
        if len(values) < 2:
            raise ValueError("cannot threshold a constant Ising landscape")
        cumulative = np.cumsum(counts)[:-1] / counts.sum()
        return float(values[int(np.argmin(np.abs(cumulative - 0.5)))])

    @cached_property
    def scores(self) -> Tensor:
        energy = self.landscape[2]
        if self.objective == "threshold":
            values = (energy <= self.threshold).astype(np.float64)
            if values.std() == 0:
                raise ValueError("threshold is constant for this Ising landscape")
            values = (values - values.mean()) / values.std()
        else:
            scale = energy.std()
            if scale == 0:
                raise ValueError("cannot standardize a constant Ising landscape")
            values = -(energy - energy.mean()) / scale
        return torch.from_numpy(self.beta * values)

    @cached_property
    def probabilities(self) -> Tensor:
        return torch.softmax(self.scores, dim=0)

    @property
    def log_partition(self) -> float:
        multiplicity = self.n_ip * math.log(2) if self.encoding == "signed_product" else 0.0
        return float(torch.logsumexp(self.scores, dim=0)) + multiplicity

    @property
    def entropy(self) -> float:
        return self.log_partition - float(self.probabilities @ self.scores)

    @property
    def mean_sign(self) -> float:
        if self.encoding == "signed_product":
            return float(self.probabilities[0])
        signs = inner_product_sign(torch.from_numpy(_bits(self.n_visible)).double())
        return float(self.probabilities @ signs)

    def sign(self, states: Tensor) -> Tensor:
        return inner_product_sign(states)

    def indices(self, states: Tensor) -> Tensor:
        if states.ndim != 2 or states.shape[1] != self.n_visible:
            raise ValueError(f"states must have shape (batch, {self.n_visible})")
        bits = (
            states
            if self.encoding == "direct"
            else (states[:, : self.n_ip] == states[:, self.n_ip :])
        )
        powers = 2 ** torch.arange(self.encoded_variables, device=states.device)
        return (bits.long() * powers).sum(dim=1)

    def log_unnormalized(self, states: Tensor) -> Tensor:
        return self.scores.to(device=states.device, dtype=states.dtype)[self.indices(states)]

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
            raise ValueError("exact Ising sampling requires a CPU generator")
        indices = torch.multinomial(
            self.probabilities, sample_size, replacement=True, generator=generator
        )
        bits = (indices[:, None] >> torch.arange(self.encoded_variables)) & 1
        if self.encoding == "signed_product":
            left = torch.randint(2, bits.shape, generator=generator)
            right = torch.where(bits.bool(), left, 1 - left)
            bits = torch.cat((left, right), dim=1)
        return bits.to(device=device, dtype=dtype)

    def save_spec(self, path: Path) -> None:
        """Save actual graph, weights, exact landscape and target normalization."""
        edges, weights, energy = self.landscape
        np.savez_compressed(
            path,
            edges=edges,
            weights=weights,
            energy=energy,
            scores=self.scores.numpy(),
            encoding=self.encoding,
            kind=self.kind,
            objective=self.objective,
            beta=self.beta,
            graph_seed=self.graph_seed,
            n_visible=self.n_visible,
            log_partition=self.log_partition,
            entropy=self.entropy,
            uniform_energy_mean=energy.mean(),
            uniform_energy_std=energy.std(),
            threshold=self.threshold,
            threshold_fraction=np.mean(energy <= self.threshold),
        )
