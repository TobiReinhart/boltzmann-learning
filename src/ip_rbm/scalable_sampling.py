"""Persistent and tempered Gibbs samplers for trainable energy models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ip_rbm.scalable_data import uniform_binary
from ip_rbm.trainable_models import TrainableEnergyModel


@dataclass
class PersistentSampler:
    """Persistent visible chains for stochastic maximum likelihood."""

    model: TrainableEnergyModel
    visible: Tensor
    generator: torch.Generator | None = None

    @classmethod
    def initialize(
        cls,
        model: TrainableEnergyModel,
        n_chains: int,
        *,
        generator: torch.Generator | None = None,
    ) -> PersistentSampler:
        visible = uniform_binary(
            n_chains,
            model.n_visible,
            device=model.device,
            dtype=model.dtype,
            generator=generator,
        )
        return cls(model=model, visible=visible, generator=generator)

    @torch.no_grad()
    def step(self, sweeps: int) -> Tensor:
        """Advance and return the persistent visible chains."""
        self.visible = self.model.gibbs_visible(
            self.visible,
            sweeps,
            generator=self.generator,
        )
        return self.visible


@dataclass(frozen=True)
class TemperingDiagnostics:
    """Swap statistics accumulated by a tempered sampler."""

    proposed: Tensor
    accepted: Tensor

    @property
    def acceptance_rate(self) -> Tensor:
        denominator = torch.clamp(self.proposed, min=1)
        return self.accepted / denominator


class TemperedPersistentSampler:
    """Replica-exchange persistent sampler on the joint visible-hidden model."""

    def __init__(
        self,
        model: TrainableEnergyModel,
        n_chains: int,
        n_replicas: int,
        *,
        minimum_inverse_temperature: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> None:
        if n_chains < 1:
            raise ValueError("n_chains must be positive")
        if n_replicas < 2:
            raise ValueError("n_replicas must be at least two")
        if not 0 <= minimum_inverse_temperature < 1:
            raise ValueError("minimum_inverse_temperature must lie in [0, 1)")
        self.model = model
        self.generator = generator
        self.inverse_temperatures = torch.linspace(
            minimum_inverse_temperature,
            1.0,
            n_replicas,
            device=model.device,
            dtype=model.dtype,
        )
        self.visible = uniform_binary(
            n_replicas * n_chains,
            model.n_visible,
            device=model.device,
            dtype=model.dtype,
            generator=generator,
        ).reshape(n_replicas, n_chains, model.n_visible)
        self.hidden = uniform_binary(
            n_replicas * n_chains,
            model.n_hidden,
            device=model.device,
            dtype=model.dtype,
            generator=generator,
        ).reshape(n_replicas, n_chains, model.n_hidden)
        # These are diagnostics only. Keep them in the model dtype because MPS
        # does not support device-side float64 tensors.
        self.proposed_swaps = torch.zeros(
            n_replicas - 1,
            device=model.device,
            dtype=model.dtype,
        )
        self.accepted_swaps = torch.zeros_like(self.proposed_swaps)
        self._swap_parity = 0

    @property
    def n_replicas(self) -> int:
        return self.visible.shape[0]

    @property
    def n_chains(self) -> int:
        return self.visible.shape[1]

    @torch.no_grad()
    def _gibbs_sweep(self) -> None:
        for replica, inverse_temperature in enumerate(self.inverse_temperatures):
            visible, hidden = self.model.gibbs_joint_step(
                self.visible[replica],
                inverse_temperature=float(inverse_temperature),
                generator=self.generator,
            )
            self.visible[replica] = visible
            self.hidden[replica] = hidden

    @torch.no_grad()
    def _swap(self) -> None:
        for lower in range(self._swap_parity, self.n_replicas - 1, 2):
            upper = lower + 1
            lower_score = self.model.joint_log_unnormalized(self.visible[lower], self.hidden[lower])
            upper_score = self.model.joint_log_unnormalized(self.visible[upper], self.hidden[upper])
            beta_lower = self.inverse_temperatures[lower]
            beta_upper = self.inverse_temperatures[upper]
            log_acceptance = (beta_upper - beta_lower) * (lower_score - upper_score)
            uniforms = torch.rand(
                self.n_chains,
                device=self.model.device,
                dtype=self.model.dtype,
                generator=self.generator,
            )
            accept = torch.log(uniforms) < torch.minimum(
                log_acceptance, torch.zeros_like(log_acceptance)
            )
            self.proposed_swaps[lower] += self.n_chains
            self.accepted_swaps[lower] += accept.sum().to(self.accepted_swaps.dtype)
            if bool(torch.any(accept)):
                lower_visible = self.visible[lower, accept].clone()
                lower_hidden = self.hidden[lower, accept].clone()
                self.visible[lower, accept] = self.visible[upper, accept]
                self.hidden[lower, accept] = self.hidden[upper, accept]
                self.visible[upper, accept] = lower_visible
                self.hidden[upper, accept] = lower_hidden
        self._swap_parity = 1 - self._swap_parity

    @torch.no_grad()
    def step(self, sweeps: int) -> Tensor:
        """Advance all replicas and return the inverse-temperature-one chains."""
        if sweeps < 1:
            raise ValueError("sweeps must be positive")
        for _ in range(sweeps):
            self._gibbs_sweep()
            self._swap()
        return self.visible[-1]

    def diagnostics(self) -> TemperingDiagnostics:
        """Return immutable copies of the accumulated swap statistics."""
        return TemperingDiagnostics(
            proposed=self.proposed_swaps.detach().cpu().clone(),
            accepted=self.accepted_swaps.detach().cpu().clone(),
        )


@dataclass(frozen=True)
class AISResult:
    """Annealed-importance estimate of the joint partition function."""

    log_partition: float
    log_weight_standard_error: float
    effective_sample_size: float
    log_weights: Tensor


@torch.no_grad()
def estimate_log_partition_ais(
    model: TrainableEnergyModel,
    *,
    n_particles: int,
    n_intermediate: int,
    sweeps_per_temperature: int = 1,
    generator: torch.Generator | None = None,
) -> AISResult:
    """Estimate ``log Z`` by annealing from the uniform joint distribution."""
    if n_particles < 2:
        raise ValueError("n_particles must be at least two")
    if n_intermediate < 2:
        raise ValueError("n_intermediate must be at least two")
    if sweeps_per_temperature < 1:
        raise ValueError("sweeps_per_temperature must be positive")

    visible = uniform_binary(
        n_particles,
        model.n_visible,
        device=model.device,
        dtype=model.dtype,
        generator=generator,
    )
    hidden = uniform_binary(
        n_particles,
        model.n_hidden,
        device=model.device,
        dtype=model.dtype,
        generator=generator,
    )
    schedule = torch.linspace(
        0.0,
        1.0,
        n_intermediate,
        device=model.device,
        dtype=model.dtype,
    )
    # Accumulate in float64 when the backend supports it. Apple MPS has no
    # float64 tensors, so retaining the model's float32 dtype avoids copying
    # every AIS increment to the CPU.
    accumulator_dtype = torch.float32 if model.device.type == "mps" else torch.float64
    log_weights = torch.zeros(
        n_particles,
        device=model.device,
        dtype=accumulator_dtype,
    )
    for index in range(1, n_intermediate):
        previous_beta = schedule[index - 1]
        beta = schedule[index]
        joint_score = model.joint_log_unnormalized(visible, hidden)
        log_weights += ((beta - previous_beta) * joint_score).to(accumulator_dtype)
        for _ in range(sweeps_per_temperature):
            hidden = model.sample_hidden(
                visible,
                inverse_temperature=float(beta),
                generator=generator,
            )
            visible = model.sample_visible(
                hidden,
                visible,
                inverse_temperature=float(beta),
                generator=generator,
            )

    log_mean_weight = torch.logsumexp(log_weights, dim=0) - math.log(n_particles)
    log_partition_base = (model.n_visible + model.n_hidden) * math.log(2)
    normalized_weights = torch.softmax(log_weights, dim=0)
    effective_sample_size = float(1.0 / torch.sum(normalized_weights.square()))
    stabilized_weights = torch.exp(log_weights - torch.max(log_weights))
    relative_standard_error = stabilized_weights.std(unbiased=True) / (
        n_particles**0.5 * stabilized_weights.mean()
    )
    return AISResult(
        log_partition=float(log_partition_base + log_mean_weight),
        log_weight_standard_error=float(relative_standard_error),
        effective_sample_size=effective_sample_size,
        log_weights=log_weights.detach().cpu(),
    )
