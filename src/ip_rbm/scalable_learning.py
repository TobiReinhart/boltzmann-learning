"""Minibatch learning algorithms for scalable RBM and 3RBM experiments."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from ip_rbm.scalable_data import uniform_binary
from ip_rbm.scalable_sampling import PersistentSampler, TemperedPersistentSampler
from ip_rbm.trainable_models import TrainableEnergyModel, TrainableThreeBodyRBM

LearningAlgorithm = Literal[
    "cd",
    "pcd",
    "tempered_pcd",
    "pseudolikelihood",
    "nce",
    "score_regression",
]
OptimizerName = Literal["adam", "sgd"]
TargetScoreFunction = Callable[[Tensor], Tensor]


def learning_rate_at_update(settings: ScalableTrainingSettings, update: int) -> float:
    """Constant prefix followed by cosine decay; old defaults remain constant."""
    start = settings.lr_decay_start_fraction
    progress = update / settings.updates
    if progress <= start or start == 1:
        return settings.learning_rate
    phase = (progress - start) / (1 - start)
    ratio = (
        settings.lr_final_ratio
        + (1 - settings.lr_final_ratio) * (1 + math.cos(math.pi * phase)) / 2
    )
    return settings.learning_rate * ratio


@dataclass(frozen=True)
class ScalableTrainingSettings:
    """Hyperparameters shared across scalable learning algorithms."""

    algorithm: LearningAlgorithm
    updates: int
    batch_size: int
    learning_rate: float
    optimizer: OptimizerName = "adam"
    gibbs_steps: int = 1
    persistent_chains: int | None = None
    weight_bound: float | None = None
    weight_decay: float = 0.0
    gradient_clip_norm: float | None = None
    nce_noise_ratio: int = 1
    tempering_replicas: int = 8
    minimum_inverse_temperature: float = 0.0
    record_every: int = 100
    seed: int = 1729
    minibatch_seed: int | None = None
    sampler_seed: int | None = None
    all_pairs_sampler: str = "legacy"
    all_pairs_block_size: int = 4
    lr_decay_start_fraction: float = 1.0
    lr_final_ratio: float = 1.0

    def __post_init__(self) -> None:
        if self.algorithm not in {
            "cd",
            "pcd",
            "tempered_pcd",
            "pseudolikelihood",
            "nce",
            "score_regression",
        }:
            raise ValueError(f"unknown learning algorithm: {self.algorithm}")
        if self.optimizer not in {"adam", "sgd"}:
            raise ValueError(f"unknown optimizer: {self.optimizer}")
        if self.updates < 1 or self.batch_size < 1:
            raise ValueError("updates and batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.lr_decay_start_fraction <= 1 or not 0 < self.lr_final_ratio <= 1:
            raise ValueError("invalid learning-rate decay settings")
        if self.gibbs_steps < 1:
            raise ValueError("gibbs_steps must be positive")
        if self.all_pairs_sampler not in {"legacy", "cached", "block"}:
            raise ValueError("unknown all-pairs sampler")
        if not 1 <= self.all_pairs_block_size <= 8:
            raise ValueError("all_pairs_block_size must be between 1 and 8")
        if self.persistent_chains is not None and self.persistent_chains < 1:
            raise ValueError("persistent_chains must be positive")
        if self.weight_bound is not None and self.weight_bound <= 0:
            raise ValueError("weight_bound must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.nce_noise_ratio < 1:
            raise ValueError("nce_noise_ratio must be positive")
        if self.tempering_replicas < 2:
            raise ValueError("tempering_replicas must be at least two")
        if not 0 <= self.minimum_inverse_temperature < 1:
            raise ValueError("minimum_inverse_temperature must lie in [0, 1)")
        if self.record_every < 1:
            raise ValueError("record_every must be positive")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.minibatch_seed is not None and self.minibatch_seed < 0:
            raise ValueError("minibatch_seed must be nonnegative")
        if self.sampler_seed is not None and self.sampler_seed < 0:
            raise ValueError("sampler_seed must be nonnegative")


@dataclass(frozen=True)
class TrainingRecord:
    """One progress record from stochastic training."""

    update: int
    objective: float
    data_score: float
    negative_score: float
    gradient_norm: float
    maximum_absolute_parameter: float
    elapsed_seconds: float
    tempering_acceptance: float


@dataclass(frozen=True)
class TrainingCheckpoint:
    """One CPU parameter snapshot captured without evaluation overhead."""

    update: int
    elapsed_seconds: float
    theta: Tensor
    nce_log_normalizer: float


@dataclass(frozen=True)
class ScalableTrainingResult:
    """Completed training history and algorithm-specific state."""

    history: tuple[TrainingRecord, ...]
    elapsed_seconds: float
    nce_log_normalizer: float
    final_negative_visible: Tensor | None
    tempering_acceptance_rates: Tensor | None
    checkpoints: tuple[TrainingCheckpoint, ...]


TrainingCallback = Callable[[TrainingRecord], None]


class _MinibatchStream:
    """Reproducible shuffled traversal of an in-memory state matrix."""

    def __init__(self, states: Tensor, batch_size: int, generator: torch.Generator) -> None:
        if states.ndim != 2 or states.shape[0] < 1:
            raise ValueError("training states must be a nonempty matrix")
        self.states = states
        self.batch_size = batch_size
        self.generator = generator
        self.permutation = torch.empty(0, dtype=torch.int64)
        self.cursor = 0

    def next(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self.cursor + self.batch_size > self.permutation.numel():
            self.permutation = torch.randperm(self.states.shape[0], generator=self.generator)
            self.cursor = 0
        indices = self.permutation[self.cursor : self.cursor + self.batch_size]
        self.cursor += self.batch_size
        if indices.shape[0] < self.batch_size:
            extra = torch.randint(
                0,
                self.states.shape[0],
                (self.batch_size - indices.shape[0],),
                generator=self.generator,
            )
            indices = torch.cat((indices, extra))
        if self.states.device.type != "cpu":
            indices = indices.to(self.states.device)
        return self.states.index_select(0, indices).to(device=device, dtype=dtype)


def make_device_generator(device: torch.device, seed: int) -> torch.Generator | None:
    """Construct a device generator, falling back to the device-global stream."""
    try:
        return torch.Generator(device=device).manual_seed(seed)
    except (RuntimeError, TypeError):
        torch.manual_seed(seed)
        return None


def _make_optimizer(
    parameters: list[nn.Parameter], settings: ScalableTrainingSettings
) -> torch.optim.Optimizer:
    if settings.optimizer == "adam":
        return torch.optim.Adam(
            parameters,
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )
    return torch.optim.SGD(
        parameters,
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )


def _random_bit_pseudolikelihood(
    model: TrainableEnergyModel,
    visible: Tensor,
    generator: torch.Generator | None,
) -> Tensor:
    bit = torch.randint(
        0,
        model.n_visible,
        (visible.shape[0],),
        device=model.device,
        generator=generator,
    )
    flipped = visible.clone()
    row = torch.arange(visible.shape[0], device=model.device)
    flipped[row, bit] = 1.0 - flipped[row, bit]
    score_difference = model.log_unnormalized(visible) - model.log_unnormalized(flipped)
    return -functional.logsigmoid(score_difference).mean()


def mean_log_pseudolikelihood(model: TrainableEnergyModel, visible: Tensor) -> Tensor:
    """Evaluate the deterministic mean single-bit log pseudolikelihood."""
    terms: list[Tensor] = []
    score = model.log_unnormalized(visible)
    for bit in range(model.n_visible):
        flipped = visible.clone()
        flipped[:, bit] = 1.0 - flipped[:, bit]
        terms.append(functional.logsigmoid(score - model.log_unnormalized(flipped)))
    return torch.stack(terms, dim=1).mean()


def _nce_objective(
    model: TrainableEnergyModel,
    visible: Tensor,
    log_normalizer: Tensor,
    noise_ratio: int,
    generator: torch.Generator | None,
) -> Tensor:
    noise = uniform_binary(
        visible.shape[0] * noise_ratio,
        model.n_visible,
        device=model.device,
        dtype=model.dtype,
        generator=generator,
    )
    noise_log_probability = -model.n_visible * math.log(2.0)
    class_prior_offset = math.log(noise_ratio)
    data_logit = (
        model.log_unnormalized(visible)
        - log_normalizer
        - class_prior_offset
        - noise_log_probability
    )
    noise_logit = (
        model.log_unnormalized(noise) - log_normalizer - class_prior_offset - noise_log_probability
    )
    data_loss = functional.softplus(-data_logit).mean()
    noise_loss = functional.softplus(noise_logit).mean()
    return (data_loss + noise_ratio * noise_loss) / (1 + noise_ratio)


def _score_regression_objective(
    model: TrainableEnergyModel,
    visible: Tensor,
    target_score_function: TargetScoreFunction,
) -> Tensor:
    """Match target/model score differences, ignoring their additive constant."""
    model_score = model.log_unnormalized(visible)
    with torch.no_grad():
        target_score = target_score_function(visible).to(
            device=model.device,
            dtype=model.dtype,
        )
    if target_score.shape != model_score.shape:
        raise ValueError("target_score_function must return one score per visible configuration")
    residual = model_score - target_score
    centered_residual = residual - residual.mean()
    return centered_residual.square().mean()


def _gradient_norm(parameters: list[nn.Parameter]) -> float:
    squared = torch.zeros((), dtype=torch.float64)
    for parameter in parameters:
        if parameter.grad is not None:
            # MPS does not implement float64 tensors. Transfer the gradient to
            # the CPU before promoting it for the diagnostic accumulation.
            gradient = parameter.grad.detach().cpu().to(torch.float64)
            squared += gradient.square().sum()
    return float(torch.sqrt(squared))


def _maximum_absolute_parameter(model: TrainableEnergyModel) -> float:
    return max(float(parameter.detach().abs().max()) for parameter in model.parameters())


def _synchronize_device(device: torch.device) -> None:
    """Wait for queued accelerator work before recording a wall-clock timestamp."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def train_scalable(
    model: TrainableEnergyModel,
    train_states: Tensor,
    settings: ScalableTrainingSettings,
    *,
    callback: TrainingCallback | None = None,
    checkpoint_updates: tuple[int, ...] = (),
    target_score_function: TargetScoreFunction | None = None,
    batch_provider: Callable[[], Tensor] | None = None,
) -> ScalableTrainingResult:
    """Train an energy model using the selected scalable objective."""
    if train_states.ndim != 2 or train_states.shape[1] != model.n_visible:
        raise ValueError(f"train_states must have shape (observations, {model.n_visible})")
    if not train_states.is_floating_point():
        raise ValueError("train_states must use a floating-point dtype")
    if isinstance(model, TrainableThreeBodyRBM):
        model.configure_all_pairs_sampler(settings.all_pairs_sampler, settings.all_pairs_block_size)
    elif settings.all_pairs_sampler != "legacy":
        raise ValueError("all-pairs sampler requires an all-pairs 3RBM")
    if settings.algorithm == "score_regression" and target_score_function is None:
        raise ValueError("score_regression requires target_score_function")
    requested_checkpoints = frozenset(checkpoint_updates)
    if any(update < 1 or update > settings.updates for update in requested_checkpoints):
        raise ValueError("checkpoint updates must lie between 1 and settings.updates")

    minibatch_seed = settings.seed if settings.minibatch_seed is None else settings.minibatch_seed
    sampler_seed = settings.seed + 1 if settings.sampler_seed is None else settings.sampler_seed
    cpu_generator = torch.Generator().manual_seed(minibatch_seed)
    device_generator = make_device_generator(model.device, sampler_seed)
    minibatches = (
        _MinibatchStream(train_states, settings.batch_size, cpu_generator)
        if batch_provider is None
        else None
    )
    n_chains = settings.persistent_chains or settings.batch_size
    persistent: PersistentSampler | None = None
    tempered: TemperedPersistentSampler | None = None
    if settings.algorithm == "pcd":
        persistent = PersistentSampler.initialize(
            model,
            n_chains,
            generator=device_generator,
        )
    elif settings.algorithm == "tempered_pcd":
        tempered = TemperedPersistentSampler(
            model,
            n_chains,
            settings.tempering_replicas,
            minimum_inverse_temperature=settings.minimum_inverse_temperature,
            generator=device_generator,
        )

    trainable_parameters = list(model.parameters())
    log_normalizer: nn.Parameter | None = None
    if settings.algorithm == "nce":
        initial_log_normalizer = (model.n_visible + model.n_hidden) * math.log(2.0)
        log_normalizer = nn.Parameter(
            torch.tensor(initial_log_normalizer, device=model.device, dtype=model.dtype)
        )
        trainable_parameters.append(log_normalizer)
    optimizer = _make_optimizer(trainable_parameters, settings)
    model.project_parameters(settings.weight_bound)

    records: list[TrainingRecord] = []
    checkpoints: list[TrainingCheckpoint] = []
    _synchronize_device(model.device)
    started = perf_counter()
    final_negative: Tensor | None = None
    for update in range(1, settings.updates + 1):
        rate = learning_rate_at_update(settings, update)
        for group in optimizer.param_groups:
            group["lr"] = rate
        if batch_provider is not None:
            batch = batch_provider().to(device=model.device, dtype=model.dtype)
            if batch.shape != (settings.batch_size, model.n_visible):
                raise ValueError("fresh batch has incorrect shape")
        else:
            assert minibatches is not None
            batch = minibatches.next(model.device, model.dtype)
        optimizer.zero_grad(set_to_none=True)
        data_score_tensor: Tensor | None = None
        negative_score_tensor: Tensor | None = None

        if settings.algorithm == "score_regression":
            if target_score_function is None:
                raise AssertionError("score-regression target was not initialized")
            loss = _score_regression_objective(model, batch, target_score_function)
        elif settings.algorithm == "pseudolikelihood":
            loss = _random_bit_pseudolikelihood(model, batch, device_generator)
        elif settings.algorithm == "nce":
            if log_normalizer is None:
                raise AssertionError("NCE log normalizer was not initialized")
            loss = _nce_objective(
                model,
                batch,
                log_normalizer,
                settings.nce_noise_ratio,
                device_generator,
            )
        else:
            with torch.no_grad():
                if settings.algorithm == "cd":
                    negative = model.gibbs_visible(
                        batch,
                        settings.gibbs_steps,
                        generator=device_generator,
                    )
                elif settings.algorithm == "pcd":
                    if persistent is None:
                        raise AssertionError("persistent sampler was not initialized")
                    negative = persistent.step(settings.gibbs_steps)
                else:
                    if tempered is None:
                        raise AssertionError("tempered sampler was not initialized")
                    negative = tempered.step(settings.gibbs_steps)
                if update == settings.updates:
                    final_negative = negative.detach().cpu().clone()
            positive_scores = model.log_unnormalized(batch)
            negative_scores = model.log_unnormalized(negative.detach())
            loss = -positive_scores.mean() + negative_scores.mean()
            data_score_tensor = positive_scores.detach().mean()
            negative_score_tensor = negative_scores.detach().mean()

        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite objective at update {update}")
        torch.autograd.backward(loss)
        if settings.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(trainable_parameters, settings.gradient_clip_norm)
        should_record = (
            update == 1
            or update % settings.record_every == 0
            or update == settings.updates
            or update in requested_checkpoints
        )
        gradient_norm = _gradient_norm(trainable_parameters) if should_record else float("nan")
        optimizer.step()
        model.project_parameters(settings.weight_bound)

        if should_record:
            _synchronize_device(model.device)
            elapsed_seconds = perf_counter() - started
            tempering_acceptance = float("nan")
            if tempered is not None:
                rates = tempered.diagnostics().acceptance_rate
                tempering_acceptance = float(rates.mean())
            record = TrainingRecord(
                update=update,
                objective=float(loss.detach()),
                data_score=(
                    float(data_score_tensor) if data_score_tensor is not None else float("nan")
                ),
                negative_score=(
                    float(negative_score_tensor)
                    if negative_score_tensor is not None
                    else float("nan")
                ),
                gradient_norm=gradient_norm,
                maximum_absolute_parameter=_maximum_absolute_parameter(model),
                elapsed_seconds=elapsed_seconds,
                tempering_acceptance=tempering_acceptance,
            )
            records.append(record)
            if callback is not None:
                callback(record)
            if update in requested_checkpoints:
                checkpoints.append(
                    TrainingCheckpoint(
                        update=update,
                        elapsed_seconds=elapsed_seconds,
                        theta=model.flat_parameters().detach().cpu().clone(),
                        nce_log_normalizer=(
                            float(log_normalizer.detach())
                            if log_normalizer is not None
                            else math.nan
                        ),
                    )
                )

    _synchronize_device(model.device)
    elapsed = perf_counter() - started
    acceptance_rates = None
    if tempered is not None:
        acceptance_rates = tempered.diagnostics().acceptance_rate
    return ScalableTrainingResult(
        history=tuple(records),
        elapsed_seconds=elapsed,
        nce_log_normalizer=(
            float(log_normalizer.detach()) if log_normalizer is not None else math.nan
        ),
        final_negative_visible=final_negative,
        tempering_acceptance_rates=acceptance_rates,
        checkpoints=tuple(checkpoints),
    )
