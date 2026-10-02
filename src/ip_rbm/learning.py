"""Exact finite-sample maximum-likelihood learning on enumerated state spaces."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import Bounds, minimize
from torch import Tensor

from ip_rbm.models import ExactModel
from ip_rbm.objectives import (
    cross_entropy_from_log_prob,
    exact_kl,
    ip_correlation,
    normalized_log_prob,
)
from ip_rbm.optimization import OptimizationSettings
from ip_rbm.targets import ExactTarget

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ExactDataset:
    """Empirical distribution represented by counts on an exact state table."""

    counts: Tensor
    empirical_prob: Tensor
    sample_size: int
    seed: int

    def __post_init__(self) -> None:
        if self.sample_size < 1:
            raise ValueError("sample_size must be positive")
        if self.seed < 0:
            raise ValueError("dataset seed must be nonnegative")
        if self.counts.ndim != 1 or self.empirical_prob.ndim != 1:
            raise ValueError("dataset counts and probabilities must be vectors")
        if self.counts.shape != self.empirical_prob.shape:
            raise ValueError("dataset counts and probabilities must have equal shapes")
        if self.counts.dtype != torch.int64:
            raise ValueError("dataset counts must use torch.int64")
        if self.empirical_prob.dtype != torch.float64:
            raise ValueError("dataset probabilities must use torch.float64")
        if self.counts.device != self.empirical_prob.device:
            raise ValueError("dataset counts and probabilities must use the same device")
        if int(self.counts.sum()) != self.sample_size:
            raise ValueError("dataset counts do not sum to sample_size")
        if torch.any(self.counts < 0):
            raise ValueError("dataset counts must be nonnegative")
        expected_prob = self.counts.to(dtype=torch.float64) / self.sample_size
        if not torch.equal(self.empirical_prob, expected_prob):
            raise ValueError("empirical probabilities must equal counts / sample_size")
        if not torch.allclose(
            self.empirical_prob.sum(),
            torch.tensor(1.0, dtype=torch.float64, device=self.empirical_prob.device),
        ):
            raise ValueError("empirical probabilities must sum to one")

    @property
    def entropy(self) -> float:
        """Return the plug-in entropy of the empirical distribution."""
        positive = self.empirical_prob > 0
        probabilities = self.empirical_prob[positive]
        return float(-torch.sum(probabilities * torch.log(probabilities)))


def _dataset_from_counts(counts: NDArray[np.int64], sample_size: int, seed: int) -> ExactDataset:
    count_tensor = torch.from_numpy(np.asarray(counts, dtype=np.int64).copy())
    empirical_prob = count_tensor.to(dtype=torch.float64) / sample_size
    return ExactDataset(
        counts=count_tensor,
        empirical_prob=empirical_prob,
        sample_size=sample_size,
        seed=seed,
    )


def sample_exact_dataset(target: ExactTarget, sample_size: int, seed: int) -> ExactDataset:
    """Draw an exact i.i.d. dataset from a fully enumerated target distribution."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    probabilities = target.prob.detach().cpu().numpy()
    probabilities = probabilities / probabilities.sum()
    rng = np.random.default_rng(seed)
    indices = rng.choice(probabilities.size, size=sample_size, replace=True, p=probabilities)
    counts = np.bincount(indices, minlength=probabilities.size).astype(np.int64, copy=False)
    return _dataset_from_counts(counts, sample_size, seed)


def sample_nested_exact_datasets(
    target: ExactTarget,
    sample_sizes: Sequence[int],
    seed: int,
) -> dict[int, ExactDataset]:
    """Draw nested i.i.d. datasets, one for each requested prefix length."""
    sizes = sorted(int(size) for size in sample_sizes)
    if not sizes or sizes[0] < 1:
        raise ValueError("sample_sizes must contain positive integers")
    if len(set(sizes)) != len(sizes):
        raise ValueError("sample_sizes must not contain duplicates")

    probabilities = target.prob.detach().cpu().numpy()
    probabilities = probabilities / probabilities.sum()
    rng = np.random.default_rng(seed)
    indices = rng.choice(probabilities.size, size=sizes[-1], replace=True, p=probabilities)
    datasets: dict[int, ExactDataset] = {}
    for sample_size in sizes:
        counts = np.bincount(indices[:sample_size], minlength=probabilities.size).astype(
            np.int64, copy=False
        )
        datasets[sample_size] = _dataset_from_counts(counts, sample_size, seed)
    return datasets


def empirical_nll(
    model: ExactModel,
    theta: Tensor,
    target: ExactTarget,
    dataset: ExactDataset,
) -> Tensor:
    """Return exact empirical negative log-likelihood per observation."""
    if model.n_visible != target.states.shape[1]:
        raise ValueError("model and target visible dimensions differ")
    if dataset.empirical_prob.shape != target.prob.shape:
        raise ValueError("dataset and target state spaces differ")
    logits = model.log_unnormalized(target.states, theta)
    return cross_entropy_from_log_prob(dataset.empirical_prob, normalized_log_prob(logits))


@dataclass(frozen=True)
class LearningRestartResult:
    """Diagnostics for one local empirical-MLE optimization."""

    restart: int
    initialization: str
    empirical_nll: float
    empirical_kl: float
    population_nll: float
    population_kl: float
    validation_nll: float
    validation_kl: float
    theta: FloatArray
    success: bool
    message: str
    iterations: int
    evaluations: int
    gradient_inf_norm: float
    projected_gradient_inf_norm: float
    elapsed_seconds: float


@dataclass(frozen=True)
class LearningMultiStartResult:
    """All restarts for one dataset, selected by lowest empirical NLL."""

    runs: tuple[LearningRestartResult, ...]
    best_restart: int
    ip_correlation: float

    @property
    def best(self) -> LearningRestartResult:
        return self.runs[self.best_restart]


LearningRestartCallback = Callable[[LearningRestartResult, float], None]


def _empirical_objective_and_gradient(
    theta_numpy: FloatArray,
    model: ExactModel,
    target: ExactTarget,
    dataset: ExactDataset,
) -> tuple[float, FloatArray]:
    theta = torch.tensor(theta_numpy, dtype=torch.float64, requires_grad=True)
    loss = empirical_nll(model, theta, target, dataset)
    (gradient,) = torch.autograd.grad(loss, theta)
    return float(loss.detach()), gradient.detach().numpy().astype(np.float64, copy=False)


def _projected_gradient_inf_norm(
    theta: FloatArray,
    gradient: FloatArray,
    weight_bound: float | None,
) -> float:
    if weight_bound is None:
        return float(np.linalg.norm(gradient, ord=np.inf))
    projected = gradient.copy()
    tolerance = 1e-10 * max(1.0, weight_bound)
    at_lower = theta <= -weight_bound + tolerance
    at_upper = theta >= weight_bound - tolerance
    projected[at_lower & (gradient > 0)] = 0.0
    projected[at_upper & (gradient < 0)] = 0.0
    return float(np.linalg.norm(projected, ord=np.inf))


def _validate_initial_theta(
    initial_theta: FloatArray | None,
    model: ExactModel,
    weight_bound: float | None,
) -> FloatArray | None:
    if initial_theta is None:
        return None
    initial = np.asarray(initial_theta, dtype=np.float64)
    if initial.shape != (model.n_parameters,):
        raise ValueError(
            f"initial_theta has shape {initial.shape}; expected ({model.n_parameters},)"
        )
    if not np.all(np.isfinite(initial)):
        raise ValueError("initial_theta contains non-finite values")
    if weight_bound is not None:
        tolerance = 1e-10 * max(1.0, weight_bound)
        if np.any(np.abs(initial) > weight_bound + tolerance):
            raise ValueError("initial_theta violates the requested weight bound")
        initial = np.clip(initial, -weight_bound, weight_bound)
    return initial.copy()


def optimize_exact_mle(
    model: ExactModel,
    target: ExactTarget,
    train_dataset: ExactDataset,
    *,
    weight_bound: float | None,
    settings: OptimizationSettings,
    validation_dataset: ExactDataset | None = None,
    initial_theta: FloatArray | None = None,
    perturb_scale: float | None = None,
    restart_callback: LearningRestartCallback | None = None,
) -> LearningMultiStartResult:
    """Optimize exact empirical NLL with reproducible multi-start L-BFGS-B.

    The partition function and every reported population quantity are evaluated
    by complete enumeration. The best restart is selected only by training NLL.
    """
    if weight_bound is not None and weight_bound <= 0:
        raise ValueError("weight_bound must be positive")
    if target.states.device.type != "cpu" or target.states.dtype != torch.float64:
        raise ValueError("SciPy reference optimization requires a CPU float64 target")
    if train_dataset.empirical_prob.shape != target.prob.shape:
        raise ValueError("training dataset and target state spaces differ")
    if (
        validation_dataset is not None
        and validation_dataset.empirical_prob.shape != target.prob.shape
    ):
        raise ValueError("validation dataset and target state spaces differ")
    if perturb_scale is not None and perturb_scale < 0:
        raise ValueError("perturb_scale must be nonnegative")
    if initial_theta is None and perturb_scale is not None:
        raise ValueError("perturb_scale requires initial_theta")

    reference = _validate_initial_theta(initial_theta, model, weight_bound)
    rng = np.random.default_rng(settings.seed)
    bounds = None if weight_bound is None else Bounds(-weight_bound, weight_bound)
    runs: list[LearningRestartResult] = []

    for restart in range(settings.restarts):
        if reference is not None and restart == 0:
            initial = reference.copy()
            initialization = "population_reference"
        elif reference is not None:
            scale = settings.random_scale if perturb_scale is None else perturb_scale
            initial = reference + rng.normal(0.0, scale, model.n_parameters)
            if weight_bound is not None:
                initial = np.clip(initial, -weight_bound, weight_bound)
            initial = initial.astype(np.float64)
            initialization = "population_reference_perturbed"
        elif restart == 0:
            initial = np.zeros(model.n_parameters, dtype=np.float64)
            initialization = "zero"
        else:
            initial = rng.normal(0.0, settings.random_scale, model.n_parameters)
            if weight_bound is not None:
                initial = np.clip(initial, -weight_bound, weight_bound)
            initial = initial.astype(np.float64)
            initialization = "random"

        started = perf_counter()
        result = minimize(
            _empirical_objective_and_gradient,
            initial,
            args=(model, target, train_dataset),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": settings.maxiter,
                "maxfun": settings.effective_maxfun,
                "ftol": settings.ftol,
                "gtol": settings.gtol,
                "maxls": 50,
            },
        )
        elapsed = perf_counter() - started
        fitted_theta = np.asarray(result.x, dtype=np.float64)
        theta_tensor = torch.from_numpy(fitted_theta)
        logits = model.log_unnormalized(target.states, theta_tensor)
        model_log_prob = normalized_log_prob(logits)
        train_nll = float(cross_entropy_from_log_prob(train_dataset.empirical_prob, model_log_prob))
        population_nll = float(cross_entropy_from_log_prob(target.prob, model_log_prob))
        if validation_dataset is None:
            validation_nll = float("nan")
            validation_kl = float("nan")
        else:
            validation_nll = float(
                cross_entropy_from_log_prob(validation_dataset.empirical_prob, model_log_prob)
            )
            validation_kl = validation_nll - validation_dataset.entropy

        gradient = np.asarray(result.jac, dtype=np.float64)
        run = LearningRestartResult(
            restart=restart,
            initialization=initialization,
            empirical_nll=train_nll,
            empirical_kl=train_nll - train_dataset.entropy,
            population_nll=population_nll,
            population_kl=float(exact_kl(model, theta_tensor, target)),
            validation_nll=validation_nll,
            validation_kl=validation_kl,
            theta=fitted_theta,
            success=bool(result.success),
            message=str(result.message),
            iterations=int(result.nit),
            evaluations=int(result.nfev),
            gradient_inf_norm=float(np.linalg.norm(gradient, ord=np.inf)),
            projected_gradient_inf_norm=_projected_gradient_inf_norm(
                fitted_theta, gradient, weight_bound
            ),
            elapsed_seconds=elapsed,
        )
        runs.append(run)
        if restart_callback is not None:
            restart_callback(run, min(completed.empirical_nll for completed in runs))

    best_restart = min(range(len(runs)), key=lambda index: runs[index].empirical_nll)
    best_theta = torch.from_numpy(runs[best_restart].theta)
    return LearningMultiStartResult(
        runs=tuple(runs),
        best_restart=best_restart,
        ip_correlation=float(ip_correlation(model, best_theta, target)),
    )
