"""Deterministic multi-start optimization of exact population objectives."""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import Bounds, minimize

from ip_rbm.models import ExactModel
from ip_rbm.objectives import exact_kl, ip_correlation
from ip_rbm.targets import ExactTarget

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OptimizationSettings:
    restarts: int = 20
    random_scale: float = 0.25
    maxiter: int = 1000
    maxfun: int | None = None
    ftol: float = 1e-13
    gtol: float = 1e-9
    seed: int = 1729

    def __post_init__(self) -> None:
        if self.restarts < 1:
            raise ValueError("restarts must be positive")
        if self.random_scale < 0:
            raise ValueError("random_scale must be nonnegative")
        if self.maxiter < 1:
            raise ValueError("maxiter must be positive")
        if self.maxfun is not None and self.maxfun < 1:
            raise ValueError("maxfun must be positive when specified")
        if self.ftol < 0 or self.gtol < 0:
            raise ValueError("optimizer tolerances must be nonnegative")

    @property
    def effective_maxfun(self) -> int:
        """Return the explicit function-evaluation budget passed to SciPy."""
        return self.maxfun if self.maxfun is not None else 2 * self.maxiter


@dataclass(frozen=True)
class RestartResult:
    restart: int
    initialization: str
    kl: float
    theta: FloatArray
    success: bool
    message: str
    iterations: int
    evaluations: int
    gradient_inf_norm: float
    projected_gradient_inf_norm: float
    elapsed_seconds: float


@dataclass(frozen=True)
class MultiStartResult:
    runs: tuple[RestartResult, ...]
    best_restart: int
    ip_correlation: float

    @property
    def best(self) -> RestartResult:
        return self.runs[self.best_restart]


RestartCallback = Callable[[RestartResult, float], None]


def _objective_and_gradient(
    theta_numpy: FloatArray,
    model: ExactModel,
    target: ExactTarget,
) -> tuple[float, FloatArray]:
    theta = torch.tensor(theta_numpy, dtype=torch.float64, requires_grad=True)
    loss = exact_kl(model, theta, target)
    (gradient,) = torch.autograd.grad(loss, theta)
    return float(loss.detach()), gradient.detach().numpy().astype(np.float64, copy=False)


def _projected_gradient_inf_norm(
    theta: FloatArray,
    gradient: FloatArray,
    weight_bound: float | None,
) -> float:
    """Return the KKT-relevant projected gradient norm for box constraints."""
    if weight_bound is None:
        return float(np.linalg.norm(gradient, ord=np.inf))
    projected = gradient.copy()
    tolerance = 1e-10 * max(1.0, weight_bound)
    at_lower = theta <= -weight_bound + tolerance
    at_upper = theta >= weight_bound - tolerance
    projected[at_lower & (gradient > 0)] = 0.0
    projected[at_upper & (gradient < 0)] = 0.0
    return float(np.linalg.norm(projected, ord=np.inf))


def optimize_exact(
    model: ExactModel,
    target: ExactTarget,
    *,
    weight_bound: float | None,
    settings: OptimizationSettings,
    initial_theta: FloatArray | None = None,
    perturb_scale: float | None = None,
    restart_callback: RestartCallback | None = None,
) -> MultiStartResult:
    """Run bounded L-BFGS-B from reproducible initial points.

    Without ``initial_theta``, restart zero uses zero and the remaining starts
    are random around zero. With ``initial_theta``, restart zero continues that
    point and the remaining starts perturb it. The returned best KL is an upper
    bound on the true global minimum.
    """
    if weight_bound is not None and weight_bound <= 0:
        raise ValueError("weight_bound must be positive")
    if target.states.device.type != "cpu" or target.states.dtype != torch.float64:
        raise ValueError("SciPy reference optimization requires a CPU float64 target")
    if perturb_scale is not None and perturb_scale < 0:
        raise ValueError("perturb_scale must be nonnegative")
    if initial_theta is None and perturb_scale is not None:
        raise ValueError("perturb_scale requires initial_theta")

    continuation: FloatArray | None = None
    if initial_theta is not None:
        continuation = np.asarray(initial_theta, dtype=np.float64)
        if continuation.shape != (model.n_parameters,):
            raise ValueError(
                f"initial_theta has shape {continuation.shape}; expected ({model.n_parameters},)"
            )
        if not np.all(np.isfinite(continuation)):
            raise ValueError("initial_theta contains non-finite values")
        if weight_bound is not None:
            tolerance = 1e-10 * max(1.0, weight_bound)
            if np.any(np.abs(continuation) > weight_bound + tolerance):
                raise ValueError("initial_theta violates the requested weight bound")
            continuation = np.clip(continuation, -weight_bound, weight_bound)
        continuation = continuation.copy()

    rng = np.random.default_rng(settings.seed)
    bounds = None if weight_bound is None else Bounds(-weight_bound, weight_bound)
    runs: list[RestartResult] = []

    for restart in range(settings.restarts):
        if continuation is not None and restart == 0:
            initial = continuation.copy()
            initialization = "continued"
        elif continuation is not None:
            scale = settings.random_scale if perturb_scale is None else perturb_scale
            initial = continuation + rng.normal(0.0, scale, model.n_parameters)
            if weight_bound is not None:
                initial = np.clip(initial, -weight_bound, weight_bound)
            initial = initial.astype(np.float64)
            initialization = "continued_perturbed"
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
            _objective_and_gradient,
            initial,
            args=(model, target),
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
        gradient = np.asarray(result.jac, dtype=np.float64)
        run = RestartResult(
            restart=restart,
            initialization=initialization,
            kl=float(result.fun),
            theta=np.asarray(result.x, dtype=np.float64),
            success=bool(result.success),
            message=str(result.message),
            iterations=int(result.nit),
            evaluations=int(result.nfev),
            gradient_inf_norm=float(np.linalg.norm(gradient, ord=np.inf)),
            projected_gradient_inf_norm=_projected_gradient_inf_norm(
                np.asarray(result.x, dtype=np.float64), gradient, weight_bound
            ),
            elapsed_seconds=elapsed,
        )
        runs.append(run)
        if restart_callback is not None:
            restart_callback(run, min(completed.kl for completed in runs))

    best_restart = min(range(len(runs)), key=lambda index: runs[index].kl)
    best_theta = torch.from_numpy(runs[best_restart].theta)
    correlation = float(ip_correlation(model, best_theta, target))
    return MultiStartResult(
        runs=tuple(runs),
        best_restart=best_restart,
        ip_correlation=correlation,
    )
