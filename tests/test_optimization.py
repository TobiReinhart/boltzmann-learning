import numpy as np
import pytest
import torch

from ip_rbm.models import RBM
from ip_rbm.optimization import OptimizationSettings, optimize_exact
from ip_rbm.targets import make_ip_target


def test_optimizer_recovers_uniform_target() -> None:
    target = make_ip_target(n_ip=2, beta=0.0)
    model = RBM(n_visible=4, n_hidden=1)
    result = optimize_exact(
        model,
        target,
        weight_bound=1.0,
        settings=OptimizationSettings(restarts=2, maxiter=50, seed=1),
    )
    assert result.best.kl < 1e-12
    assert torch.isfinite(torch.tensor(result.best.gradient_inf_norm))


def test_optimizer_reports_completed_restarts() -> None:
    target = make_ip_target(n_ip=1, beta=0.0)
    model = RBM(n_visible=2, n_hidden=1)
    reported: list[tuple[int, float]] = []

    optimize_exact(
        model,
        target,
        weight_bound=1.0,
        settings=OptimizationSettings(restarts=3, maxiter=5, seed=1),
        restart_callback=lambda run, best_kl: reported.append((run.restart, best_kl)),
    )

    assert [restart for restart, _ in reported] == [0, 1, 2]
    assert reported[-1][1] == min(best_kl for _, best_kl in reported)


def test_function_evaluation_budget_is_explicit() -> None:
    assert OptimizationSettings(maxiter=123).effective_maxfun == 246
    assert OptimizationSettings(maxiter=123, maxfun=321).effective_maxfun == 321


def test_optimizer_continues_and_perturbs_saved_parameters() -> None:
    target = make_ip_target(n_ip=1, beta=0.0)
    model = RBM(n_visible=2, n_hidden=1)
    initial = np.zeros(model.n_parameters, dtype=np.float64)

    result = optimize_exact(
        model,
        target,
        weight_bound=1.0,
        settings=OptimizationSettings(restarts=3, maxiter=5, seed=1),
        initial_theta=initial,
        perturb_scale=0.01,
    )

    assert [run.initialization for run in result.runs] == [
        "continued",
        "continued_perturbed",
        "continued_perturbed",
    ]


def test_optimizer_rejects_incompatible_continuation() -> None:
    target = make_ip_target(n_ip=1, beta=0.0)
    model = RBM(n_visible=2, n_hidden=1)

    with pytest.raises(ValueError, match="initial_theta has shape"):
        optimize_exact(
            model,
            target,
            weight_bound=1.0,
            settings=OptimizationSettings(restarts=1, maxiter=5),
            initial_theta=np.zeros(model.n_parameters + 1),
        )


def test_optimizer_supports_unbounded_parameters() -> None:
    target = make_ip_target(n_ip=1, beta=0.0)
    model = RBM(n_visible=2, n_hidden=1)

    result = optimize_exact(
        model,
        target,
        weight_bound=None,
        settings=OptimizationSettings(restarts=2, maxiter=10, seed=1),
    )

    assert result.best.kl < 1e-12
    assert result.best.projected_gradient_inf_norm == result.best.gradient_inf_norm
