import itertools

import numpy as np
import torch

from ip_rbm.learning import (
    ExactDataset,
    empirical_nll,
    optimize_exact_mle,
    sample_exact_dataset,
    sample_nested_exact_datasets,
)
from ip_rbm.models import RBM
from ip_rbm.objectives import exact_kl
from ip_rbm.optimization import OptimizationSettings
from ip_rbm.targets import make_ip_target


def test_exact_sampling_is_reproducible_and_nested() -> None:
    target = make_ip_target(n_ip=2, beta=0.7)

    first = sample_nested_exact_datasets(target, [10, 25], seed=42)
    second = sample_nested_exact_datasets(target, [10, 25], seed=42)

    assert torch.equal(first[10].counts, second[10].counts)
    assert torch.equal(first[25].counts, second[25].counts)
    assert torch.all(first[10].counts <= first[25].counts)
    assert int(first[10].counts.sum()) == 10
    assert int(first[25].counts.sum()) == 25


def test_independent_exact_sample_has_requested_shape_and_size() -> None:
    target = make_ip_target(n_ip=3, beta=1.0)
    dataset = sample_exact_dataset(target, sample_size=100, seed=7)

    assert dataset.counts.shape == target.prob.shape
    assert int(dataset.counts.sum()) == 100
    assert torch.allclose(dataset.empirical_prob, dataset.counts.to(torch.float64) / 100)


def test_empirical_nll_gradient_matches_central_difference() -> None:
    target = make_ip_target(n_ip=2, beta=0.4)
    dataset = sample_exact_dataset(target, sample_size=50, seed=8)
    model = RBM(n_visible=4, n_hidden=1)
    theta = torch.linspace(-0.3, 0.4, model.n_parameters, dtype=torch.float64)
    theta.requires_grad_(True)
    loss = empirical_nll(model, theta, target, dataset)
    (gradient,) = torch.autograd.grad(loss, theta)

    step = 1e-6
    for index in itertools.islice(range(model.n_parameters), 0, model.n_parameters, 2):
        direction = torch.zeros_like(theta)
        direction[index] = step
        finite_difference = (
            empirical_nll(model, theta.detach() + direction, target, dataset)
            - empirical_nll(model, theta.detach() - direction, target, dataset)
        ) / (2 * step)
        assert torch.allclose(gradient[index], finite_difference, atol=1e-8, rtol=1e-6)


def test_exact_mle_recovers_an_exactly_uniform_empirical_distribution() -> None:
    target = make_ip_target(n_ip=1, beta=0.0)
    counts = torch.ones(4, dtype=torch.int64)
    dataset = ExactDataset(
        counts=counts,
        empirical_prob=counts.to(torch.float64) / 4,
        sample_size=4,
        seed=1,
    )
    model = RBM(n_visible=2, n_hidden=1)

    result = optimize_exact_mle(
        model,
        target,
        dataset,
        weight_bound=1.0,
        settings=OptimizationSettings(restarts=2, maxiter=50, seed=2),
    )

    assert result.best.empirical_kl < 1e-12
    assert result.best.population_kl < 1e-12
    assert result.best_restart == int(np.argmin([run.empirical_nll for run in result.runs]))
    assert float(exact_kl(model, torch.from_numpy(result.best.theta), target)) < 1e-12
