import math

import pytest
import torch

from ip_rbm.objectives import model_probability
from ip_rbm.scalable_data import IPTarget
from ip_rbm.scalable_evaluation import ais_population_metrics, exact_overlap_metrics
from ip_rbm.scalable_sampling import estimate_log_partition_ais
from ip_rbm.states import all_binary_states
from ip_rbm.targets import make_ip_target
from ip_rbm.trainable_models import TrainableRBM, TrainableThreeBodyRBM


@pytest.mark.parametrize("kind", ["rbm", "3rbm_cross", "3rbm_all"])
def test_trainable_scores_match_exact_parameter_order(kind: str) -> None:
    if kind == "rbm":
        model = TrainableRBM(4, 3, dtype=torch.float64)
    elif kind == "3rbm_cross":
        model = TrainableThreeBodyRBM(4, 3, pair_mode="cross", dtype=torch.float64)
    else:
        model = TrainableThreeBodyRBM(4, 3, pair_mode="all", dtype=torch.float64)
    theta = torch.linspace(-0.4, 0.5, model.n_parameters, dtype=torch.float64)
    model.load_flat_parameters(theta)
    states = all_binary_states(4)

    actual = model.log_unnormalized(states)
    expected = model.exact_model().log_unnormalized(states, theta)

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert torch.equal(model.flat_parameters(), theta)


@pytest.mark.parametrize(
    "model",
    [
        TrainableRBM(2, 1, dtype=torch.float64),
        TrainableThreeBodyRBM(2, 1, pair_mode="cross", dtype=torch.float64),
        TrainableThreeBodyRBM(2, 1, pair_mode="all", dtype=torch.float64),
    ],
)
def test_gibbs_sampler_recovers_exact_small_model(model: TrainableRBM) -> None:
    theta = torch.linspace(-0.35, 0.45, model.n_parameters, dtype=torch.float64)
    model.load_flat_parameters(theta)
    generator = torch.Generator().manual_seed(44)
    visible = torch.randint(0, 2, (6000, 2), generator=generator).to(torch.float64)

    samples = model.gibbs_visible(visible, 100, generator=generator)
    indices = (2 * samples[:, 0] + samples[:, 1]).to(torch.int64)
    empirical = torch.bincount(indices, minlength=4).to(torch.float64) / samples.shape[0]
    target = make_ip_target(1, 0.0)
    exact = model_probability(model.exact_model(), theta, target)

    assert torch.max(torch.abs(empirical - exact)) < 0.035


def test_analytic_ip_target_matches_enumeration_and_sampling() -> None:
    target = IPTarget(3, 0.7)
    exact = make_ip_target(3, 0.7)

    assert math.isclose(target.log_partition, float(torch.logsumexp(0.7 * exact.ip_sign, 0)))
    assert math.isclose(target.entropy, float(-torch.sum(exact.prob * exact.log_prob)))
    assert torch.allclose(target.log_prob(exact.states), exact.log_prob)

    samples = target.sample(
        30000,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(91),
    )
    assert abs(float(target.sign(samples).mean()) - target.mean_sign) < 0.02


def test_ais_is_exact_for_zero_parameters() -> None:
    model = TrainableThreeBodyRBM(4, 2, init_std=0.0, dtype=torch.float64)
    result = estimate_log_partition_ais(
        model,
        n_particles=32,
        n_intermediate=8,
        generator=torch.Generator().manual_seed(5),
    )

    assert result.log_partition == pytest.approx((4 + 2) * math.log(2), abs=1e-12)
    assert result.effective_sample_size == pytest.approx(32.0)


def test_ais_population_kl_uses_analytic_target_entropy() -> None:
    model = TrainableRBM(4, 2, init_std=0.0, dtype=torch.float64)
    target = IPTarget(2, 1.0)
    target_states = target.sample(
        16,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(12),
    )

    result = ais_population_metrics(
        model,
        target,
        target_states,
        n_particles=16,
        n_intermediate=8,
        generator=torch.Generator().manual_seed(13),
    )

    expected_uniform_kl = target.n_visible * math.log(2.0) - target.entropy
    assert result.population_kl == pytest.approx(expected_uniform_kl, abs=1e-12)


def test_ais_agrees_with_exact_partition_for_nonzero_model() -> None:
    model = TrainableThreeBodyRBM(4, 2, dtype=torch.float64)
    model.load_flat_parameters(torch.linspace(-0.3, 0.4, model.n_parameters, dtype=torch.float64))
    exact = float(torch.logsumexp(model.log_unnormalized(all_binary_states(4)), dim=0).detach())

    result = estimate_log_partition_ais(
        model,
        n_particles=500,
        n_intermediate=50,
        generator=torch.Generator().manual_seed(8),
    )

    assert result.log_partition == pytest.approx(exact, abs=0.02)


def test_exact_overlap_uses_existing_reference_path() -> None:
    model = TrainableRBM(4, 2, init_std=0.0, dtype=torch.float32)
    metrics = exact_overlap_metrics(model, IPTarget(2, 0.0))

    assert metrics.population_kl == pytest.approx(0.0, abs=1e-12)
    assert metrics.ip_correlation == pytest.approx(0.25, abs=1e-12)


@pytest.mark.parametrize(
    "model",
    [
        TrainableRBM(4, 2, dtype=torch.float64),
        TrainableThreeBodyRBM(4, 2, dtype=torch.float64),
    ],
)
def test_phase_decomposition_matches_exact_likelihood_gradient(model: TrainableRBM) -> None:
    theta = torch.linspace(-0.2, 0.3, model.n_parameters, dtype=torch.float64)
    model.load_flat_parameters(theta)
    states = all_binary_states(4)
    data = states[[1, 4, 7, 13]]

    scores = model.log_unnormalized(states)
    exact_loss = torch.logsumexp(scores, dim=0) - model.log_unnormalized(data).mean()
    exact_gradient = torch.autograd.grad(exact_loss, tuple(model.parameters()))

    scores = model.log_unnormalized(states)
    probability = torch.softmax(scores.detach(), dim=0)
    phase_loss = (probability * scores).sum() - model.log_unnormalized(data).mean()
    phase_gradient = torch.autograd.grad(phase_loss, tuple(model.parameters()))

    for exact, phase in zip(exact_gradient, phase_gradient, strict=True):
        assert torch.allclose(exact, phase, atol=1e-12, rtol=1e-12)
