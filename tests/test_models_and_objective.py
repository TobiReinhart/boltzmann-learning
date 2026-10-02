import itertools

import torch

from ip_rbm.models import RBM, ThreeBodyRBM
from ip_rbm.objectives import exact_kl, normalized_log_prob
from ip_rbm.states import all_binary_states, pair_features
from ip_rbm.targets import make_ip_target


def _explicit_hidden_sum_rbm(model: RBM, states: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    a, b, weights = model.unpack(theta)
    hidden_states = all_binary_states(model.n_hidden) if model.n_hidden else torch.zeros((1, 0))
    rows = []
    for visible in states:
        joint_logits = visible @ a + hidden_states @ b + (visible @ weights) @ hidden_states.T
        rows.append(torch.logsumexp(joint_logits, dim=0))
    return torch.stack(rows)


def _explicit_hidden_sum_3rbm(
    model: ThreeBodyRBM, states: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    a, b, weights, cubic = model.unpack(theta)
    hidden_states = all_binary_states(model.n_hidden) if model.n_hidden else torch.zeros((1, 0))
    features = pair_features(states, model.pair_mode)
    rows = []
    for visible, feature in zip(states, features, strict=True):
        hidden_fields = b + visible @ weights + feature @ cubic
        rows.append(visible @ a + torch.logsumexp(hidden_states @ hidden_fields, dim=0))
    return torch.stack(rows)


def test_rbm_free_energy_matches_explicit_hidden_sum() -> None:
    generator = torch.Generator().manual_seed(3)
    model = RBM(n_visible=4, n_hidden=2)
    states = all_binary_states(4)
    theta = torch.randn(model.n_parameters, generator=generator, dtype=torch.float64)
    assert torch.allclose(
        model.log_unnormalized(states, theta),
        _explicit_hidden_sum_rbm(model, states, theta),
        atol=1e-12,
    )


def test_3rbm_free_energy_matches_explicit_hidden_sum() -> None:
    generator = torch.Generator().manual_seed(4)
    model = ThreeBodyRBM(n_visible=4, n_hidden=2, pair_mode="cross")
    states = all_binary_states(4)
    theta = torch.randn(model.n_parameters, generator=generator, dtype=torch.float64)
    assert torch.allclose(
        model.log_unnormalized(states, theta),
        _explicit_hidden_sum_3rbm(model, states, theta),
        atol=1e-12,
    )


def test_3rbm_contains_corresponding_rbm_when_cubic_terms_are_zero() -> None:
    generator = torch.Generator().manual_seed(5)
    rbm = RBM(n_visible=4, n_hidden=3)
    three_body = ThreeBodyRBM(n_visible=4, n_hidden=3, pair_mode="cross")
    states = all_binary_states(4)
    rbm_theta = torch.randn(rbm.n_parameters, generator=generator, dtype=torch.float64)
    a, b, weights = rbm.unpack(rbm_theta)
    cubic = torch.zeros((three_body.n_pairs, three_body.n_hidden), dtype=torch.float64)
    three_body_theta = torch.cat([a, b, weights.reshape(-1), cubic.reshape(-1)])
    assert torch.allclose(
        normalized_log_prob(rbm.log_unnormalized(states, rbm_theta)),
        normalized_log_prob(three_body.log_unnormalized(states, three_body_theta)),
        atol=1e-12,
    )


def test_dormant_hidden_unit_preserves_rbm_distribution() -> None:
    generator = torch.Generator().manual_seed(6)
    smaller = RBM(n_visible=4, n_hidden=2)
    larger = RBM(n_visible=4, n_hidden=3)
    states = all_binary_states(4)
    theta = torch.randn(smaller.n_parameters, generator=generator, dtype=torch.float64)
    a, b, weights = smaller.unpack(theta)
    embedded = torch.cat(
        [
            a,
            torch.cat([b, torch.zeros(1)]),
            torch.cat([weights, torch.zeros((4, 1))], dim=1).reshape(-1),
        ]
    )
    assert torch.allclose(
        normalized_log_prob(smaller.log_unnormalized(states, theta)),
        normalized_log_prob(larger.log_unnormalized(states, embedded)),
        atol=1e-12,
    )


def test_exact_kl_gradient_matches_central_difference() -> None:
    target = make_ip_target(n_ip=2, beta=0.4)
    model = RBM(n_visible=4, n_hidden=1)
    theta = torch.linspace(-0.3, 0.4, model.n_parameters, dtype=torch.float64)
    theta.requires_grad_(True)
    loss = exact_kl(model, theta, target)
    (gradient,) = torch.autograd.grad(loss, theta)

    step = 1e-6
    for index in itertools.islice(range(model.n_parameters), 0, model.n_parameters, 2):
        direction = torch.zeros_like(theta)
        direction[index] = step
        finite_difference = (
            exact_kl(model, theta.detach() + direction, target)
            - exact_kl(model, theta.detach() - direction, target)
        ) / (2 * step)
        assert torch.allclose(gradient[index], finite_difference, atol=1e-8, rtol=1e-6)


def test_uniform_target_is_exactly_represented_at_zero_parameters() -> None:
    target = make_ip_target(n_ip=2, beta=0.0)
    model = RBM(n_visible=4, n_hidden=2)
    theta = torch.zeros(model.n_parameters, dtype=torch.float64)
    assert abs(float(exact_kl(model, theta, target))) < 1e-14
