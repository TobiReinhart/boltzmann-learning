import math

import pytest
import torch

from ip_rbm.scalable_data import AnchoredTeacherTarget
from ip_rbm.scalable_evaluation import mean_log_pseudolikelihood, target_score_metrics
from ip_rbm.states import all_binary_states
from ip_rbm.trainable_models import TrainableThreeBodyRBM


def _load_lifted_teacher(
    model: TrainableThreeBodyRBM,
    target: AnchoredTeacherTarget,
) -> None:
    n_ip = target.n_ip
    dense_hidden = target.teacher_hidden
    with torch.no_grad():
        model.visible_bias.zero_()
        model.hidden_bias.zero_()
        model.weights.zero_()
        model.cubic.zero_()
        cubic = model.cubic.reshape(n_ip, n_ip, model.n_hidden)
        teacher_weights = target.teacher_weights.to(model.dtype)
        model.hidden_bias[:dense_hidden] = target.teacher_bias.to(model.dtype) - target.center * (
            teacher_weights.sum(dim=1)
        )
        diagonal = torch.arange(n_ip)
        cubic[diagonal, diagonal, :dense_hidden] = teacher_weights.T
        anchor_value = math.log(2.0) + target.anchor_weights
        anchor_coupling = torch.log(torch.expm1(anchor_value)).to(model.dtype)
        cubic[diagonal, diagonal, dense_hidden:] = torch.diag(anchor_coupling)


def test_anchored_teacher_normalization_and_exact_sampling() -> None:
    target = AnchoredTeacherTarget(
        n_ip=3,
        teacher_hidden=3,
        anchor_strength=0.25,
        interaction_scale=1.75,
        bias_span=0.75,
        teacher_seed=101,
    )
    visible = all_binary_states(6)
    log_prob = target.log_prob(visible)

    assert torch.logsumexp(log_prob, dim=0) == pytest.approx(0.0, abs=1e-12)
    assert target.entropy == pytest.approx(float(-torch.sum(torch.exp(log_prob) * log_prob)))

    samples = target.sample(
        50000,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(71),
    )
    products = samples[:, :3] * samples[:, 3:]
    indices = (products * torch.tensor([4.0, 2.0, 1.0])).sum(dim=1).to(torch.int64)
    empirical = torch.bincount(indices, minlength=8).to(torch.float64) / samples.shape[0]
    assert torch.max(torch.abs(empirical - torch.exp(target.product_log_prob))) < 0.015


def test_anchored_teacher_has_exact_linear_width_3rbm_lift() -> None:
    target = AnchoredTeacherTarget(
        n_ip=3,
        teacher_hidden=3,
        anchor_strength=0.25,
        interaction_scale=1.75,
        bias_span=0.75,
        teacher_seed=20260901,
    )
    model = TrainableThreeBodyRBM(6, 6, pair_mode="cross", init_std=0.0, dtype=torch.float64)
    _load_lifted_teacher(model, target)
    states = all_binary_states(6)
    difference = model.log_unnormalized(states) - target.log_unnormalized(states)

    assert torch.max(torch.abs(difference - difference.mean())) < 1e-12
    metrics = target_score_metrics(model, target, states)
    assert metrics.target_score_rmse < 1e-12
    assert metrics.target_score_correlation == pytest.approx(1.0, abs=1e-12)


def test_anchored_teacher_seed_is_reproducible() -> None:
    first = AnchoredTeacherTarget(8, 8, 0.25, 1.75, 0.75, 44)
    second = AnchoredTeacherTarget(8, 8, 0.25, 1.75, 0.75, 44)

    assert torch.equal(first.anchor_weights, second.anchor_weights)
    assert torch.equal(first.teacher_weights, second.teacher_weights)
    assert torch.equal(first.teacher_bias, second.teacher_bias)
    normalized = first.teacher_weights / first.interaction_scale * math.sqrt(first.n_ip)
    correlations = normalized @ normalized.T / first.n_ip
    off_diagonal = correlations[~torch.eye(first.teacher_hidden, dtype=torch.bool)]
    assert torch.max(torch.abs(off_diagonal)) < 1.0
    assert torch.linalg.matrix_rank(normalized) == first.n_ip - 1


def test_target_score_metrics_uses_separate_pseudolikelihood_states() -> None:
    target = AnchoredTeacherTarget(3, 3, 0.25, 1.75, 0.75, 44)
    model = TrainableThreeBodyRBM(6, 6, pair_mode="cross", init_std=0.0, dtype=torch.float64)
    _load_lifted_teacher(model, target)
    score_states = all_binary_states(6)
    pseudolikelihood_states = target.sample(
        17,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(19),
    )

    metrics = target_score_metrics(
        model,
        target,
        score_states,
        pseudolikelihood_states=pseudolikelihood_states,
    )

    assert metrics.mean_log_pseudolikelihood == pytest.approx(
        float(mean_log_pseudolikelihood(model, pseudolikelihood_states).detach())
    )
