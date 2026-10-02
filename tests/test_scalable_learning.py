import math

import pytest
import torch

from ip_rbm.scalable_data import IPTarget
from ip_rbm.scalable_learning import (
    ScalableTrainingSettings,
    _score_regression_objective,
    train_scalable,
)
from ip_rbm.states import all_binary_states
from ip_rbm.trainable_models import TrainableRBM, TrainableThreeBodyRBM


@pytest.mark.parametrize(
    "algorithm",
    ["cd", "pcd", "tempered_pcd", "pseudolikelihood", "nce", "score_regression"],
)
@pytest.mark.parametrize("kind", ["rbm", "3rbm"])
def test_all_scalable_algorithms_run_and_respect_bounds(algorithm: str, kind: str) -> None:
    target = IPTarget(2, 0.8)
    states = target.sample(
        64,
        generator=torch.Generator().manual_seed(12),
    )
    model = TrainableRBM(4, 3) if kind == "rbm" else TrainableThreeBodyRBM(4, 2)
    settings = ScalableTrainingSettings(
        algorithm=algorithm,  # type: ignore[arg-type]
        updates=4,
        batch_size=16,
        learning_rate=0.01,
        gibbs_steps=2,
        persistent_chains=12,
        weight_bound=0.2,
        tempering_replicas=3,
        record_every=2,
        seed=7,
    )

    result = train_scalable(
        model,
        states,
        settings,
        target_score_function=(
            target.log_unnormalized if algorithm == "score_regression" else None
        ),
    )

    assert [record.update for record in result.history] == [1, 2, 4]
    assert all(math.isfinite(record.objective) for record in result.history)
    assert max(float(parameter.detach().abs().max()) for parameter in model.parameters()) <= 0.2
    if algorithm == "nce":
        assert math.isfinite(result.nce_log_normalizer)
    else:
        assert math.isnan(result.nce_log_normalizer)
    if algorithm in {"cd", "pcd", "tempered_pcd"}:
        assert result.final_negative_visible is not None


def test_score_regression_is_invariant_to_target_offset() -> None:
    states = all_binary_states(4, dtype=torch.float64)
    model = TrainableRBM(4, 3, init_std=0.02, dtype=torch.float64)

    def target_score(visible: torch.Tensor) -> torch.Tensor:
        return 0.7 * visible[:, 0] - 0.4 * visible[:, 1] + visible[:, 2] * visible[:, 3]

    baseline = _score_regression_objective(model, states, target_score)
    shifted = _score_regression_objective(model, states, lambda visible: target_score(visible) + 17)

    assert float(shifted.detach()) == pytest.approx(float(baseline.detach()), abs=1e-12)


def test_score_regression_learns_a_simple_target_without_negative_samples() -> None:
    states = all_binary_states(2, dtype=torch.float64)
    model = TrainableRBM(2, 1, init_std=0.0, dtype=torch.float64)

    def target_score(visible: torch.Tensor) -> torch.Tensor:
        return 0.7 * visible[:, 0] - 0.4 * visible[:, 1]

    initial = float(_score_regression_objective(model, states, target_score).detach())
    settings = ScalableTrainingSettings(
        algorithm="score_regression",
        updates=100,
        batch_size=4,
        learning_rate=0.05,
        record_every=100,
        seed=123,
    )

    result = train_scalable(model, states, settings, target_score_function=target_score)
    final = float(_score_regression_objective(model, states, target_score).detach())

    assert final < initial * 1e-3
    assert result.final_negative_visible is None


def test_scalable_training_is_reproducible() -> None:
    states = IPTarget(2, 1.0).sample(
        64,
        generator=torch.Generator().manual_seed(99),
    )
    settings = ScalableTrainingSettings(
        algorithm="pcd",
        updates=5,
        batch_size=16,
        learning_rate=0.01,
        seed=123,
    )
    first = TrainableRBM(4, 3, init_std=0.0)
    second = TrainableRBM(4, 3, init_std=0.0)

    train_scalable(first, states, settings)
    train_scalable(second, states, settings)

    assert torch.equal(first.flat_parameters(), second.flat_parameters())


def test_scalable_training_captures_requested_checkpoints() -> None:
    states = IPTarget(2, 1.0).sample(
        32,
        generator=torch.Generator().manual_seed(99),
    )
    model = TrainableRBM(4, 3, init_std=0.0)
    settings = ScalableTrainingSettings(
        algorithm="cd",
        updates=4,
        batch_size=8,
        learning_rate=0.01,
        minibatch_seed=123,
        sampler_seed=456,
    )

    result = train_scalable(model, states, settings, checkpoint_updates=(2, 4))

    assert [checkpoint.update for checkpoint in result.checkpoints] == [2, 4]
    assert all(checkpoint.theta.device.type == "cpu" for checkpoint in result.checkpoints)
    assert torch.equal(result.checkpoints[-1].theta, model.flat_parameters().detach().cpu())
    assert result.checkpoints[0].elapsed_seconds <= result.checkpoints[1].elapsed_seconds
