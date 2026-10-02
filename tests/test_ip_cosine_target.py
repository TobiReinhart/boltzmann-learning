import math
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from ip_rbm import scalable_experiment
from ip_rbm.plotting import plot_scalable_objective_curves
from ip_rbm.scalable_data import BlockIPTarget, CountCosineTarget, IPCosineTarget, IPTarget
from ip_rbm.scalable_evaluation import target_score_metrics
from ip_rbm.scalable_experiment import run_scalable_experiment
from ip_rbm.states import all_binary_states


class _KnownScoreModel:
    def __init__(self, target: IPCosineTarget, *, include_ip: bool) -> None:
        self.target = target
        self.include_ip = include_ip
        self.n_visible = target.n_visible
        self.device = torch.device("cpu")
        self.dtype = torch.float64

    def log_unnormalized(self, visible: torch.Tensor) -> torch.Tensor:
        cosine = self.target.cosine_component(visible)
        score = self.target.beta * self.target.rho * cosine
        if self.include_ip:
            score = score + self.target.beta * self.target.sign(visible)
        return score + 7.25


def test_ip_cosine_is_sign_preserving_and_normalized() -> None:
    target = IPCosineTarget(n_ip=4, beta=0.9, w=2, rho=0.75)
    states = all_binary_states(8)
    scores = target.log_unnormalized(states)

    assert torch.equal(torch.sign(scores), target.sign(states))
    assert torch.logsumexp(target.log_prob(states), dim=0) == pytest.approx(0.0, abs=1e-12)
    assert target.entropy == pytest.approx(
        float(-torch.sum(torch.exp(target.log_prob(states)) * target.log_prob(states)))
    )


def test_ip_cosine_exact_count_sampling() -> None:
    target = IPCosineTarget(n_ip=5, beta=1.0, w=3, rho=0.5)
    samples = target.sample(
        50000,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(71),
    )
    counts = (samples[:, :5] * samples[:, 5:]).sum(dim=1).to(torch.int64)
    empirical = torch.bincount(counts, minlength=6).to(torch.float64) / samples.shape[0]

    assert torch.max(torch.abs(empirical - torch.exp(target.count_log_prob))) < 0.012


def test_component_diagnostic_separates_ip_from_cosine() -> None:
    target = IPCosineTarget(n_ip=4, beta=0.9, w=2, rho=0.5)
    states = all_binary_states(8)

    exact = target_score_metrics(_KnownScoreModel(target, include_ip=True), target, states)  # type: ignore[arg-type]
    scaffold_only = target_score_metrics(
        _KnownScoreModel(target, include_ip=False),  # type: ignore[arg-type]
        target,
        states,
    )

    assert exact.learned_ip_component_ratio == pytest.approx(1.0, abs=1e-12)
    assert exact.learned_cosine_component_ratio == pytest.approx(1.0, abs=1e-12)
    assert scaffold_only.learned_ip_component_ratio == pytest.approx(0.0, abs=1e-12)
    assert scaffold_only.learned_cosine_component_ratio == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(
    ("w", "rho"),
    [(1, 0.5), (2, 0.0), (2, 1.0), (2, -0.1)],
)
def test_ip_cosine_rejects_non_sign_preserving_parameters(w: int, rho: float) -> None:
    with pytest.raises(ValueError):
        IPCosineTarget(n_ip=4, beta=1.0, w=w, rho=rho)


def test_comparison_configuration_is_matched_and_uses_block_counts() -> None:
    config_path = (
        Path(__file__).parents[1] / "configs" / "stage2b_n16_theory_learning_comparison.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    targets = scalable_experiment._parse_targets(config)
    models = scalable_experiment._parse_models(config)

    assert [point.name for point in targets] == [
        "pure-IP",
        "hybrid-rho0p5-w4",
        "hybrid-rho0p75-w4",
        "pure-cosine-w4",
        "block-q4-L4",
        "block-q2-L8",
    ]
    block_targets = [point for point in targets if point.kind == "block_ip"]
    assert [point.block_count for point in block_targets] == [4, 2]
    assert [point.resolved_block_size(16) for point in block_targets] == [4, 8]
    assert all(isinstance(point.make(16), BlockIPTarget) for point in block_targets)

    parameter_counts = [
        scalable_experiment.make_trainable_model(
            point.kind,
            32,
            point.n_hidden,
            pair_mode=point.pair_mode,
            init_std=0.0,
        ).n_parameters
        for point in models
    ]
    assert parameter_counts == [18512, 18528]

    for point in targets:
        target = point.make(16)
        if isinstance(target, (CountCosineTarget, IPCosineTarget)):
            uniform_probability = torch.exp(
                target.count_log_multiplicity - target.n_visible * math.log(2.0)
            )
            mean = torch.sum(uniform_probability * target.count_scores)
            variance = torch.sum(uniform_probability * (target.count_scores - mean).square())
        elif isinstance(target, BlockIPTarget):
            variance = torch.tensor(
                target.beta**2 * target.n_blocks * (1.0 - 4.0 ** (-target.block_size)),
                dtype=torch.float64,
            )
        elif isinstance(target, IPTarget):
            variance = torch.tensor(
                target.beta**2 * (1.0 - 4.0 ** (-target.n_ip)),
                dtype=torch.float64,
            )
        else:
            raise AssertionError("unexpected target type")
        assert float(variance) == pytest.approx(1.0, rel=2e-9)


def test_ip_cosine_smoke_runs_and_plots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = Path(__file__).parents[1] / "configs" / "stage2b_ip_cosine_smoke.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["output_dir"] = "results/ip_cosine_test"
    config_path = Path("ip_cosine.yaml")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    results_path = run_scalable_experiment(config_path)
    results = pd.read_csv(results_path)
    checkpoints = pd.read_csv(results_path.with_name("checkpoints.csv"))

    assert len(results) == 4
    assert set(results["target_kind"]) == {"ip_cosine", "block_ip"}
    hybrid = checkpoints.loc[checkpoints["target_kind"] == "ip_cosine"]
    assert hybrid["learned_ip_component_ratio"].notna().all()
    assert hybrid["learned_cosine_component_ratio"].notna().all()
    plot_dir = plot_scalable_objective_curves(results_path.with_name("checkpoints.csv"))
    assert len(list(plot_dir.glob("*.pdf"))) == 3
