from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from ip_rbm.benchmark_targets import BenchmarkTarget, four_body_terms
from ip_rbm.paper_studies import (
    _report_lattice,
    lattice_observables,
    run_study,
    select_checkpoints,
    statewise_metrics,
    study_plan,
)
from ip_rbm.plotting import plot_scalable_objective_curves
from ip_rbm.scalable_learning import ScalableTrainingSettings, learning_rate_at_update
from ip_rbm.states import all_binary_states
from ip_rbm.targets import inner_product_sign


@pytest.mark.parametrize(
    "family",
    [
        "four_lifted",
        "four_rewired",
        "four_disjoint",
        "independent",
        "mixture",
        "rbm_teacher",
        "random_pairwise",
        "random_table",
        "lattice_ising",
    ],
)
def test_benchmark_normalization_and_sampling(family):
    n = 8 if family == "lattice_ising" else 4
    beta = 0.44 if family == "lattice_ising" else 1.0
    target = BenchmarkTarget(n, beta, family, 914201)
    states = all_binary_states(2 * n)
    logp = target.log_prob(states)
    assert logp.exp().sum().item() == pytest.approx(1, abs=1e-12)
    assert -(logp.exp() * logp).sum().item() == pytest.approx(target.entropy)
    a = target.sample(10000, generator=torch.Generator().manual_seed(5))
    b = target.sample(10000, generator=torch.Generator().manual_seed(5))
    assert torch.equal(a, b)
    expected = (logp.exp() * target.log_unnormalized(states)).sum().item()
    observed = target.log_unnormalized(a).double()
    assert abs(observed.mean().item() - expected) < 6 * observed.std().item() / 100 + 0.01


@pytest.mark.parametrize("width", [8, 16])
def test_four_body_degree_matching_and_pair_breaking(width):
    for seed in [914201, 914202]:
        lifted = four_body_terms(width, "four_lifted", seed)
        rewired = four_body_terms(width, "four_rewired", seed)
        assert lifted.shape == rewired.shape == (width // 2, 4)
        assert np.array_equal(np.bincount(lifted.ravel()), np.bincount(rewired.ravel()))
        assert (np.bincount(rewired.ravel()) == 2).all()
        incidence = np.array([np.isin(np.arange(width), term) for term in rewired]).T
        assert np.any(np.unique(incidence, axis=0, return_counts=True)[1] % 2)
        target = BenchmarkTarget(width // 2, 1, "four_rewired", seed)
        assert target.scores.mean().item() == pytest.approx(0, abs=1e-12)
        assert target.scores.square().mean().item() == pytest.approx(1)


def test_decay_defaults_and_endpoints():
    settings = ScalableTrainingSettings(
        algorithm="cd", updates=100, batch_size=8, learning_rate=0.002
    )
    assert learning_rate_at_update(settings, 100) == 0.002
    decay = replace(settings, lr_decay_start_fraction=0.5, lr_final_ratio=0.1)
    assert learning_rate_at_update(decay, 50) == 0.002
    assert learning_rate_at_update(decay, 100) == pytest.approx(0.0002)
    assert 0.0002 < learning_rate_at_update(decay, 75) < 0.002


def test_statewise_metrics_ignore_additive_constant():
    states = all_binary_states(4)
    score = inner_product_sign(states)
    logp = score - torch.logsumexp(score, 0)
    result = statewise_metrics(score + 17, score, logp, states)
    assert result["population_kl"] == pytest.approx(0, abs=1e-12)
    assert result["score_linf_up_to_constant"] == 0
    assert result["global_ip_margin"] == 2


def test_all_frozen_configs_validate():
    root = Path(__file__).parents[1] / "configs/paper"
    expected = {
        "study1_population": 81,
        "study1_learning": 108,
        "study2_physics": 240,
        "study2_critical": 36,
        "study3_controls": 240,
    }
    for name, count in expected.items():
        plan = study_plan(yaml.safe_load((root / f"{name}.yaml").read_text()))
        assert plan["fits"] == count


def test_learning_workflow_selection_and_single_series_plot(tmp_path):
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/paper/smoke_learning.yaml").read_text()
    )
    config["output_dir"] = str(tmp_path / "run")
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(config))
    results = run_study(path)
    output = results.parent
    assert (output / "source_snapshot.zip").exists()
    checkpoints = pd.read_csv(output / "checkpoints.csv")
    assert checkpoints.validation_log_pseudolikelihood.notna().all()
    assert (
        checkpoints.groupby(["target_name", "dataset_repeat"]).selection_seed.nunique().eq(1).all()
    )
    selected = select_checkpoints(checkpoints)
    altered = checkpoints.copy()
    altered["population_kl"] = np.arange(len(altered))[::-1] * 1000
    altered["normalized_target_score_rmse"] = np.arange(len(altered)) * 1000
    assert selected.parameter_file.tolist() == select_checkpoints(altered).parameter_file.tolist()
    assert (selected.total_tuning_training_seconds >= selected.training_elapsed_seconds).all()
    # Regression: multiple targets, one architecture/learner, finite KL.
    single = output / "single.csv"
    checkpoints[checkpoints.training_name == "rbm-cd1"].to_csv(single, index=False)
    plot_scalable_objective_curves(single, output / "single_plots")
    assert list((output / "single_plots").glob("*.pdf"))
    with pytest.raises(FileExistsError):
        run_study(path)


def test_population_workflow(tmp_path):
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/paper/smoke_population.yaml").read_text()
    )
    config["output_dir"] = str(tmp_path / "population")
    path = tmp_path / "population.yaml"
    path.write_text(yaml.safe_dump(config))
    result = pd.read_csv(run_study(path))
    assert len(result) == 9
    assert result.population_kl.ge(-1e-10).all()
    assert result.score_linf_up_to_constant.ge(0).all()


def test_lattice_observables_and_report(tmp_path):
    states = all_binary_states(16)
    probabilities = torch.full((len(states),), 1 / len(states), dtype=torch.float64)
    metrics = lattice_observables(probabilities, states, beta=0.4)
    assert metrics["energy_per_spin"] == pytest.approx(0)
    assert metrics["susceptibility_per_spin"] == pytest.approx(0.4)
    assert metrics["heat_capacity_per_spin"] == pytest.approx(2 * 0.4**2)
    assert metrics["binder_cumulant"] == pytest.approx(2 / (3 * 16))
    path = tmp_path / "uniform.npz"
    np.savez(path, theta=np.zeros(33))  # V=16, m=1 RBM.
    selected = pd.DataFrame(
        [
            {
                "target_family": "lattice_ising",
                "target_name": "critical",
                "n_ip": 8,
                "model": "rbm",
                "n_hidden": 1,
                "pair_mode": "none",
                "parameter_file": str(path),
                "beta": 0.4,
                "dataset_repeat": 0,
                "training_name": "rbm-cd1",
                "update": 1,
            }
        ]
    )
    _report_lattice(selected, tmp_path)
    report = pd.read_csv(tmp_path / "lattice_observables.csv").iloc[0]
    assert report.model_energy_per_spin == pytest.approx(0)
    assert report.target_energy_per_spin < 0
