from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from ip_rbm import scalable_experiment
from ip_rbm.plotting import (
    plot_scalable_cdk_curves,
    plot_scalable_learning,
    plot_scalable_objective_curves,
    plot_scalable_resource_curves,
)
from ip_rbm.scalable_data import IPTarget
from ip_rbm.scalable_evaluation import IPScoreMetrics, TargetScoreMetrics
from ip_rbm.scalable_experiment import run_scalable_experiment


def test_scalable_experiment_runs_every_algorithm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = Path("scalable.yaml")
    config_path.write_text(
        """
name: scalable_smoke_test
n_ip: [1]
beta: [0.5]
sample_sizes: [16]
dataset_repetitions: 1
data_seed: 10
initialization_seed: 20
initialization_std: 0.01
pair_minibatches_across_models: true
pair_evaluation_states_across_models: true
device: cpu
dtype: float32
weight_bound: [1.0]
models:
  - kind: rbm
    hidden: [1]
  - kind: 3rbm
    hidden: [1]
    pair_mode: cross
training_defaults:
  updates: 2
  batch_size: 8
  learning_rate: 0.01
  record_every: 1
  tempering_replicas: 3
training:
  - name: cd1
    algorithm: cd
  - name: pcd1
    algorithm: pcd
  - name: tpcd1
    algorithm: tempered_pcd
  - name: pseudo
    algorithm: pseudolikelihood
  - name: nce
    algorithm: nce
checkpoints:
  updates: [1, 2]
  ais_updates: [2]
evaluation:
  target_samples: 32
  exact_max_n: 1
  ais_enabled: true
  ais_particles: 8
  ais_intermediate: 8
  model_sample_chains: 8
  model_sample_burn_in: 2
  model_sample_rounds: 2
  model_sample_thinning: 1
clean_output_dir: true
output_dir: results/scalable_smoke_test
""".strip(),
        encoding="utf-8",
    )

    results_path = run_scalable_experiment(config_path)

    results = pd.read_csv(results_path)
    history = pd.read_csv(results_path.with_name("history.csv"))
    summary = pd.read_csv(results_path.with_name("summary.csv"))
    checkpoints = pd.read_csv(results_path.with_name("checkpoints.csv"))
    checkpoint_summary = pd.read_csv(results_path.with_name("checkpoint_summary.csv"))
    assert len(results) == 10
    assert len(history) == 20
    assert len(summary) == 10
    assert len(checkpoints) == 20
    assert len(checkpoint_summary) == 20
    assert set(results["algorithm"]) == {
        "cd",
        "pcd",
        "tempered_pcd",
        "pseudolikelihood",
        "nce",
    }
    assert results["population_kl"].ge(-1e-10).all()
    assert set(results["population_kl_source"]) == {"exact"}
    assert results["ais_population_kl"].notna().all()
    assert results["parameter_file"].map(lambda value: Path(value).is_file()).all()
    assert results["minibatch_seed"].nunique() == 5
    assert results.groupby("training_name")["minibatch_seed"].nunique().eq(1).all()
    assert results["training_peak_rss_bytes"].gt(0).all()
    assert checkpoints["update"].drop_duplicates().tolist() == [1, 2]
    assert checkpoints["parameter_file"].map(lambda value: Path(value).is_file()).all()
    dataset_file = results_path.with_name("datasets") / "ip_n1_beta0.5_data000.npz"
    with np.load(dataset_file, allow_pickle=False) as archive:
        assert archive["train_states"].shape == (16, 2)
        assert archive["validation_states"].shape == (32, 2)
    assert results_path.with_name("metadata.json").is_file()
    assert results_path.with_name("config.yaml").is_file()

    plot_path = plot_scalable_learning(results_path)
    assert plot_path.is_file()
    resource_plot_dir = plot_scalable_resource_curves(results_path.with_name("checkpoints.csv"))
    assert len(list(resource_plot_dir.glob("*.pdf"))) == 10


def test_scalable_experiment_can_pair_initializations_across_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    initial_parameters: list[torch.Tensor] = []
    make_model = scalable_experiment.make_trainable_model

    def capture_initial_parameters(*args: Any, **kwargs: Any):
        model = make_model(*args, **kwargs)
        initial_parameters.append(model.flat_parameters().detach().clone())
        return model

    monkeypatch.setattr(
        scalable_experiment,
        "make_trainable_model",
        capture_initial_parameters,
    )
    config_path = Path("paired_initialization.yaml")
    config_path.write_text(
        """
name: paired_initialization_test
n_ip: [1]
beta: [0.0]
sample_sizes: [8]
dataset_repetitions: 1
initialization_seed: 20
pair_initializations_across_training: true
initialization_std: 0.01
device: cpu
dtype: float32
weight_bound: [1.0]
models:
  - kind: rbm
    hidden: [1]
training_defaults:
  updates: 1
  batch_size: 4
  learning_rate: 0.01
  record_every: 1
training:
  - name: cd1
    algorithm: cd
  - name: nce
    algorithm: nce
evaluation:
  target_samples: 8
  exact_max_n: 1
  model_sample_chains: 4
  model_sample_burn_in: 1
  model_sample_rounds: 2
  model_sample_thinning: 1
clean_output_dir: true
output_dir: results/paired_initialization_test
""".strip(),
        encoding="utf-8",
    )

    results = pd.read_csv(run_scalable_experiment(config_path))

    assert len(initial_parameters) == 2
    assert torch.equal(initial_parameters[0], initial_parameters[1])
    assert results["initialization_seed"].nunique() == 1
    assert results["training_seed"].nunique() == 2


def test_cd_scaling_configuration_is_parameter_matched() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "stage2b_cd_scaling_resources.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_points = scalable_experiment._parse_models(config)
    expected_parameters = {
        (8, "small"): (662, 664),
        (8, "large"): (1308, 1312),
        (16, "small"): (2342, 2344),
        (16, "large"): (4652, 4656),
        (32, "small"): (8774, 8776),
        (32, "large"): (17484, 17488),
    }

    for (n_ip, comparison_group), expected in expected_parameters.items():
        supported = [
            point
            for point in model_points
            if point.supports(n_ip) and point.comparison_group == comparison_group
        ]
        assert [point.kind for point in supported] == ["rbm", "3rbm"]
        parameter_counts = tuple(
            scalable_experiment.make_trainable_model(
                point.kind,
                2 * n_ip,
                point.n_hidden,
                pair_mode=point.pair_mode,
                init_std=0.0,
            ).n_parameters
            for point in supported
        )
        assert parameter_counts == expected

    checkpoints = scalable_experiment._parse_checkpoints(config)
    assert checkpoints.updates == (250, 1000, 3000, 10000, 20000, 30000)
    assert checkpoints.ais_updates == frozenset({3000, 10000, 30000})


def test_score_shape_rmse_penalizes_gap_error_and_sector_variance() -> None:
    target = IPTarget(2, 1.0)
    ideal = IPScoreMetrics(
        mean_log_pseudolikelihood=0.0,
        even_mean_score=1.0,
        odd_mean_score=-1.0,
        score_gap=2.0,
        even_score_variance=0.0,
        odd_score_variance=0.0,
    )
    variable = IPScoreMetrics(
        mean_log_pseudolikelihood=0.0,
        even_mean_score=1.0,
        odd_mean_score=-1.0,
        score_gap=2.0,
        even_score_variance=1.0,
        odd_score_variance=3.0,
    )
    target_metrics = TargetScoreMetrics(
        mean_log_pseudolikelihood=0.0,
        target_score_rmse=0.0,
        normalized_target_score_rmse=0.0,
        target_score_correlation=1.0,
        target_score_standard_deviation=1.0,
    )

    ideal_columns = scalable_experiment._checkpoint_metric_columns(
        target=target,
        score_metrics=ideal,
        target_metrics=target_metrics,
        exact_population_kl=0.0,
        exact_ip_correlation=target.mean_sign,
        ais_metrics=None,
    )
    variable_columns = scalable_experiment._checkpoint_metric_columns(
        target=target,
        score_metrics=variable,
        target_metrics=target_metrics,
        exact_population_kl=0.0,
        exact_ip_correlation=target.mean_sign,
        ais_metrics=None,
    )

    assert ideal_columns["normalized_score_shape_rmse"] == 0.0
    assert variable_columns["normalized_score_shape_rmse"] == pytest.approx(2.0**0.5)


def test_anchored_teacher_smoke_runs_and_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = Path(__file__).parents[1] / "configs" / "stage2b_anchored_teacher_smoke.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["output_dir"] = "results/anchored_teacher_test"
    config_path = Path("anchored_teacher.yaml")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    results_path = run_scalable_experiment(config_path)
    results = pd.read_csv(results_path)
    checkpoints = pd.read_csv(results_path.with_name("checkpoints.csv"))

    assert len(results) == 4
    assert set(results["target_name"]) == {"anchors-only", "anchored-g1p75"}
    assert set(results["target_kind"]) == {"anchored_teacher"}
    assert results["population_kl"].ge(-1e-10).all()
    assert checkpoints["normalized_target_score_rmse"].notna().all()
    assert checkpoints["target_score_correlation"].notna().all()
    plot_dir = plot_scalable_objective_curves(results_path.with_name("checkpoints.csv"))
    assert len(list(plot_dir.glob("*.pdf"))) == 3


def test_anchored_teacher_configuration_is_parameter_matched() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "stage2b_anchored_teacher_objectives.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_points = scalable_experiment._parse_models(config)
    expected_parameters = {8: (1308, 1312), 16: (9272, 9280)}

    for n_ip, expected in expected_parameters.items():
        supported = [point for point in model_points if point.supports(n_ip)]
        parameter_counts = tuple(
            scalable_experiment.make_trainable_model(
                point.kind,
                2 * n_ip,
                point.n_hidden,
                pair_mode=point.pair_mode,
                init_std=0.0,
            ).n_parameters
            for point in supported
        )
        assert parameter_counts == expected

    targets = scalable_experiment._parse_targets(config)
    assert [target.name for target in targets] == [
        "anchors-only",
        "anchored-g1p25",
        "anchored-g1p75",
        "anchored-g2p25",
    ]


def test_count_cosine_capacity_configuration_is_parameter_matched() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "stage2b_n16_count_cosine_capacity.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_points = scalable_experiment._parse_models(config)
    target_points = scalable_experiment._parse_targets(config)

    expected = {
        "m3-064": (18512, 18528),
        "m3-128": (37025, 37024),
        "m3-256": (74018, 74016),
        "m3-512": (148004, 148000),
    }
    for comparison_group, expected_parameters in expected.items():
        supported = [point for point in model_points if point.comparison_group == comparison_group]
        assert [point.kind for point in supported] == ["rbm", "3rbm"]
        parameter_counts = tuple(
            scalable_experiment.make_trainable_model(
                point.kind,
                32,
                point.n_hidden,
                pair_mode=point.pair_mode,
                init_std=0.0,
            ).n_parameters
            for point in supported
        )
        assert parameter_counts == expected_parameters

    assert [point.w for point in target_points] == [1, 2, 3, 4]
    assert all(point.kind == "count_cosine" for point in target_points)


def test_cdk_smoke_runs_records_compute_and_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = Path(__file__).parents[1] / "configs" / "stage2b_cdk_ipw_smoke.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["output_dir"] = "results/cdk_smoke_test"
    config_path = Path("cdk_smoke.yaml")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    results_path = run_scalable_experiment(config_path)
    results = pd.read_csv(results_path)
    checkpoints = pd.read_csv(results_path.with_name("checkpoints.csv"))
    checkpoint_summary = pd.read_csv(results_path.with_name("checkpoint_summary.csv"))

    assert len(results) == 4
    assert set(results["gibbs_steps"]) == {1, 2}
    assert set(checkpoint_summary["gibbs_steps"]) == {1, 2}
    assert np.array_equal(
        checkpoints["gibbs_chain_sweeps"].to_numpy(),
        (checkpoints["update"] * checkpoints["gibbs_steps"] * checkpoints["batch_size"]).to_numpy(),
    )
    assert results.groupby("model", dropna=False)["initialization_seed"].nunique().eq(1).all()
    plot_dir = plot_scalable_cdk_curves(results_path.with_name("checkpoints.csv"))
    assert len(list(plot_dir.glob("*.pdf"))) == 1


def test_n16_cdk_configuration_is_parameter_matched_and_compute_comparable() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "stage2b_n16_cdk_ipw.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_points = scalable_experiment._parse_models(config)
    training_points = scalable_experiment._parse_training(config)
    target_points = scalable_experiment._parse_targets(config)

    parameter_counts = tuple(
        scalable_experiment.make_trainable_model(
            point.kind,
            32,
            point.n_hidden,
            pair_mode=point.pair_mode,
            init_std=0.0,
        ).n_parameters
        for point in model_points
    )
    assert parameter_counts == (18512, 18528)
    assert [point.settings.gibbs_steps for point in training_points] == [1, 2, 4, 8]
    assert all(point.settings.algorithm == "cd" for point in training_points)
    assert all(point.settings.updates == 25000 for point in training_points)
    assert [point.w for point in target_points] == [1, 3, 4]
    assert config["dataset_repetitions"] == 2
    checkpoints = scalable_experiment._parse_checkpoints(config)
    assert {250, 1000, 2000, 4000, 8000, 16000, 25000}.issubset(checkpoints.updates)


def test_score_regression_smoke_runs_without_training_gibbs_sweeps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = Path(__file__).parents[1] / "configs" / "stage2b_score_regression_smoke.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["output_dir"] = "results/score_regression_smoke_test"
    config_path = Path("score_regression_smoke.yaml")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    results_path = run_scalable_experiment(config_path)
    results = pd.read_csv(results_path)
    checkpoints = pd.read_csv(results_path.with_name("checkpoints.csv"))

    assert len(results) == 2
    assert set(results["algorithm"]) == {"score_regression"}
    assert set(results["training_distribution"]) == {"uniform"}
    assert results["gibbs_chain_sweeps"].eq(0).all()
    assert results["population_kl"].ge(-1e-10).all()
    assert checkpoints["normalized_target_score_rmse"].notna().all()
    assert checkpoints["target_score_correlation"].notna().all()
    dataset_file = (
        results_path.with_name("datasets") / "uniform-count-cosine-w1_n3_beta1_data000.npz"
    )
    with np.load(dataset_file, allow_pickle=False) as archive:
        assert archive["train_states"].shape == (256, 6)
        assert archive["validation_states"].shape == (64, 6)
    plot_dir = plot_scalable_objective_curves(results_path.with_name("checkpoints.csv"))
    assert len(list(plot_dir.glob("*.pdf"))) == 1


def test_n16_score_regression_diagnostic_configuration_is_parameter_matched() -> None:
    config_path = (
        Path(__file__).parents[1] / "configs" / "stage2b_n16_score_regression_diagnostic.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_points = scalable_experiment._parse_models(config)
    training_points = scalable_experiment._parse_training(config)
    target_points = scalable_experiment._parse_targets(config)

    expected = {
        "m3-064": (18512, 18528),
        "m3-256": (74018, 74016),
    }
    for comparison_group, expected_parameters in expected.items():
        supported = [point for point in model_points if point.comparison_group == comparison_group]
        assert [point.kind for point in supported] == ["rbm", "3rbm"]
        parameter_counts = tuple(
            scalable_experiment.make_trainable_model(
                point.kind,
                32,
                point.n_hidden,
                pair_mode=point.pair_mode,
                init_std=0.0,
            ).n_parameters
            for point in supported
        )
        assert parameter_counts == expected_parameters

    assert config["training_distribution"] == "uniform"
    assert [point.w for point in target_points] == [1, 3, 4]
    assert len(training_points) == 1
    assert training_points[0].settings.algorithm == "score_regression"
    assert training_points[0].settings.updates == 20000
    assert training_points[0].settings.batch_size == 512
    assert config["dataset_repetitions"] == 3
