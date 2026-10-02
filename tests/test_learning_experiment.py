from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ip_rbm.learning_experiment import run_learning_experiment
from ip_rbm.plotting import plot_learning_curve


def test_learning_experiment_writes_paired_exact_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = Path("learning.yaml")
    config_path.write_text(
        """
name: exact_learning_test
n_ip: [1]
beta: [0.0]
sample_sizes: [4, 8]
dataset_repetitions: 2
validation_size: 8
data_seed: 123
weight_bound: [1.0]
models:
  - kind: rbm
    hidden: [0]
  - kind: 3rbm
    hidden: [0]
    pair_mode: cross
optimization:
  restarts: 1
  random_scale: 0.1
  maxiter: 10
  maxfun: 20
  ftol: 1.0e-12
  gtol: 1.0e-8
  seed: 1
clean_output_dir: true
output_dir: results/learning_test
""".strip(),
        encoding="utf-8",
    )

    results_path = run_learning_experiment(config_path)

    results = pd.read_csv(results_path)
    fits = pd.read_csv(results_path.with_name("fits.csv"))
    summary = pd.read_csv(results_path.with_name("summary.csv"))
    assert len(results) == 8
    assert len(fits) == 8
    assert len(summary) == 4
    assert set(results["model"]) == {"rbm", "3rbm"}
    assert set(results["sample_size"]) == {4, 8}
    assert results["population_kl"].ge(-1e-12).all()
    assert results["dataset_file"].map(lambda value: Path(value).is_file()).all()
    assert results["parameter_file"].map(lambda value: Path(value).is_file()).all()

    for (_, sample_size, _dataset_repeat), group in results.groupby(
        ["n_ip", "sample_size", "dataset_repeat"]
    ):
        assert group["dataset_seed"].nunique() == 1
        assert group["dataset_file"].nunique() == 1
        dataset_path = Path(group["dataset_file"].iloc[0])
        with np.load(dataset_path, allow_pickle=False) as archive:
            stored_sizes = archive["sample_sizes"]
            stored_counts = archive["train_counts"]
        index = int(np.flatnonzero(stored_sizes == sample_size)[0])
        assert int(stored_counts[index].sum()) == sample_size

    assert results_path.with_name("metadata.json").is_file()
    assert results_path.with_name("config.yaml").is_file()


def test_learning_plot_is_created(tmp_path: Path) -> None:
    results_path = tmp_path / "results.csv"
    pd.DataFrame(
        {
            "model": ["rbm", "rbm", "3rbm", "3rbm"],
            "n_hidden": [4, 4, 2, 2],
            "n_parameters": [24, 24, 22, 22],
            "n_ip": [2, 2, 2, 2],
            "beta": [0.5, 0.5, 0.5, 0.5],
            "sample_size": [16, 64, 16, 64],
            "population_kl": [0.4, 0.2, 0.3, 0.1],
            "weight_bound": [2.0, 2.0, 2.0, 2.0],
        }
    ).to_csv(results_path, index=False)

    output = plot_learning_curve(results_path)

    assert output == tmp_path / "population_kl_vs_samples.pdf"
    assert output.is_file()


def test_learning_experiment_can_initialize_from_population_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source_parameter_dir = Path("results/stage1_reference/parameters")
    source_parameter_dir.mkdir(parents=True)
    np.savez_compressed(
        source_parameter_dir / "rbm_n1_m0_beta0_bound1_pairs-none.npz",
        theta_best=np.zeros(2),
    )
    config_path = Path("reference_learning.yaml")
    config_path.write_text(
        """
name: reference_learning_test
n_ip: [1]
beta: [0.0]
sample_sizes: [4]
dataset_repetitions: 1
data_seed: 12
weight_bound: [1.0]
models:
  - kind: rbm
    hidden: [0]
    population_reference:
      source_dir: results/stage1_reference
      initialize: true
      perturb_scale: 0.01
optimization:
  restarts: 2
  random_scale: 0.1
  maxiter: 10
  maxfun: 20
  ftol: 1.0e-12
  gtol: 1.0e-8
  seed: 1
clean_output_dir: true
output_dir: results/reference_learning
""".strip(),
        encoding="utf-8",
    )

    results_path = run_learning_experiment(config_path)

    results = pd.read_csv(results_path)
    fits = pd.read_csv(results_path.with_name("fits.csv"))
    assert results.loc[0, "initialization_mode"] == "population_reference"
    assert results.loc[0, "population_reference_kl"] < 1e-12
    assert list(fits["initialization"]) == [
        "population_reference",
        "population_reference_perturbed",
    ]


def test_learning_experiment_supports_n_specific_model_grids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = Path("n_specific_learning.yaml")
    config_path.write_text(
        """
name: n_specific_learning_test
n_ip: [1, 2]
beta: [0.0]
sample_sizes: [4]
dataset_repetitions: 1
data_seed: 12
weight_bound: [1.0]
models:
  - kind: rbm
    n_ip: [1]
    hidden: [0]
  - kind: 3rbm
    n_ip: [2]
    hidden: [0]
    pair_mode: cross
optimization:
  restarts: 1
  random_scale: 0.1
  maxiter: 5
  maxfun: 10
  ftol: 1.0e-12
  gtol: 1.0e-8
  seed: 1
clean_output_dir: true
output_dir: results/n_specific_learning
""".strip(),
        encoding="utf-8",
    )

    results = pd.read_csv(run_learning_experiment(config_path))

    assert list(results[["n_ip", "model"]].itertuples(index=False, name=None)) == [
        (1, "rbm"),
        (2, "3rbm"),
    ]
