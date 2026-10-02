from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ip_rbm.experiment import (
    _estimated_objective_work,
    _load_continuation_theta,
    _parameter_stem,
    _parse_continuation,
    _prepare_output_directory,
    run_experiment,
)
from ip_rbm.models import RBM


def test_estimated_work_accounts_for_state_space_and_model_size() -> None:
    n5_m8 = _estimated_objective_work(5, "3rbm", 8, "cross")
    n6_m8 = _estimated_objective_work(6, "3rbm", 8, "cross")
    n6_m16 = _estimated_objective_work(6, "3rbm", 16, "cross")

    assert n6_m8 > 4 * n5_m8
    assert n6_m16 > n6_m8


def test_unbounded_parameter_stem_is_explicit() -> None:
    stem = _parameter_stem("rbm", 6, 136, 1.0, None, "cross")

    assert stem == "rbm_n6_m136_beta1_boundunbounded_pairs-none"


def test_load_continuation_theta_validates_and_loads_best_parameters(tmp_path: Path) -> None:
    model = RBM(n_visible=4, n_hidden=2)
    source_dir = tmp_path / "previous"
    parameter_dir = source_dir / "parameters"
    parameter_dir.mkdir(parents=True)
    theta = np.linspace(-0.5, 0.5, model.n_parameters)
    source_file = parameter_dir / "point.npz"
    np.savez_compressed(source_file, theta_best=theta)

    loaded, loaded_from = _load_continuation_theta(source_dir, "point", model, 1.0)

    np.testing.assert_array_equal(loaded, theta)
    assert loaded_from == source_file


def test_continuation_source_cannot_be_inside_cleaned_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "results" / "new"
    config = {"continuation": {"source_dir": str(output_dir / "old")}}

    with pytest.raises(ValueError, match="must not be cleaned"):
        _parse_continuation(config, output_dir)


def test_experiment_continues_matching_saved_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source_parameter_dir = Path("results/source/parameters")
    source_parameter_dir.mkdir(parents=True)
    np.savez_compressed(
        source_parameter_dir / "rbm_n1_m0_beta0_bound1_pairs-none.npz",
        theta_best=np.zeros(2),
    )
    config_path = Path("continue.yaml")
    config_path.write_text(
        """
name: continued_test
n_ip: [1]
beta: [0.0]
weight_bound: [1.0]
models:
  - kind: rbm
    hidden: [0]
optimization:
  restarts: 1
  random_scale: 0.1
  maxiter: 2
  maxfun: 4
  ftol: 1.0e-12
  gtol: 1.0e-8
  seed: 1
continuation:
  source_dir: results/source
  perturb_scale: 0.01
clean_output_dir: true
output_dir: results/continued
""".strip(),
        encoding="utf-8",
    )

    results_path = run_experiment(config_path)

    results = pd.read_csv(results_path)
    restarts = pd.read_csv(results_path.with_name("restarts.csv"))
    assert results.loc[0, "best_initialization"] == "continued"
    assert restarts.loc[0, "initialization"] == "continued"
    assert results.loc[0, "best_kl"] < 1e-12


def test_experiment_records_unbounded_weight_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = Path("unbounded.yaml")
    config_path.write_text(
        """
name: unbounded_test
n_ip: [1]
beta: [0.0]
weight_bound: [null]
models:
  - kind: rbm
    hidden: [0]
optimization:
  restarts: 1
  random_scale: 0.1
  maxiter: 2
  maxfun: 4
  ftol: 1.0e-12
  gtol: 1.0e-8
  seed: 1
clean_output_dir: true
output_dir: results/unbounded
""".strip(),
        encoding="utf-8",
    )

    results = pd.read_csv(run_experiment(config_path))

    assert results.loc[0, "weight_bound"] == "unbounded"
    assert "boundunbounded" in results.loc[0, "parameter_file"]


def test_prepare_output_directory_removes_stale_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    output_dir = Path("results/test_run")
    output_dir.mkdir(parents=True)
    stale_file = output_dir / "stale.txt"
    stale_file.write_text("old result", encoding="utf-8")

    parameter_dir = _prepare_output_directory(output_dir, clean=True)

    assert not stale_file.exists()
    assert parameter_dir == output_dir / "parameters"
    assert parameter_dir.is_dir()


def test_prepare_output_directory_preserves_existing_files_when_not_cleaning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    output_dir = Path("results/test_run")
    output_dir.mkdir(parents=True)
    existing_file = output_dir / "existing.txt"
    existing_file.write_text("keep me", encoding="utf-8")

    _prepare_output_directory(output_dir, clean=False)

    assert existing_file.read_text(encoding="utf-8") == "keep me"


def test_prepare_output_directory_refuses_results_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    results_root = Path("results")
    results_root.mkdir()
    sentinel = results_root / "keep.txt"
    sentinel.write_text("important", encoding="utf-8")

    with pytest.raises(ValueError, match="named subdirectory"):
        _prepare_output_directory(results_root, clean=True)

    assert sentinel.read_text(encoding="utf-8") == "important"


def test_prepare_output_directory_refuses_unrelated_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    unrelated = Path("unrelated")
    unrelated.mkdir()
    sentinel = unrelated / "keep.txt"
    sentinel.write_text("important", encoding="utf-8")

    with pytest.raises(ValueError, match="named subdirectory"):
        _prepare_output_directory(unrelated, clean=True)

    assert sentinel.read_text(encoding="utf-8") == "important"
