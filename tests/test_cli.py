from pathlib import Path

import pytest

from ip_rbm import cli, experiment, learning_experiment, plotting, scalable_experiment


def test_run_plot_open_executes_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = Path("configs/example.yaml")
    results_path = tmp_path / "results.csv"
    plot_path = tmp_path / "kl_vs_hidden.pdf"
    opened: list[Path] = []

    def fake_run_experiment(received: str | Path) -> Path:
        assert Path(received) == config_path
        return results_path

    def fake_plot_frontier(received: str | Path) -> Path:
        assert Path(received) == results_path
        return plot_path

    monkeypatch.setattr(experiment, "run_experiment", fake_run_experiment)
    monkeypatch.setattr(plotting, "plot_frontier", fake_plot_frontier)
    monkeypatch.setattr(cli, "_open_path", opened.append)

    cli.main(["run-plot-open", str(config_path)])

    assert opened == [plot_path]
    output = capsys.readouterr().out
    assert f"Wrote {results_path}" in output
    assert f"Wrote {plot_path}" in output


def test_learn_plot_open_executes_learning_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = Path("configs/learning.yaml")
    results_path = tmp_path / "results.csv"
    plot_path = tmp_path / "population_kl_vs_samples.pdf"
    opened: list[Path] = []

    def fake_run_learning_experiment(received: str | Path) -> Path:
        assert Path(received) == config_path
        return results_path

    def fake_plot_learning_curve(received: str | Path) -> Path:
        assert Path(received) == results_path
        return plot_path

    monkeypatch.setattr(
        learning_experiment, "run_learning_experiment", fake_run_learning_experiment
    )
    monkeypatch.setattr(plotting, "plot_learning_curve", fake_plot_learning_curve)
    monkeypatch.setattr(cli, "_open_path", opened.append)

    cli.main(["learn-plot-open", str(config_path)])

    assert opened == [plot_path]
    output = capsys.readouterr().out
    assert f"Wrote {results_path}" in output
    assert f"Wrote {plot_path}" in output


def test_train_scalable_plot_open_executes_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = Path("configs/scalable.yaml")
    results_path = tmp_path / "results.csv"
    plot_path = tmp_path / "scalable_population_kl_vs_samples.pdf"
    opened: list[Path] = []

    def fake_run_scalable_experiment(received: str | Path) -> Path:
        assert Path(received) == config_path
        return results_path

    def fake_plot_scalable_learning(received: str | Path) -> Path:
        assert Path(received) == results_path
        return plot_path

    monkeypatch.setattr(
        scalable_experiment,
        "run_scalable_experiment",
        fake_run_scalable_experiment,
    )
    monkeypatch.setattr(plotting, "plot_scalable_learning", fake_plot_scalable_learning)
    monkeypatch.setattr(cli, "_open_path", opened.append)

    cli.main(["train-scalable-plot-open", str(config_path)])

    assert opened == [plot_path]
    output = capsys.readouterr().out
    assert f"Wrote {results_path}" in output
    assert f"Wrote {plot_path}" in output


def test_plot_scalable_resources_writes_plot_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoints_path = tmp_path / "checkpoints.csv"
    output_dir = tmp_path / "focused_plots"

    def fake_plot_scalable_resource_curves(
        received: str | Path, received_output_dir: str | Path | None
    ) -> Path:
        assert Path(received) == checkpoints_path
        assert Path(received_output_dir) == output_dir
        return output_dir

    monkeypatch.setattr(
        plotting,
        "plot_scalable_resource_curves",
        fake_plot_scalable_resource_curves,
    )

    cli.main(
        [
            "plot-scalable-resources",
            str(checkpoints_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert f"Wrote plots to {output_dir}" in capsys.readouterr().out


def test_plot_scalable_objectives_writes_plot_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoints_path = tmp_path / "checkpoints.csv"
    output_dir = tmp_path / "objective_plots"

    def fake_plot_scalable_objective_curves(
        received: str | Path, received_output_dir: str | Path | None
    ) -> Path:
        assert Path(received) == checkpoints_path
        assert Path(received_output_dir) == output_dir
        return output_dir

    monkeypatch.setattr(
        plotting,
        "plot_scalable_objective_curves",
        fake_plot_scalable_objective_curves,
    )

    cli.main(
        [
            "plot-scalable-objectives",
            str(checkpoints_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert f"Wrote plots to {output_dir}" in capsys.readouterr().out


def test_plot_scalable_cdk_writes_plot_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoints_path = tmp_path / "checkpoints.csv"
    output_dir = tmp_path / "cdk_plots"

    def fake_plot_scalable_cdk_curves(
        received: str | Path, received_output_dir: str | Path | None
    ) -> Path:
        assert Path(received) == checkpoints_path
        assert Path(received_output_dir) == output_dir
        return output_dir

    monkeypatch.setattr(
        plotting,
        "plot_scalable_cdk_curves",
        fake_plot_scalable_cdk_curves,
    )

    cli.main(
        [
            "plot-scalable-cdk",
            str(checkpoints_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert f"Wrote plots to {output_dir}" in capsys.readouterr().out
