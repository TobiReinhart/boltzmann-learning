"""Configuration-driven exact empirical-MLE experiments (Stage 2a)."""

from __future__ import annotations

import itertools
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil  # type: ignore[import-untyped]
import torch
import yaml
from numpy.typing import NDArray
from tqdm import tqdm

from ip_rbm.experiment import (
    WeightBound,
    _estimated_objective_work,
    _format_bytes,
    _load_config,
    _load_continuation_theta,
    _make_model,
    _parameter_stem,
    _prepare_output_directory,
    _weight_bound_label,
)
from ip_rbm.learning import (
    ExactDataset,
    LearningRestartResult,
    optimize_exact_mle,
    sample_exact_dataset,
    sample_nested_exact_datasets,
)
from ip_rbm.models import ExactModel
from ip_rbm.objectives import exact_kl
from ip_rbm.optimization import OptimizationSettings
from ip_rbm.targets import ExactTarget, make_ip_target


@dataclass(frozen=True)
class PopulationReferenceSettings:
    """Best-known Stage-I parameters used as a baseline and optional initialization."""

    source_dir: Path
    initialize: bool = False
    perturb_scale: float = 0.01

    def __post_init__(self) -> None:
        if self.perturb_scale < 0:
            raise ValueError("population-reference perturb_scale must be nonnegative")


@dataclass(frozen=True)
class LearningModelPoint:
    """One architecture and hidden width in the learning grid."""

    kind: str
    n_hidden: int
    pair_mode: str
    population_reference: PopulationReferenceSettings | None
    n_ip_values: frozenset[int] | None = None

    def supports_n_ip(self, n_ip: int) -> bool:
        """Return whether this model point belongs to the requested target size."""
        return self.n_ip_values is None or n_ip in self.n_ip_values


@dataclass(frozen=True)
class LoadedPopulationReference:
    """Validated Stage-I parameters and their recomputed exact KL."""

    theta: NDArray[np.float64]
    source_file: Path
    population_kl: float


def _parse_model_points(config: dict[str, Any]) -> list[LearningModelPoint]:
    raw_models = config.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("models must be a nonempty YAML list")

    points: list[LearningModelPoint] = []
    for raw_model in raw_models:
        if not isinstance(raw_model, dict):
            raise ValueError("each model entry must be a YAML mapping")
        kind = str(raw_model["kind"])
        pair_mode = str(raw_model.get("pair_mode", "cross"))
        hidden = raw_model.get("hidden")
        if not isinstance(hidden, list) or not hidden:
            raise ValueError("each model entry requires a nonempty hidden list")

        raw_n_ip_values = raw_model.get("n_ip")
        n_ip_values: frozenset[int] | None = None
        if raw_n_ip_values is not None:
            if not isinstance(raw_n_ip_values, list) or not raw_n_ip_values:
                raise ValueError("model n_ip must be a nonempty YAML list")
            n_ip_values = frozenset(int(value) for value in raw_n_ip_values)
            if any(n_ip < 1 for n_ip in n_ip_values):
                raise ValueError("model n_ip must contain positive integers")

        raw_reference = raw_model.get("population_reference")
        reference: PopulationReferenceSettings | None = None
        if raw_reference is not None:
            if not isinstance(raw_reference, dict):
                raise ValueError("population_reference must be a YAML mapping")
            if "source_dir" not in raw_reference:
                raise ValueError("population_reference requires source_dir")
            initialize = raw_reference.get("initialize", False)
            if not isinstance(initialize, bool):
                raise ValueError("population_reference initialize must be true or false")
            reference = PopulationReferenceSettings(
                source_dir=Path(str(raw_reference["source_dir"])),
                initialize=initialize,
                perturb_scale=float(raw_reference.get("perturb_scale", 0.01)),
            )

        for n_hidden in hidden:
            point = LearningModelPoint(
                kind=kind,
                n_hidden=int(n_hidden),
                pair_mode=pair_mode,
                population_reference=reference,
                n_ip_values=n_ip_values,
            )
            _make_model(point.kind, 2, point.n_hidden, point.pair_mode)
            points.append(point)
    return points


def _derived_seed(base_seed: int, *coordinates: int) -> int:
    """Derive a stable independent 64-bit seed from integer grid coordinates."""
    if base_seed < 0:
        raise ValueError("data_seed must be nonnegative")
    sequence = np.random.SeedSequence([base_seed, *coordinates])
    words = sequence.generate_state(2, dtype=np.uint32)
    return (int(words[0]) << 32) | int(words[1])


def _dataset_stem(n_ip: int, beta: float, dataset_repeat: int) -> str:
    return f"ip_n{n_ip}_beta{beta:g}_data{dataset_repeat:03d}"


def _save_datasets(
    path: Path,
    datasets: dict[int, ExactDataset],
    validation_dataset: ExactDataset | None,
) -> None:
    sample_sizes = np.asarray(sorted(datasets), dtype=np.int64)
    train_counts = np.stack(
        [datasets[int(sample_size)].counts.numpy() for sample_size in sample_sizes]
    )
    if validation_dataset is None:
        validation_counts = np.empty(0, dtype=np.int64)
        validation_size = 0
        validation_seed = -1
    else:
        validation_counts = validation_dataset.counts.numpy()
        validation_size = validation_dataset.sample_size
        validation_seed = validation_dataset.seed
    np.savez_compressed(
        path,
        sample_sizes=sample_sizes,
        train_counts=train_counts,
        train_seed=next(iter(datasets.values())).seed,
        validation_counts=validation_counts,
        validation_size=validation_size,
        validation_seed=validation_seed,
    )


def _fraction_at_bound(theta: NDArray[np.float64], weight_bound: WeightBound) -> float:
    if weight_bound is None:
        return float("nan")
    tolerance = 1e-8 * max(1.0, weight_bound)
    return float(np.mean(np.abs(theta) >= weight_bound - tolerance))


def _validate_reference_location(source_dir: Path, output_dir: Path) -> None:
    resolved_output = output_dir.resolve()
    resolved_source = source_dir.resolve()
    if resolved_source == resolved_output or resolved_output in resolved_source.parents:
        raise ValueError("population-reference source_dir must not be cleaned by output_dir")


def _load_population_reference(
    reference: PopulationReferenceSettings,
    model_point: LearningModelPoint,
    model: ExactModel,
    target: ExactTarget,
    weight_bound: WeightBound,
) -> LoadedPopulationReference:
    stem = _parameter_stem(
        model_point.kind,
        target.n_ip,
        model_point.n_hidden,
        target.beta,
        weight_bound,
        model_point.pair_mode,
    )
    theta, source_file = _load_continuation_theta(reference.source_dir, stem, model, weight_bound)
    population_kl = float(exact_kl(model, torch.from_numpy(theta), target))
    return LoadedPopulationReference(theta, source_file, population_kl)


def _make_summary(results: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "experiment",
        "model",
        "pair_mode",
        "n_ip",
        "n_visible",
        "n_hidden",
        "n_parameters",
        "beta",
        "weight_bound",
        "sample_size",
        "initialization_mode",
    ]
    grouped = results.groupby(group_columns, dropna=False, sort=False)
    return grouped.agg(
        dataset_repetitions=("dataset_repeat", "nunique"),
        population_kl_mean=("population_kl", "mean"),
        population_kl_median=("population_kl", "median"),
        population_kl_q25=("population_kl", lambda values: values.quantile(0.25)),
        population_kl_q75=("population_kl", lambda values: values.quantile(0.75)),
        empirical_nll_mean=("empirical_nll", "mean"),
        population_nll_mean=("population_nll", "mean"),
        generalization_gap_mean=("generalization_gap", "mean"),
        validation_nll_mean=("validation_nll", "mean"),
        optimizer_success_rate=("optimizer_success", "mean"),
        elapsed_seconds_mean=("elapsed_seconds", "mean"),
        population_reference_kl=("population_reference_kl", "first"),
    ).reset_index()


def run_learning_experiment(config_path: str | Path) -> Path:
    """Run a YAML-defined Stage 2a grid and return its per-dataset result CSV."""
    config_path = Path(config_path)
    config = _load_config(config_path)
    output_dir = Path(config["output_dir"])
    clean_output_dir = config.get("clean_output_dir", False)
    if not isinstance(clean_output_dir, bool):
        raise ValueError("clean_output_dir must be true or false")

    n_ip_values = [int(value) for value in config["n_ip"]]
    beta_values = [float(value) for value in config["beta"]]
    if not n_ip_values or any(n_ip < 1 for n_ip in n_ip_values):
        raise ValueError("n_ip must contain positive integers")
    if not beta_values or any(beta < 0 for beta in beta_values):
        raise ValueError("beta must contain nonnegative values")
    sample_sizes = sorted(int(value) for value in config["sample_sizes"])
    if not sample_sizes or sample_sizes[0] < 1 or len(set(sample_sizes)) != len(sample_sizes):
        raise ValueError("sample_sizes must be unique positive integers")
    dataset_repetitions = int(config.get("dataset_repetitions", 1))
    if dataset_repetitions < 1:
        raise ValueError("dataset_repetitions must be positive")
    validation_size = int(config.get("validation_size", 0))
    if validation_size < 0:
        raise ValueError("validation_size must be nonnegative")
    data_seed = int(config.get("data_seed", 314159))
    if data_seed < 0:
        raise ValueError("data_seed must be nonnegative")

    settings = OptimizationSettings(**config["optimization"])
    model_points = _parse_model_points(config)
    weight_bounds: list[WeightBound] = [
        None if value is None else float(value) for value in config["weight_bound"]
    ]
    if not weight_bounds or any(bound is not None and bound <= 0 for bound in weight_bounds):
        raise ValueError("weight_bound must contain positive values or null")
    target_points = list(itertools.product(n_ip_values, beta_values))
    targets = [make_ip_target(n_ip, beta) for n_ip, beta in target_points]
    configured_n_ip = set(n_ip_values)
    for model_point in model_points:
        if model_point.n_ip_values is not None and not model_point.n_ip_values.issubset(
            configured_n_ip
        ):
            raise ValueError("model n_ip values must also appear in the top-level n_ip grid")
    for n_ip in configured_n_ip:
        if not any(model_point.supports_n_ip(n_ip) for model_point in model_points):
            raise ValueError(f"no model points are configured for n_ip={n_ip}")

    references: dict[tuple[int, int, int], LoadedPopulationReference | None] = {}
    for target_index, target in enumerate(targets):
        for bound_index, weight_bound in enumerate(weight_bounds):
            for model_index, model_point in enumerate(model_points):
                if not model_point.supports_n_ip(target.n_ip):
                    continue
                reference_settings = model_point.population_reference
                if reference_settings is None:
                    references[(target_index, bound_index, model_index)] = None
                    continue
                _validate_reference_location(reference_settings.source_dir, output_dir)
                model = _make_model(
                    model_point.kind,
                    2 * target.n_ip,
                    model_point.n_hidden,
                    model_point.pair_mode,
                )
                references[(target_index, bound_index, model_index)] = _load_population_reference(
                    reference_settings, model_point, model, target, weight_bound
                )

    parameter_dir = _prepare_output_directory(output_dir, clean=clean_output_dir)
    dataset_dir = output_dir / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_datasets: dict[tuple[int, int, int], ExactDataset] = {}
    validation_datasets: dict[tuple[int, int], ExactDataset | None] = {}
    dataset_files: dict[tuple[int, int], Path] = {}
    for target_index, target in enumerate(targets):
        for dataset_repeat in range(dataset_repetitions):
            train_seed = _derived_seed(data_seed, target_index, dataset_repeat, 0)
            nested = sample_nested_exact_datasets(target, sample_sizes, train_seed)
            for sample_size, dataset in nested.items():
                train_datasets[(target_index, dataset_repeat, sample_size)] = dataset
            validation_seed = _derived_seed(data_seed, target_index, dataset_repeat, 1)
            validation_dataset = (
                sample_exact_dataset(target, validation_size, validation_seed)
                if validation_size > 0
                else None
            )
            validation_datasets[(target_index, dataset_repeat)] = validation_dataset
            dataset_path = dataset_dir / (
                _dataset_stem(target.n_ip, target.beta, dataset_repeat) + ".npz"
            )
            _save_datasets(dataset_path, nested, validation_dataset)
            dataset_files[(target_index, dataset_repeat)] = dataset_path

    grid = [
        (target_index, bound_index, model_index, dataset_repeat, sample_size)
        for target_index, target in enumerate(targets)
        for bound_index in range(len(weight_bounds))
        for model_index, model_point in enumerate(model_points)
        if model_point.supports_n_ip(target.n_ip)
        for dataset_repeat in range(dataset_repetitions)
        for sample_size in sample_sizes
    ]
    work_per_point = [
        _estimated_objective_work(
            targets[target_index].n_ip,
            model_points[model_index].kind,
            model_points[model_index].n_hidden,
            model_points[model_index].pair_mode,
        )
        for target_index, _, model_index, _, _ in grid
    ]
    total_work = settings.restarts * sum(work_per_point)
    process = psutil.Process()
    process.cpu_percent(interval=None)
    progress = tqdm(
        total=total_work,
        desc=f"{config['name']} exact-MLE work",
        unit="work",
        unit_scale=True,
        dynamic_ncols=True,
    )

    rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    try:
        for point_index, point in enumerate(grid):
            target_index, bound_index, model_index, dataset_repeat, sample_size = point
            target = targets[target_index]
            weight_bound = weight_bounds[bound_index]
            model_point = model_points[model_index]
            model = _make_model(
                model_point.kind,
                2 * target.n_ip,
                model_point.n_hidden,
                model_point.pair_mode,
            )
            train_dataset = train_datasets[(target_index, dataset_repeat, sample_size)]
            validation_dataset = validation_datasets[(target_index, dataset_repeat)]
            reference = references[(target_index, bound_index, model_index)]
            reference_settings = model_point.population_reference
            initialize_from_reference = bool(
                reference is not None
                and reference_settings is not None
                and reference_settings.initialize
            )
            initial_theta = reference.theta if initialize_from_reference and reference else None
            perturb_scale = (
                reference_settings.perturb_scale
                if initialize_from_reference and reference_settings
                else None
            )
            point_work = work_per_point[point_index]

            def report_restart(
                run: LearningRestartResult,
                best_empirical_nll: float,
                point_work: int = point_work,
                point_index: int = point_index,
                model_point: LearningModelPoint = model_point,
                target: ExactTarget = target,
                sample_size: int = sample_size,
                dataset_repeat: int = dataset_repeat,
            ) -> None:
                memory = psutil.virtual_memory()
                progress.update(point_work)
                progress.set_postfix(
                    {
                        "point": f"{point_index + 1}/{len(grid)}",
                        "model": model_point.kind,
                        "n": target.n_ip,
                        "m": model_point.n_hidden,
                        "N": sample_size,
                        "data": f"{dataset_repeat + 1}/{dataset_repetitions}",
                        "restart": f"{run.restart + 1}/{settings.restarts}",
                        "train": f"{best_empirical_nll:.3e}",
                        "pop_KL": f"{run.population_kl:.2e}",
                        "CPU": f"{process.cpu_percent(interval=None):.0f}%",
                        "RSS": _format_bytes(process.memory_info().rss),
                        "free": _format_bytes(memory.available),
                    },
                    refresh=False,
                )

            result = optimize_exact_mle(
                model,
                target,
                train_dataset,
                weight_bound=weight_bound,
                settings=settings,
                validation_dataset=validation_dataset,
                initial_theta=initial_theta,
                perturb_scale=perturb_scale,
                restart_callback=report_restart,
            )
            best = result.best
            effective_pair_mode = model_point.pair_mode if model_point.kind == "3rbm" else "none"
            initialization_mode = (
                "population_reference" if initialize_from_reference else "zero_random"
            )
            model_stem = _parameter_stem(
                model_point.kind,
                target.n_ip,
                model_point.n_hidden,
                target.beta,
                weight_bound,
                model_point.pair_mode,
            )
            fit_stem = f"{model_stem}_N{sample_size}_data{dataset_repeat:03d}"
            parameter_path = parameter_dir / f"{fit_stem}.npz"
            np.savez_compressed(
                parameter_path,
                theta_best=best.theta,
                theta_runs=np.stack([run.theta for run in result.runs]),
                empirical_nll_runs=np.asarray([run.empirical_nll for run in result.runs]),
                population_kl_runs=np.asarray([run.population_kl for run in result.runs]),
                validation_nll_runs=np.asarray([run.validation_nll for run in result.runs]),
                initialization_runs=np.asarray([run.initialization for run in result.runs]),
                ip_correlation=result.ip_correlation,
            )
            reference_kl = reference.population_kl if reference is not None else float("nan")
            reference_source = str(reference.source_file) if reference is not None else ""
            validation_seed = validation_dataset.seed if validation_dataset is not None else -1
            validation_nll = best.validation_nll
            row_common: dict[str, Any] = {
                "experiment": config["name"],
                "model": model_point.kind,
                "pair_mode": effective_pair_mode,
                "n_ip": target.n_ip,
                "n_visible": 2 * target.n_ip,
                "n_hidden": model_point.n_hidden,
                "n_parameters": model.n_parameters,
                "beta": target.beta,
                "weight_bound": _weight_bound_label(weight_bound),
                "sample_size": sample_size,
                "dataset_repeat": dataset_repeat,
                "dataset_seed": train_dataset.seed,
                "dataset_file": str(dataset_files[(target_index, dataset_repeat)]),
                "validation_size": validation_size,
                "validation_seed": validation_seed,
                "initialization_mode": initialization_mode,
                "population_reference_source": reference_source,
                "population_reference_kl": reference_kl,
            }
            rows.append(
                {
                    **row_common,
                    "target_entropy": float(-torch.sum(target.prob * target.log_prob)),
                    "empirical_entropy": train_dataset.entropy,
                    "empirical_nll": best.empirical_nll,
                    "empirical_kl": best.empirical_kl,
                    "population_nll": best.population_nll,
                    "population_kl": best.population_kl,
                    "population_kl_minus_reference": best.population_kl - reference_kl,
                    "generalization_gap": best.population_nll - best.empirical_nll,
                    "validation_nll": validation_nll,
                    "validation_kl": best.validation_kl,
                    "validation_gap": validation_nll - best.empirical_nll,
                    "ip_correlation": result.ip_correlation,
                    "best_restart": best.restart,
                    "best_initialization": best.initialization,
                    "optimizer_success": best.success,
                    "optimizer_message": best.message,
                    "iterations": best.iterations,
                    "evaluations": best.evaluations,
                    "gradient_inf_norm": best.gradient_inf_norm,
                    "projected_gradient_inf_norm": best.projected_gradient_inf_norm,
                    "fraction_at_bound": _fraction_at_bound(best.theta, weight_bound),
                    "max_abs_weight": float(np.max(np.abs(best.theta))),
                    "successful_restarts": sum(run.success for run in result.runs),
                    "elapsed_seconds": sum(run.elapsed_seconds for run in result.runs),
                    "parameter_file": str(parameter_path),
                }
            )
            for run in result.runs:
                fit_rows.append(
                    {
                        **row_common,
                        "restart": run.restart,
                        "initialization": run.initialization,
                        "empirical_nll": run.empirical_nll,
                        "empirical_kl": run.empirical_kl,
                        "population_nll": run.population_nll,
                        "population_kl": run.population_kl,
                        "population_kl_minus_reference": run.population_kl - reference_kl,
                        "generalization_gap": run.population_nll - run.empirical_nll,
                        "validation_nll": run.validation_nll,
                        "validation_kl": run.validation_kl,
                        "validation_gap": run.validation_nll - run.empirical_nll,
                        "optimizer_success": run.success,
                        "optimizer_message": run.message,
                        "iterations": run.iterations,
                        "evaluations": run.evaluations,
                        "gradient_inf_norm": run.gradient_inf_norm,
                        "projected_gradient_inf_norm": run.projected_gradient_inf_norm,
                        "fraction_at_bound": _fraction_at_bound(run.theta, weight_bound),
                        "max_abs_weight": float(np.max(np.abs(run.theta))),
                        "elapsed_seconds": run.elapsed_seconds,
                        "parameter_file": str(parameter_path),
                    }
                )
    finally:
        progress.close()

    results_path = output_dir / "results.csv"
    results = pd.DataFrame(rows)
    results.to_csv(results_path, index=False)
    pd.DataFrame(fit_rows).to_csv(output_dir / "fits.csv", index=False)
    _make_summary(results).to_csv(output_dir / "summary.csv", index=False)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    optimization_metadata = asdict(settings)
    optimization_metadata["effective_maxfun"] = settings.effective_maxfun
    metadata = {
        "stage": "2a_exact_empirical_mle",
        "created_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "data_seed": data_seed,
        "nested_training_sets": True,
        "exact_partition_function": True,
        "exact_population_evaluation": True,
        "optimization": optimization_metadata,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return results_path
