"""Configuration-driven Stage-I parameter sweeps."""

from __future__ import annotations

import itertools
import json
import platform
import shutil
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

from ip_rbm.models import RBM, ExactModel, ThreeBodyRBM
from ip_rbm.optimization import OptimizationSettings, RestartResult, optimize_exact
from ip_rbm.targets import make_ip_target

WeightBound = float | None


@dataclass(frozen=True)
class ContinuationSettings:
    """Initialization from the best parameters of an earlier experiment."""

    source_dir: Path
    perturb_scale: float = 0.01

    def __post_init__(self) -> None:
        if self.perturb_scale < 0:
            raise ValueError("continuation perturb_scale must be nonnegative")


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("the experiment configuration must be a YAML mapping")
    return config


def _make_model(kind: str, n_visible: int, n_hidden: int, pair_mode: str) -> ExactModel:
    if kind == "rbm":
        return RBM(n_visible=n_visible, n_hidden=n_hidden)
    if kind == "3rbm":
        if pair_mode not in {"cross", "all"}:
            raise ValueError(f"invalid pair_mode: {pair_mode}")
        return ThreeBodyRBM(
            n_visible=n_visible,
            n_hidden=n_hidden,
            pair_mode=pair_mode,  # type: ignore[arg-type]
        )
    raise ValueError(f"unknown model kind: {kind}")


def _parameter_stem(
    kind: str,
    n_ip: int,
    n_hidden: int,
    beta: float,
    weight_bound: WeightBound,
    pair_mode: str,
) -> str:
    effective_pair_mode = pair_mode if kind == "3rbm" else "none"
    bound_label = "unbounded" if weight_bound is None else f"{weight_bound:g}"
    return f"{kind}_n{n_ip}_m{n_hidden}_beta{beta:g}_bound{bound_label}_pairs-{effective_pair_mode}"


def _weight_bound_label(weight_bound: WeightBound) -> float | str:
    return "unbounded" if weight_bound is None else weight_bound


def _parse_continuation(config: dict[str, Any], output_dir: Path) -> ContinuationSettings | None:
    raw = config.get("continuation")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("continuation must be a YAML mapping")
    if "source_dir" not in raw:
        raise ValueError("continuation requires source_dir")
    continuation = ContinuationSettings(
        source_dir=Path(str(raw["source_dir"])),
        perturb_scale=float(raw.get("perturb_scale", 0.01)),
    )

    resolved_output = output_dir.resolve()
    resolved_source = continuation.source_dir.resolve()
    if resolved_source == resolved_output or resolved_output in resolved_source.parents:
        raise ValueError("continuation source_dir must not be cleaned by output_dir")
    return continuation


def _load_continuation_theta(
    source_dir: Path,
    stem: str,
    model: ExactModel,
    weight_bound: WeightBound,
) -> tuple[NDArray[np.float64], Path]:
    source_file = source_dir / "parameters" / f"{stem}.npz"
    if not source_file.is_file():
        raise FileNotFoundError(f"continuation parameter file not found: {source_file}")
    with np.load(source_file, allow_pickle=False) as archive:
        if "theta_best" not in archive:
            raise ValueError(f"continuation file has no theta_best array: {source_file}")
        theta = np.asarray(archive["theta_best"], dtype=np.float64)
    if theta.shape != (model.n_parameters,):
        raise ValueError(
            f"continuation parameters in {source_file} have shape {theta.shape}; "
            f"expected ({model.n_parameters},)"
        )
    if not np.all(np.isfinite(theta)):
        raise ValueError(f"continuation parameters contain non-finite values: {source_file}")
    if weight_bound is not None:
        tolerance = 1e-10 * max(1.0, weight_bound)
        if np.any(np.abs(theta) > weight_bound + tolerance):
            raise ValueError(
                f"continuation parameters in {source_file} exceed weight_bound={weight_bound:g}"
            )
        theta = np.clip(theta, -weight_bound, weight_bound)
    return theta.copy(), source_file


def _prepare_output_directory(output_dir: Path, *, clean: bool) -> Path:
    """Create an experiment output directory, optionally removing an earlier run.

    Destructive cleaning is deliberately restricted to a named subdirectory of
    ``results/``. This prevents a malformed configuration from deleting the
    repository root, the complete results tree, or an unrelated directory.
    """
    if clean:
        results_root = (Path.cwd() / "results").resolve()
        resolved_output = output_dir.resolve()
        if resolved_output == results_root or results_root not in resolved_output.parents:
            raise ValueError(
                "clean_output_dir requires output_dir to be a named subdirectory of results/"
            )
        if output_dir.is_symlink():
            raise ValueError("refusing to clean a symbolic-link output directory")
        if output_dir.exists():
            if not output_dir.is_dir():
                raise ValueError("output_dir exists but is not a directory")
            print(f"Cleaning output directory: {output_dir}")
            shutil.rmtree(output_dir)

    parameter_dir = output_dir / "parameters"
    parameter_dir.mkdir(parents=True, exist_ok=True)
    return parameter_dir


def _estimated_objective_work(
    n_ip: int,
    kind: str,
    n_hidden: int,
    pair_mode: str,
) -> int:
    """Return a relative work estimate for one optimizer restart.

    Exact evaluation visits all ``4**n_ip`` visible states. The remaining
    factor approximates the tensor operations needed to evaluate the model
    and its gradient. These units are deliberately relative rather than a
    prediction of seconds.
    """
    model = _make_model(kind, 2 * n_ip, n_hidden, pair_mode)
    pair_feature_cost = model.n_pairs if isinstance(model, ThreeBodyRBM) else 0
    n_states = 1 << (2 * n_ip)
    return n_states * (model.n_parameters + pair_feature_cost)


def _format_bytes(n_bytes: int) -> str:
    value = float(n_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def run_experiment(config_path: str | Path) -> Path:
    """Execute every point in a YAML grid and return the result CSV path."""
    config_path = Path(config_path)
    config = _load_config(config_path)
    output_dir = Path(config["output_dir"])
    clean_output_dir = config.get("clean_output_dir", False)
    if not isinstance(clean_output_dir, bool):
        raise ValueError("clean_output_dir must be true or false")
    continuation = _parse_continuation(config, output_dir)

    settings = OptimizationSettings(**config["optimization"])
    model_points: list[tuple[str, int, str]] = []
    for model_config in config["models"]:
        kind = str(model_config["kind"])
        pair_mode = str(model_config.get("pair_mode", "cross"))
        model_points.extend((kind, int(hidden), pair_mode) for hidden in model_config["hidden"])

    weight_bounds: list[WeightBound] = [
        None if value is None else float(value) for value in config["weight_bound"]
    ]

    grid = list(
        itertools.product(
            [int(value) for value in config["n_ip"]],
            [float(value) for value in config["beta"]],
            weight_bounds,
            model_points,
        )
    )
    continuation_points: list[tuple[NDArray[np.float64], Path] | None] = []
    for n_ip, beta, weight_bound, (kind, n_hidden, pair_mode) in grid:
        if continuation is None:
            continuation_points.append(None)
            continue
        model = _make_model(kind, 2 * n_ip, n_hidden, pair_mode)
        stem = _parameter_stem(kind, n_ip, n_hidden, beta, weight_bound, pair_mode)
        continuation_points.append(
            _load_continuation_theta(
                continuation.source_dir,
                stem,
                model,
                weight_bound,
            )
        )
    parameter_dir = _prepare_output_directory(output_dir, clean=clean_output_dir)

    rows: list[dict[str, Any]] = []
    restart_rows: list[dict[str, Any]] = []
    work_per_point = [
        _estimated_objective_work(n_ip, kind, n_hidden, pair_mode)
        for n_ip, _, _, (kind, n_hidden, pair_mode) in grid
    ]
    total_work = settings.restarts * sum(work_per_point)
    process = psutil.Process()
    process.cpu_percent(interval=None)
    progress = tqdm(
        total=total_work,
        desc=f"{config['name']} estimated work",
        unit="work",
        unit_scale=True,
        dynamic_ncols=True,
    )
    try:
        for point_index, (n_ip, beta, weight_bound, model_point) in enumerate(grid):
            kind, n_hidden, pair_mode = model_point
            target = make_ip_target(n_ip, beta)
            model = _make_model(kind, 2 * n_ip, n_hidden, pair_mode)
            point_work = work_per_point[point_index]
            continuation_point = continuation_points[point_index]
            initial_theta = continuation_point[0] if continuation_point is not None else None
            continuation_source = (
                str(continuation_point[1]) if continuation_point is not None else ""
            )

            def report_restart(
                run: RestartResult,
                best_kl: float,
                point_work: int = point_work,
                point_index: int = point_index,
                kind: str = kind,
                n_ip: int = n_ip,
                n_hidden: int = n_hidden,
            ) -> None:
                memory = psutil.virtual_memory()
                progress.update(point_work)
                progress.set_postfix(
                    {
                        "point": f"{point_index + 1}/{len(grid)}",
                        "model": kind,
                        "n": n_ip,
                        "m": n_hidden,
                        "restart": f"{run.restart + 1}/{settings.restarts}",
                        "init": run.initialization,
                        "iter": run.iterations,
                        "best_KL": f"{best_kl:.2e}",
                        "CPU": f"{process.cpu_percent(interval=None):.0f}%",
                        "RSS": _format_bytes(process.memory_info().rss),
                        "free": _format_bytes(memory.available),
                    },
                    refresh=False,
                )

            result = optimize_exact(
                model,
                target,
                weight_bound=weight_bound,
                settings=settings,
                initial_theta=initial_theta,
                perturb_scale=(continuation.perturb_scale if continuation is not None else None),
                restart_callback=report_restart,
            )
            best = result.best
            effective_pair_mode = pair_mode if kind == "3rbm" else "none"
            stem = _parameter_stem(kind, n_ip, n_hidden, beta, weight_bound, pair_mode)
            np.savez_compressed(
                parameter_dir / f"{stem}.npz",
                theta_best=best.theta,
                theta_runs=np.stack([run.theta for run in result.runs]),
                kl_runs=np.asarray([run.kl for run in result.runs]),
                initialization_runs=np.asarray([run.initialization for run in result.runs]),
                ip_correlation=result.ip_correlation,
            )
            restart_kls = np.asarray([run.kl for run in result.runs])
            rows.append(
                {
                    "experiment": config["name"],
                    "model": kind,
                    "pair_mode": effective_pair_mode,
                    "n_ip": n_ip,
                    "n_visible": 2 * n_ip,
                    "n_hidden": n_hidden,
                    "n_parameters": model.n_parameters,
                    "beta": beta,
                    "weight_bound": _weight_bound_label(weight_bound),
                    "best_kl": best.kl,
                    "ip_correlation": result.ip_correlation,
                    "best_restart": best.restart,
                    "best_initialization": best.initialization,
                    "continuation_source": continuation_source,
                    "optimizer_success": best.success,
                    "optimizer_message": best.message,
                    "iterations": best.iterations,
                    "evaluations": best.evaluations,
                    "gradient_inf_norm": best.gradient_inf_norm,
                    "projected_gradient_inf_norm": best.projected_gradient_inf_norm,
                    "restart_kl_median": float(np.median(restart_kls)),
                    "restart_kl_q25": float(np.quantile(restart_kls, 0.25)),
                    "restart_kl_q75": float(np.quantile(restart_kls, 0.75)),
                    "successful_restarts": sum(run.success for run in result.runs),
                    "elapsed_seconds": sum(run.elapsed_seconds for run in result.runs),
                    "parameter_file": str(parameter_dir / f"{stem}.npz"),
                }
            )
            for run in result.runs:
                restart_rows.append(
                    {
                        "experiment": config["name"],
                        "model": kind,
                        "pair_mode": effective_pair_mode,
                        "n_ip": n_ip,
                        "n_visible": 2 * n_ip,
                        "n_hidden": n_hidden,
                        "n_parameters": model.n_parameters,
                        "beta": beta,
                        "weight_bound": _weight_bound_label(weight_bound),
                        "restart": run.restart,
                        "initialization": run.initialization,
                        "continuation_source": continuation_source,
                        "kl": run.kl,
                        "optimizer_success": run.success,
                        "optimizer_message": run.message,
                        "iterations": run.iterations,
                        "evaluations": run.evaluations,
                        "gradient_inf_norm": run.gradient_inf_norm,
                        "projected_gradient_inf_norm": run.projected_gradient_inf_norm,
                        "elapsed_seconds": run.elapsed_seconds,
                        "parameter_file": str(parameter_dir / f"{stem}.npz"),
                    }
                )
    finally:
        progress.close()

    results_path = output_dir / "results.csv"
    pd.DataFrame(rows).to_csv(results_path, index=False)
    pd.DataFrame(restart_rows).to_csv(output_dir / "restarts.csv", index=False)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    optimization_metadata = asdict(settings)
    optimization_metadata["effective_maxfun"] = settings.effective_maxfun
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "optimization": optimization_metadata,
        "continuation": (
            {
                "source_dir": str(continuation.source_dir),
                "perturb_scale": continuation.perturb_scale,
            }
            if continuation is not None
            else None
        ),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return results_path
