"""Frozen supplementary-study workflow: manifests, exact frontiers and reporting."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import zipfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import torch
import yaml

from ip_rbm.learning_experiment import _derived_seed
from ip_rbm.models import RBM, ThreeBodyRBM
from ip_rbm.optimization import OptimizationSettings, optimize_exact
from ip_rbm.scalable_experiment import (
    _parse_models,
    _parse_targets,
    _parse_training,
    run_scalable_experiment,
)
from ip_rbm.states import all_binary_states
from ip_rbm.targets import ExactTarget, inner_product_sign


def study_plan(config: dict[str, Any]) -> dict[str, Any]:
    phase = config.get("study_type", "learning")
    if phase not in {"learning", "population"}:
        raise ValueError("study_type must be learning or population")
    if int(config.get("cpu_threads", 2)) < 1:
        raise ValueError("cpu_threads must be positive")
    models = _parse_models(config)
    targets = _parse_targets(config)
    fits = 0
    updates = 0
    for n in config["n_ip"]:
        if phase == "population" and n > 6:
            raise ValueError("population study is capped at n_ip=6")
        for target in targets:
            actual = target.make(n)
            if not math.isfinite(actual.entropy):
                raise ValueError("nonfinite target entropy")
            for model in models:
                if not model.supports(n):
                    continue
                if phase == "population":
                    fits += len(config["weight_bound"])
                else:
                    for learner in _parse_training(config):
                        mode = model.pair_mode if model.kind == "3rbm" else "none"
                        if mode in learner.pair_modes:
                            count = (
                                config["dataset_repetitions"]
                                * len(config["sample_sizes"])
                                * len(config["weight_bound"])
                            )
                            fits += count
                            updates += count * learner.settings.updates
    return {
        "name": config["name"],
        "study_type": phase,
        "fits": fits,
        "optimizer_updates": updates,
        "population_restarts": fits * config["optimization"]["restarts"]
        if phase == "population"
        else 0,
    }


def _snapshot(config_path: Path, output: Path, plan: dict[str, Any]) -> None:
    root = Path(__file__).resolve().parents[2]
    paths = sorted(
        {
            p
            for folder in ["src", "tests", "configs", "docs"]
            for p in (root / folder).rglob("*")
            if p.is_file() and p.suffix in {".py", ".yaml", ".md", ".tex"}
        }
    )
    paths += [root / name for name in ["pyproject.toml", "uv.lock", "README.md"]]
    hashes = {}
    with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            relative = str(path.relative_to(root))
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.write(path, relative)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False
    ).stdout.strip()
    manifest = {
        **plan,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "git_commit": commit,
        "dirty_worktree": bool(dirty),
        "source_sha256": hashes,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cpu_threads": torch.get_num_threads(),
        "cpu_interop_threads": torch.get_num_interop_threads(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "mps_available": torch.backends.mps.is_available(),
        "selection_rule": "maximum independent validation PLL; ties: time, learner, update",
    }
    (output / "study_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def statewise_metrics(
    scores: torch.Tensor,
    target_scores: torch.Tensor,
    logp: torch.Tensor,
    states: torch.Tensor,
    *,
    block_count: int = 0,
) -> dict[str, float]:
    logq = scores - torch.logsumexp(scores, 0)
    residual = scores - target_scores
    result = {
        "population_kl": float((logp.exp() * (logp - logq)).sum()),
        "total_variation": float(0.5 * (logp.exp() - logq.exp()).abs().sum()),
        "score_linf_up_to_constant": float((residual.max() - residual.min()) / 2),
        "log_probability_linf": float((logp - logq).abs().max()),
    }
    sign = inner_product_sign(states) > 0
    result["global_ip_margin"] = float(scores[sign].min() - scores[~sign].max())
    result["conditional_block_margin"] = math.nan
    if block_count:
        n = states.shape[1] // 2
        size = n // block_count
        margins = []
        for block in range(block_count):
            coords = list(range(block * size, (block + 1) * size))
            coords += [n + i for i in coords]
            others = [i for i in range(2 * n) if i not in coords]
            group = (states[:, others].long() * (2 ** torch.arange(len(others)))).sum(1)
            signs = inner_product_sign(states[:, coords]) > 0
            for value in torch.unique(group):
                mask = group == value
                margins.append(float(scores[mask & signs].min() - scores[mask & ~signs].max()))
        result["conditional_block_margin"] = min(margins)
    return result


def run_population_study(config: dict[str, Any], output: Path) -> Path:
    torch.set_num_threads(int(config.get("cpu_threads", 2)))
    settings = OptimizationSettings(**config["optimization"])
    parameter_dir = output / "parameters"
    parameter_dir.mkdir(exist_ok=True)
    rows: list[dict[str, Any]] = []
    for n in config["n_ip"]:
        states = all_binary_states(2 * n)
        for ti, point in enumerate(_parse_targets(config)):
            target = point.make(n)
            score = target.log_unnormalized(states)
            logp = target.log_prob(states)
            exact = ExactTarget(
                states, inner_product_sign(states), logp, logp.exp(), n, target.beta
            )
            for mi, model_point in enumerate(_parse_models(config)):
                if not model_point.supports(n):
                    continue
                model = (
                    RBM(2 * n, model_point.n_hidden)
                    if model_point.kind == "rbm"
                    else ThreeBodyRBM(2 * n, model_point.n_hidden, model_point.pair_mode)
                )
                for bi, bound in enumerate(config["weight_bound"]):
                    fitted = optimize_exact(
                        model,
                        exact,
                        weight_bound=bound,
                        settings=replace(
                            settings, seed=_derived_seed(settings.seed, n, ti, mi, bi)
                        ),
                    )
                    for run in fitted.runs:
                        path = (
                            parameter_dir / f"{point.name}_n{n}_model{mi}_B{bi}_r{run.restart}.npz"
                        )
                        np.savez_compressed(path, theta=run.theta)
                        with torch.no_grad():
                            metrics = statewise_metrics(
                                model.log_unnormalized(states, torch.from_numpy(run.theta)),
                                score,
                                logp,
                                states,
                                block_count=point.block_count if point.kind == "block_ip" else 0,
                            )
                        row = {
                            **point.columns(n),
                            "n_ip": n,
                            "n_visible": 2 * n,
                            "model": model_point.kind,
                            "pair_mode": model_point.pair_mode
                            if model_point.kind == "3rbm"
                            else "none",
                            "comparison_group": model_point.comparison_group,
                            "n_hidden": model.n_hidden,
                            "n_parameters": model.n_parameters,
                            "weight_bound": bound,
                            "parameter_file": str(path),
                            **{k: v for k, v in asdict(run).items() if k != "theta"},
                            **metrics,
                        }
                        rows.append(row)
                    # Incremental persistence also retains unsuccessful optimizer returns.
                    pd.DataFrame(rows).to_csv(output / "restarts.csv", index=False)
                    print(
                        f"{point.name} n={n} {model_point.kind}/{model_point.pair_mode} "
                        f"m={model.n_hidden}: best KL={fitted.best.kl:.6g}",
                        flush=True,
                    )
    restarts = pd.DataFrame(rows)
    keys = [
        "target_name",
        "n_ip",
        "model",
        "pair_mode",
        "n_hidden",
        "comparison_group",
        "weight_bound",
    ]
    best = restarts.loc[restarts.groupby(keys, dropna=False).population_kl.idxmin()]
    path = output / "results.csv"
    best.to_csv(path, index=False)
    return path


def select_checkpoints(checkpoints: pd.DataFrame) -> pd.DataFrame:
    """No test or population metric enters selection; capacities remain separate."""
    metric = "validation_log_pseudolikelihood"
    if metric not in checkpoints or not np.isfinite(checkpoints[metric]).all():
        raise ValueError("selection requires finite independent validation scores for every row")
    keys = [
        "target_name",
        "n_ip",
        "model",
        "pair_mode",
        "n_hidden",
        "comparison_group",
        "weight_bound",
        "sample_size",
        "dataset_repeat",
    ]
    ordered = checkpoints.sort_values(
        [metric, "training_elapsed_seconds", "training_name", "update"],
        ascending=[False, True, True, True],
        kind="stable",
    )
    selected = ordered.drop_duplicates(keys).copy()
    totals = checkpoints.groupby(
        [*keys, "training_name"], dropna=False
    ).training_elapsed_seconds.max()
    totals = (
        totals.groupby(level=list(range(len(keys))), dropna=False)
        .sum()
        .rename("total_tuning_training_seconds")
    )
    return selected.merge(totals.reset_index(), on=keys, validate="one_to_one")


def lattice_observables(
    probabilities: torch.Tensor, states: torch.Tensor, beta: float
) -> dict[str, float]:
    """Exact finite-volume observables; susceptibility uses the signed magnetization."""
    side = math.isqrt(states.shape[1])
    if side * side != states.shape[1] or side < 3:
        raise ValueError("a square lattice with side >= 3 is required")
    spins = (2 * states - 1).reshape(-1, side, side)
    energy = -(spins * (spins.roll(1, 1) + spins.roll(1, 2))).sum((1, 2))
    magnetization = spins.sum((1, 2))

    def mean(values: torch.Tensor) -> float:
        return float(probabilities @ values)

    volume = states.shape[1]
    m2 = mean(magnetization**2)
    return {
        "energy_per_spin": mean(energy) / volume,
        "absolute_magnetization_per_spin": mean(magnetization.abs()) / volume,
        "susceptibility_per_spin": beta * (m2 - mean(magnetization) ** 2) / volume,
        "heat_capacity_per_spin": beta**2 * (mean(energy**2) - mean(energy) ** 2) / volume,
        "binder_cumulant": 1 - mean(magnetization**4) / (3 * m2**2),
    }


def _report_lattice(selected: pd.DataFrame, output: Path) -> None:
    from ip_rbm.benchmark_targets import BenchmarkTarget

    rows = []
    for _, row in selected.loc[selected.target_family == "lattice_ising"].iterrows():
        n = int(row.n_ip)
        states = all_binary_states(2 * n)
        model = (
            RBM(2 * n, int(row.n_hidden))
            if row.model == "rbm"
            else ThreeBodyRBM(
                2 * n, int(row.n_hidden), cast(Literal["cross", "all"], row.pair_mode)
            )
        )
        with np.load(row.parameter_file) as parameters:
            theta = torch.tensor(parameters["theta"], dtype=torch.float64)
        with torch.no_grad():
            scores = torch.cat(
                [
                    model.log_unnormalized(states[start : start + 1024], theta)
                    for start in range(0, len(states), 1024)
                ]
            )
        target = BenchmarkTarget(n, float(row.beta), "lattice_ising")
        record = {
            key: row[key]
            for key in [
                "target_name",
                "n_ip",
                "model",
                "pair_mode",
                "dataset_repeat",
                "training_name",
                "update",
            ]
        }
        for label, probabilities in [
            ("target", target.probabilities),
            ("model", scores.softmax(0)),
        ]:
            record.update(
                {
                    f"{label}_{key}": value
                    for key, value in lattice_observables(
                        probabilities, states, float(row.beta)
                    ).items()
                }
            )
        rows.append(record)
    if rows:
        pd.DataFrame(rows).to_csv(output / "lattice_observables.csv", index=False)


def _resource_figures(checkpoints: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    for (target, n, tier), data in checkpoints.groupby(["target_name", "n_ip", "comparison_group"]):
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        metric = (
            "population_kl" if data.population_kl.notna().all() else "normalized_target_score_rmse"
        )
        for (model, learner), group in data.groupby(["model", "training_name"]):
            label = f"{model}/{learner}"
            medians = group.groupby("update")[["training_elapsed_seconds", metric]].median()
            axes[0].plot(medians.training_elapsed_seconds, medians[metric], ".-", label=label)
            final = group.sort_values("update").drop_duplicates("dataset_repeat", keep="last")
            axes[1].scatter(final.training_peak_rss_bytes / 2**20, final[metric], label=label)
            axes[2].scatter(
                final.training_cpu_user_seconds + final.training_cpu_system_seconds,
                final[metric],
                label=label,
            )
        axes[0].set_xlabel("Training wall time (s); median by checkpoint")
        axes[1].set_xlabel("Full-fit process peak RSS (MiB)")
        axes[2].set_xlabel("Full-fit process CPU time (s)")
        for ax in axes:
            ax.set_ylabel(metric)
        axes[0].legend(fontsize=6)
        fig.suptitle(f"{target}, n={n}, {tier}")
        fig.tight_layout()
        fig.savefig(output / f"resources_{target}_n{n}_{tier}.pdf")
        plt.close(fig)


def report_study(results_path: str | Path) -> Path:
    from ip_rbm.plotting import plot_scalable_cdk_curves, plot_scalable_objective_curves

    path = Path(results_path)
    output = path.parent
    if (output / "checkpoints.csv").exists():
        checkpoints = pd.read_csv(output / "checkpoints.csv")
        selected = select_checkpoints(checkpoints)
        selected.to_csv(output / "selected.csv", index=False)
        _report_lattice(selected, output)
        _resource_figures(checkpoints, output)
        keys = ["target_name", "n_ip", "model", "pair_mode", "n_parameters", "comparison_group"]
        metrics = [
            "population_kl",
            "normalized_target_score_rmse",
            "training_elapsed_seconds",
            "total_tuning_training_seconds",
        ]
        selected.groupby(keys)[metrics].agg(["median", "min", "max"]).to_csv(
            output / "selected_summary.csv"
        )
        attainment = []
        for key, group in checkpoints.groupby([*keys, "training_name", "dataset_repeat"]):
            error = group.normalized_target_score_rmse
            metric = "normalized_target_score_rmse"
            if error.isna().all():
                error = group.population_kl / group.uniform_baseline_kl
                metric = "population_kl_over_uniform_baseline"
            for threshold in [0.25, 0.5]:
                reached = group[error <= threshold].sort_values("training_elapsed_seconds")
                attainment.append(
                    {
                        **dict(zip([*keys, "training_name", "dataset_repeat"], key, strict=True)),
                        "threshold": threshold,
                        "metric": metric,
                        "reached": not reached.empty,
                        "first_observed_seconds": float(reached.training_elapsed_seconds.iloc[0])
                        if not reached.empty
                        else math.nan,
                        "censoring_seconds": float(group.training_elapsed_seconds.max()),
                    }
                )
        pd.DataFrame(attainment).to_csv(output / "attainment.csv", index=False)
        if checkpoints.gibbs_steps.nunique() > 1:
            plot_scalable_cdk_curves(output / "checkpoints.csv")
        if checkpoints.normalized_target_score_rmse.notna().any():
            plot_scalable_objective_curves(output / "checkpoints.csv")
    else:
        import matplotlib.pyplot as plt

        results = pd.read_csv(path)
        for (target, n), data in results.groupby(["target_name", "n_ip"]):
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            for (model, pairs), group in data.groupby(["model", "pair_mode"]):
                for ax, x in zip(axes, ["n_hidden", "n_parameters"], strict=True):
                    ordered = group.sort_values(x)
                    ax.plot(ordered[x], ordered.population_kl, "o-", label=f"{model}/{pairs}")
                    ax.set_xlabel(x)
                    ax.set_ylabel("Exact population KL (best found)")
            axes[0].legend()
            fig.suptitle(f"{target}, n={n}")
            fig.tight_layout()
            fig.savefig(output / f"frontier_{target}_n{n}.pdf")
            plt.close(fig)
    return output


def run_study(config_path: str | Path, *, dry_run: bool = False) -> Path:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    plan = study_plan(config)
    print(json.dumps(plan, indent=2), flush=True)
    output = Path(config["output_dir"])
    if dry_run:
        return output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Study output is not empty: {output}; choose a fresh output directory"
        )
    if config.get("clean_output_dir", False):
        raise ValueError("paper studies must use clean_output_dir: false")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(config.get("cpu_threads", 2)))
    (output / "protocol.yaml").write_text(config_path.read_text())
    _snapshot(config_path, output, plan)
    path = (
        run_population_study(config, output)
        if plan["study_type"] == "population"
        else run_scalable_experiment(config_path)
    )
    report_study(path)
    return path
