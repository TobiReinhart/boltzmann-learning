"""Standard plots for exact representability and learning result tables."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes


def plot_frontier(results_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Plot best observed KL against hidden width."""
    results_path = Path(results_path)
    results = pd.read_csv(results_path)
    if output_path is None:
        output_path = results_path.with_name("kl_vs_hidden.pdf")
    output_path = Path(output_path)

    sns.set_theme(context="paper", style="ticks")
    results = results.assign(plot_kl=np.maximum(results["best_kl"], 1e-15))
    grid = sns.relplot(
        data=results,
        x="n_hidden",
        y="plot_kl",
        hue="model",
        style="weight_bound",
        col="n_ip",
        row="beta",
        kind="line",
        marker="o",
        markersize=4.5,
        dashes=False,
        estimator=None,
        facet_kws={"sharex": True, "sharey": True, "margin_titles": False},
        height=3.0,
        aspect=1.15,
    )
    grid.set_axis_labels(
        "Hidden units $m$", r"Best observed $D_{\mathrm{KL}}(p^\star\Vert q_\theta)$"
    )
    grid.set_titles(template=r"$n={col_name},\ \beta={row_name}$")
    grid.set(yscale="log")
    grid.figure.suptitle("Exact-enumeration approximation frontier", y=1.02)
    sns.despine(fig=grid.figure)
    grid.savefig(output_path, bbox_inches="tight")
    plt.close(grid.figure)
    return output_path


def plot_learning_curve(results_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Plot exact population KL against the number of training observations."""
    results_path = Path(results_path)
    results = pd.read_csv(results_path)
    if output_path is None:
        output_path = results_path.with_name("population_kl_vs_samples.pdf")
    output_path = Path(output_path)

    required = {
        "model",
        "n_hidden",
        "n_parameters",
        "n_ip",
        "beta",
        "sample_size",
        "population_kl",
        "weight_bound",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"learning result table is missing columns: {sorted(missing)}")

    sns.set_theme(context="paper", style="ticks")
    results = results.assign(
        plot_kl=np.maximum(results["population_kl"], 1e-15),
        model_label=results.apply(
            lambda row: (
                f"{str(row['model']).upper()} "
                f"($m={int(row['n_hidden'])}$, $P={int(row['n_parameters'])}$)"
            ),
            axis=1,
        ),
    )
    grid = sns.relplot(
        data=results,
        x="sample_size",
        y="plot_kl",
        hue="model_label",
        style="weight_bound",
        col="n_ip",
        row="beta",
        kind="line",
        markers=True,
        dashes=False,
        estimator="median",
        errorbar=("pi", 50),
        facet_kws={"sharex": True, "sharey": True, "margin_titles": False},
        height=3.0,
        aspect=1.2,
    )
    grid.set_axis_labels(
        "Training observations $N$",
        r"Exact $D_{\mathrm{KL}}(p^\star\Vert q_{\hat\theta_N})$",
    )
    grid.set_titles(template=r"$n={col_name},\ \beta={row_name}$")
    grid.set(xscale="log", yscale="log")
    grid.figure.suptitle("Exact finite-sample maximum-likelihood learning", y=1.02)
    sns.despine(fig=grid.figure)
    grid.savefig(output_path, bbox_inches="tight")
    plt.close(grid.figure)
    return output_path


def plot_scalable_learning(results_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Plot exact or AIS-estimated population KL for scalable learners."""
    results_path = Path(results_path)
    results = pd.read_csv(results_path)
    if output_path is None:
        output_path = results_path.with_name("scalable_population_kl_vs_samples.pdf")
    output_path = Path(output_path)
    required = {
        "model",
        "n_hidden",
        "n_parameters",
        "n_ip",
        "beta",
        "sample_size",
        "population_kl",
        "training_name",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"scalable result table is missing columns: {sorted(missing)}")
    finite = np.isfinite(results["population_kl"])
    if not finite.any():
        raise ValueError("no exact or AIS population-KL estimates are available to plot")

    sns.set_theme(context="paper", style="ticks")
    results = results.loc[finite].copy()
    facet_row = "target_name" if "target_name" in results.columns else "beta"
    results = results.assign(
        plot_kl=np.maximum(results["population_kl"], 1e-15),
        model_label=results.apply(
            lambda row: (
                f"{str(row['model']).upper()} {row['training_name']} "
                f"($m={int(row['n_hidden'])}$, $P={int(row['n_parameters'])}$)"
            ),
            axis=1,
        ),
    )
    grid = sns.relplot(
        data=results,
        x="sample_size",
        y="plot_kl",
        hue="model_label",
        style="model_label",
        col="n_ip",
        row=facet_row,
        kind="line",
        markers=True,
        dashes=False,
        estimator="median",
        errorbar=("pi", 50),
        facet_kws={"sharex": True, "sharey": True, "margin_titles": False},
        height=3.1,
        aspect=1.25,
    )
    grid.set_axis_labels(
        "Training observations $N$",
        r"Exact/AIS $D_{\mathrm{KL}}(p^\star\Vert q_{\hat\theta})$",
    )
    if facet_row == "target_name":
        grid.set_titles(template=r"$n={col_name}$, target={row_name}")
    else:
        grid.set_titles(template=r"$n={col_name},\ \beta={row_name}$")
    grid.set(xscale="log", yscale="log")
    grid.figure.suptitle("Scalable RBM and 3RBM learning", y=1.02)
    sns.despine(fig=grid.figure)
    grid.savefig(output_path, bbox_inches="tight")
    plt.close(grid.figure)
    return output_path


def _filename_value(value: object) -> str:
    return str(value).replace(".", "p").replace("/", "-").replace(" ", "-")


def _architecture_label(row: pd.Series) -> str:
    model = str(row["model"]).upper()
    if str(row["model"]).lower() == "3rbm":
        pair_mode = str(row.get("pair_mode", "cross"))
        return f"{model}-{pair_mode}"
    return model


def _resource_model_label(row: pd.Series) -> str:
    return (
        f"{_architecture_label(row)} {row['training_name']} "
        f"($m={int(row['n_hidden'])}$, $P={int(row['n_parameters'])}$)"
    )


def _cdk_model_label(row: pd.Series) -> str:
    sampler = str(row.get("all_pairs_sampler", "legacy"))
    suffix = ""
    if row.get("pair_mode") == "all":
        suffix = f" {sampler}"
        if sampler == "block":
            suffix += f"{int(row['all_pairs_block_size'])}"
    return (
        f"{_architecture_label(row)}{suffix} CD-{int(row['gibbs_steps'])} "
        f"($m={int(row['n_hidden'])}$, $P={int(row['n_parameters'])}$)"
    )


def _optional_group_columns(data: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in data.columns]


def _plot_resource_metric(
    axis: Axes,
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    target: float | None = None,
) -> None:
    finite = np.isfinite(data[x]) & np.isfinite(data[y])
    if not finite.any():
        axis.text(0.5, 0.5, "Metric unavailable", ha="center", va="center")
        axis.set_axis_off()
        return
    sns.lineplot(
        data=data.loc[finite],
        x=x,
        y=y,
        hue="model_label",
        style="model_label",
        markers=True,
        dashes=False,
        estimator="median",
        errorbar=("pi", 50),
        ax=axis,
    )
    axis.set_xscale("log")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    if target is not None:
        axis.axhline(target, color="black", linewidth=0.8, linestyle=":")
    legend = axis.get_legend()
    if legend is not None:
        legend.remove()


def plot_scalable_resource_curves(
    checkpoints_path: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    """Write one focused resource-curve PDF per fixed experimental condition."""
    checkpoints_path = Path(checkpoints_path)
    checkpoints = pd.read_csv(checkpoints_path)
    required = {
        "model",
        "comparison_group",
        "n_ip",
        "n_hidden",
        "n_parameters",
        "beta",
        "weight_bound",
        "sample_size",
        "training_name",
        "dataset_repeat",
        "update",
        "training_examples_processed",
        "training_elapsed_seconds",
        "normalized_kl_improvement",
        "score_gap_ratio",
        "normalized_score_shape_rmse",
        "training_peak_rss_bytes",
        "training_rss_increase_bytes",
        "training_cpu_user_seconds",
        "training_cpu_system_seconds",
        "training_peak_mps_tensor_bytes",
        "training_peak_cuda_allocated_bytes",
    }
    missing = required.difference(checkpoints.columns)
    if missing:
        raise ValueError(f"checkpoint table is missing columns: {sorted(missing)}")
    if output_dir is None:
        output_dir = checkpoints_path.with_name("resource_plots")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="paper", style="ticks")
    checkpoints = checkpoints.assign(model_label=checkpoints.apply(_resource_model_label, axis=1))
    condition_columns = _optional_group_columns(
        checkpoints,
        [
            "target_name",
            "n_ip",
            "sample_size",
            "beta",
            "weight_bound",
            "comparison_group",
            "training_name",
        ],
    )
    for _, data in checkpoints.groupby(condition_columns, dropna=False, sort=True):
        condition = data.iloc[0]
        target_name = condition.get("target_name", f"ip-beta{condition['beta']:g}")
        n_ip = condition["n_ip"]
        sample_size = condition["sample_size"]
        beta = condition["beta"]
        weight_bound = condition["weight_bound"]
        comparison_group = condition["comparison_group"]
        training_name = condition["training_name"]
        figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.6))
        _plot_resource_metric(
            axes[0, 0],
            data,
            x="training_examples_processed",
            y="normalized_kl_improvement",
            xlabel="Training examples processed",
            ylabel="KL improvement over uniform",
            target=1.0,
        )
        _plot_resource_metric(
            axes[0, 1],
            data,
            x="training_elapsed_seconds",
            y="normalized_kl_improvement",
            xlabel="Training wall time (s)",
            ylabel="KL improvement over uniform",
            target=1.0,
        )
        _plot_resource_metric(
            axes[1, 0],
            data,
            x="training_examples_processed",
            y="normalized_score_shape_rmse",
            xlabel="Training examples processed",
            ylabel=r"Normalized score-shape RMSE",
            target=0.0,
        )
        _plot_resource_metric(
            axes[1, 1],
            data,
            x="training_elapsed_seconds",
            y="normalized_score_shape_rmse",
            xlabel="Training wall time (s)",
            ylabel=r"Normalized score-shape RMSE",
            target=0.0,
        )
        handles, labels = axes[1, 1].get_legend_handles_labels()
        if not handles:
            handles, labels = axes[1, 0].get_legend_handles_labels()
        if handles:
            figure.legend(
                handles,
                labels,
                loc="outside lower center",
                ncol=min(len(handles), 3),
                frameon=False,
            )
        figure.suptitle(
            f"Resource comparison: {target_name}, n={int(n_ip)}, N={int(sample_size)}, "
            f"beta={beta:g}, B={weight_bound}, {comparison_group}, {training_name}"
        )
        figure.tight_layout(rect=(0, 0.08, 1, 0.95))
        sns.despine(fig=figure)
        filename = (
            f"{_filename_value(target_name)}_n{int(n_ip)}_N{int(sample_size)}_"
            f"beta{_filename_value(beta)}_"
            f"B{_filename_value(weight_bound)}_{comparison_group}_{training_name}.pdf"
        )
        figure.savefig(output_dir / filename, bbox_inches="tight")
        plt.close(figure)

    final_group_columns = _optional_group_columns(
        checkpoints,
        [
            "target_name",
            "n_ip",
            "sample_size",
            "beta",
            "weight_bound",
            "comparison_group",
            "training_name",
            "model",
            "n_hidden",
            "dataset_repeat",
        ],
    )
    final_update = checkpoints.groupby(
        final_group_columns,
        dropna=False,
    )["update"].transform("max")
    final = checkpoints.loc[checkpoints["update"] == final_update].copy()
    final = final.assign(
        peak_rss_mib=final["training_peak_rss_bytes"] / (1024.0**2),
        rss_increase_mib=final["training_rss_increase_bytes"] / (1024.0**2),
        cpu_seconds=(final["training_cpu_user_seconds"] + final["training_cpu_system_seconds"]),
        cpu_seconds_per_wall_second=(
            final["training_cpu_user_seconds"] + final["training_cpu_system_seconds"]
        )
        / final["training_elapsed_seconds"],
        peak_accelerator_mib=np.maximum(
            final["training_peak_mps_tensor_bytes"],
            final["training_peak_cuda_allocated_bytes"],
        )
        / (1024.0**2),
    )
    scaling_columns = _optional_group_columns(
        final,
        [
            "target_name",
            "sample_size",
            "beta",
            "weight_bound",
            "comparison_group",
            "training_name",
        ],
    )
    for _, data in final.groupby(scaling_columns, dropna=False, sort=True):
        condition = data.iloc[0]
        target_name = condition.get("target_name", f"ip-beta{condition['beta']:g}")
        sample_size = condition["sample_size"]
        beta = condition["beta"]
        weight_bound = condition["weight_bound"]
        comparison_group = condition["comparison_group"]
        training_name = condition["training_name"]
        figure, axes = plt.subplots(2, 3, figsize=(10.5, 6.0))
        for axis, metric, label in zip(
            axes.flat,
            [
                "training_elapsed_seconds",
                "cpu_seconds",
                "cpu_seconds_per_wall_second",
                "peak_rss_mib",
                "rss_increase_mib",
                "peak_accelerator_mib",
            ],
            [
                "Training wall time (s)",
                "Process CPU time (s)",
                "CPU seconds / wall second",
                "Peak process RSS (MiB)",
                "RSS increase during fit (MiB)",
                "Peak accelerator tensors (MiB)",
            ],
            strict=True,
        ):
            sns.lineplot(
                data=data,
                x="n_ip",
                y=metric,
                hue="model_label",
                style="model_label",
                markers=True,
                dashes=False,
                estimator="median",
                errorbar=("pi", 50),
                ax=axis,
            )
            axis.set_xlabel("IP register size $n$")
            axis.set_ylabel(label)
            legend = axis.get_legend()
            if legend is not None:
                legend.remove()
        handles, labels = axes[-1, -1].get_legend_handles_labels()
        if not handles:
            handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            figure.legend(
                handles,
                labels,
                loc="outside lower center",
                ncol=min(len(handles), 3),
                frameon=False,
            )
        figure.suptitle(
            f"Resource scaling: {target_name}, N={int(sample_size)}, beta={beta:g}, "
            f"B={weight_bound}, {comparison_group}, {training_name}"
        )
        figure.tight_layout(rect=(0, 0.1, 1, 0.94))
        sns.despine(fig=figure)
        filename = (
            f"resource_scaling_{_filename_value(target_name)}_N{int(sample_size)}_"
            f"beta{_filename_value(beta)}_"
            f"B{_filename_value(weight_bound)}_{comparison_group}_{training_name}.pdf"
        )
        figure.savefig(output_dir / filename, bbox_inches="tight")
        plt.close(figure)
    return output_dir


def _plot_objective_metric(
    axis: Axes,
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    target: float | None = None,
    log_y: bool = False,
) -> None:
    _plot_resource_metric(
        axis,
        data,
        x=x,
        y=y,
        xlabel=xlabel,
        ylabel=ylabel,
        target=target,
    )
    finite_positive = np.isfinite(data[y]) & (data[y] > 0)
    if log_y and finite_positive.any() and axis.axison:
        axis.set_yscale("log")


def plot_scalable_objective_curves(
    checkpoints_path: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    """Plot focused architecture comparisons for each named target objective."""
    checkpoints_path = Path(checkpoints_path)
    checkpoints = pd.read_csv(checkpoints_path)
    required = {
        "target_name",
        "model",
        "comparison_group",
        "n_ip",
        "n_hidden",
        "n_parameters",
        "weight_bound",
        "sample_size",
        "training_name",
        "dataset_repeat",
        "update",
        "training_examples_processed",
        "training_elapsed_seconds",
        "population_kl",
        "normalized_target_score_rmse",
        "target_score_correlation",
    }
    missing = required.difference(checkpoints.columns)
    if missing:
        raise ValueError(f"checkpoint table is missing columns: {sorted(missing)}")
    if output_dir is None:
        output_dir = checkpoints_path.with_name("objective_plots")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="paper", style="ticks")
    checkpoints = checkpoints.assign(model_label=checkpoints.apply(_resource_model_label, axis=1))
    condition_columns = [
        "n_ip",
        "sample_size",
        "weight_bound",
        "comparison_group",
    ]
    checkpoints["model_label"] = checkpoints["model_label"] + " / " + checkpoints["training_name"]
    for _, condition_data in checkpoints.groupby(
        condition_columns,
        dropna=False,
        sort=True,
    ):
        condition = condition_data.iloc[0]
        n_ip = int(condition["n_ip"])
        sample_size = int(condition["sample_size"])
        weight_bound = condition["weight_bound"]
        comparison_group = condition["comparison_group"]
        training_name = "all-learners"
        for target_name, data in condition_data.groupby("target_name", sort=True):
            figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.6))
            _plot_objective_metric(
                axes[0, 0],
                data,
                x="training_examples_processed",
                y="normalized_target_score_rmse",
                xlabel="Training examples processed",
                ylabel="Normalized target-score RMSE",
                target=0.0,
            )
            _plot_objective_metric(
                axes[0, 1],
                data,
                x="training_elapsed_seconds",
                y="normalized_target_score_rmse",
                xlabel="Training wall time (s)",
                ylabel="Normalized target-score RMSE",
                target=0.0,
            )
            component_columns = {
                "learned_ip_component_ratio",
                "learned_cosine_component_ratio",
            }
            has_component_diagnostics = component_columns.issubset(data.columns) and bool(
                np.isfinite(data["learned_ip_component_ratio"]).any()
            )
            if has_component_diagnostics:
                _plot_objective_metric(
                    axes[1, 0],
                    data,
                    x="training_examples_processed",
                    y="learned_ip_component_ratio",
                    xlabel="Training examples processed",
                    ylabel="Recovered IP coefficient / target",
                    target=1.0,
                )
                _plot_objective_metric(
                    axes[1, 1],
                    data,
                    x="training_examples_processed",
                    y="learned_cosine_component_ratio",
                    xlabel="Training examples processed",
                    ylabel="Recovered cosine coefficient / target",
                    target=1.0,
                )
            else:
                _plot_objective_metric(
                    axes[1, 0],
                    data,
                    x="training_examples_processed",
                    y="population_kl",
                    xlabel="Training examples processed",
                    ylabel=r"Exact/AIS $D_{\mathrm{KL}}(p^\star\Vert q)$",
                    target=0.0,
                    log_y=True,
                )
                _plot_objective_metric(
                    axes[1, 1],
                    data,
                    x="training_examples_processed",
                    y="target_score_correlation",
                    xlabel="Training examples processed",
                    ylabel="Target/model score correlation",
                    target=1.0,
                )
            handles, labels = axes[0, 0].get_legend_handles_labels()
            if handles:
                figure.legend(
                    handles,
                    labels,
                    loc="outside lower center",
                    ncol=min(len(handles), 3),
                    frameon=False,
                )
            figure.suptitle(
                f"{target_name}: n={n_ip}, N={sample_size}, B={weight_bound}, "
                f"{comparison_group}, {training_name}"
            )
            figure.tight_layout(rect=(0, 0.08, 1, 0.95))
            sns.despine(fig=figure)
            filename = (
                f"{_filename_value(target_name)}_n{n_ip}_N{sample_size}_"
                f"B{_filename_value(weight_bound)}_{comparison_group}_{training_name}.pdf"
            )
            figure.savefig(output_dir / filename, bbox_inches="tight")
            plt.close(figure)

        final_group_columns = [
            "target_name",
            "model",
            "n_hidden",
            "dataset_repeat",
            "training_name",
        ]
        final_update = condition_data.groupby(final_group_columns, dropna=False)[
            "update"
        ].transform("max")
        final = condition_data.loc[condition_data["update"] == final_update]
        if final["target_name"].nunique() < 2:
            continue
        figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
        sns.pointplot(
            data=final,
            x="target_name",
            y="normalized_target_score_rmse",
            hue="model_label",
            estimator="median",
            errorbar=("pi", 50),
            dodge=final["model_label"].nunique() > 1,
            linestyles="none",
            ax=axes[0],
        )
        axes[0].set_xlabel("Target objective")
        axes[0].set_ylabel("Final normalized target-score RMSE")
        axes[0].tick_params(axis="x", rotation=20)
        finite_kl = np.isfinite(final["population_kl"]) & (final["population_kl"] > 0)
        if finite_kl.any():
            sns.pointplot(
                data=final.loc[finite_kl],
                x="target_name",
                y="population_kl",
                hue="model_label",
                estimator="median",
                errorbar=("pi", 50),
                dodge=final.loc[finite_kl, "model_label"].nunique() > 1,
                linestyles="none",
                ax=axes[1],
            )
            axes[1].set_yscale("log")
            axes[1].set_xlabel("Target objective")
            axes[1].set_ylabel(r"Final exact/AIS $D_{\mathrm{KL}}(p^\star\Vert q)$")
            axes[1].tick_params(axis="x", rotation=20)
        else:
            axes[1].text(0.5, 0.5, "KL unavailable", ha="center", va="center")
            axes[1].set_axis_off()
        for axis in axes:
            legend = axis.get_legend()
            if legend is not None:
                legend.remove()
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            figure.legend(
                handles,
                labels,
                loc="outside lower center",
                ncol=min(len(handles), 3),
                frameon=False,
            )
        figure.suptitle(
            f"Objective difficulty: n={n_ip}, N={sample_size}, B={weight_bound}, "
            f"{comparison_group}, {training_name}"
        )
        figure.tight_layout(rect=(0, 0.14, 1, 0.92))
        sns.despine(fig=figure)
        filename = (
            f"objective_summary_n{n_ip}_N{sample_size}_B{_filename_value(weight_bound)}_"
            f"{comparison_group}_{training_name}.pdf"
        )
        figure.savefig(output_dir / filename, bbox_inches="tight")
        plt.close(figure)
    return output_dir


def plot_scalable_cdk_curves(
    checkpoints_path: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    """Compare CD-k fits at equal update, Gibbs-sweep, and wall-time budgets."""
    checkpoints_path = Path(checkpoints_path)
    checkpoints = pd.read_csv(checkpoints_path)
    required = {
        "target_name",
        "algorithm",
        "model",
        "comparison_group",
        "n_ip",
        "n_hidden",
        "n_parameters",
        "weight_bound",
        "sample_size",
        "dataset_repeat",
        "update",
        "gibbs_steps",
        "gibbs_chain_sweeps",
        "training_elapsed_seconds",
        "normalized_target_score_rmse",
        "target_score_correlation",
    }
    missing = required.difference(checkpoints.columns)
    if missing:
        raise ValueError(f"checkpoint table is missing columns: {sorted(missing)}")
    checkpoints = checkpoints.loc[checkpoints["algorithm"] == "cd"].copy()
    if checkpoints.empty:
        raise ValueError("checkpoint table contains no CD fits")
    if checkpoints["gibbs_steps"].nunique() < 2:
        raise ValueError("CD-k plotting requires at least two distinct gibbs_steps values")
    if output_dir is None:
        output_dir = checkpoints_path.with_name("cdk_plots")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(context="paper", style="ticks")
    checkpoints = checkpoints.assign(model_label=checkpoints.apply(_cdk_model_label, axis=1))
    condition_columns = [
        "target_name",
        "n_ip",
        "sample_size",
        "weight_bound",
        "comparison_group",
    ]
    for _, data in checkpoints.groupby(condition_columns, dropna=False, sort=True):
        condition = data.iloc[0]
        figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
        _plot_resource_metric(
            axes[0, 0],
            data,
            x="update",
            y="normalized_target_score_rmse",
            xlabel="Optimizer updates (equal data exposure)",
            ylabel="Normalized target-score RMSE",
            target=0.0,
        )
        _plot_resource_metric(
            axes[0, 1],
            data,
            x="gibbs_chain_sweeps",
            y="normalized_target_score_rmse",
            xlabel="Full Gibbs sweeps (cost varies by kernel)",
            ylabel="Normalized target-score RMSE",
            target=0.0,
        )
        _plot_resource_metric(
            axes[1, 0],
            data,
            x="training_elapsed_seconds",
            y="normalized_target_score_rmse",
            xlabel="Training wall time (s)",
            ylabel="Normalized target-score RMSE",
            target=0.0,
        )
        _plot_resource_metric(
            axes[1, 1],
            data,
            x="update",
            y="target_score_correlation",
            xlabel="Optimizer updates (equal data exposure)",
            ylabel="Target/model score correlation",
            target=1.0,
        )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            figure.legend(
                handles,
                labels,
                loc="outside lower center",
                ncol=min(len(handles), 3),
                frameon=False,
            )
        target_name = str(condition["target_name"])
        n_ip = int(condition["n_ip"])
        sample_size = int(condition["sample_size"])
        weight_bound = condition["weight_bound"]
        comparison_group = str(condition["comparison_group"])
        figure.suptitle(
            f"CD-k comparison: {target_name}, n={n_ip}, N={sample_size}, "
            f"B={weight_bound}, {comparison_group}"
        )
        figure.tight_layout(rect=(0, 0.14, 1, 0.95))
        sns.despine(fig=figure)
        filename = (
            f"cdk_{_filename_value(target_name)}_n{n_ip}_N{sample_size}_"
            f"B{_filename_value(weight_bound)}_{comparison_group}.pdf"
        )
        figure.savefig(output_dir / filename, bbox_inches="tight")
        plt.close(figure)
    return output_dir
