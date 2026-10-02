"""Compare exact n8 beta=1 and hard-support IP learning curves.

Run: uv run python scripts/plot_ip_beta_comparison.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(root: Path | None = None) -> None:
    root = Path(__file__).resolve().parents[1] if root is None else root
    frames = []
    for name, label in [("beta1", "beta=1"), ("beta_infinity", "beta=infinity")]:
        path = root / f"results/paper_study1_n8_{name}/checkpoints.csv"
        frame = pd.read_csv(path)
        if not frame.population_kl_source.eq("exact").all():
            raise ValueError("This comparison requires exact KL at every checkpoint")
        frame["target_label"] = label
        frame["kl_fraction"] = frame.population_kl / frame.uniform_baseline_kl
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    output = root / "results/paper_study1_n8_beta_comparison"
    output.mkdir(parents=True, exist_ok=True)
    for axis_key, axis_label in [
        ("update", "Parameter updates"),
        ("training_elapsed_seconds", "Training time (s)"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for (model, label), group in data.groupby(["model", "target_label"]):
            color = "tab:blue" if model == "rbm" else "tab:purple"
            style = "--" if label == "beta=1" else "-"
            for _, run in group.groupby("dataset_repeat"):
                run = run.sort_values("update")
                for ax, metric in zip(axes, ["kl_fraction", "population_kl"], strict=True):
                    ax.plot(run[axis_key], run[metric], style, color=color, alpha=0.2)
            median = group.groupby("update")[
                [axis_key, "kl_fraction", "population_kl"]
                if axis_key != "update"
                else ["kl_fraction", "population_kl"]
            ].median()
            x = median.index if axis_key == "update" else median[axis_key]
            for ax, metric in zip(axes, ["kl_fraction", "population_kl"], strict=True):
                ax.plot(x, median[metric], style, color=color, label=f"{model}, {label}")
        for ax, ylabel in zip(axes, ["KL / uniform-baseline KL", "Exact forward KL"], strict=True):
            ax.set(xscale="log", xlabel=axis_label, ylabel=ylabel, ylim=(0, None))
            ax.legend(fontsize=8)
        axes[0].axhline(1, color="gray", linewidth=0.6)
        fig.tight_layout()
        fig.savefig(output / f"beta_comparison_{axis_key}.pdf")
        plt.close(fig)
    rows = []
    for keys, group in data.groupby(["model", "target_label", "dataset_repeat"]):
        for threshold in [0.75, 0.5, 0.25]:
            reached = group[group.kl_fraction <= threshold].sort_values("update")
            rows.append(
                dict(zip(["model", "target_label", "dataset_repeat"], keys, strict=True))
                | {
                    "kl_fraction_threshold": threshold,
                    "reached": not reached.empty,
                    "first_update": reached["update"].iloc[0] if len(reached) else np.nan,
                    "first_seconds": reached.training_elapsed_seconds.iloc[0]
                    if len(reached)
                    else np.nan,
                }
            )
    pd.DataFrame(rows).to_csv(output / "kl_attainment.csv", index=False)
    print(output)


if __name__ == "__main__":
    main()
