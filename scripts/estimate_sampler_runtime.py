"""Project a sampler experiment from calibrated, post-warmup timings.

Run with uv run python scripts/estimate_sampler_runtime.py TIMING/checkpoints.csv CONFIG.yaml.
This reads files only and never launches the long experiment.
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from ip_rbm.scalable_experiment import _parse_models, _parse_targets, _parse_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    timing = pd.read_csv(args.checkpoints)
    config = yaml.safe_load(args.config.read_text())
    total = 0.0
    fits = 0
    for n in config["n_ip"]:
        for model in _parse_models(config):
            if not model.supports(n):
                continue
            mode = model.pair_mode if model.kind == "3rbm" else "none"
            for learner in _parse_training(config):
                if mode not in learner.pair_modes:
                    continue
                settings = learner.settings
                rows = timing[
                    (timing.n_ip == n)
                    & (timing.model == model.kind)
                    & (timing.pair_mode == mode)
                    & (timing.n_hidden == model.n_hidden)
                    & (timing.training_name == learner.name)
                    & (timing.batch_size == settings.batch_size)
                    & (timing.gibbs_steps == settings.gibbs_steps)
                ].sort_values("update")
                if mode == "all":
                    rows = rows[rows.all_pairs_sampler == settings.all_pairs_sampler]
                    if settings.all_pairs_sampler == "block":
                        rows = rows[rows.all_pairs_block_size == settings.all_pairs_block_size]
                if rows["update"].nunique() < 2:
                    raise ValueError(
                        f"Missing two matching calibration checkpoints: {learner.name}"
                    )
                # Median per update also permits repeated calibration fits.
                median = rows.groupby("update").training_elapsed_seconds.median().sort_index()
                rate = (median.iloc[-1] - median.iloc[-2]) / (median.index[-1] - median.index[-2])
                if rate <= 0:
                    raise ValueError("Nonpositive calibration rate")
                repetitions = (
                    len(_parse_targets(config))
                    * config["dataset_repetitions"]
                    * len(config["sample_sizes"])
                    * len(config.get("weight_bound", [None]))
                )
                seconds = rate * settings.updates
                total += seconds * repetitions
                fits += repetitions
                print(
                    f"{mode:5s} {learner.name:14s}: {rate * 1000:7.2f} ms/update; "
                    f"{seconds / 60:6.1f} min/fit x {repetitions}"
                )
    print(f"Calibration device(s): {', '.join(timing.device.unique())}")
    print(f"Projected training: {total / 3600:.1f} hours for {fits} fits.")
    print(
        f"Planning allowance (1.25-1.75x): {total * 1.25 / 3600:.1f}"
        f"-{total * 1.75 / 3600:.1f} hours; not a confidence interval or time cap."
    )
    print(
        "Projection assumes the same device and load; evaluation, thermals, and learned "
        "parameter-dependent execution costs can change runtime."
    )


if __name__ == "__main__":
    main()
