"""Command-line interface for exact representability and learning experiments."""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import torch


def _open_path(path: Path) -> None:
    """Open a generated file with the operating system's default application."""
    if sys.platform == "darwin":
        command = ["open", str(path)]
    elif sys.platform.startswith("win"):
        command = ["cmd", "/c", "start", "", str(path)]
    else:
        command = ["xdg-open", str(path)]
    subprocess.run(command, check=True)


def _print_system() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("Reference optimizer: CPU float64 (SciPy L-BFGS-B)")
    scalable_device = (
        "CUDA"
        if torch.cuda.is_available()
        else "MPS"
        if torch.backends.mps.is_available()
        else "CPU"
    )
    print(f"Scalable-training default: {scalable_device} float32 (PyTorch)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ip-rbm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("system", help="show numerical backend and accelerator availability")
    study_parser = subparsers.add_parser("run-study", help="run a frozen supplementary study")
    study_parser.add_argument("config")
    study_parser.add_argument(
        "--dry-run", action="store_true", help="validate and count; do not train"
    )
    study_report = subparsers.add_parser(
        "report-study", help="regenerate supplementary-study reports"
    )
    study_report.add_argument("results")

    run_parser = subparsers.add_parser("run", help="run a YAML-defined exact experiment grid")
    run_parser.add_argument("config", help="path to the YAML experiment configuration")

    workflow_parser = subparsers.add_parser(
        "run-plot-open",
        help="run an experiment, generate its standard plot, and open the PDF",
    )
    workflow_parser.add_argument("config", help="path to the YAML experiment configuration")

    plot_parser = subparsers.add_parser("plot", help="plot a Stage-I results.csv file")
    plot_parser.add_argument("results", help="path to results.csv")
    plot_parser.add_argument("--output", help="optional output PDF path")

    learn_parser = subparsers.add_parser(
        "learn", help="run a YAML-defined exact empirical-MLE grid (Stage 2a)"
    )
    learn_parser.add_argument("config", help="path to the YAML learning configuration")

    learn_workflow_parser = subparsers.add_parser(
        "learn-plot-open",
        help="run exact empirical MLE, plot its learning curve, and open the PDF",
    )
    learn_workflow_parser.add_argument("config", help="path to the YAML learning configuration")

    learning_plot_parser = subparsers.add_parser(
        "plot-learning", help="plot a Stage-2a results.csv file"
    )
    learning_plot_parser.add_argument("results", help="path to results.csv")
    learning_plot_parser.add_argument("--output", help="optional output PDF path")

    scalable_parser = subparsers.add_parser(
        "train-scalable", help="run a YAML-defined scalable learning grid"
    )
    scalable_parser.add_argument("config", help="path to the scalable learning configuration")

    scalable_workflow_parser = subparsers.add_parser(
        "train-scalable-plot-open",
        help="run scalable learning, plot its population KL, and open the PDF",
    )
    scalable_workflow_parser.add_argument(
        "config", help="path to the scalable learning configuration"
    )

    scalable_plot_parser = subparsers.add_parser(
        "plot-scalable", help="plot a scalable-learning results.csv file"
    )
    scalable_plot_parser.add_argument("results", help="path to results.csv")
    scalable_plot_parser.add_argument("--output", help="optional output PDF path")

    resource_plot_parser = subparsers.add_parser(
        "plot-scalable-resources",
        help="plot checkpoint learning/resource curves as a directory of focused PDFs",
    )
    resource_plot_parser.add_argument("checkpoints", help="path to checkpoints.csv")
    resource_plot_parser.add_argument("--output-dir", help="optional output directory")

    objective_plot_parser = subparsers.add_parser(
        "plot-scalable-objectives",
        help="plot named-target RBM/3RBM learning curves as focused PDFs",
    )
    objective_plot_parser.add_argument("checkpoints", help="path to checkpoints.csv")
    objective_plot_parser.add_argument("--output-dir", help="optional output directory")

    cdk_plot_parser = subparsers.add_parser(
        "plot-scalable-cdk",
        help="compare CD-k at equal updates, Gibbs sweeps, and wall time",
    )
    cdk_plot_parser.add_argument("checkpoints", help="path to checkpoints.csv")
    cdk_plot_parser.add_argument("--output-dir", help="optional output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "system":
        _print_system()
    elif args.command == "run-study":
        from ip_rbm.paper_studies import run_study

        print(run_study(args.config, dry_run=args.dry_run))
    elif args.command == "report-study":
        from ip_rbm.paper_studies import report_study

        print(report_study(args.results))
    elif args.command == "run":
        from ip_rbm.experiment import run_experiment

        result = run_experiment(args.config)
        print(f"Wrote {result}")
    elif args.command == "run-plot-open":
        from ip_rbm.experiment import run_experiment
        from ip_rbm.plotting import plot_frontier

        results_path = run_experiment(args.config)
        print(f"Wrote {results_path}")
        plot_path = plot_frontier(results_path)
        print(f"Wrote {plot_path}")
        _open_path(plot_path)
    elif args.command == "plot":
        from ip_rbm.plotting import plot_frontier

        result = plot_frontier(args.results, args.output)
        print(f"Wrote {result}")
    elif args.command == "learn":
        from ip_rbm.learning_experiment import run_learning_experiment

        result = run_learning_experiment(args.config)
        print(f"Wrote {result}")
    elif args.command == "learn-plot-open":
        from ip_rbm.learning_experiment import run_learning_experiment
        from ip_rbm.plotting import plot_learning_curve

        results_path = run_learning_experiment(args.config)
        print(f"Wrote {results_path}")
        plot_path = plot_learning_curve(results_path)
        print(f"Wrote {plot_path}")
        _open_path(plot_path)
    elif args.command == "plot-learning":
        from ip_rbm.plotting import plot_learning_curve

        result = plot_learning_curve(args.results, args.output)
        print(f"Wrote {result}")
    elif args.command == "train-scalable":
        from ip_rbm.scalable_experiment import run_scalable_experiment

        result = run_scalable_experiment(args.config)
        print(f"Wrote {result}")
    elif args.command == "train-scalable-plot-open":
        from ip_rbm.plotting import plot_scalable_learning
        from ip_rbm.scalable_experiment import run_scalable_experiment

        results_path = run_scalable_experiment(args.config)
        print(f"Wrote {results_path}")
        plot_path = plot_scalable_learning(results_path)
        print(f"Wrote {plot_path}")
        _open_path(plot_path)
    elif args.command == "plot-scalable":
        from ip_rbm.plotting import plot_scalable_learning

        result = plot_scalable_learning(args.results, args.output)
        print(f"Wrote {result}")
    elif args.command == "plot-scalable-resources":
        from ip_rbm.plotting import plot_scalable_resource_curves

        result = plot_scalable_resource_curves(args.checkpoints, args.output_dir)
        print(f"Wrote plots to {result}")
    elif args.command == "plot-scalable-objectives":
        from ip_rbm.plotting import plot_scalable_objective_curves

        result = plot_scalable_objective_curves(args.checkpoints, args.output_dir)
        print(f"Wrote plots to {result}")
    elif args.command == "plot-scalable-cdk":
        from ip_rbm.plotting import plot_scalable_cdk_curves

        result = plot_scalable_cdk_curves(args.checkpoints, args.output_dir)
        print(f"Wrote plots to {result}")


if __name__ == "__main__":
    main()
