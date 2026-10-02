"""Configuration-driven scalable learning experiments."""

from __future__ import annotations

import json
import math
import platform
import sys
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import yaml
from numpy.typing import NDArray
from torch import Tensor
from tqdm import tqdm

from ip_rbm.benchmark_targets import BenchmarkTarget
from ip_rbm.experiment import _load_config, _prepare_output_directory
from ip_rbm.ising_targets import IsingTarget
from ip_rbm.learning_experiment import _derived_seed
from ip_rbm.resource_monitor import TrainingResourceMonitor, TrainingResourceUsage
from ip_rbm.scalable_data import (
    AnchoredTeacherTarget,
    BlockIPTarget,
    CountCosineTarget,
    FreshTargetBatches,
    IPCosineTarget,
    IPTarget,
    ScalableTarget,
    uniform_binary,
)
from ip_rbm.scalable_evaluation import (
    IPScoreMetrics,
    TargetScoreMetrics,
    ais_population_metrics,
    exact_overlap_metrics,
    ip_score_metrics,
    model_sample_metrics,
    target_score_metrics,
)
from ip_rbm.scalable_learning import (
    ScalableTrainingSettings,
    TrainingRecord,
    make_device_generator,
    mean_log_pseudolikelihood,
    train_scalable,
)
from ip_rbm.states import PairMode
from ip_rbm.trainable_models import ModelKind, TrainableEnergyModel, make_trainable_model


@dataclass(frozen=True)
class ScalableModelPoint:
    """One architecture-width point in a scalable experiment grid."""

    kind: ModelKind
    n_hidden: int
    pair_mode: PairMode
    n_ip_values: frozenset[int] | None = None
    comparison_group: str = "default"
    seed_index: int | None = None

    def supports(self, n_ip: int) -> bool:
        return self.n_ip_values is None or n_ip in self.n_ip_values


@dataclass(frozen=True)
class TrainingPoint:
    """Named stochastic-training configuration."""

    name: str
    settings: ScalableTrainingSettings
    pair_modes: tuple[str, ...] = ("none", "cross", "all")
    seed_index: int | None = None


TargetKind = Literal[
    "benchmark",
    "ising_energy",
    "ising_ground",
    "ip",
    "count_cosine",
    "ip_cosine",
    "block_ip",
    "anchored_teacher",
]


@dataclass(frozen=True)
class ScalableTargetPoint:
    """One named target family point in a scalable experiment grid."""

    name: str
    kind: TargetKind
    beta: float = 1.0
    teacher_hidden_multiplier: float = 1.0
    anchor_strength: float = 0.0
    interaction_scale: float = 0.0
    bias_span: float = 0.0
    teacher_seed: int = 0
    center: float = 0.25
    w: int = 1
    rho: float = 0.0
    block_size: int = 1
    block_count: int = 0
    graph: str = "ring_chords"
    couplings: str = "mixed"
    graph_seed: int = 11
    encoding: str = "signed_product"
    objective: str = "smooth"
    family: str = "independent"

    def resolved_block_size(self, n_ip: int) -> int:
        if self.block_count:
            if n_ip % self.block_count != 0:
                raise ValueError("block-IP block_count must divide n_ip")
            return n_ip // self.block_count
        return self.block_size

    def make(self, n_ip: int) -> ScalableTarget:
        if self.kind == "benchmark":
            return BenchmarkTarget(n_ip, self.beta, self.family, self.graph_seed)
        if self.kind in {"ising_energy", "ising_ground"}:
            return IsingTarget(
                n_ip,
                self.beta,
                self.kind,
                self.graph,
                self.couplings,
                self.graph_seed,
                self.encoding,
                self.objective,
            )
        if self.kind == "ip":
            return IPTarget(n_ip, self.beta)
        if self.kind == "count_cosine":
            return CountCosineTarget(n_ip, self.beta, self.w)
        if self.kind == "ip_cosine":
            return IPCosineTarget(n_ip, self.beta, self.w, self.rho)
        if self.kind == "block_ip":
            return BlockIPTarget(n_ip, self.beta, self.resolved_block_size(n_ip))
        teacher_hidden = max(1, round(self.teacher_hidden_multiplier * n_ip))
        return AnchoredTeacherTarget(
            n_ip=n_ip,
            teacher_hidden=teacher_hidden,
            anchor_strength=self.anchor_strength,
            interaction_scale=self.interaction_scale,
            bias_span=self.bias_span,
            teacher_seed=self.teacher_seed,
            center=self.center,
        )

    def columns(self, n_ip: int) -> dict[str, str | int | float]:
        teacher_hidden = (
            max(1, round(self.teacher_hidden_multiplier * n_ip))
            if self.kind == "anchored_teacher"
            else 0
        )
        return {
            "target_name": self.name,
            "target_kind": self.kind,
            "target_family": self.family if self.kind == "benchmark" else self.kind,
            "target_instance_seed": self.graph_seed if self.kind == "benchmark" else 0,
            "target_graph": self.graph if self.kind.startswith("ising_") else "none",
            "target_couplings": self.couplings if self.kind.startswith("ising_") else "none",
            "target_graph_seed": self.graph_seed if self.kind.startswith("ising_") else 0,
            "target_encoding": self.encoding if self.kind.startswith("ising_") else "none",
            "target_objective": self.objective if self.kind.startswith("ising_") else "none",
            "target_beta": self.beta,
            "target_teacher_hidden": teacher_hidden,
            "target_anchor_strength": self.anchor_strength,
            "target_interaction_scale": self.interaction_scale,
            "target_bias_span": self.bias_span,
            "target_teacher_seed": self.teacher_seed,
            "target_center": self.center,
            "target_w": self.w if self.kind in {"count_cosine", "ip_cosine"} else 0,
            "target_rho": self.rho if self.kind == "ip_cosine" else 0.0,
            "target_block_size": (self.resolved_block_size(n_ip) if self.kind == "block_ip" else 0),
            "target_block_count": (
                n_ip // self.resolved_block_size(n_ip) if self.kind == "block_ip" else 0
            ),
        }

    def dataset_stem(self, n_ip: int) -> str:
        if self.kind == "ip":
            return f"ip_n{n_ip}_beta{self.beta:g}"
        if self.kind == "count_cosine":
            return f"count-cosine-w{self.w}_n{n_ip}_beta{self.beta:g}"
        if self.kind == "ip_cosine":
            return f"ip-cosine-w{self.w}_rho{self.rho:g}_n{n_ip}_beta{self.beta:g}"
        if self.kind == "block_ip":
            block_size = self.resolved_block_size(n_ip)
            block_count = n_ip // block_size
            if not self.block_count:
                return f"block-ip-k{block_size}_n{n_ip}_beta{self.beta:g}"
            return f"block-ip-q{block_count}-L{block_size}_n{n_ip}_beta{self.beta:g}"
        return f"target-{self.name}_n{n_ip}"


@dataclass(frozen=True)
class ScalableEvaluationSettings:
    """Exact-overlap and Monte Carlo evaluation settings."""

    target_samples: int = 4096
    selection_samples: int = 0
    pseudolikelihood_samples: int = 2048
    exact_max_n: int = 6
    ais_enabled: bool = False
    ais_particles: int = 256
    ais_intermediate: int = 1000
    ais_sweeps: int = 1
    ais_repetitions: int = 1
    model_sample_chains: int = 256
    model_sample_burn_in: int = 100
    model_sample_rounds: int = 20
    model_sample_thinning: int = 5

    def __post_init__(self) -> None:
        if self.target_samples < 2:
            raise ValueError("evaluation target_samples must be at least two")
        if self.selection_samples < 0:
            raise ValueError("selection_samples must be nonnegative")
        if self.pseudolikelihood_samples < 1:
            raise ValueError("evaluation pseudolikelihood_samples must be positive")
        if self.exact_max_n < 0:
            raise ValueError("evaluation exact_max_n must be nonnegative")
        if (
            self.ais_particles < 2
            or self.ais_intermediate < 2
            or self.ais_sweeps < 1
            or self.ais_repetitions < 1
        ):
            raise ValueError("invalid AIS settings")
        if (
            self.model_sample_chains < 1
            or self.model_sample_burn_in < 0
            or self.model_sample_rounds < 2
            or self.model_sample_thinning < 1
        ):
            raise ValueError("invalid model-sampling settings")


@dataclass(frozen=True)
class CheckpointSettings:
    """Post-training evaluation requested for saved parameter snapshots."""

    updates: tuple[int, ...] = ()
    ais_updates: frozenset[int] = frozenset()
    save_parameters: bool = True


@dataclass(frozen=True)
class RepeatedAISMetrics:
    """Aggregate repeated AIS evaluations without hiding Monte Carlo spread."""

    population_kl_mean: float
    population_kl_std: float
    log_partition_mean: float
    log_partition_std: float
    effective_sample_size_mean: float
    effective_sample_size_min: float


def _seed_index(value: Any) -> int | None:
    """Optional stable seed identity when subsetting a previously defined grid."""
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise ValueError("seed_index must be a nonnegative integer or null")
    return value


def _parse_models(config: dict[str, Any]) -> list[ScalableModelPoint]:
    raw_models = config.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("models must be a nonempty YAML list")
    points: list[ScalableModelPoint] = []
    for raw in raw_models:
        if not isinstance(raw, dict):
            raise ValueError("each model entry must be a YAML mapping")
        kind = str(raw["kind"])
        if kind not in {"rbm", "3rbm"}:
            raise ValueError(f"unknown model kind: {kind}")
        pair_mode = str(raw.get("pair_mode", "cross"))
        if pair_mode not in {"cross", "all"}:
            raise ValueError(f"unknown pair mode: {pair_mode}")
        hidden = raw.get("hidden")
        if not isinstance(hidden, list) or not hidden:
            raise ValueError("each model entry requires a nonempty hidden list")
        raw_n_ip = raw.get("n_ip")
        supported: frozenset[int] | None = None
        if raw_n_ip is not None:
            if not isinstance(raw_n_ip, list) or not raw_n_ip:
                raise ValueError("model n_ip must be a nonempty list")
            supported = frozenset(int(value) for value in raw_n_ip)
            if any(value < 1 for value in supported):
                raise ValueError("model n_ip values must be positive")
        comparison_group = str(raw.get("comparison_group", "default"))
        if not comparison_group or not comparison_group.replace("-", "").replace("_", "").isalnum():
            raise ValueError("model comparison_group may contain only letters, digits, '-' and '_'")
        for n_hidden in hidden:
            parsed_hidden = int(n_hidden)
            if parsed_hidden < 1:
                raise ValueError("scalable model hidden widths must be positive")
            points.append(
                ScalableModelPoint(
                    kind=kind,  # type: ignore[arg-type]
                    n_hidden=parsed_hidden,
                    pair_mode=pair_mode,  # type: ignore[arg-type]
                    n_ip_values=supported,
                    comparison_group=comparison_group,
                    seed_index=_seed_index(raw.get("seed_index")),
                )
            )
    return points


def _parse_targets(config: dict[str, Any]) -> list[ScalableTargetPoint]:
    """Parse named targets, retaining the legacy top-level ``beta`` grid."""
    raw_targets = config.get("targets")
    if raw_targets is None:
        beta_values = [float(value) for value in config["beta"]]
        if not beta_values or any(value < 0 for value in beta_values):
            raise ValueError("beta must contain nonnegative values")
        return [
            ScalableTargetPoint(name=f"ip-beta{beta:g}", kind="ip", beta=beta)
            for beta in beta_values
        ]
    if "beta" in config:
        raise ValueError("use either targets or the legacy top-level beta grid, not both")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("targets must be a nonempty YAML list")
    points: list[ScalableTargetPoint] = []
    names: set[str] = set()
    allowed = {
        "family",
        "graph",
        "couplings",
        "graph_seed",
        "encoding",
        "objective",
        "name",
        "kind",
        "beta",
        "teacher_hidden_multiplier",
        "anchor_strength",
        "interaction_scale",
        "bias_span",
        "teacher_seed",
        "center",
        "w",
        "rho",
        "block_size",
        "block_count",
    }
    for raw in raw_targets:
        if not isinstance(raw, dict):
            raise ValueError("each target entry must be a YAML mapping")
        unknown = set(raw).difference(allowed)
        if unknown:
            raise ValueError(f"unknown target fields: {sorted(unknown)}")
        name = str(raw.get("name", ""))
        if not name or not name.replace("-", "").replace("_", "").isalnum():
            raise ValueError("target names may contain only letters, digits, '-' and '_'")
        if name in names:
            raise ValueError(f"duplicate target name: {name}")
        names.add(name)
        kind = str(raw.get("kind", ""))
        if kind not in {
            "benchmark",
            "ising_energy",
            "ising_ground",
            "ip",
            "count_cosine",
            "ip_cosine",
            "block_ip",
            "anchored_teacher",
        }:
            raise ValueError(f"unknown target kind: {kind}")
        beta = float(raw.get("beta", 1.0))
        if beta < 0:
            raise ValueError("target beta must be nonnegative")
        if kind == "anchored_teacher" and beta != 1.0:
            raise ValueError("anchored_teacher targets require beta=1")
        point = ScalableTargetPoint(
            name=name,
            kind=kind,  # type: ignore[arg-type]
            beta=beta,
            teacher_hidden_multiplier=float(raw.get("teacher_hidden_multiplier", 1.0)),
            anchor_strength=float(raw.get("anchor_strength", 0.0)),
            interaction_scale=float(raw.get("interaction_scale", 0.0)),
            bias_span=float(raw.get("bias_span", 0.0)),
            teacher_seed=int(raw.get("teacher_seed", 0)),
            center=float(raw.get("center", 0.25)),
            w=int(raw.get("w", 1)),
            rho=float(raw.get("rho", 0.0)),
            block_size=int(raw.get("block_size", 1)),
            block_count=int(raw.get("block_count", 0)),
            graph=str(raw.get("graph", "ring_chords")),
            couplings=str(raw.get("couplings", "unit" if kind == "ising_ground" else "mixed")),
            graph_seed=int(raw.get("graph_seed", 11)),
            encoding=str(raw.get("encoding", "signed_product")),
            objective=str(raw.get("objective", "smooth")),
            family=str(raw.get("family", "independent")),
        )
        if point.teacher_hidden_multiplier <= 0:
            raise ValueError("teacher_hidden_multiplier must be positive")
        if point.anchor_strength < 0 or point.interaction_scale < 0 or point.bias_span < 0:
            raise ValueError("target strengths and bias_span must be nonnegative")
        if point.teacher_seed < 0:
            raise ValueError("teacher_seed must be nonnegative")
        if not 0 <= point.center <= 1:
            raise ValueError("target center must lie in [0, 1]")
        if point.w < 1:
            raise ValueError("target w must be a positive integer")
        if kind == "ip_cosine" and not 0.0 < point.rho < 1.0:
            raise ValueError("ip_cosine target rho must lie strictly between zero and one")
        if point.block_size < 1:
            raise ValueError("target block_size must be a positive integer")
        if point.block_count < 0:
            raise ValueError("target block_count must be nonnegative")
        if "block_size" in raw and "block_count" in raw:
            raise ValueError("block_ip targets must use either block_size or block_count, not both")
        if kind == "block_ip" and point.block_count == 0 and "block_count" in raw:
            raise ValueError("target block_count must be positive")
        points.append(point)
    return points


def _parse_training(config: dict[str, Any]) -> list[TrainingPoint]:
    raw_training = config.get("training")
    if not isinstance(raw_training, list) or not raw_training:
        raise ValueError("training must be a nonempty YAML list")
    defaults = config.get("training_defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("training_defaults must be a YAML mapping")
    points: list[TrainingPoint] = []
    names: set[str] = set()
    for raw in raw_training:
        if not isinstance(raw, dict):
            raise ValueError("each training entry must be a YAML mapping")
        merged = {**defaults, **raw}
        name = str(merged.pop("name"))
        if not name or not name.replace("-", "").replace("_", "").isalnum():
            raise ValueError("training names may contain only letters, digits, '-' and '_'")
        if name in names:
            raise ValueError(f"duplicate training name: {name}")
        names.add(name)
        pair_modes = merged.pop("pair_modes", ["none", "cross", "all"])
        if (
            not isinstance(pair_modes, list)
            or not pair_modes
            or any(mode not in {"none", "cross", "all"} for mode in pair_modes)
        ):
            raise ValueError("training pair_modes must be a nonempty list of none/cross/all")
        seed_index = _seed_index(merged.pop("seed_index", None))
        settings = ScalableTrainingSettings(**merged)
        if settings.all_pairs_sampler != "legacy" and set(pair_modes) != {"all"}:
            raise ValueError("cached/block samplers require training pair_modes: [all]")
        points.append(TrainingPoint(name, settings, tuple(pair_modes), seed_index))
    return points


def _parse_checkpoints(config: dict[str, Any]) -> CheckpointSettings:
    raw = config.get("checkpoints", {})
    if not isinstance(raw, dict):
        raise ValueError("checkpoints must be a YAML mapping")
    updates = tuple(sorted({int(value) for value in raw.get("updates", [])}))
    ais_updates = frozenset(int(value) for value in raw.get("ais_updates", []))
    if any(update < 1 for update in updates):
        raise ValueError("checkpoint updates must be positive")
    if any(update < 1 for update in ais_updates):
        raise ValueError("checkpoint AIS updates must be positive")
    if not ais_updates.issubset(updates):
        raise ValueError("checkpoint ais_updates must be included in checkpoint updates")
    save_parameters = raw.get("save_parameters", True)
    if not isinstance(save_parameters, bool):
        raise ValueError("checkpoint save_parameters must be true or false")
    return CheckpointSettings(
        updates=updates,
        ais_updates=ais_updates,
        save_parameters=save_parameters,
    )


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError("dtype must be float32 or float64")


def _make_summary(results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "experiment",
        "target_name",
        "target_kind",
        "target_w",
        "target_rho",
        "target_block_size",
        "target_block_count",
        "model",
        "pair_mode",
        "n_ip",
        "n_visible",
        "n_hidden",
        "n_parameters",
        "comparison_group",
        "beta",
        "weight_bound",
        "sample_size",
        "training_name",
        "algorithm",
        "all_pairs_sampler",
        "all_pairs_block_size",
        "training_distribution",
    ]
    return (
        results.groupby(columns, dropna=False, sort=False)
        .agg(
            dataset_repetitions=("dataset_repeat", "nunique"),
            population_kl_mean=("population_kl", "mean"),
            population_kl_median=("population_kl", "median"),
            population_kl_q25=("population_kl", lambda values: values.quantile(0.25)),
            population_kl_q75=("population_kl", lambda values: values.quantile(0.75)),
            ais_population_kl_median=("ais_population_kl", "median"),
            ais_population_kl_std_median=("ais_population_kl_std", "median"),
            ais_effective_sample_size_median=("ais_effective_sample_size", "median"),
            ais_effective_sample_size_min=("ais_effective_sample_size_min", "min"),
            score_gap_mean=("score_gap", "mean"),
            mean_log_pseudolikelihood=("mean_log_pseudolikelihood", "mean"),
            learned_ip_component_ratio_median=("learned_ip_component_ratio", "median"),
            learned_cosine_component_ratio_median=(
                "learned_cosine_component_ratio",
                "median",
            ),
            elapsed_seconds_mean=("elapsed_seconds", "mean"),
        )
        .reset_index()
    )


def _make_checkpoint_summary(checkpoints: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "experiment",
        "target_name",
        "target_kind",
        "target_w",
        "target_rho",
        "target_block_size",
        "target_block_count",
        "model",
        "pair_mode",
        "comparison_group",
        "n_ip",
        "n_visible",
        "n_hidden",
        "n_parameters",
        "beta",
        "weight_bound",
        "sample_size",
        "training_name",
        "algorithm",
        "all_pairs_sampler",
        "all_pairs_block_size",
        "training_distribution",
        "batch_size",
        "gibbs_steps",
        "update",
    ]
    return (
        checkpoints.groupby(columns, dropna=False, sort=False)
        .agg(
            dataset_repetitions=("dataset_repeat", "nunique"),
            population_kl_median=("population_kl", "median"),
            population_kl_q25=("population_kl", lambda values: values.quantile(0.25)),
            population_kl_q75=("population_kl", lambda values: values.quantile(0.75)),
            ais_population_kl_median=("ais_population_kl", "median"),
            ais_population_kl_std_median=("ais_population_kl_std", "median"),
            ais_effective_sample_size_median=("ais_effective_sample_size", "median"),
            ais_effective_sample_size_min=("ais_effective_sample_size_min", "min"),
            normalized_kl_improvement_median=("normalized_kl_improvement", "median"),
            score_gap_ratio_median=("score_gap_ratio", "median"),
            score_gap_ratio_q25=("score_gap_ratio", lambda values: values.quantile(0.25)),
            score_gap_ratio_q75=("score_gap_ratio", lambda values: values.quantile(0.75)),
            normalized_score_shape_rmse_median=("normalized_score_shape_rmse", "median"),
            normalized_score_shape_rmse_q25=(
                "normalized_score_shape_rmse",
                lambda values: values.quantile(0.25),
            ),
            normalized_score_shape_rmse_q75=(
                "normalized_score_shape_rmse",
                lambda values: values.quantile(0.75),
            ),
            normalized_target_score_rmse_median=(
                "normalized_target_score_rmse",
                "median",
            ),
            normalized_target_score_rmse_q25=(
                "normalized_target_score_rmse",
                lambda values: values.quantile(0.25),
            ),
            normalized_target_score_rmse_q75=(
                "normalized_target_score_rmse",
                lambda values: values.quantile(0.75),
            ),
            target_score_correlation_median=("target_score_correlation", "median"),
            learned_ip_component_ratio_median=("learned_ip_component_ratio", "median"),
            learned_ip_component_ratio_q25=(
                "learned_ip_component_ratio",
                lambda values: values.quantile(0.25),
            ),
            learned_ip_component_ratio_q75=(
                "learned_ip_component_ratio",
                lambda values: values.quantile(0.75),
            ),
            learned_cosine_component_ratio_median=(
                "learned_cosine_component_ratio",
                "median",
            ),
            learned_cosine_component_ratio_q25=(
                "learned_cosine_component_ratio",
                lambda values: values.quantile(0.25),
            ),
            learned_cosine_component_ratio_q75=(
                "learned_cosine_component_ratio",
                lambda values: values.quantile(0.75),
            ),
            training_elapsed_seconds_median=("training_elapsed_seconds", "median"),
            checkpoint_evaluation_seconds_mean=("checkpoint_evaluation_seconds", "mean"),
            training_peak_rss_bytes_median=("training_peak_rss_bytes", "median"),
            training_rss_increase_bytes_median=("training_rss_increase_bytes", "median"),
            training_cpu_user_seconds_median=("training_cpu_user_seconds", "median"),
            training_cpu_system_seconds_median=("training_cpu_system_seconds", "median"),
            training_peak_mps_tensor_bytes_median=(
                "training_peak_mps_tensor_bytes",
                "median",
            ),
            training_peak_cuda_allocated_bytes_median=(
                "training_peak_cuda_allocated_bytes",
                "median",
            ),
        )
        .reset_index()
    )


@torch.no_grad()
def _selection_pll(model: TrainableEnergyModel, states: Tensor | None) -> float:
    return float(mean_log_pseudolikelihood(model, states)) if states is not None else math.nan


def _fraction_at_bound(model_parameters: Tensor, weight_bound: float | None) -> float:
    if weight_bound is None:
        return math.nan
    tolerance = 1e-6 * max(1.0, weight_bound)
    return float((model_parameters.abs() >= weight_bound - tolerance).to(torch.float32).mean())


def _uniform_baseline_kl(target: ScalableTarget) -> float:
    return target.n_visible * math.log(2.0) - target.entropy


def _normalized_kl_improvement(population_kl: float, uniform_kl: float) -> float:
    if not math.isfinite(population_kl) or uniform_kl <= 0:
        return math.nan
    return 1.0 - population_kl / uniform_kl


def _run_repeated_ais(
    model: TrainableEnergyModel,
    target: ScalableTarget,
    held_out: Tensor,
    evaluation: ScalableEvaluationSettings,
    *,
    seed: int,
) -> RepeatedAISMetrics:
    population_kls: list[float] = []
    log_partitions: list[float] = []
    effective_sample_sizes: list[float] = []
    for repetition in range(evaluation.ais_repetitions):
        generator = make_device_generator(model.device, _derived_seed(seed, repetition))
        metrics = ais_population_metrics(
            model,
            target,
            held_out,
            n_particles=evaluation.ais_particles,
            n_intermediate=evaluation.ais_intermediate,
            sweeps_per_temperature=evaluation.ais_sweeps,
            generator=generator,
        )
        population_kls.append(metrics.population_kl)
        log_partitions.append(metrics.ais.log_partition)
        effective_sample_sizes.append(metrics.ais.effective_sample_size)
    return RepeatedAISMetrics(
        population_kl_mean=float(np.mean(population_kls)),
        population_kl_std=float(np.std(population_kls)),
        log_partition_mean=float(np.mean(log_partitions)),
        log_partition_std=float(np.std(log_partitions)),
        effective_sample_size_mean=float(np.mean(effective_sample_sizes)),
        effective_sample_size_min=float(np.min(effective_sample_sizes)),
    )


def _resource_columns(usage: TrainingResourceUsage) -> dict[str, int | float]:
    return {
        "training_baseline_rss_bytes": usage.baseline_rss_bytes,
        "training_peak_rss_bytes": usage.peak_rss_bytes,
        "training_rss_increase_bytes": usage.rss_increase_bytes,
        "training_cpu_user_seconds": usage.cpu_user_seconds,
        "training_cpu_system_seconds": usage.cpu_system_seconds,
        "training_peak_mps_tensor_bytes": usage.peak_mps_tensor_bytes,
        "training_peak_mps_driver_bytes": usage.peak_mps_driver_bytes,
        "training_peak_cuda_allocated_bytes": usage.peak_cuda_allocated_bytes,
        "training_peak_cuda_reserved_bytes": usage.peak_cuda_reserved_bytes,
    }


def _checkpoint_metric_columns(
    *,
    target: ScalableTarget,
    score_metrics: IPScoreMetrics | None,
    target_metrics: TargetScoreMetrics,
    exact_population_kl: float,
    exact_ip_correlation: float,
    ais_metrics: RepeatedAISMetrics | None,
) -> dict[str, float | str]:
    population_kl = (
        exact_population_kl
        if math.isfinite(exact_population_kl)
        else (ais_metrics.population_kl_mean if ais_metrics is not None else math.nan)
    )
    population_kl_source = (
        "exact"
        if math.isfinite(exact_population_kl)
        else ("ais" if ais_metrics is not None else "unavailable")
    )
    uniform_kl = _uniform_baseline_kl(target)
    target_gap = (
        2.0 * target.beta
        if isinstance(target, IPTarget) and math.isfinite(target.beta)
        else math.nan
    )
    if score_metrics is None:
        ip_columns = {
            "even_mean_score": math.nan,
            "odd_mean_score": math.nan,
            "score_gap": math.nan,
            "even_score_variance": math.nan,
            "odd_score_variance": math.nan,
        }
        score_gap_error = math.nan
        score_gap_ratio = math.nan
        score_shape_rmse = target_metrics.target_score_rmse
        normalized_score_shape_rmse = target_metrics.normalized_target_score_rmse
    else:
        ip_columns = {
            key: value
            for key, value in asdict(score_metrics).items()
            if key != "mean_log_pseudolikelihood"
        }
        score_gap_error = score_metrics.score_gap - target_gap
        score_gap_ratio = score_metrics.score_gap / target_gap if target_gap > 0 else math.nan
        score_shape_rmse = math.sqrt(
            max(
                0.0,
                0.5 * (score_metrics.even_score_variance + score_metrics.odd_score_variance)
                + 0.25 * score_gap_error**2,
            )
        )
        normalized_score_shape_rmse = (
            score_shape_rmse / target.beta if target.beta > 0 else math.nan
        )
        if isinstance(target, IPTarget) and math.isinf(target.beta):
            score_gap_error = score_gap_ratio = math.nan
            score_shape_rmse = normalized_score_shape_rmse = math.nan
    exact_even_mass = (1.0 + exact_ip_correlation) / 2.0
    conditional_even_kl = (
        exact_population_kl + math.log(exact_even_mass)
        if isinstance(target, IPTarget) and math.isinf(target.beta) and exact_even_mass > 0
        else math.nan
    )
    return {
        "exact_even_probability": exact_even_mass,
        "exact_even_conditional_kl": conditional_even_kl,
        "population_kl": population_kl,
        "population_kl_source": population_kl_source,
        "uniform_baseline_kl": uniform_kl,
        "normalized_kl_improvement": _normalized_kl_improvement(population_kl, uniform_kl),
        "exact_ip_correlation": exact_ip_correlation,
        **asdict(target_metrics),
        **ip_columns,
        "score_gap_error": score_gap_error,
        "score_gap_ratio": score_gap_ratio,
        "score_shape_rmse": score_shape_rmse,
        "normalized_score_shape_rmse": normalized_score_shape_rmse,
        "ais_population_kl": (
            ais_metrics.population_kl_mean if ais_metrics is not None else math.nan
        ),
        "ais_population_kl_std": (
            ais_metrics.population_kl_std if ais_metrics is not None else math.nan
        ),
        "ais_log_partition": (
            ais_metrics.log_partition_mean if ais_metrics is not None else math.nan
        ),
        "ais_log_partition_std": (
            ais_metrics.log_partition_std if ais_metrics is not None else math.nan
        ),
        "ais_effective_sample_size": (
            ais_metrics.effective_sample_size_mean if ais_metrics is not None else math.nan
        ),
        "ais_effective_sample_size_min": (
            ais_metrics.effective_sample_size_min if ais_metrics is not None else math.nan
        ),
    }


def _generate_training_states(
    target: ScalableTarget,
    sample_size: int,
    generator: torch.Generator,
    *,
    distribution: str,
    chunk_size: int | None,
) -> NDArray[np.uint8]:
    """Store binary data compactly, optionally bounding sampling temporaries.

    With no chunk size, retain the original one-call RNG sequence. Chunked
    sampling remains iid but has its own reproducible sequence, recorded in YAML.
    """
    if sample_size < 1 or (chunk_size is not None and chunk_size < 1):
        raise ValueError("sample_size and dataset_generation_batch_size must be positive")
    if distribution not in {"target", "uniform"}:
        raise ValueError("unknown training distribution")
    size = sample_size if chunk_size is None else min(chunk_size, sample_size)
    stored = np.empty((sample_size, target.n_visible), dtype=np.uint8)
    for start in range(0, sample_size, size):
        count = min(size, sample_size - start)
        batch = (
            uniform_binary(
                count,
                target.n_visible,
                device=torch.device("cpu"),
                dtype=torch.float32,
                generator=generator,
            )
            if distribution == "uniform"
            else target.sample(count, dtype=torch.float32, generator=generator)
        )
        stored[start : start + count] = batch.numpy()
    return stored


def run_scalable_experiment(config_path: str | Path) -> Path:
    """Run a YAML-defined scalable-learning grid and return ``results.csv``."""
    config_path = Path(config_path)
    config = _load_config(config_path)
    output_dir = Path(config["output_dir"])
    clean = config.get("clean_output_dir", False)
    if not isinstance(clean, bool):
        raise ValueError("clean_output_dir must be true or false")

    n_ip_values = [int(value) for value in config["n_ip"]]
    sample_sizes = sorted(int(value) for value in config["sample_sizes"])
    if not n_ip_values or any(value < 1 for value in n_ip_values):
        raise ValueError("n_ip must contain positive integers")
    if not sample_sizes or sample_sizes[0] < 1 or len(set(sample_sizes)) != len(sample_sizes):
        raise ValueError("sample_sizes must contain unique positive integers")
    generation_batch_size = config.get("dataset_generation_batch_size")
    if generation_batch_size is not None and (
        isinstance(generation_batch_size, bool)
        or not isinstance(generation_batch_size, int)
        or generation_batch_size < 1
    ):
        raise ValueError("dataset_generation_batch_size must be a positive integer")
    repetitions = int(config.get("dataset_repetitions", 1))
    if repetitions < 1:
        raise ValueError("dataset_repetitions must be positive")
    data_seed = int(config.get("data_seed", 314159))
    initialization_seed = int(config.get("initialization_seed", 1729))
    init_std = float(config.get("initialization_std", 0.01))
    pair_initializations = config.get("pair_initializations_across_training", False)
    if not isinstance(pair_initializations, bool):
        raise ValueError("pair_initializations_across_training must be true or false")
    pair_minibatches = config.get("pair_minibatches_across_models", False)
    if not isinstance(pair_minibatches, bool):
        raise ValueError("pair_minibatches_across_models must be true or false")
    pair_training_batches = config.get("pair_minibatches_across_training", False)
    if not isinstance(pair_training_batches, bool):
        raise ValueError("pair_minibatches_across_training must be true or false")
    if pair_training_batches and not pair_minibatches:
        raise ValueError("pair_minibatches_across_training requires pairing across models")
    pair_evaluation_states = config.get("pair_evaluation_states_across_models", False)
    if not isinstance(pair_evaluation_states, bool):
        raise ValueError("pair_evaluation_states_across_models must be true or false")
    if data_seed < 0 or initialization_seed < 0:
        raise ValueError("data and initialization seeds must be nonnegative")
    if init_std < 0:
        raise ValueError("initialization_std must be nonnegative")
    training_distribution = str(config.get("training_distribution", "target"))
    data_mode = str(config.get("training_data_mode", "fixed"))
    if data_mode not in {"fixed", "fresh"}:
        raise ValueError("training_data_mode must be fixed or fresh")
    if data_mode == "fresh" and len(sample_sizes) != 1:
        raise ValueError("fresh training uses one sample_sizes entry as reference provenance")
    if training_distribution not in {"target", "uniform"}:
        raise ValueError("training_distribution must be target or uniform")
    weight_bounds = [
        None if value is None else float(value) for value in config.get("weight_bound", [None])
    ]
    if not weight_bounds or any(value is not None and value <= 0 for value in weight_bounds):
        raise ValueError("weight_bound must contain positive values or null")
    models = _parse_models(config)
    target_points = _parse_targets(config)
    training_points = _parse_training(config)
    algorithms = {point.settings.algorithm for point in training_points}
    if data_mode == "fresh" and (algorithms != {"cd"} or training_distribution != "target"):
        raise ValueError("fresh training currently requires CD and target observations")
    if "score_regression" in algorithms and training_distribution != "uniform":
        raise ValueError("score_regression requires training_distribution: uniform")
    if training_distribution == "uniform" and algorithms != {"score_regression"}:
        raise ValueError(
            "training_distribution: uniform is reserved for score_regression experiments"
        )
    checkpoints = _parse_checkpoints(config)
    evaluation_raw = config.get("evaluation", {})
    if not isinstance(evaluation_raw, dict):
        raise ValueError("evaluation must be a YAML mapping")
    evaluation = ScalableEvaluationSettings(**evaluation_raw)
    device = _resolve_device(str(config.get("device", "auto")))
    dtype = _resolve_dtype(str(config.get("dtype", "float32")))
    if device.type == "mps" and dtype == torch.float64:
        raise ValueError("MPS does not support float64 training")

    parameter_dir = _prepare_output_directory(output_dir, clean=clean)
    checkpoint_parameter_dir = parameter_dir / "checkpoints"
    if checkpoints.updates and checkpoints.save_parameters:
        checkpoint_parameter_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset_paths: dict[tuple[int, int, int], Path] = {}
    for n_index, n_ip in enumerate(n_ip_values):
        for target_index, target_point in enumerate(target_points):
            target = target_point.make(n_ip)
            if isinstance(target, (IsingTarget, BenchmarkTarget)):
                spec_dir = output_dir / "target_specs"
                spec_dir.mkdir(exist_ok=True)
                target.save_spec(spec_dir / f"{target_point.name}_n{n_ip}.npz")
            for repetition in range(repetitions):
                train_seed = _derived_seed(data_seed, n_index, target_index, repetition, 0)
                validation_seed = _derived_seed(data_seed, n_index, target_index, repetition, 1)
                train_generator = torch.Generator().manual_seed(train_seed)
                validation_generator = torch.Generator().manual_seed(validation_seed)
                generated_train = _generate_training_states(
                    target,
                    sample_sizes[-1] if data_mode == "fixed" else 1,
                    train_generator,
                    distribution=training_distribution,
                    chunk_size=generation_batch_size,
                )
                if data_mode == "fresh":
                    # No finite training dataset is stored or reused.
                    generated_train = np.empty((0, target.n_visible), dtype=np.uint8)
                generated_validation = target.sample(
                    evaluation.target_samples,
                    dtype=torch.float32,
                    generator=validation_generator,
                )
                distribution_prefix = "uniform-" if training_distribution == "uniform" else ""
                dataset_path = dataset_dir / (
                    f"{distribution_prefix}{target_point.dataset_stem(n_ip)}_"
                    f"data{repetition:03d}.npz"
                )
                dataset_paths[(n_index, target_index, repetition)] = dataset_path
                np.savez_compressed(
                    dataset_path,
                    train_states=generated_train,
                    validation_states=generated_validation.numpy().astype(np.uint8),
                    train_seed=train_seed,
                    validation_seed=validation_seed,
                )
                del generated_train, generated_validation

    grid = [
        (
            n_index,
            target_index,
            repetition,
            sample_size,
            model_index,
            training_index,
            bound_index,
        )
        for n_index, n_ip in enumerate(n_ip_values)
        for target_index in range(len(target_points))
        for repetition in range(repetitions)
        for sample_size in sample_sizes
        for model_index, model in enumerate(models)
        if model.supports(n_ip)
        for training_index in range(len(training_points))
        if (model.pair_mode if model.kind == "3rbm" else "none")
        in training_points[training_index].pair_modes
        for bound_index in range(len(weight_bounds))
    ]
    total_updates = sum(training_points[point[5]].settings.updates for point in grid)
    progress = tqdm(
        total=total_updates,
        desc=f"{config['name']} scalable training",
        unit="update",
        dynamic_ncols=True,
    )
    result_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    active_data_key: tuple[int, int, int, int] | None = None
    active_train_states: Tensor | None = None
    active_validation_states: Tensor | None = None
    try:
        for point_index, point in enumerate(grid):
            (
                n_index,
                target_index,
                repetition,
                sample_size,
                model_index,
                training_index,
                bound_index,
            ) = point
            n_ip = n_ip_values[n_index]
            target_point = target_points[target_index]
            target = target_point.make(n_ip)
            beta = target.beta
            model_point = models[model_index]
            training_point = training_points[training_index]
            weight_bound = weight_bounds[bound_index]
            model_seed_index = (
                model_index if model_point.seed_index is None else model_point.seed_index
            )
            training_seed_index = (
                training_index if training_point.seed_index is None else training_point.seed_index
            )
            run_seed = _derived_seed(
                initialization_seed,
                n_index,
                target_index,
                repetition,
                sample_sizes.index(sample_size),
                model_seed_index,
                training_seed_index,
                bound_index,
            )
            model_initialization_seed = (
                _derived_seed(
                    initialization_seed,
                    n_index,
                    target_index,
                    repetition,
                    sample_sizes.index(sample_size),
                    model_seed_index,
                    bound_index,
                )
                if pair_initializations
                else run_seed
            )
            minibatch_seed = (
                _derived_seed(
                    initialization_seed,
                    1_000_003,
                    n_index,
                    target_index,
                    repetition,
                    sample_sizes.index(sample_size),
                    0 if pair_training_batches else training_seed_index,
                    bound_index,
                )
                if pair_minibatches
                else run_seed
            )
            sampler_seed = run_seed + 1
            torch.manual_seed(model_initialization_seed)
            model = make_trainable_model(
                model_point.kind,
                2 * n_ip,
                model_point.n_hidden,
                pair_mode=model_point.pair_mode,
                init_std=init_std,
                device=device,
                dtype=dtype,
            )
            settings = replace(
                training_point.settings,
                weight_bound=weight_bound,
                seed=run_seed,
                minibatch_seed=minibatch_seed,
                sampler_seed=sampler_seed,
            )
            checkpoint_updates = tuple(
                update for update in checkpoints.updates if update <= settings.updates
            )
            last_update = 0

            def report(
                record: TrainingRecord,
                point_index: int = point_index,
                n_ip: int = n_ip,
                model_point: ScalableModelPoint = model_point,
                training_point: TrainingPoint = training_point,
                sample_size: int = sample_size,
            ) -> None:
                nonlocal last_update
                progress.update(record.update - last_update)
                last_update = record.update
                progress.set_postfix(
                    {
                        "point": f"{point_index + 1}/{len(grid)}",
                        "n": n_ip,
                        "model": model_point.kind,
                        "m": model_point.n_hidden,
                        "algorithm": training_point.name,
                        "N": sample_size,
                        "loss": f"{record.objective:.3g}",
                    },
                    refresh=False,
                )

            data_key = (n_index, target_index, repetition, sample_size)
            if data_key != active_data_key:
                with np.load(
                    dataset_paths[(n_index, target_index, repetition)], allow_pickle=False
                ) as archive:
                    active_train_states = torch.from_numpy(
                        np.array(
                            archive["train_states"][:sample_size],
                            dtype=np.float32,
                            copy=True,
                        )
                    )
                    active_validation_states = torch.from_numpy(
                        np.array(
                            archive["validation_states"],
                            dtype=np.float32,
                            copy=True,
                        )
                    )
                active_data_key = data_key
            if active_train_states is None or active_validation_states is None:
                raise AssertionError("active scalable dataset was not loaded")
            train_states = active_train_states
            fresh_seed = _derived_seed(data_seed, n_index, target_index, repetition, 2)
            fresh_batches = (
                FreshTargetBatches(target, settings.batch_size, fresh_seed)
                if data_mode == "fresh"
                else None
            )
            resource_monitor = TrainingResourceMonitor(device)
            resource_monitor.start()
            try:
                result = train_scalable(
                    model,
                    train_states,
                    settings,
                    callback=report,
                    checkpoint_updates=checkpoint_updates,
                    batch_provider=fresh_batches,
                    target_score_function=(
                        target.log_unnormalized
                        if settings.algorithm == "score_regression"
                        else None
                    ),
                )
            finally:
                resource_usage = resource_monitor.stop()
            if last_update < settings.updates:
                progress.update(settings.updates - last_update)

            pair_mode = model_point.pair_mode if model_point.kind == "3rbm" else "none"
            bound_label: float | str = "unbounded" if weight_bound is None else weight_bound
            stem = (
                f"{model_point.kind}_{target_point.name}_n{n_ip}_m{model_point.n_hidden}_"
                f"N{sample_size}_data{repetition:03d}_{training_point.name}_"
                f"pairs-{pair_mode}_"
                f"bound{'unbounded' if weight_bound is None else f'{weight_bound:g}'}"
            )
            if settings.algorithm == "cd":
                gibbs_chains_per_update = settings.batch_size
            elif settings.algorithm == "pcd":
                gibbs_chains_per_update = settings.persistent_chains or settings.batch_size
            elif settings.algorithm == "tempered_pcd":
                gibbs_chains_per_update = (
                    settings.persistent_chains or settings.batch_size
                ) * settings.tempering_replicas
            else:
                gibbs_chains_per_update = 0
            common: dict[str, Any] = {
                "experiment": str(config["name"]),
                **target_point.columns(n_ip),
                "model": model_point.kind,
                "pair_mode": pair_mode,
                "comparison_group": model_point.comparison_group,
                "n_ip": n_ip,
                "n_visible": 2 * n_ip,
                "n_hidden": model_point.n_hidden,
                "n_parameters": model.n_parameters,
                "beta": beta,
                "weight_bound": bound_label,
                "sample_size": sample_size if data_mode == "fixed" else 0,
                "reference_sample_size": sample_size,
                "training_data_mode": data_mode,
                "fresh_data_seed": fresh_seed if data_mode == "fresh" else None,
                "fresh_target_draws_total": fresh_batches.draws if fresh_batches else 0,
                "dataset_repeat": repetition,
                "training_name": training_point.name,
                "algorithm": settings.algorithm,
                "training_distribution": training_distribution,
                "initialization_seed": model_initialization_seed,
                "training_seed": run_seed,
                "minibatch_seed": minibatch_seed,
                "sampler_seed": sampler_seed,
                "model_seed_index": model_seed_index,
                "training_seed_index": training_seed_index,
                "device": str(device),
                "dtype": str(dtype).removeprefix("torch."),
                "learning_rate": settings.learning_rate,
                "lr_decay_start_fraction": settings.lr_decay_start_fraction,
                "lr_final_ratio": settings.lr_final_ratio,
                "optimizer": settings.optimizer,
                "batch_size": settings.batch_size,
                "gibbs_steps": settings.gibbs_steps,
                "all_pairs_sampler": settings.all_pairs_sampler
                if pair_mode == "all"
                else "register",
                "all_pairs_block_size": settings.all_pairs_block_size
                if pair_mode == "all" and settings.all_pairs_sampler == "block"
                else 1,
                "persistent_chains": settings.persistent_chains or settings.batch_size,
                "weight_decay": settings.weight_decay,
                "nce_noise_ratio": settings.nce_noise_ratio,
                "tempering_replicas": settings.tempering_replicas,
            }

            evaluation_seed = (
                _derived_seed(
                    data_seed,
                    2_000_003,
                    n_index,
                    target_index,
                    repetition,
                    sample_sizes.index(sample_size),
                )
                if pair_evaluation_states
                else run_seed
            )
            held_out = active_validation_states
            pseudolikelihood_states = held_out[: evaluation.pseudolikelihood_samples]
            score_generator = torch.Generator().manual_seed(evaluation_seed + 3)
            score_states = uniform_binary(
                evaluation.target_samples,
                target.n_visible,
                device="cpu",
                dtype=torch.float32,
                generator=score_generator,
            )
            sector_states: Tensor | None = None
            if isinstance(target, IPTarget):
                sector_count = max(16, evaluation.target_samples // 2)
                sector_states = torch.cat(
                    (
                        target.sample_sector(
                            sector_count,
                            even=True,
                            dtype=torch.float32,
                            generator=score_generator,
                        ),
                        target.sample_sector(
                            sector_count,
                            even=False,
                            dtype=torch.float32,
                            generator=score_generator,
                        ),
                    )
                )
            final_theta = model.flat_parameters().detach().cpu().clone()
            resource_columns = _resource_columns(resource_usage)
            selection_seed = _derived_seed(data_seed, 9_000_019, n_index, target_index, repetition)
            selection_states = (
                target.sample(
                    evaluation.selection_samples,
                    generator=torch.Generator().manual_seed(selection_seed),
                ).to(device=model.device, dtype=model.dtype)
                if evaluation.selection_samples
                else None
            )
            common["selection_seed"] = selection_seed
            common["selection_samples"] = evaluation.selection_samples
            if selection_states is not None:
                selection_path = (
                    dataset_dir / f"selection-{target_point.dataset_stem(n_ip)}-r{repetition}.npz"
                )
                if not selection_path.exists():
                    np.savez_compressed(
                        selection_path, states=selection_states.cpu().numpy(), seed=selection_seed
                    )

            for checkpoint in result.checkpoints:
                if checkpoint.update == settings.updates:
                    continue
                model.load_flat_parameters(checkpoint.theta)
                checkpoint_started = perf_counter()
                checkpoint_target_score = target_score_metrics(
                    model,
                    target,
                    score_states,
                    pseudolikelihood_states=pseudolikelihood_states,
                )
                checkpoint_score = (
                    ip_score_metrics(
                        model,
                        target,
                        held_out,
                        sector_states=sector_states,
                    )
                    if isinstance(target, IPTarget)
                    else None
                )
                checkpoint_exact = (
                    exact_overlap_metrics(model, target) if n_ip <= evaluation.exact_max_n else None
                )
                checkpoint_ais = None
                if evaluation.ais_enabled and checkpoint.update in checkpoints.ais_updates:
                    checkpoint_ais = _run_repeated_ais(
                        model,
                        target,
                        held_out,
                        evaluation,
                        seed=_derived_seed(evaluation_seed, checkpoint.update, 7),
                    )
                checkpoint_evaluation_seconds = perf_counter() - checkpoint_started
                checkpoint_parameter_path: Path | None = None
                if checkpoints.save_parameters:
                    checkpoint_parameter_path = checkpoint_parameter_dir / (
                        f"{stem}_update{checkpoint.update:06d}.npz"
                    )
                    np.savez_compressed(
                        checkpoint_parameter_path,
                        theta=checkpoint.theta.numpy(),
                        nce_log_normalizer=checkpoint.nce_log_normalizer,
                        training_seed=run_seed,
                        minibatch_seed=minibatch_seed,
                        sampler_seed=sampler_seed,
                        update=checkpoint.update,
                        model=model_point.kind,
                        pair_mode=pair_mode,
                        all_pairs_sampler=settings.all_pairs_sampler,
                        all_pairs_block_size=settings.all_pairs_block_size,
                        n_visible=2 * n_ip,
                        n_hidden=model_point.n_hidden,
                    )
                checkpoint_rows.append(
                    {
                        **common,
                        "update": checkpoint.update,
                        "validation_log_pseudolikelihood": _selection_pll(model, selection_states),
                        "training_examples_processed": checkpoint.update * settings.batch_size,
                        "gibbs_chain_sweeps": checkpoint.update
                        * settings.gibbs_steps
                        * gibbs_chains_per_update,
                        "training_elapsed_seconds": checkpoint.elapsed_seconds,
                        "checkpoint_evaluation_seconds": checkpoint_evaluation_seconds,
                        **_checkpoint_metric_columns(
                            target=target,
                            score_metrics=checkpoint_score,
                            target_metrics=checkpoint_target_score,
                            exact_population_kl=(
                                checkpoint_exact.population_kl
                                if checkpoint_exact is not None
                                else math.nan
                            ),
                            exact_ip_correlation=(
                                checkpoint_exact.ip_correlation
                                if checkpoint_exact is not None
                                else math.nan
                            ),
                            ais_metrics=checkpoint_ais,
                        ),
                        "maximum_absolute_parameter": float(checkpoint.theta.abs().max()),
                        "fraction_at_bound": _fraction_at_bound(checkpoint.theta, weight_bound),
                        "nce_log_normalizer": checkpoint.nce_log_normalizer,
                        "parameter_file": (
                            str(checkpoint_parameter_path)
                            if checkpoint_parameter_path is not None
                            else ""
                        ),
                        **resource_columns,
                    }
                )

            model.load_flat_parameters(final_theta)
            final_evaluation_started = perf_counter()
            target_metrics = target_score_metrics(
                model,
                target,
                score_states,
                pseudolikelihood_states=pseudolikelihood_states,
            )
            score_metrics = (
                ip_score_metrics(
                    model,
                    target,
                    held_out,
                    sector_states=sector_states,
                )
                if isinstance(target, IPTarget)
                else None
            )
            exact_metrics = (
                exact_overlap_metrics(model, target) if n_ip <= evaluation.exact_max_n else None
            )
            repeated_ais = None
            if evaluation.ais_enabled:
                repeated_ais = _run_repeated_ais(
                    model,
                    target,
                    held_out,
                    evaluation,
                    seed=_derived_seed(evaluation_seed, settings.updates, 7),
                )
            evaluation_generator = make_device_generator(model.device, evaluation_seed + 2)
            sample_metrics = model_sample_metrics(
                model,
                target,
                n_chains=evaluation.model_sample_chains,
                burn_in=evaluation.model_sample_burn_in,
                rounds=evaluation.model_sample_rounds,
                thinning=evaluation.model_sample_thinning,
                generator=evaluation_generator,
            )
            final_evaluation_seconds = perf_counter() - final_evaluation_started
            metric_columns = _checkpoint_metric_columns(
                target=target,
                score_metrics=score_metrics,
                target_metrics=target_metrics,
                exact_population_kl=(
                    exact_metrics.population_kl if exact_metrics is not None else math.nan
                ),
                exact_ip_correlation=(
                    exact_metrics.ip_correlation if exact_metrics is not None else math.nan
                ),
                ais_metrics=repeated_ais,
            )
            metric_columns["validation_log_pseudolikelihood"] = _selection_pll(
                model, selection_states
            )

            parameter_path = parameter_dir / f"{stem}.npz"
            np.savez_compressed(
                parameter_path,
                theta=final_theta.numpy(),
                nce_log_normalizer=result.nce_log_normalizer,
                training_seed=run_seed,
                minibatch_seed=minibatch_seed,
                sampler_seed=sampler_seed,
                model=model_point.kind,
                pair_mode=pair_mode,
                all_pairs_sampler=settings.all_pairs_sampler,
                all_pairs_block_size=settings.all_pairs_block_size,
                n_visible=2 * n_ip,
                n_hidden=model_point.n_hidden,
            )
            last_record = result.history[-1]
            result_rows.append(
                {
                    **common,
                    "updates": settings.updates,
                    "training_examples_processed": settings.updates * settings.batch_size,
                    "gibbs_chain_sweeps": (
                        settings.updates * settings.gibbs_steps * gibbs_chains_per_update
                    ),
                    **metric_columns,
                    "sampled_ip_correlation": sample_metrics.ip_correlation,
                    "target_ip_correlation": target.mean_sign,
                    "ip_lag_one_autocorrelation": sample_metrics.lag_one_autocorrelation,
                    "final_objective": last_record.objective,
                    "final_gradient_norm": last_record.gradient_norm,
                    "maximum_absolute_parameter": last_record.maximum_absolute_parameter,
                    "fraction_at_bound": _fraction_at_bound(
                        model.flat_parameters().detach(), weight_bound
                    ),
                    "elapsed_seconds": result.elapsed_seconds,
                    "training_elapsed_seconds": result.elapsed_seconds,
                    "evaluation_seconds": final_evaluation_seconds,
                    "nce_log_normalizer": result.nce_log_normalizer,
                    "parameter_file": str(parameter_path),
                    **resource_columns,
                }
            )
            if settings.updates in checkpoint_updates:
                checkpoint_rows.append(
                    {
                        **common,
                        "update": settings.updates,
                        "training_examples_processed": settings.updates * settings.batch_size,
                        "gibbs_chain_sweeps": settings.updates
                        * settings.gibbs_steps
                        * gibbs_chains_per_update,
                        "training_elapsed_seconds": result.elapsed_seconds,
                        "checkpoint_evaluation_seconds": final_evaluation_seconds,
                        **metric_columns,
                        "maximum_absolute_parameter": last_record.maximum_absolute_parameter,
                        "fraction_at_bound": _fraction_at_bound(final_theta, weight_bound),
                        "nce_log_normalizer": result.nce_log_normalizer,
                        "parameter_file": str(parameter_path),
                        **resource_columns,
                    }
                )
            for record in result.history:
                history_rows.append({**common, **asdict(record)})
    finally:
        progress.close()

    results = pd.DataFrame(result_rows)
    results_path = output_dir / "results.csv"
    results.to_csv(results_path, index=False)
    pd.DataFrame(history_rows).to_csv(output_dir / "history.csv", index=False)
    _make_summary(results).to_csv(output_dir / "summary.csv", index=False)
    if checkpoint_rows:
        checkpoint_table = pd.DataFrame(checkpoint_rows)
        checkpoint_table.to_csv(output_dir / "checkpoints.csv", index=False)
        _make_checkpoint_summary(checkpoint_table).to_csv(
            output_dir / "checkpoint_summary.csv", index=False
        )
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    metadata = {
        "stage": "scalable_learning",
        "created_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "dtype": str(dtype),
        "exact_partition_function_during_training": False,
        "nested_training_sets": data_mode == "fixed",
        "training_data_mode": data_mode,
        "pair_initializations_across_training": pair_initializations,
        "pair_minibatches_across_models": pair_minibatches,
        "pair_minibatches_across_training": pair_training_batches,
        "pair_evaluation_states_across_models": pair_evaluation_states,
        "training_distribution": training_distribution,
        "checkpoint_updates": list(checkpoints.updates),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return results_path
