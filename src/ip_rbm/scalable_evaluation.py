"""Exact-overlap and sampling-based evaluation for scalable learners."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ip_rbm.objectives import exact_kl, ip_correlation
from ip_rbm.scalable_data import IPCosineTarget, IPTarget, ScalableTarget, uniform_binary
from ip_rbm.scalable_learning import mean_log_pseudolikelihood
from ip_rbm.scalable_sampling import AISResult, estimate_log_partition_ais
from ip_rbm.states import all_binary_states
from ip_rbm.targets import ExactTarget
from ip_rbm.trainable_models import TrainableEnergyModel


@dataclass(frozen=True)
class ExactOverlapMetrics:
    """Exact population quantities available in the enumeration overlap regime."""

    population_kl: float
    ip_correlation: float


@torch.no_grad()
def exact_overlap_metrics(
    model: TrainableEnergyModel,
    target: ScalableTarget,
) -> ExactOverlapMetrics:
    """Evaluate a trained model through the existing CPU-float64 exact path."""
    states = all_binary_states(target.n_visible)
    log_prob = target.log_prob(states)
    exact_target = ExactTarget(
        states=states,
        ip_sign=target.sign(states),
        log_prob=log_prob,
        prob=torch.exp(log_prob),
        n_ip=target.n_ip,
        beta=target.beta,
    )
    exact_model = model.exact_model()
    theta = model.flat_parameters().detach().cpu().to(torch.float64)
    return ExactOverlapMetrics(
        population_kl=float(exact_kl(exact_model, theta, exact_target)),
        ip_correlation=float(ip_correlation(exact_model, theta, exact_target)),
    )


@dataclass(frozen=True)
class IPScoreMetrics:
    """Normalization-free diagnostics specialized to the IP target."""

    mean_log_pseudolikelihood: float
    even_mean_score: float
    odd_mean_score: float
    score_gap: float
    even_score_variance: float
    odd_score_variance: float


@dataclass(frozen=True)
class TargetScoreMetrics:
    """Normalization-free agreement with an arbitrary target score."""

    mean_log_pseudolikelihood: float
    target_score_rmse: float
    normalized_target_score_rmse: float
    target_score_correlation: float
    target_score_standard_deviation: float
    learned_ip_component_ratio: float = math.nan
    learned_cosine_component_ratio: float = math.nan


@torch.no_grad()
def target_score_metrics(
    model: TrainableEnergyModel,
    target: ScalableTarget,
    states: Tensor,
    *,
    pseudolikelihood_states: Tensor | None = None,
) -> TargetScoreMetrics:
    """Compare model and target scores after optimizing the additive constant."""
    visible = states.to(device=model.device, dtype=model.dtype)
    pseudolikelihood_visible = (
        visible
        if pseudolikelihood_states is None
        else pseudolikelihood_states.to(device=model.device, dtype=model.dtype)
    )
    model_score = model.log_unnormalized(visible)
    target_score = target.log_unnormalized(visible)
    if isinstance(target, IPTarget) and math.isinf(target.beta):
        # Infinite target energies do not admit a finite score-shape RMSE.
        return TargetScoreMetrics(
            mean_log_pseudolikelihood=float(
                mean_log_pseudolikelihood(model, pseudolikelihood_visible)
            ),
            target_score_rmse=math.nan,
            normalized_target_score_rmse=math.nan,
            target_score_correlation=math.nan,
            target_score_standard_deviation=math.nan,
        )
    residual = model_score - target_score
    centered_residual = residual - residual.mean()
    rmse = float(torch.sqrt(torch.mean(centered_residual.square())))
    centered_model = model_score - model_score.mean()
    centered_target = target_score - target_score.mean()
    model_variance = torch.mean(centered_model.square())
    target_variance = torch.mean(centered_target.square())
    target_standard_deviation = float(torch.sqrt(target_variance))
    if float(model_variance) == 0.0 or float(target_variance) == 0.0:
        correlation = math.nan
    else:
        correlation = float(
            torch.mean(centered_model * centered_target)
            / torch.sqrt(model_variance * target_variance)
        )
    learned_ip_component_ratio = math.nan
    learned_cosine_component_ratio = math.nan
    if isinstance(target, IPCosineTarget) and target.beta > 0:
        # The intercept absorbs the arbitrary additive score constant. Joint
        # least squares separates the parity coefficient from the correlated
        # low-frequency cosine scaffold.
        sign = target.sign(visible).detach().cpu().to(torch.float64)
        cosine = target.cosine_component(visible).detach().cpu().to(torch.float64)
        design = torch.stack((torch.ones_like(sign), sign, cosine), dim=1)
        response = model_score.detach().cpu().to(torch.float64)
        coefficients = torch.linalg.lstsq(design, response).solution
        learned_ip_component_ratio = float(coefficients[1] / target.beta)
        learned_cosine_component_ratio = float(coefficients[2] / (target.beta * target.rho))
    return TargetScoreMetrics(
        mean_log_pseudolikelihood=float(mean_log_pseudolikelihood(model, pseudolikelihood_visible)),
        target_score_rmse=rmse,
        normalized_target_score_rmse=(
            rmse / target_standard_deviation if target_standard_deviation > 0 else math.nan
        ),
        target_score_correlation=correlation,
        target_score_standard_deviation=target_standard_deviation,
        learned_ip_component_ratio=learned_ip_component_ratio,
        learned_cosine_component_ratio=learned_cosine_component_ratio,
    )


@torch.no_grad()
def ip_score_metrics(
    model: TrainableEnergyModel,
    target: IPTarget,
    states: Tensor,
    *,
    sector_states: Tensor | None = None,
) -> IPScoreMetrics:
    """Measure the desired inter-sector gap and unwanted intra-sector variation."""
    visible = states.to(device=model.device, dtype=model.dtype)
    sector_visible = (
        visible
        if sector_states is None
        else sector_states.to(device=model.device, dtype=model.dtype)
    )
    sign = target.sign(sector_visible)
    even = sign > 0
    odd = sign < 0
    if not bool(torch.any(even)) or not bool(torch.any(odd)):
        raise ValueError("evaluation states must contain both IP sectors")
    scores = model.log_unnormalized(sector_visible)
    even_scores = scores[even]
    odd_scores = scores[odd]
    even_mean = even_scores.mean()
    odd_mean = odd_scores.mean()
    return IPScoreMetrics(
        mean_log_pseudolikelihood=float(mean_log_pseudolikelihood(model, visible)),
        even_mean_score=float(even_mean),
        odd_mean_score=float(odd_mean),
        score_gap=float(even_mean - odd_mean),
        even_score_variance=float(even_scores.var(unbiased=False)),
        odd_score_variance=float(odd_scores.var(unbiased=False)),
    )


@dataclass(frozen=True)
class AISPopulationMetrics:
    """AIS-normalized population estimate from exact target samples."""

    population_kl: float
    target_cross_entropy: float
    ais: AISResult


@torch.no_grad()
def ais_population_metrics(
    model: TrainableEnergyModel,
    target: ScalableTarget,
    target_states: Tensor,
    *,
    n_particles: int,
    n_intermediate: int,
    sweeps_per_temperature: int = 1,
    generator: torch.Generator | None = None,
) -> AISPopulationMetrics:
    """Estimate population KL using target Monte Carlo and AIS normalization."""
    visible = target_states.to(device=model.device, dtype=model.dtype)
    ais = estimate_log_partition_ais(
        model,
        n_particles=n_particles,
        n_intermediate=n_intermediate,
        sweeps_per_temperature=sweeps_per_temperature,
        generator=generator,
    )
    model_score = model.log_unnormalized(visible)
    cross_entropy = -float(model_score.mean()) + ais.log_partition
    return AISPopulationMetrics(
        population_kl=cross_entropy - target.entropy,
        target_cross_entropy=cross_entropy,
        ais=ais,
    )


@dataclass(frozen=True)
class ModelSampleMetrics:
    """Diagnostics obtained from Gibbs samples of a fitted model."""

    ip_correlation: float
    lag_one_autocorrelation: float
    n_samples: int


@torch.no_grad()
def model_sample_metrics(
    model: TrainableEnergyModel,
    target: ScalableTarget,
    *,
    n_chains: int,
    burn_in: int,
    rounds: int,
    thinning: int,
    generator: torch.Generator | None = None,
) -> ModelSampleMetrics:
    """Estimate the model IP expectation and a basic mixing diagnostic."""
    if n_chains < 1 or burn_in < 0 or rounds < 2 or thinning < 1:
        raise ValueError("invalid model-sampling settings")
    visible = uniform_binary(
        n_chains,
        model.n_visible,
        device=model.device,
        dtype=model.dtype,
        generator=generator,
    )
    if burn_in:
        visible = model.gibbs_visible(visible, burn_in, generator=generator)
    signs: list[Tensor] = []
    for _ in range(rounds):
        visible = model.gibbs_visible(visible, thinning, generator=generator)
        signs.append(target.sign(visible))
    values = torch.stack(signs)
    centered = values - values.mean()
    variance = centered.square().mean()
    if float(variance) == 0.0:
        autocorrelation = float("nan")
    else:
        autocorrelation = float((centered[:-1] * centered[1:]).mean() / variance)
    return ModelSampleMetrics(
        ip_correlation=float(values.mean()),
        lag_one_autocorrelation=autocorrelation,
        n_samples=values.numel(),
    )
