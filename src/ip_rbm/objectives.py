"""Exact normalized probabilities, KL divergences, and diagnostics."""

import torch
from torch import Tensor

from ip_rbm.models import ExactModel
from ip_rbm.targets import ExactTarget


def normalized_log_prob(log_unnormalized: Tensor) -> Tensor:
    """Normalize a vector of log weights by exact log-sum-exp."""
    return log_unnormalized - torch.logsumexp(log_unnormalized, dim=0)


def cross_entropy_from_log_prob(distribution: Tensor, model_log_prob: Tensor) -> Tensor:
    """Return ``-sum_v distribution(v) log q(v)`` on an enumerated state space."""
    if distribution.shape != model_log_prob.shape:
        raise ValueError("distribution and model state spaces differ")
    return -torch.sum(distribution * model_log_prob)


def kl_from_log_prob(target: ExactTarget, model_log_prob: Tensor) -> Tensor:
    """Compute ``D_KL(target || model)`` from fully enumerated probabilities."""
    if model_log_prob.shape != target.log_prob.shape:
        raise ValueError("target and model state spaces differ")
    support = target.prob > 0
    return torch.sum(target.prob[support] * (target.log_prob[support] - model_log_prob[support]))


def exact_kl(model: ExactModel, theta: Tensor, target: ExactTarget) -> Tensor:
    """Compute the exact visible-space population KL divergence."""
    if model.n_visible != target.states.shape[1]:
        raise ValueError("model and target visible dimensions differ")
    logits = model.log_unnormalized(target.states, theta)
    return kl_from_log_prob(target, normalized_log_prob(logits))


def model_probability(model: ExactModel, theta: Tensor, target: ExactTarget) -> Tensor:
    """Return the model distribution on the target's enumerated state table."""
    logits = model.log_unnormalized(target.states, theta)
    return torch.exp(normalized_log_prob(logits))


def ip_correlation(model: ExactModel, theta: Tensor, target: ExactTarget) -> Tensor:
    """Compute ``E_q[(-1)**IP]`` exactly."""
    return torch.sum(model_probability(model, theta, target) * target.ip_sign)
