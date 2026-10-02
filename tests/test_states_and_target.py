import math

import torch

from ip_rbm.states import all_binary_states, pair_features
from ip_rbm.targets import make_ip_target


def test_binary_state_enumeration() -> None:
    states = all_binary_states(3)
    assert states.shape == (8, 3)
    assert torch.equal(states[0], torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64))
    assert torch.equal(states[-1], torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64))
    assert torch.unique(states, dim=0).shape[0] == 8


def test_cross_pair_features() -> None:
    states = torch.tensor([[1.0, 0.0, 1.0, 1.0]], dtype=torch.float64)
    features = pair_features(states, "cross")
    assert torch.equal(features, torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float64))


def test_ip_target_normalization_and_sector_ratio() -> None:
    beta = 0.7
    target = make_ip_target(n_ip=3, beta=beta)
    assert torch.allclose(target.prob.sum(), torch.tensor(1.0, dtype=torch.float64))
    sector_zero = target.log_prob[target.ip_sign == 1][0]
    sector_one = target.log_prob[target.ip_sign == -1][0]
    assert math.isclose(float(sector_zero - sector_one), 2 * beta, abs_tol=1e-13)
