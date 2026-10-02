import pytest
import torch

from ip_rbm.scalable_data import CountCosineTarget, IPTarget
from ip_rbm.states import all_binary_states


def test_count_cosine_w1_matches_ip_target() -> None:
    cosine = CountCosineTarget(n_ip=4, beta=1.25, w=1)
    ip = IPTarget(n_ip=4, beta=1.25)
    states = all_binary_states(8)

    assert torch.allclose(cosine.log_unnormalized(states), ip.log_unnormalized(states))
    assert cosine.log_partition == pytest.approx(ip.log_partition)
    assert cosine.entropy == pytest.approx(ip.entropy)
    assert cosine.mean_sign == pytest.approx(ip.mean_sign)


def test_count_cosine_normalization_and_exact_count_sampling() -> None:
    target = CountCosineTarget(n_ip=5, beta=1.0, w=3)
    visible = all_binary_states(10)
    log_prob = target.log_prob(visible)

    assert torch.logsumexp(log_prob, dim=0) == pytest.approx(0.0, abs=1e-12)
    assert target.entropy == pytest.approx(float(-torch.sum(torch.exp(log_prob) * log_prob)))

    samples = target.sample(
        60000,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(71),
    )
    counts = (samples[:, :5] * samples[:, 5:]).sum(dim=1).to(torch.int64)
    empirical = torch.bincount(counts, minlength=6).to(torch.float64) / samples.shape[0]
    assert torch.max(torch.abs(empirical - torch.exp(target.count_log_prob))) < 0.012


@pytest.mark.parametrize("w", [0, -1])
def test_count_cosine_rejects_nonpositive_width(w: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        CountCosineTarget(n_ip=4, beta=1.0, w=w)
