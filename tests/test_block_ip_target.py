from pathlib import Path

import pytest
import torch
import yaml

from ip_rbm import scalable_experiment
from ip_rbm.scalable_data import BlockIPTarget, IPTarget
from ip_rbm.states import all_binary_states


def test_one_block_matches_ip_target() -> None:
    block = BlockIPTarget(n_ip=4, beta=1.25, block_size=4)
    ip = IPTarget(n_ip=4, beta=1.25)
    states = all_binary_states(8)

    assert torch.equal(block.sign(states), ip.sign(states))
    assert torch.allclose(block.log_unnormalized(states), ip.log_unnormalized(states))
    assert block.log_partition == pytest.approx(ip.log_partition)
    assert block.entropy == pytest.approx(ip.entropy)
    assert block.mean_sign == pytest.approx(ip.mean_sign)


def test_block_target_normalization_and_sampling() -> None:
    target = BlockIPTarget(n_ip=4, beta=0.8, block_size=2)
    states = all_binary_states(8)
    log_prob = target.log_prob(states)

    assert torch.logsumexp(log_prob, dim=0) == pytest.approx(0.0, abs=1e-12)
    assert target.entropy == pytest.approx(float(-torch.sum(torch.exp(log_prob) * log_prob)))

    samples = target.sample(
        50000,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(91),
    )
    products = samples[:, :4] * samples[:, 4:]
    block_counts = products.reshape(-1, 2, 2).sum(dim=2).to(torch.int64)
    block_signs = 1.0 - 2.0 * torch.remainder(block_counts, 2).to(torch.float64)
    empirical = block_signs.mean(dim=0)
    expected = torch.full_like(empirical, target.block_target.mean_sign)

    assert samples.shape == (50000, 8)
    assert torch.max(torch.abs(empirical - expected)) < 0.012


@pytest.mark.parametrize("block_size", [0, 3, 5])
def test_block_target_rejects_invalid_block_size(block_size: int) -> None:
    with pytest.raises(ValueError, match="block_size"):
        BlockIPTarget(n_ip=4, beta=1.0, block_size=block_size)


def test_block_ip_long_configuration_is_parameter_matched() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "stage2b_n16_block_ip_learning.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    targets = scalable_experiment._parse_targets(config)
    models = scalable_experiment._parse_models(config)

    assert [point.block_size for point in targets] == [1, 2, 4, 8, 16]
    assert all(isinstance(point.make(16), BlockIPTarget) for point in targets)

    by_group: dict[str, list[int]] = {}
    for point in models:
        model = scalable_experiment.make_trainable_model(
            point.kind,
            32,
            point.n_hidden,
            pair_mode=point.pair_mode,
            init_std=0.0,
        )
        by_group.setdefault(point.comparison_group, []).append(model.n_parameters)
    assert by_group == {
        "m3-032": [9272, 9280],
        "m3-064": [18512, 18528],
    }
