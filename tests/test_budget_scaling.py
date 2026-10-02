"""Large-budget follow-up: bounded data generation and single-depth reporting."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from ip_rbm.paper_studies import run_study, study_plan
from ip_rbm.scalable_data import IPTarget, uniform_binary
from ip_rbm.scalable_experiment import _generate_training_states


@pytest.mark.parametrize("distribution", ["target", "uniform"])
def test_unchunked_generation_preserves_original_rng(distribution):
    target = IPTarget(3, 1.0)
    rng = torch.Generator().manual_seed(123)
    expected = (
        target.sample(71, generator=rng)
        if distribution == "target"
        else uniform_binary(71, 6, device=torch.device("cpu"), dtype=torch.float32, generator=rng)
    )
    actual = _generate_training_states(
        target,
        71,
        torch.Generator().manual_seed(123),
        distribution=distribution,
        chunk_size=None,
    )
    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual, expected.numpy())


def test_chunked_generation_is_bounded_reproducible_and_correct():
    target = IPTarget(3, 1.0)
    calls = []

    class RecordingTarget:
        n_visible = target.n_visible

        def sample(self, count, **kwargs):
            calls.append(count)
            return target.sample(count, **kwargs)

    kwargs = dict(distribution="target", chunk_size=1024)
    actual = _generate_training_states(
        RecordingTarget(), 20003, torch.Generator().manual_seed(5), **kwargs
    )
    repeat = _generate_training_states(target, 20003, torch.Generator().manual_seed(5), **kwargs)
    assert max(calls) == 1024 and calls[-1] == 547
    assert actual.shape == (20003, 6)
    np.testing.assert_array_equal(actual, repeat)
    states = torch.from_numpy(actual).float()
    assert abs(float(target.sign(states).mean()) - target.mean_sign) < 0.025
    assert np.isin(actual, [0, 1]).all()


def test_budget_config_has_compensated_data_and_matched_parameters():
    root = Path(__file__).parents[1]
    config = yaml.safe_load((root / "configs/paper/study1_budget_n12.yaml").read_text())
    assert study_plan(config)["fits"] == 2
    assert config["sample_sizes"] == [131072 * 256]
    assert config["training_defaults"]["batch_size"] == 256 * 256
    assert config["training_defaults"]["updates"] >= 50000
    assert config["dataset_generation_batch_size"] == 65536
    rbm, cubic = config["models"]
    assert cubic["hidden"] == [12**2]
    assert abs((24 + 25 * rbm["hidden"][0]) - (24 + 301 * cubic["hidden"][0])) == 6


def test_single_depth_study_and_chunked_dataset_complete(tmp_path):
    root = Path(__file__).parents[1]
    config = yaml.safe_load((root / "configs/paper/study1_budget_n12.yaml").read_text())
    config.update(
        n_ip=[2],
        sample_sizes=[23],
        dataset_generation_batch_size=7,
        device="cpu",
        output_dir=str(tmp_path / "results"),
    )
    for model in config["models"]:
        model["hidden"] = [2]
    config["training_defaults"].update(updates=2, batch_size=8, record_every=1)
    config["checkpoints"]["updates"] = [1, 2]
    config["evaluation"].update(
        selection_samples=16,
        target_samples=32,
        pseudolikelihood_samples=16,
        model_sample_chains=4,
        model_sample_burn_in=1,
        model_sample_rounds=2,
        model_sample_thinning=1,
    )
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(config))
    result = run_study(path)
    assert result.is_file()
    assert (result.parent / "selected.csv").is_file()
    assert list(result.parent.glob("resources_*.pdf"))
    with np.load(next((result.parent / "datasets").glob("ip_*.npz"))) as data:
        assert data["train_states"].shape == (23, 4)
        assert data["train_states"].dtype == np.uint8
