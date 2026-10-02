"""Protocol and seed-pairing checks for the n=12 confirmation/ablation jobs."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ip_rbm.paper_studies import study_plan
from ip_rbm.scalable_experiment import _parse_models, _parse_training, run_scalable_experiment
from ip_rbm.scalable_learning import learning_rate_at_update

CONFIGS = Path(__file__).parents[1] / "configs/paper"


def load(name):
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())


def test_followup_protocols_change_only_planned_budgets():
    original = load("study1_budget_n12")
    a = load("study1_n12_ablation_small_data")
    b = load("study1_n12_ablation_small_batch")
    for config in [a, b]:
        assert study_plan(config)["fits"] == 1
        for key in [
            "n_ip",
            "targets",
            "data_seed",
            "initialization_seed",
            "initialization_std",
            "dataset_generation_batch_size",
            "weight_bound",
            "evaluation",
            "checkpoints",
        ]:
            assert config[key] == original[key]
        assert config["models"][0] == {**original["models"][1], "seed_index": 1}
        assert config["training"][0] == {**original["training"][1], "seed_index": 1}
    assert a["sample_sizes"] == [131072]
    assert a["training_defaults"] == original["training_defaults"]
    assert b["sample_sizes"] == original["sample_sizes"]
    assert b["training_defaults"] == {**original["training_defaults"], "batch_size": 256}


def test_confirmation_seeds_are_new_and_paired_between_architectures():
    original = load("study1_budget_n12")
    cubic = load("study1_n12_confirm_3rbm")
    rbm = load("study1_n12_confirm_rbm_optional")
    for config in [cubic, rbm]:
        assert study_plan(config)["fits"] == 2
        assert config["training_defaults"] == original["training_defaults"]
        assert config["sample_sizes"] == original["sample_sizes"]
        assert config["data_seed"] != original["data_seed"]
        assert config["initialization_seed"] != original["initialization_seed"]
        assert config["clean_output_dir"] is False
        assert config["output_dir"] != original["output_dir"]
    assert cubic["data_seed"] == rbm["data_seed"]
    assert cubic["initialization_seed"] == rbm["initialization_seed"]


def test_matched_examples_preserves_protocol_and_exposure():
    reference = load("study1_n12_ablation_small_batch")
    new = load("study1_n12_batch4096_matched_examples")
    original = load("study1_budget_n12")
    assert study_plan(new)["fits"] == 1
    for key in reference.keys() - {"name", "output_dir", "training_defaults", "checkpoints"}:
        assert new[key] == reference[key]
    old_settings = original["training_defaults"]
    settings = new["training_defaults"]
    assert settings == {
        **old_settings,
        "batch_size": 4096,
        "updates": 1200000,
        "record_every": 16000,
    }
    assert settings["batch_size"] * settings["updates"] == 4915200000
    assert settings["batch_size"] * settings["updates"] == (
        old_settings["batch_size"] * old_settings["updates"]
    )
    assert {16 * step for step in original["checkpoints"]["updates"]} <= set(
        new["checkpoints"]["updates"]
    )
    assert 8192 * settings["batch_size"] == new["sample_sizes"][0]
    assert new["output_dir"] != original["output_dir"]


def test_paired_batch4096_protocol():
    new = load("study1_n12_paired_batch4096")
    reference = load("study1_n12_batch4096_matched_examples")
    timing = load("study1_n12_paired_batch4096_timing")
    assert study_plan(new)["fits"] == 6
    assert study_plan(new)["optimizer_updates"] == 2880000
    assert study_plan(timing)["fits"] == 2
    for key in [
        "sample_sizes",
        "targets",
        "n_ip",
        "weight_bound",
        "evaluation",
        "dataset_generation_batch_size",
        "initialization_std",
    ]:
        assert new[key] == reference[key]
    for key in ["models", "training"]:
        assert timing[key] == new[key]
    assert new["training_defaults"]["batch_size"] == 4096
    assert new["training_defaults"]["updates"] * 4096 == 1966080000
    for previous in [reference, load("study1_n12_confirm_3rbm")]:
        assert new["data_seed"] != previous["data_seed"]
        assert new["initialization_seed"] != previous["initialization_seed"]
    old_settings = _parse_training(reference)[0].settings
    for learner in _parse_training(new):
        for step in new["checkpoints"]["updates"]:
            assert learning_rate_at_update(learner.settings, step) == (
                learning_rate_at_update(old_settings, step)
            )
    assert 24 + 25 * 1734 == 43374
    assert 24 + (1 + 24 + 276) * 144 == 43368
    assert new["clean_output_dir"] is False


def test_n16_timing_uses_matched_capacity_and_successful_learner():
    config = load("study1_n16_paired_batch4096_timing")
    previous = load("study1_n12_paired_batch4096_timing")
    assert study_plan(config)["fits"] == 2
    assert study_plan(config)["optimizer_updates"] == 2000
    assert config["training_defaults"] == previous["training_defaults"]
    assert config["training"] == previous["training"]
    assert config["n_ip"] == [16]
    assert config["models"][0]["hidden"] == [4104]
    assert config["models"][1]["hidden"] == [256]
    assert 32 + 33 * 4104 == 135464
    assert 32 + (1 + 32 + 496) * 256 == 135456
    assert config["sample_sizes"] == [65536]
    assert config["clean_output_dir"] is False
    assert config["device"] == "mps"


def test_n16_long_run_transfers_n12_schedule_and_calibrated_models():
    config = load("study1_n16_paired_batch4096")
    previous = load("study1_n12_paired_batch4096")
    timing = load("study1_n16_paired_batch4096_timing")
    assert study_plan(config)["fits"] == 2
    assert study_plan(config)["optimizer_updates"] == 960000
    for key in ["models", "training", "n_ip", "device"]:
        assert config[key] == timing[key]
    for key in [
        "training_defaults",
        "sample_sizes",
        "evaluation",
        "checkpoints",
        "weight_bound",
        "targets",
        "dataset_generation_batch_size",
    ]:
        assert config[key] == previous[key]
    assert config["data_seed"] != timing["data_seed"]
    assert config["initialization_seed"] != timing["initialization_seed"]
    assert config["output_dir"] not in [previous["output_dir"], timing["output_dir"]]
    assert config["clean_output_dir"] is False
    assert config["pair_minibatches_across_models"] is True
    assert config["pair_evaluation_states_across_models"] is True


def test_paired_batch4096_tiny_cpu_smoke(tmp_path):
    config = load("study1_n12_paired_batch4096")
    config.update(
        n_ip=[2],
        sample_sizes=[32],
        dataset_repetitions=1,
        device="cpu",
        dataset_generation_batch_size=16,
        output_dir=str(tmp_path / "paired"),
    )
    for model in config["models"]:
        model["hidden"] = [2]
    config["training_defaults"].update(updates=2, batch_size=8, record_every=1)
    config["checkpoints"]["updates"] = [1, 2]
    config["evaluation"].update(
        selection_samples=8,
        target_samples=16,
        pseudolikelihood_samples=8,
        model_sample_chains=4,
        model_sample_burn_in=1,
        model_sample_rounds=2,
        model_sample_thinning=1,
    )
    path = tmp_path / "paired.yaml"
    path.write_text(yaml.safe_dump(config))
    result = pd.read_csv(run_scalable_experiment(path), dtype=str)
    assert set(result.model) == {"rbm", "3rbm"}
    assert result.minibatch_seed.nunique() == 1
    assert result.selection_seed.nunique() == 1
    assert np.isfinite(result.normalized_target_score_rmse.astype(float)).all()


@pytest.mark.parametrize("value", [-1, True, 1.5, "1"])
def test_invalid_seed_identity_is_rejected(value):
    config = load("study1_n12_confirm_3rbm")
    config["models"][0]["seed_index"] = value
    with pytest.raises(ValueError, match="seed_index"):
        _parse_models(config)
    config["training"][0]["seed_index"] = value
    with pytest.raises(ValueError, match="seed_index"):
        _parse_training(config)


def test_subset_reproduces_full_grid_initialization_and_sampler(tmp_path):
    full = load("study1_budget_n12")
    full.update(n_ip=[2], sample_sizes=[32], device="cpu", dataset_generation_batch_size=16)
    for model in full["models"]:
        model["hidden"] = [2]
    full["training_defaults"].update(updates=2, batch_size=8, record_every=1)
    full["checkpoints"]["updates"] = [1, 2]
    full["evaluation"].update(
        selection_samples=8,
        target_samples=16,
        pseudolikelihood_samples=8,
        model_sample_chains=4,
        model_sample_burn_in=1,
        model_sample_rounds=2,
        model_sample_thinning=1,
    )
    subset = deepcopy(full)
    subset["models"] = [{**full["models"][1], "seed_index": 1}]
    subset["training"] = [{**full["training"][1], "seed_index": 1}]
    results = []
    for name, config in [("full", full), ("subset", subset)]:
        config["output_dir"] = str(tmp_path / name)
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config))
        result = pd.read_csv(run_scalable_experiment(path), dtype=str)
        results.append(result.loc[result.model == "3rbm"].iloc[0])
    old, new = results
    for key in [
        "initialization_seed",
        "training_seed",
        "minibatch_seed",
        "sampler_seed",
        "selection_seed",
        "model_seed_index",
        "training_seed_index",
    ]:
        assert old[key] == new[key]
    with np.load(old.parameter_file) as old_file, np.load(new.parameter_file) as new_file:
        np.testing.assert_array_equal(old_file["theta"], new_file["theta"])
