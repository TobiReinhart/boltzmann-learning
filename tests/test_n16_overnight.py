"""The overnight attempt trades update count for a larger batch and CD depth."""

from pathlib import Path

import yaml

from ip_rbm.paper_studies import study_plan
from ip_rbm.scalable_experiment import _parse_training
from ip_rbm.scalable_learning import learning_rate_at_update

CONFIGS = Path(__file__).parents[1] / "configs/paper"


def load(name):
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())


def test_overnight_budget_and_preserved_reference():
    new = load("study1_n16_overnight_3rbm")
    old = load("study1_n16_fresh_3rbm")
    timing = load("study1_n16_overnight_timing")
    assert study_plan(new)["fits"] == 1
    assert study_plan(new)["optimizer_updates"] == 160000
    for key in [
        "training_data_mode",
        "models",
        "targets",
        "n_ip",
        "sample_sizes",
        "weight_bound",
        "initialization_seed",
        "data_seed",
        "initialization_std",
        "evaluation",
    ]:
        assert new[key] == old[key]
    assert new["training_data_mode"] == "fresh"
    settings = _parse_training(new)[0].settings
    short = _parse_training(timing)[0].settings
    assert settings.batch_size == 32768 == short.batch_size
    assert settings.gibbs_steps == 4 == short.gibbs_steps
    assert settings.updates * settings.batch_size == 5242880000
    assert 60000 * settings.batch_size == 480000 * 4096
    assert new["models"] == timing["models"]
    assert new["training"] == timing["training"]
    for step in new["checkpoints"]["updates"]:
        assert 1 <= step <= settings.updates
        assert learning_rate_at_update(settings, step) == settings.learning_rate
    assert settings.updates in new["checkpoints"]["updates"]
    assert not new["clean_output_dir"]
    assert len({new["output_dir"], old["output_dir"], timing["output_dir"]}) == 3
