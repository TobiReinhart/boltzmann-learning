"""Fresh-data CD must not recycle a finite dataset or perturb sampler RNG."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from ip_rbm.paper_studies import report_study, study_plan
from ip_rbm.scalable_data import FreshTargetBatches, IPTarget
from ip_rbm.scalable_experiment import run_scalable_experiment

CONFIGS = Path(__file__).parents[1] / "configs/paper"


def load(name):
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())


def test_fresh_stream_reproducible_and_rng_isolated():
    target = IPTarget(3, 1.0)
    first = FreshTargetBatches(target, 1024, 99)
    second = FreshTargetBatches(target, 1024, 99)
    before = torch.random.get_rng_state().clone()
    a, b = first(), first()
    assert torch.equal(a, second())
    assert torch.equal(b, second())
    assert not torch.equal(a, b)
    assert first.draws == 2048
    assert torch.equal(before, torch.random.get_rng_state())
    assert abs(float(target.sign(torch.cat([a, b])).mean()) - target.mean_sign) < 0.08


def test_fresh_protocol_preserves_reference_budget():
    old = load("study1_n16_paired_batch4096")
    new = load("study1_n16_fresh_3rbm")
    rbm = load("study1_n16_fresh_rbm_optional")
    for c in [new, rbm]:
        assert study_plan(c)["fits"] == 1
        assert study_plan(c)["optimizer_updates"] == 480000
        for key in [
            "training_defaults",
            "sample_sizes",
            "data_seed",
            "initialization_seed",
            "checkpoints",
            "evaluation",
            "weight_bound",
            "targets",
        ]:
            assert c[key] == old[key]
    assert new["models"] == [old["models"][1]]
    assert new["training"] == [old["training"][1]]
    assert rbm["models"] == [old["models"][0]]
    assert rbm["training"] == [old["training"][0]]


def test_fresh_cpu_smoke_and_reports(tmp_path):
    c = load("study1_n16_fresh_3rbm")
    c.update(n_ip=[2], sample_sizes=[32], device="cpu", output_dir=str(tmp_path / "fresh"))
    c["models"][0]["hidden"] = [2]
    c["training_defaults"].update(updates=3, batch_size=8, record_every=1)
    c["checkpoints"]["updates"] = [1, 3]
    c["evaluation"].update(
        selection_samples=8,
        target_samples=16,
        pseudolikelihood_samples=8,
        model_sample_chains=4,
        model_sample_burn_in=1,
        model_sample_rounds=2,
        model_sample_thinning=1,
    )
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(c))
    result = run_scalable_experiment(p)
    report_study(result)
    r = pd.read_csv(result).iloc[0]
    assert r.training_data_mode == "fresh"
    assert r.sample_size == 0
    assert r.reference_sample_size == 32
    assert r.fresh_target_draws_total == r.training_examples_processed == 24
    assert np.isfinite(r.population_kl)
    for archive in (result.parent / "datasets").glob("ip_*.npz"):
        with np.load(archive) as f:
            assert f["train_states"].shape == (0, 4)
