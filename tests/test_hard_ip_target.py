"""Hard-support IP sampling, evaluation and complete report checks."""

import math
import runpy
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from ip_rbm.objectives import kl_from_log_prob
from ip_rbm.paper_studies import report_study, study_plan
from ip_rbm.scalable_data import IPTarget
from ip_rbm.scalable_experiment import run_scalable_experiment
from ip_rbm.states import all_binary_states
from ip_rbm.targets import ExactTarget


def test_hard_ip_exact_and_sampling():
    target = IPTarget(3, math.inf)
    states = all_binary_states(6)
    logp = target.log_prob(states)
    prob = logp.exp()
    assert int((prob > 0).sum()) == 36
    assert float(prob.sum()) == 1.0
    assert target.entropy == math.log(36)
    assert target.mean_sign == 1
    sample = target.sample(8192, generator=torch.Generator().manual_seed(7))
    assert bool((target.sign(sample) == 1).all())
    ids = (sample.to(torch.int64) * 2 ** torch.arange(6)).sum(1)
    counts = torch.bincount(ids, minlength=64)
    positive = counts[counts > 0].float()
    assert len(positive) == 36
    assert float(((positive - 8192 / 36) ** 2 / (8192 / 36)).sum()) < 85
    exact = ExactTarget(states, target.sign(states), logp, prob, 3, math.inf)
    uniform = torch.full_like(logp, -math.log(64))
    assert abs(float(kl_from_log_prob(exact, uniform)) - math.log(64 / 36)) < 1e-12
    assert float(kl_from_log_prob(exact, logp)) == 0


def test_beta_protocol_pairing_and_hard_target_smoke(tmp_path):
    root = Path(__file__).parents[1] / "configs/paper"
    configs = [
        yaml.safe_load((root / f"study1_n8_{name}.yaml").read_text())
        for name in ["beta1", "beta_infinity"]
    ]
    for key in configs[0].keys() - {"name", "output_dir", "targets"}:
        assert configs[0][key] == configs[1][key]
    assert all(study_plan(c)["fits"] == 6 for c in configs)
    results = []
    for i, c in enumerate(configs):
        c.update(
            n_ip=[2],
            sample_sizes=[32],
            dataset_repetitions=1,
            device="cpu",
            dataset_generation_batch_size=16,
            output_dir=str(tmp_path / f"run{i}"),
        )
        for m in c["models"]:
            m["hidden"] = [2]
        c["training_defaults"].update(updates=2, batch_size=8, record_every=1)
        c["checkpoints"]["updates"] = [1, 2]
        c["evaluation"].update(
            selection_samples=8,
            target_samples=16,
            pseudolikelihood_samples=8,
            model_sample_chains=4,
            model_sample_burn_in=1,
            model_sample_rounds=2,
            model_sample_thinning=1,
        )
        path = tmp_path / f"c{i}.yaml"
        path.write_text(yaml.safe_dump(c))
        result = run_scalable_experiment(path)
        report_study(result)
        results.append(pd.read_csv(result, dtype=str).sort_values("model"))
    for key in ["initialization_seed", "minibatch_seed", "sampler_seed"]:
        assert results[0][key].tolist() == results[1][key].tolist()
    hard = results[1]
    assert np.isfinite(hard.population_kl.astype(float)).all()
    assert hard.normalized_target_score_rmse.isna().all()
    kl = hard.population_kl.astype(float)
    mass = hard.exact_even_probability.astype(float)
    within = hard.exact_even_conditional_kl.astype(float)
    np.testing.assert_allclose(kl, -np.log(mass) + within)
    assert (within >= -1e-12).all()
    for i, suffix in enumerate(["beta1", "beta_infinity"]):
        destination = tmp_path / f"results/paper_study1_n8_{suffix}"
        destination.mkdir(parents=True)
        shutil.copyfile(tmp_path / f"run{i}/checkpoints.csv", destination / "checkpoints.csv")
    plot = runpy.run_path(str(root.parents[1] / "scripts/plot_ip_beta_comparison.py"))
    plot["main"](tmp_path)
    output = tmp_path / "results/paper_study1_n8_beta_comparison"
    assert (output / "beta_comparison_update.pdf").stat().st_size > 0
    assert (output / "beta_comparison_training_elapsed_seconds.pdf").stat().st_size > 0
    assert len(pd.read_csv(output / "kl_attainment.csv")) == 12
