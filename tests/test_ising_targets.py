from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from ip_rbm.ising_targets import IsingTarget, frustration_table, ground_graph
from ip_rbm.scalable_experiment import (
    _parse_models,
    _parse_targets,
    _parse_training,
    run_scalable_experiment,
)
from ip_rbm.states import all_binary_states


@pytest.mark.parametrize("weighted", [False, True])
def test_frustration_transform_matches_brute_force(weighted):
    edges = ground_graph(8, "ring_chords", 11)
    weights = np.linspace(0.5, 1.5, 8) if weighted else np.ones(8)
    spins = 2 * all_binary_states(int(edges.max()) + 1).numpy() - 1
    disorder = 2 * ((np.arange(256)[:, None] >> np.arange(8)) & 1) - 1
    products = spins[:, edges[:, 0]] * spins[:, edges[:, 1]]
    expected = np.min(-disorder @ (products * weights).T, axis=1)
    actual = 2 * frustration_table(edges, weights) - weights.sum()
    assert np.allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("encoding,n", [("signed_product", 4), ("direct", 2)])
@pytest.mark.parametrize("kind,couplings", [("ising_energy", "mixed"), ("ising_ground", "unit")])
@pytest.mark.parametrize("objective", ["smooth", "threshold"])
def test_exact_normalization_entropy_and_sampling(encoding, n, kind, couplings, objective):
    target = IsingTarget(n, 0.8, kind, "cycles", couplings, 11, encoding, objective)
    states = all_binary_states(2 * n)
    probabilities = target.log_prob(states).exp()
    assert probabilities.sum().item() == pytest.approx(1, abs=1e-12)
    assert -(probabilities * target.log_prob(states)).sum().item() == pytest.approx(target.entropy)
    assert (probabilities * target.sign(states)).sum().item() == pytest.approx(target.mean_sign)
    sample = target.sample(40000, generator=torch.Generator().manual_seed(72))
    frequencies = torch.bincount(target.indices(sample), minlength=len(target.scores)) / len(sample)
    assert (frequencies - target.probabilities).abs().max() < 0.008
    if encoding == "signed_product":
        assert (sample[:, :n].mean(0) - 0.5).abs().max() < 0.012


def test_ground_gauge_invariance_and_energy_convention():
    target = IsingTarget(8, 1, "ising_ground", "ring_chords", "weighted")
    edges, weights, energy = target.landscape
    gauge = np.random.default_rng(18).choice([-1, 1], int(edges.max()) + 1)
    mask = int((gauge[edges[:, 0]] * gauge[edges[:, 1]] < 0) @ (1 << np.arange(8)))
    assert np.allclose(energy, energy[np.arange(256) ^ mask])
    assert energy[-1] == pytest.approx(-weights.sum())


def test_energy_is_fixed_ising_hamiltonian_and_standardized():
    target = IsingTarget(8, 1, "ising_energy", "ladder", "mixed")
    edges, weights, energy = target.landscape
    bits = (np.arange(256)[:, None] >> np.arange(8)) & 1
    spins = 2 * bits - 1
    assert np.array_equal(energy, -(spins[:, edges[:, 0]] * spins[:, edges[:, 1]]) @ weights)
    assert target.scores.mean().item() == pytest.approx(0, abs=1e-12)
    assert target.scores.square().mean().item() == pytest.approx(1)


def test_invalid_size_is_rejected_before_enumeration():
    with pytest.raises(ValueError, match=r"4\.\.16"):
        IsingTarget(16, 1, "ising_energy", encoding="direct")


@pytest.mark.parametrize("name", ["comparison", "direct_control"])
def test_prepared_grids_have_exact_targets_and_matched_parameters(name):
    path = Path(__file__).parents[1] / f"configs/stage2b_ising_{name}.yaml"
    config = yaml.safe_load(path.read_text())
    count = 0
    for n in config["n_ip"]:
        models = [m for m in _parse_models(config) if m.supports(n)]
        parameters = [
            (
                2 * n
                + ((2 * n + 1) if m.kind == "rbm" else (2 * n + 1 + (2 * n) * (2 * n - 1) // 2))
                * m.n_hidden
            )
            for m in models
        ]
        assert max(parameters) / min(parameters) < 1.004
        for point in _parse_targets(config):
            target = point.make(n)
            assert target.entropy > 0
            assert target.scores.square().mean().item() == pytest.approx(1, abs=1e-10)
            for model in models:
                mode = model.pair_mode if model.kind == "3rbm" else "none"
                count += sum(mode in t.pair_modes for t in _parse_training(config))
    assert count * config["dataset_repetitions"] == 128


def test_ising_end_to_end_and_target_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {
        "name": "ising-test",
        "n_ip": [4],
        "sample_sizes": [64],
        "device": "cpu",
        "dataset_repetitions": 1,
        "targets": [
            {"name": "energy", "kind": "ising_energy", "graph": "cycles"},
            {"name": "ground", "kind": "ising_ground", "graph": "cycles"},
        ],
        "models": [
            {"kind": "rbm", "hidden": [10]},
            {"kind": "3rbm", "hidden": [3], "pair_mode": "all"},
        ],
        "training_defaults": {"updates": 2, "batch_size": 8, "learning_rate": 0.002},
        "training": [
            {"name": "rbm", "algorithm": "cd", "pair_modes": ["none"]},
            {
                "name": "all",
                "algorithm": "cd",
                "pair_modes": ["all"],
                "all_pairs_sampler": "block",
                "gibbs_steps": 2,
            },
        ],
        "checkpoints": {"updates": [1, 2]},
        "evaluation": {
            "target_samples": 64,
            "pseudolikelihood_samples": 16,
            "exact_max_n": 4,
            "model_sample_chains": 8,
            "model_sample_burn_in": 1,
            "model_sample_rounds": 2,
        },
        "output_dir": "results/test",
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    result = pd.read_csv(run_scalable_experiment(path))
    assert len(result) == 4
    assert result.population_kl.ge(-1e-9).all()
    assert result.target_encoding.eq("signed_product").all()
    assert len(list((tmp_path / "results/test/target_specs").glob("*.npz"))) == 2
