import pandas as pd
import pytest
import torch
import yaml

from ip_rbm.scalable_experiment import run_scalable_experiment
from ip_rbm.states import all_binary_states
from ip_rbm.trainable_models import TrainableThreeBodyRBM


def model_fixture(n_visible: int = 6) -> TrainableThreeBodyRBM:
    model = TrainableThreeBodyRBM(n_visible, 2, pair_mode="all", dtype=torch.float64)
    generator = torch.Generator().manual_seed(57)
    model.load_flat_parameters(
        torch.randn(model.n_parameters, generator=generator, dtype=torch.float64) * 0.7
    )
    return model


@pytest.mark.parametrize("temperature", [0.0, 0.3, 1.0])
def test_cached_sweep_matches_legacy_with_identical_random_numbers(temperature: float) -> None:
    model = model_fixture()
    v = all_binary_states(6).repeat(4, 1)
    h = all_binary_states(2).repeat_interleave(64, dim=0)
    with torch.no_grad():
        expected = model.sample_visible(
            h, v, inverse_temperature=temperature, generator=torch.Generator().manual_seed(91)
        )
        before = model.flat_parameters().clone()
        model.configure_all_pairs_sampler("cached")
        actual = model.sample_visible(
            h, v, inverse_temperature=temperature, generator=torch.Generator().manual_seed(91)
        )
    assert torch.equal(expected, actual)
    assert torch.equal(model.flat_parameters(), before)


@pytest.mark.parametrize("temperature", [0.0, 0.3, 1.0])
def test_block_conditional_matches_enumerated_joint_energy(temperature: float) -> None:
    model = model_fixture()
    v = all_binary_states(6)[[5, 28, 61]]
    h = all_binary_states(2)[[1, 2, 3]]
    indices = torch.tensor([4, 0, 3])
    states = all_binary_states(3)
    with torch.no_grad():
        logits = model._block_logits(
            v,
            model.visible_bias + h @ model.weights.T,
            model._conditional_couplings(h),
            indices,
            states,
        )
        exhaustive = []
        for state in states:
            candidate = v.clone()
            candidate[:, indices] = state
            exhaustive.append(model.joint_log_unnormalized(candidate, h))
        expected = torch.stack(exhaustive, dim=1)
    assert torch.allclose(
        torch.softmax(temperature * logits, dim=1),
        torch.softmax(temperature * expected, dim=1),
        atol=1e-12,
    )


@pytest.mark.parametrize("block_size", [1, 4, 8])
def test_random_block_sweep_preserves_exact_conditional_law(block_size: int) -> None:
    # Six visible bits exercises a short final block (size 4), full update
    # (size 8), and the randomly ordered single-bit special case (size 1).
    model = model_fixture()
    model.configure_all_pairs_sampler("block", block_size)
    states = all_binary_states(6)
    fixed_hidden = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    generator = torch.Generator().manual_seed(28)
    with torch.no_grad():
        probability = torch.softmax(
            model.joint_log_unnormalized(states, fixed_hidden.expand(64, -1)), dim=0
        )
        initial = states[
            torch.multinomial(probability, 30000, replacement=True, generator=generator)
        ]
        actual = model.sample_visible(
            fixed_hidden.expand(len(initial), -1), initial, generator=generator
        )
    indices = (actual.long() * (2 ** torch.arange(5, -1, -1))).sum(dim=1)
    frequencies = torch.bincount(indices, minlength=64) / len(initial)
    uncertainty = 6 * torch.sqrt(probability * (1 - probability) / len(initial)) + 0.001
    assert torch.all((frequencies - probability).abs() < uncertainty)


def test_sampler_grid_filters_pairs_and_preserves_training_pairing(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = {
        "name": "sampler-test",
        "n_ip": [2],
        "beta": [0.5],
        "sample_sizes": [32],
        "dataset_repetitions": 1,
        "device": "cpu",
        "pair_initializations_across_training": True,
        "pair_minibatches_across_models": True,
        "pair_minibatches_across_training": True,
        "models": [{"kind": "3rbm", "hidden": [2], "pair_mode": mode} for mode in ["cross", "all"]],
        "training_defaults": {"updates": 2, "batch_size": 8, "learning_rate": 0.001},
        "training": [
            {"name": "cross", "algorithm": "cd", "pair_modes": ["cross"]},
            {
                "name": "cached",
                "algorithm": "cd",
                "pair_modes": ["all"],
                "all_pairs_sampler": "cached",
                "gibbs_steps": 2,
            },
            {
                "name": "block",
                "algorithm": "cd",
                "pair_modes": ["all"],
                "all_pairs_sampler": "block",
                "gibbs_steps": 4,
            },
        ],
        "checkpoints": {"updates": [1, 2]},
        "evaluation": {
            "target_samples": 32,
            "pseudolikelihood_samples": 8,
            "exact_max_n": 2,
            "model_sample_chains": 8,
            "model_sample_burn_in": 1,
            "model_sample_rounds": 2,
        },
        "output_dir": "results/test",
    }
    path = tmp_path / "test.yaml"
    path.write_text(yaml.safe_dump(config))
    result = pd.read_csv(run_scalable_experiment(path))
    assert len(result) == 3
    assert set(result.all_pairs_sampler) == {"register", "cached", "block"}
    assert result[result.pair_mode == "all"].initialization_seed.nunique() == 1
    assert result.minibatch_seed.nunique() == 1
    assert result.population_kl.ge(-1e-9).all()
    assert result.training_examples_processed.eq(16).all()
