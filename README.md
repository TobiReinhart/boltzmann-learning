# Boltzmann learning benchmarks

Python/PyTorch tools for studying how interaction structure affects learning
in energy-based probabilistic models. The framework compares ordinary restricted
Boltzmann machines (RBMs) with three-body RBMs (3RBMs), whose hidden units can
also couple to pairs of visible units in 3-body interactions.

**Status:** research software accompanying a manuscript in preparation. This
repository contains code and experiment definitions. A preprint link will be
added when available.

## What is implemented?

- **Models:** RBMs and 3RBMs with cross-register or all 3-body interactions.
- **Exact reference calculations:** population KL and finite-sample maximum
  likelihood with full enumeration and bounded SciPy L-BFGS-B optimization.
- **Scalable learning:** CD-k, persistent CD, replica-exchange persistent CD,
  pseudolikelihood and noise-contrastive estimation; supervised energy regression
  is available as a separate diagnostic.
- **Sampling:** conditional Gibbs updates, including exact small-block visible
  updates for all-pairs 3RBMs, and fixed-dataset or fresh-observation CD training.
- **Targets:** inner-product-modulo-two distributions, product-encoded and
  higher-order spin models, small ground-energy targets, and generic controls.
- **Evaluation:** exact KL for small systems, optional annealed importance
  sampling, energy-shape diagnostics, and CPU/wall-time/memory measurements.
- **Reproducibility:** YAML protocols, seeded random streams, paired datasets,
  saved parameter snapshots and automatic plots/reports.

The code supports CPU, Apple MPS and CUDA. The demonstration below uses CPU;
accelerator speed, reproducibility and memory usage depend on the device.

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/TobiReinhart/boltzmann-learning.git
cd boltzmann-learning
uv sync --locked --dev
uv run ip-rbm run-study configs/demo.yaml --dry-run
uv run ip-rbm run-study configs/demo.yaml
```

The demo trains a small parameter-matched RBM/3RBM pair on eight visible units
for 200 CD-2 updates each. It is intended as a quick workflow check. It requires neither downloaded data nor a GPU. Initial dependency installation can take longer than training.

Generated files appear under `results/demo/`, including `results.csv`,
`checkpoints.csv`, model parameters and diagnostic PDFs. A nonempty output
directory is protected from silent overwriting: to run again, change `name`
and `output_dir` in a copy of the YAML. Inspect all commands with:

```bash
uv run ip-rbm --help
```

## Experiment design and interpretation

For `n_ip = n`, the visible vector contains **2n binary variables**.
An RBM has linear visible-hidden couplings; the 3RBM additionally has 3-body terms
`C[i,j,r] * v[i] * v[j] * h[r]`. Hidden units remain conditionally independent
given visibles, but all-pairs 3RBM visibles do not factorize given hidden units.
One CD-k step uses k complete hidden/visible sweeps, not k individual spin updates.

Compare architectures by parameter count, not simply hidden-unit count.
Equal update or parameter budgets do not imply equal runtime or memory.
Short-chain CD approximates the likelihood gradient and can remain biased.
Small numerical RBM errors are attained fits, not certified approximation bounds.

The normalized energy diagnostic compares model and target log weights up to
an additive constant on independent uniform probes. Zero is agreement and one
is the flat-energy reference. The probes are not guaranteed
absent from the training observations. No minibatch-only normalization is used
as a substitute for the model partition function.

`configs/` contains exact, finite-sample and scalable experiment definitions;
`configs/paper/` contains the manuscript-related protocols, including controls.
Some historical configurations require earlier locally generated reference fits,
and some explicitly select MPS. Check the YAML before running. Large protocols
may require hours, substantial RAM and billions of sampled observations.

With `training_data_mode: fresh`, observations are independently generated at
each update rather than stored as a reusable dataset. `sample_sizes` then labels
the historical reference budget; actual draws are `updates * batch_size` and are
recorded separately. Parameter snapshots are saved, but **optimizer-state crash
resumption is not implemented**.

For reproducibility, the `run-study` command writes configuration, software and
source snapshots beside local results. These generated outputs are ignored by Git.
Review such snapshots before distributing them separately.

## Code map

| Module | Responsibility |
|---|---|
| `models.py`, `trainable_models.py` | Exact and minibatch-based model evaluation |
| `targets.py`, `scalable_data.py`, `ising_targets.py`, `benchmark_targets.py` | Target definitions and sampling |
| `optimization.py`, `learning.py` | Enumerated population/empirical optimization |
| `scalable_learning.py`, `scalable_sampling.py` | Stochastic learning and Monte Carlo kernels |
| `scalable_evaluation.py`, `resource_monitor.py` | Diagnostics and resource measurements |
| `paper_studies.py`, `cli.py`, `plotting.py` | Configured studies, reporting and plotting |
| `tests/` | Mathematical identities, sampler checks and integration tests |

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pre-commit install
```

GitHub Actions runs tests, linting, formatting and type checks on Linux/CPU.
Core dependencies: PyTorch, NumPy, SciPy, pandas, Matplotlib, Seaborn, PyYAML,
psutil and tqdm. The lockfile pins the environment used by these workflows.

Experimental software and computational study by Tobias Reinhart. The related
theoretical manuscript is joint work; its authorship and publication details
will be linked with the preprint. This repository is distributed under the
[MIT license](LICENSE).
