# INSIDE

Code for **INSIDE: Intra-Size Coalition Design for Shapley Value Estimation**,
including INSIDE-Coalition (`inside_greedy`), INSIDE-Orbit (`inside_orbit`),
and baseline methods.

## Code structure

- `frame_ofa/`: Shapley estimators, coalition designs, and benchmark games.
- `experiments/`: experiment runners, plotting scripts, and result validators.
- `tests/`: unit and regression tests.
- `results/`: experiment reports, reference data, and PDF figures.
- `docs/`: experiment protocols and detailed reproduction commands.
- `third_party/`: reference implementations and fixtures; see [UPSTREAM.md](UPSTREAM.md).

## Installation

Requires Python 3.10 or later. Run the following commands in a Bash shell:

```bash
git clone https://github.com/xiaokaimao/inside.git
cd inside
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[experiments,dev]"
```

## Run

Run all commands from the repository root with the environment activated.

```bash
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLBACKEND=Agg

# Small synthetic benchmark
python -m experiments.benchmark_synthetic --players 8 --samples 40 --repeats 100

# Interaction-game baseline comparison (run or resume)
python -m experiments.run_interaction_baselines --jobs 4 --design-jobs 4
python -m experiments.plot_interaction_baselines \
  --source results/json/interaction_all_baselines_n500.json

# Large-n interaction scalability experiment (run or resume)
bash experiments/run_interaction_scalability.sh

# Parameter sensitivity experiments
python -m experiments.run_airport_sensitivity
python -m experiments.plot_airport_sensitivity
python -m experiments.run_svm_sensitivity
python -m experiments.plot_sensitivity_separate --datasets wine cancer

# Introductory schematic (PDF only)
python -m experiments.plot_inside_intro

# Tests
python -m pytest -q
```

Generate the main RMSE figure from the included benchmark reports:

```bash
python -m experiments.plot_inside_four_panel \
  --voting results/json/voting_inside_with_shapdoe_orthogonal_3repeats.json \
  --airport results/json/airport_inside_with_shapdoe_orthogonal_3repeats.json \
  --wine results/json/wine_inside_with_shapdoe_orthogonal_3repeats.json \
  --cancer results/json/cancer_inside_with_shapdoe_orthogonal_3repeats.json \
  --metric rmse \
  --output results/pdf/inside_comparison_four_panel_ls_orthogonal_rmse.pdf
```

Use `python -m experiments.<module> --help` for command-line options.
