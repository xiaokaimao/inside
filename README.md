# INSIDE

This repository provides the implementation of the SIGMOD 2027 paper:

> **INSIDE: Intra-Size Coalition Design for Shapley Value Estimation**

INSIDE accelerates Shapley value estimation by designing which coalitions are
sampled within each coalition-size stratum. The implementation provides two
paper-facing variants: `inside_greedy` and `inside_orbit`.

## Installation

INSIDE requires Python 3.10 or later.

```bash
git clone https://github.com/xiaokaimao/inside.git
cd inside
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Quick start

```python
import numpy as np

from frame_ofa import FrameOFAEstimator


weights = np.array([1.0, -0.5, 2.0, 0.3])


def utility(coalition):
    return float(weights @ coalition)


result = FrameOFAEstimator(
    num_players=len(weights),
    num_samples=40,
    mode="inside_greedy",  # Use "inside_orbit" for the orbit variant.
    seed=0,
).estimate(utility)

print(result.values)
print(result.utility_evaluations)
```

## Run the synthetic experiment

```bash
python -m experiments.benchmark_synthetic \
  --players 8 \
  --samples 40 \
  --repeats 100
```

## Run the tests

```bash
python -m unittest discover -s tests -v
```
