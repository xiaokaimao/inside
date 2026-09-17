"""Execute pinned author functions and write reproducible comparison fixtures.

python -m experiments.verify_orthogonal_upstream --source /tmp/algorithms.py
Use --download to retrieve the pinned public source into a temporary directory.
Only selected function ASTs run; unrelated imports, Numba decorators, datasets
and experiment entry points are not executed. Function bodies remain unchanged.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request

import numpy as np

from frame_ofa import orthogonal_permutations

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "third_party/shap_sampling/provenance.json"
FUNCTIONS = {"_sample_sphere", "get_orthogonal_vectors", "_gram_schmidt_permutations",
             "_orthogonal_permutations", "mask_dataset", "_accumulate_samples",
             "estimate_shap_given_permutations"}


class _BlockRandom:
    """Feed identical Gaussian draws to upstream GS and local QR implementations."""

    def __init__(self, n: int, seed: int):
        self.n, self.seed, self.index = n, seed, 0

    def randn(self, ndim: int) -> np.ndarray:
        assert ndim == self.n
        block, row = divmod(self.index, self.n - 1)
        if row == 0:
            self.rng = np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=(block,)))
        self.index += 1
        return self.rng.standard_normal(ndim)


class _NumpyProxy:
    def __init__(self, n: int, seed: int):
        self.random = _BlockRandom(n, seed)

    def __getattr__(self, name):
        return getattr(np, name)


def generate(source: Path, output: Path) -> dict:
    provenance = json.loads(PROVENANCE.read_text())
    raw = source.read_bytes()
    expected = provenance["files"]["algorithms.py"]["sha256"]
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("upstream source SHA-256 mismatch")
    selected = [node for node in ast.parse(raw).body
                if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
    if {node.name for node in selected} != FUNCTIONS:
        raise ValueError("upstream function set changed")
    for node in selected:
        node.decorator_list = []
    compiled = compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec")
    cases = []
    # Single pair, partial basis, full basis, multiple bases, partial final basis.
    for n, count, seed in [(2, 2, 7), (4, 2, 19), (4, 6, 29), (4, 16, 31),
                           (8, 20, 103), (17, 64, 77), (51, 122, 177)]:
        namespace = {"np": _NumpyProxy(n, seed)}
        exec(compiled, namespace)
        permutations = namespace["_orthogonal_permutations"](count, n)
        np.testing.assert_array_equal(permutations, orthogonal_permutations(n, count, seed))
        # A general, shifted, nonlinear coalition game, represented by 0/1 features.
        weights = np.linspace(-.7, 1.2, n)

        def predict(x):
            return 2.3 + np.sin(x @ weights) + .17 * (x.sum(axis=1) ** 2)

        values = namespace["estimate_shap_given_permutations"](
            np.zeros((1, n)), np.ones((1, n)), predict, permutations)[0]
        cases.append({"num_players": n, "num_permutations": count, "seed": seed,
                      "permutations": permutations.tolist(), "values": values.tolist()})
    result = {"upstream_commit": provenance["commit"], "source_sha256": expected,
              "numpy_version": np.__version__, "rng": "PCG64 with SeedSequence(seed, spawn_key=(block,))",
              "verification": "Unchanged upstream function bodies, supplied identical Gaussian draws; Numba decorators removed",
              "game": "2.3 + sin(mask @ linspace(-0.7, 1.2, n)) + 0.17 * sum(mask)**2",
              "cases": cases}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return {"status": "passed", "cases": len(cases), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", type=Path)
    group.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "third_party/shap_sampling/fixtures.json")
    args = parser.parse_args()
    if args.download:
        with tempfile.TemporaryDirectory(prefix="orthogonal-upstream-") as temporary:
            source = Path(temporary) / "algorithms.py"
            url = json.loads(PROVENANCE.read_text())["files"]["algorithms.py"]["url"]
            with urllib.request.urlopen(url, timeout=30) as response:
                source.write_bytes(response.read())
            print(generate(source, args.output))
    else:
        print(generate(args.source, args.output))


if __name__ == "__main__":
    main()
