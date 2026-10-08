#!/usr/bin/env bash
# Resume complete three-seed cells automatically; run from any working directory.
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/frame-mpl
PYTHON_BIN="${PYTHON_BIN:-python}"
for n in 1000 2000; do
    "$PYTHON_BIN" -u -m experiments.run_interaction_baselines \
        --players "$n" --degrees 3 --budgets "$n" "$((2*n))" "$((4*n))" \
        --repeats 3 --methods inside_greedy inside_orbit ofa cc s_diff \
        --jobs 3 --design-jobs 48 --baseline-jobs 16 \
        --output "results/json/interaction_scalability_n${n}.json"
done
# Optional external figure-audit helpers; ordinary reproduction needs no skill install.
if [[ -n "${FIGURE_QA:-}" ]]; then
    export PYTHONPATH="$FIGURE_QA:$(pwd):${PYTHONPATH:-}"
    "$PYTHON_BIN" -m experiments.plot_interaction_scalability --audit-alignment
    "$PYTHON_BIN" "$FIGURE_QA/audit_pdf_text.py" results/pdf/interaction_scalability_rmse.pdf \
        --json > results/json/interaction_scalability_rmse.text-audit.json
    "$PYTHON_BIN" "$FIGURE_QA/audit_figure_collisions.py" results/pdf/interaction_scalability_rmse.pdf \
        --json-out results/json/interaction_scalability_rmse.collision-audit.json
else
    "$PYTHON_BIN" -m experiments.plot_interaction_scalability
fi
