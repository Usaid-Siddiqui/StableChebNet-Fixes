#!/usr/bin/env bash
# Phase 1 -- the decisive gamma sweep on Peptides-func (12 runs; Amendment 1).
# Predictions (seed 99):
#   dirichlet: gamma* ~ 0    (inert; matches the paper's own grid search)
#   uniform  : gamma* ~ 1/eps = 1/0.45 ~ 2.2  (largest stability gain -- the best fix)
#   fejer    : gamma* ~ 3-4  (usable, slightly worse than uniform)
# The grid includes 2.2 so the uniform arm is not artificially handicapped.
set -euo pipefail
cd "$(dirname "$0")/../Peptides/Stable"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-Fejer_Peptides_func}"
export CKPT_DIR="${CKPT_DIR:-$(cd ../.. && pwd)/results/checkpoints}"
RESULTS="${RESULTS_CSV:-$(cd ../.. && pwd)/results/phase1_func.csv}"
mkdir -p "$(dirname "$RESULTS")"

SEED=99
EPOCHS="${EPOCHS:-200}"
SUBSET="${MAX_GRAPHS:+--max_graphs $MAX_GRAPHS}"

echo "== Phase 1: Peptides-func gamma sweep -> $RESULTS =="
for kernel in dirichlet uniform fejer; do
  for gamma in 0.0 0.1 1.0 2.2; do
    echo ">>> phase1  kernel=$kernel  gamma=$gamma  seed=$SEED"
    python3 ChebStable_peptide.py \
      --seed "$SEED" \
      --damping_kernel "$kernel" \
      --dissipative_force "$gamma" \
      --epochs "$EPOCHS" \
      --results_csv "$RESULTS" \
      --run_name "p1_${kernel}_g${gamma}_s${SEED}" $SUBSET
  done
done
echo "== Phase 1 done. Results: $RESULTS  (+ *_spectral.csv) =="
