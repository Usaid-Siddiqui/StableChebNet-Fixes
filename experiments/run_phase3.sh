#!/usr/bin/env bash
# Phase 3 (optional) -- repeat the gamma sweep on Peptides-struct (K=8, regression / MAE).
# Lower is better (MAE); the CSV logs test_MAE.
set -euo pipefail
cd "$(dirname "$0")/../Peptides/Stable"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-Fejer_Peptides_struct}"
export CKPT_DIR="${CKPT_DIR:-$(cd ../.. && pwd)/results/checkpoints}"
RESULTS="${RESULTS_CSV:-$(cd ../.. && pwd)/results/phase3_struct.csv}"
mkdir -p "$(dirname "$RESULTS")"

SEED="${SEED:-12}"
EPOCHS="${EPOCHS:-350}"
SUBSET="${MAX_GRAPHS:+--max_graphs $MAX_GRAPHS}"

# uniform's predicted best gamma is 1/eps = 1/0.3 ~ 3.3; grid includes it.
echo "== Phase 3: Peptides-struct gamma sweep -> $RESULTS =="
for kernel in dirichlet uniform fejer; do
  for gamma in 0.0 0.005 1.0 3.3; do
    echo ">>> phase3  kernel=$kernel  gamma=$gamma  seed=$SEED"
    python3 ChebStable_Struc.py \
      --seed "$SEED" \
      --damping_kernel "$kernel" \
      --dissipative_force "$gamma" \
      --epochs "$EPOCHS" \
      --results_csv "$RESULTS" \
      --run_name "p3_${kernel}_g${gamma}_s${SEED}" $SUBSET
  done
done
echo "== Phase 3 done. Results: $RESULTS =="
