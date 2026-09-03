#!/usr/bin/env bash
# Phase 2 -- seed variance at the best gamma (8 runs).
# Baseline dirichlet @ gamma=0.1 (paper default) vs the WINNING arm from Phase 1,
# each over seeds {0,1,2,3}. Report mean +/- std of test AP; the std matters as much
# as the mean.
#   Set WIN_KERNEL / WIN_GAMMA to Phase 1's winner before running.
#   Defaults assume uniform @ 2.2 (the Amendment-1 prediction) wins.
set -euo pipefail
cd "$(dirname "$0")/../Peptides/Stable"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-Fejer_Peptides_func}"
RESULTS="${RESULTS_CSV:-$(cd ../.. && pwd)/results/phase2_func.csv}"
mkdir -p "$(dirname "$RESULTS")"

EPOCHS="${EPOCHS:-200}"
SUBSET="${MAX_GRAPHS:+--max_graphs $MAX_GRAPHS}"
WIN_KERNEL="${WIN_KERNEL:-uniform}"   # <-- Phase 1's winning arm
WIN_GAMMA="${WIN_GAMMA:-2.2}"         # <-- its best gamma
SEEDS="${SEEDS:-0 1 2 3}"

echo "== Phase 2: seed variance ($WIN_KERNEL gamma=$WIN_GAMMA vs dirichlet 0.1) -> $RESULTS =="
for seed in $SEEDS; do
  echo ">>> phase2  dirichlet g=0.1  seed=$seed"
  python3 ChebStable_peptide.py --seed "$seed" --damping_kernel dirichlet \
    --dissipative_force 0.1 --epochs "$EPOCHS" --results_csv "$RESULTS" \
    --run_name "p2_dirichlet_g0.1_s${seed}" $SUBSET
done
for seed in $SEEDS; do
  echo ">>> phase2  $WIN_KERNEL g=$WIN_GAMMA  seed=$seed"
  python3 ChebStable_peptide.py --seed "$seed" --damping_kernel "$WIN_KERNEL" \
    --dissipative_force "$WIN_GAMMA" --epochs "$EPOCHS" --results_csv "$RESULTS" \
    --run_name "p2_${WIN_KERNEL}_g${WIN_GAMMA}_s${seed}" $SUBSET
done
echo "== Phase 2 done. Summarize with: python3 experiments/summarize.py $RESULTS =="
