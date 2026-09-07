#!/usr/bin/env bash
# GraphProp depth sweep -- the practical-value experiment on a task that PROVABLY
# needs depth (SSSP/ecc/diam require ~diameter-hop propagation). Sweeps num_layers
# x {dirichlet, uniform, fejer} on one task, mirroring the paper's own depth study
# ({1,5,10,20} layers) so results are comparable to the Peptides depth sweep.
#
# Metric is log10(MSE), LOWER is better. The predicted story: as depth grows,
# dirichlet/uniform diverge (can't train the depth the task needs) while fejer
# stays trainable AND improves (more reach on a task that rewards reach).
set -euo pipefail
cd "$(dirname "$0")/../GraphProp/graph_prop_pred"

# ensure the dataset is extracted (idempotent)
[ -f data/train_dist_25-35_data.pt ] || tar xzf data.tar.gz

export CKPT_DIR="${CKPT_DIR:-$(cd ../.. && pwd)/results/gp_checkpoints}"
TASK="${TASK:-dist}"                              # dist (SSSP) | ecc | diam
RESULTS="${RESULTS_CSV:-$(cd ../.. && pwd)/results/graphprop_${TASK}.csv}"
mkdir -p "$(dirname "$RESULTS")"
# start clean so re-runs don't accumulate stale rows (columns are fixed here)
[ -f "$RESULTS" ] && { echo "[note] removing existing $RESULTS for a clean sweep"; rm -f "$RESULTS"; }

SEED="${SEED:-41}"
EPOCHS="${EPOCHS:-1500}"
PATIENCE="${PATIENCE:-200}"
K="${K:-4}"
EPSILON="${EPSILON:-0.8}"                         # 1/eps = 1.25 -> uniform/fejer gamma
HIDDEN="${HIDDEN:-40}"
LR="${LR:-0.003}"
LAYERS="${LAYERS:-1 5 10 20}"
SUBSET="${MAX_GRAPHS:+--max_graphs $MAX_GRAPHS}"
# dirichlet at its paper-tuned ~0 gamma; the fixes at gamma ~ 1/eps
CONFIGS="${CONFIGS:-dirichlet:0.01 uniform:1.25 fejer:1.25}"

echo "== GraphProp[$TASK] depth sweep: layers={$LAYERS} configs={$CONFIGS} -> $RESULTS =="
for cfg in $CONFIGS; do
  kernel="${cfg%%:*}"; gamma="${cfg##*:}"
  for L in $LAYERS; do
    echo ">>> graphprop  task=$TASK kernel=$kernel gamma=$gamma layers=$L seed=$SEED"
    python3 run_single.py \
      --task "$TASK" --damping_kernel "$kernel" --dissipative_force "$gamma" \
      --num_layers "$L" --K "$K" --epsilon "$EPSILON" --hidden "$HIDDEN" --lr "$LR" \
      --epochs "$EPOCHS" --patience "$PATIENCE" --seed "$SEED" \
      --results_csv "$RESULTS" \
      --run_name "gp_${TASK}_${kernel}_g${gamma}_L${L}_s${SEED}" $SUBSET \
      || echo "[warn] run failed (non-zero exit): $TASK/$kernel/L$L -- continuing"
  done
done
echo "== done. Summarize:  python3 experiments/summarize_depth.py $RESULTS =="
