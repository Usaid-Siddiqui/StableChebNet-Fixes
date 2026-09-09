#!/usr/bin/env bash
# Ring-transfer depth sweep -- the practical-value experiment on a task whose
# required depth is tunable (= RING/2 with K=2, one hop per layer). Sweeps
# num_layers x {dirichlet, uniform, fejer} at a fixed ring size.
#
# Expected money plot (test_acc, higher better): for L < RING/2 all arms fail
# (can't reach the target); for L >= RING/2, fejer solves while dirichlet/uniform
# diverge -> a solvable depth window unique to fejer.
set -euo pipefail
cd "$(dirname "$0")/.."

export CKPT_DIR="${CKPT_DIR:-$(pwd)/results/ring_checkpoints}"
RING="${RING:-16}"                                  # required depth = RING/2 (=8 by default)
SEED="${SEED:-0}"
EPSILON="${EPSILON:-2.5}"    # step size in the instability regime (calibrated: separates the arms)
# ring + eps + seed all in the filename -> different runs NEVER clobber each other
RESULTS="${RESULTS_CSV:-$(pwd)/results/ring_P${RING}_e${EPSILON}_s${SEED}.csv}"
mkdir -p "$(dirname "$RESULTS")"
# clear only THIS exact (ring,eps,seed) file so a re-run is clean but every other run is preserved
[ -f "$RESULTS" ] && { echo "[note] removing existing $RESULTS for a clean re-run"; rm -f "$RESULTS"; }

EPOCHS="${EPOCHS:-300}"
K="${K:-2}"
HIDDEN="${HIDDEN:-32}"
CLASSES="${CLASSES:-5}"
# required depth = ceil((RING/2)/(K-1)); sweep from below it (all fail: cannot reach) to past it
LAYERS="${LAYERS:-2 4 6 8 10 12 16}"
HOPS=$(( RING / 2 ))
REQ_DEPTH=$(( (HOPS + K - 2) / (K - 1) ))   # ceil(hops/(K-1))
# dirichlet at paper-like small gamma; the fixes at a fixed moderate gamma
# (kernel shape, not gamma magnitude, is what we are comparing)
CONFIGS="${CONFIGS:-dirichlet:0.01 uniform:1.25 fejer:1.25}"

echo "== Ring[P=$RING, K=$K, hops=$HOPS, req_depth=$REQ_DEPTH] sweep: layers={$LAYERS} configs={$CONFIGS} -> $RESULTS =="
for cfg in $CONFIGS; do
  kernel="${cfg%%:*}"; gamma="${cfg##*:}"
  for L in $LAYERS; do
    echo ">>> ring  kernel=$kernel gamma=$gamma layers=$L (req $REQ_DEPTH)"
    python3 LongRange/run_ring.py \
      --ring "$RING" --num_classes "$CLASSES" --num_layers "$L" --K "$K" \
      --epsilon "$EPSILON" --gamma "$gamma" --damping_kernel "$kernel" \
      --hidden "$HIDDEN" --epochs "$EPOCHS" --seed "$SEED" \
      --results_csv "$RESULTS" \
      --run_name "ring${RING}_e${EPSILON}_${kernel}_g${gamma}_L${L}_s${SEED}" \
      || echo "[warn] run failed (non-zero exit): $kernel/L$L -- continuing"
  done
done
echo "== done. Summarize:  python3 experiments/summarize_depth.py $RESULTS =="
