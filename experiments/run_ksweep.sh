#!/usr/bin/env bash
# K-sweep -- the most direct test of the diagnosis.
#
# The Dirichlet damping profile s(lambda~) = sum_{k<K} T_k is sign-changing only
# for K >= 3, and the negative fraction grows with K (-> 1/4). Fejer is nonnegative
# for every K. So the predicted dirichlet-vs-fejer gap (at MATCHED gamma) should be
# ~zero at K=2 and open monotonically as K grows.
#
# We hold the REQUIRED DEPTH fixed (~16 layers) so depth/instability is not a
# confound: one layer reaches K-1 hops, the target is P/2 hops away, so
#     P/2 = 16 * (K-1)   =>   P = 32 * (K-1)
# and sweep depth around 16 for each K. All three kernels run at the SAME gamma.
set -euo pipefail
cd "$(dirname "$0")/.."

export CKPT_DIR="${CKPT_DIR:-$(pwd)/results/ring_checkpoints}"
KS="${KS:-2 3 4 6 10}"
REQ="${REQ:-16}"                                  # target required depth (layers)
LAYERS="${LAYERS:-8 12 16 20 24}"
EPSILON="${EPSILON:-2.0}"
GAMMA="${GAMMA:-1.25}"                            # matched across all three kernels
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-300}"
CONFIGS="dirichlet:${GAMMA} uniform:${GAMMA} fejer:${GAMMA}"

echo "== K-sweep: K={$KS}  req_depth=$REQ  eps=$EPSILON  gamma=$GAMMA (matched)  seed=$SEED =="
for K in $KS; do
  P=$(( 2 * REQ * (K - 1) ))                      # ring size so that P/2 = REQ*(K-1)
  OUT="$(pwd)/results/ksweep_K${K}_P${P}_g${GAMMA}_e${EPSILON}_s${SEED}.csv"
  echo ">>> K=$K  ring P=$P  (req depth $REQ)  -> $OUT"
  K="$K" RING="$P" LAYERS="$LAYERS" EPSILON="$EPSILON" SEED="$SEED" EPOCHS="$EPOCHS" \
  CONFIGS="$CONFIGS" RESULTS_CSV="$OUT" bash experiments/run_ring.sh \
    || echo "[warn] K=$K sweep failed -- continuing"
done
echo "== done. Summarize:  python3 experiments/summarize_ksweep.py 'results/ksweep_K*_g${GAMMA}_e${EPSILON}_s${SEED}.csv' =="
