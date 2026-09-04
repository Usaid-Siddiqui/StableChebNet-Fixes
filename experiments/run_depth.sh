#!/usr/bin/env bash
# Depth sweep -- does the per-layer growth compound with depth?
# Runs the three key Phase-1 configs at increasing num_layers on Peptides-func.
#   dirichlet @ 0.1  (paper default)
#   uniform   @ 2.2  (the fix; gamma ~ 1/eps)
#   fejer     @ 2.2
# Prediction: if the fix matters, dirichlet's test AP / ||J|| should degrade with
# depth while uniform stays controlled. If all arms track together at depth too,
# "gamma is a dead hyperparameter" is airtight and the story is the integrator.
#
# Each (config, depth) run writes its own results + _spectral CSV under
# results/depth/ (per-layer spectral columns vary with depth, so files can't be
# shared); the per-run result rows are concatenated into results/depth_func.csv.
set -euo pipefail
cd "$(dirname "$0")/../Peptides/Stable"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-Fejer_Peptides_depth}"
export CKPT_DIR="${CKPT_DIR:-$(cd ../.. && pwd)/results/checkpoints}"
OUTDIR="${OUTDIR:-$(cd ../.. && pwd)/results/depth}"
SUMMARY="${SUMMARY:-$(cd ../.. && pwd)/results/depth_func.csv}"
mkdir -p "$OUTDIR"

SEED="${SEED:-99}"
EPOCHS="${EPOCHS:-200}"
SUBSET="${MAX_GRAPHS:+--max_graphs $MAX_GRAPHS}"
LAYERS="${LAYERS:-3 6 10 16}"
# configs as "kernel:gamma"
CONFIGS="${CONFIGS:-dirichlet:0.1 uniform:2.2 fejer:2.2}"

echo "== Depth sweep: layers={$LAYERS} configs={$CONFIGS} -> $SUMMARY =="
for cfg in $CONFIGS; do
  kernel="${cfg%%:*}"; gamma="${cfg##*:}"
  for L in $LAYERS; do
    tag="L${L}_${kernel}_g${gamma}_s${SEED}"
    echo ">>> depth  kernel=$kernel gamma=$gamma layers=$L"
    python3 ChebStable_peptide.py \
      --seed "$SEED" \
      --damping_kernel "$kernel" \
      --dissipative_force "$gamma" \
      --num_layers "$L" \
      --epochs "$EPOCHS" \
      --results_csv "$OUTDIR/${tag}.csv" \
      --run_name "depth_${tag}" $SUBSET
  done
done

# ---- aggregate the per-run result rows into one summary (fixed columns) ----
python3 - "$OUTDIR" "$SUMMARY" <<'PY'
import csv, glob, os, sys
outdir, summary = sys.argv[1], sys.argv[2]
files = sorted(f for f in glob.glob(os.path.join(outdir, "*.csv")) if not f.endswith("_spectral.csv"))
rows = []
for f in files:
    rows += list(csv.DictReader(open(f)))
if rows:
    with open(summary, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n== Depth sweep done. {len(rows)} runs -> {summary} ==")
    print("Summarize by depth:  python3 experiments/summarize_depth.py " + summary)
else:
    print("no result rows found")
PY
