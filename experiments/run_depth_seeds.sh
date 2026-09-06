#!/usr/bin/env bash
# Seed confirmation at a fixed depth -- is the "dirichlet+uniform diverge, fejer
# survives" split at L=16 robust across seeds, or a single-trajectory fluke?
# Runs 3 arms x N seeds at NUM_LAYERS and reports the divergence rate + mean/std
# of test AP (survivors) per arm.
set -euo pipefail
cd "$(dirname "$0")/../Peptides/Stable"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-Fejer_Peptides_depthseeds}"
export CKPT_DIR="${CKPT_DIR:-$(cd ../.. && pwd)/results/checkpoints}"
OUTDIR="${OUTDIR:-$(cd ../.. && pwd)/results/depth_seeds}"
SUMMARY="${SUMMARY:-$(cd ../.. && pwd)/results/depth_seeds.csv}"
mkdir -p "$OUTDIR"

NUM_LAYERS="${NUM_LAYERS:-16}"
EPOCHS="${EPOCHS:-200}"
SUBSET="${MAX_GRAPHS:+--max_graphs $MAX_GRAPHS}"
SEEDS="${SEEDS:-0 1 2 3}"
CONFIGS="${CONFIGS:-dirichlet:0.1 uniform:2.2 fejer:2.2}"

echo "== Seed confirmation at L=$NUM_LAYERS: configs={$CONFIGS} seeds={$SEEDS} -> $SUMMARY =="
for cfg in $CONFIGS; do
  kernel="${cfg%%:*}"; gamma="${cfg##*:}"
  for s in $SEEDS; do
    tag="L${NUM_LAYERS}_${kernel}_g${gamma}_s${s}"
    echo ">>> seed  kernel=$kernel gamma=$gamma seed=$s layers=$NUM_LAYERS"
    python3 ChebStable_peptide.py \
      --seed "$s" \
      --damping_kernel "$kernel" \
      --dissipative_force "$gamma" \
      --num_layers "$NUM_LAYERS" \
      --epochs "$EPOCHS" \
      --results_csv "$OUTDIR/${tag}.csv" \
      --run_name "ds_${tag}" $SUBSET \
      || echo "[warn] run failed (non-zero exit): $tag -- continuing"
  done
done

# ---- aggregate result rows (union of columns) ----
python3 - "$OUTDIR" "$SUMMARY" <<'PY'
import csv, glob, os, sys
outdir, summary = sys.argv[1], sys.argv[2]
files = sorted(f for f in glob.glob(os.path.join(outdir, "*.csv")) if not f.endswith("_spectral.csv"))
rows = []
for f in files:
    rows += list(csv.DictReader(open(f)))
if rows:
    fieldnames = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(summary, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        w.writeheader(); w.writerows(rows)
    print(f"\n== Seed sweep done. {len(rows)} runs -> {summary} ==")
    print("Summarize:  python3 experiments/summarize_seeds.py " + summary)
else:
    print("no result rows found")
PY
