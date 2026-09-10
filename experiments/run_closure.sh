#!/usr/bin/env bash
# Closure runs for the damping investigation (FINDINGS.md section 5).
#
# PART A -- replicate the matched-gamma Dirichlet control across seeds.
#   We have dirichlet@2.2 L16 for seed 0 only (n=1). fejer@2.2 L16 is ALREADY
#   seed-confirmed (4 seeds in depth_seeds.csv) and uniform@2.2 too (4/4 diverged),
#   so only Dirichlet needs replicating. If it survives across seeds, the claim
#   "adequate gamma rescues Dirichlet; Fejer is unnecessary" is settled.
#
# PART B -- the missing deep gamma=0 control, doing double duty:
#   (a) separates "damping" from "depth" (at gamma=0 there is no damping at all);
#   (b) at gamma=0 the three kernels are BIT-IDENTICAL models, so running all three
#       at one seed measures the L16 run-to-run noise floor for free -- the same
#       trick that gave us 0.0122 at L3 (FINDINGS.md flaw 9).
set -euo pipefail
cd "$(dirname "$0")/../Peptides/Stable"

ROOT="$(cd ../.. && pwd)"
export WANDB_MODE="${WANDB_MODE:-offline}"
export CKPT_DIR="${CKPT_DIR:-$ROOT/results/checkpoints}"
RESULTS="${RESULTS_CSV:-$ROOT/results/peptides_L16_closure.csv}"
mkdir -p "$(dirname "$RESULTS")"

LAYERS="${LAYERS:-16}"
GAMMA="${GAMMA:-2.2}"
EPOCHS="${EPOCHS:-200}"
SEEDS_A="${SEEDS_A:-1 2 3}"        # seed 0 already exists (peptides_L16_fixedg2.2.csv)
SEED_B="${SEED_B:-0}"
SUBSET="${MAX_GRAPHS:+--max_graphs $MAX_GRAPHS}"

run () {   # kernel gamma seed tag
  echo ">>> L$LAYERS kernel=$1 gamma=$2 seed=$3"
  python3 ChebStable_peptide.py \
    --num_layers "$LAYERS" --damping_kernel "$1" --dissipative_force "$2" \
    --seed "$3" --epochs "$EPOCHS" --results_csv "$RESULTS" --run_name "$4" $SUBSET \
    || echo "[warn] run failed: $4 -- continuing"
}

echo "== PART A: replicate dirichlet@$GAMMA at L$LAYERS across seeds {$SEEDS_A} =="
for s in $SEEDS_A; do run dirichlet "$GAMMA" "$s" "clo_L${LAYERS}_dirichlet_g${GAMMA}_s${s}"; done

echo "== PART B: gamma=0 control at L$LAYERS (3 identical models -> L16 noise floor) =="
for k in dirichlet uniform fejer; do run "$k" 0.0 "$SEED_B" "clo_L${LAYERS}_g0_${k}_s${SEED_B}"; done

echo
echo "== building the final combined table =="
python3 - "$ROOT" "$RESULTS" <<'PY'
import csv, os, sys
root, new = sys.argv[1], sys.argv[2]
srcs = [new,
        os.path.join(root, "results", "peptides_L16_fixedg2.2.csv"),
        os.path.join(root, "results", "depth_seeds.csv")]
rows = []
for f in srcs:
    if os.path.exists(f):
        rows += list(csv.DictReader(open(f)))
# keep only depth-16 rows at the matched gamma or the gamma=0 control
keep, seen = [], set()
for r in rows:
    try:
        if int(r["num_layers"]) != 16:
            continue
        g = float(r["gamma"])
    except (KeyError, ValueError):
        continue
    if g not in (2.2, 0.0):
        continue
    key = (r["kernel"], r["gamma"], r["seed"])
    if key in seen:      # de-duplicate across source files
        continue
    seen.add(key); keep.append(r)
out = os.path.join(root, "results", "peptides_L16_matched_final.csv")
if keep:
    fields = []
    for r in keep:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader(); w.writerows(keep)
    print(f"  {len(keep)} rows -> {out}")
    print(f"  summarize:  python3 experiments/summarize_seeds.py {out}")
else:
    print("  no matching rows found")
PY
