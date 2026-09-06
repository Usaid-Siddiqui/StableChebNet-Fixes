"""Summarize a seed-confirmation sweep: divergence rate + AP stats per arm.

Usage:  python experiments/summarize_seeds.py results/depth_seeds.csv
Groups by kernel (assumes one depth). Reports, per arm:
  - how many seeds diverged (NaN blow-up) vs trained through
  - mean +/- std of test AP over the SURVIVORS (diverged runs excluded from the mean)
"""
import csv
import math
import sys
from collections import defaultdict


def main(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print("empty CSV"); return
    metric = "test_AP" if "test_AP" in rows[0] else "test_MAE"

    groups = defaultdict(list)
    for r in rows:
        groups[r["kernel"]].append(r)

    layers = sorted({r.get("num_layers", "?") for r in rows})
    print(f"\nSeed confirmation ({metric}), layers={','.join(layers)}\n")
    print(f"{'kernel':10s} {'n':>3s} {'diverged':>9s} {'survived':>9s} "
          f"{'mean(surv)':>11s} {'std(surv)':>10s}   seeds(diverged marked *)")
    for kernel in sorted(groups):
        rs = groups[kernel]
        surv, div = [], []
        cells = []
        for r in sorted(rs, key=lambda x: int(x.get("seed", 0))):
            diverged = str(r.get("diverged", "0")) == "1"
            val = float(r[metric])
            s = r.get("seed", "?")
            if diverged or math.isnan(val):
                div.append(r); cells.append(f"{s}:div*")
            else:
                surv.append(val); cells.append(f"{s}:{val:.4f}")
        n = len(rs)
        if surv:
            mean = sum(surv) / len(surv)
            std = (sum((v - mean) ** 2 for v in surv) / len(surv)) ** 0.5 if len(surv) > 1 else 0.0
            mstr, sstr = f"{mean:.4f}", f"{std:.4f}"
        else:
            mstr, sstr = "--", "--"
        print(f"{kernel:10s} {n:>3d} {len(div):>9d} {len(surv):>9d} "
              f"{mstr:>11s} {sstr:>10s}   " + "  ".join(cells))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/depth_seeds.csv")
