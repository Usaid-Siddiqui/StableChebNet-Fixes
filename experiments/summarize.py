"""Summarize a results CSV: mean +/- std of the test metric per (kernel, gamma).

Usage:  python experiments/summarize.py results/phase2_func.csv
Works for both func (test_AP, higher better) and struct (test_MAE, lower better).
"""
import csv
import sys
from collections import defaultdict


def main(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print("empty CSV"); return
    metric = "test_AP" if "test_AP" in rows[0] else "test_MAE"
    better = "higher" if metric == "test_AP" else "lower"

    groups = defaultdict(list)
    for r in rows:
        groups[(r["kernel"], r["gamma"])].append(float(r[metric]))

    print(f"{path}   metric={metric} ({better} is better)\n")
    print(f"{'kernel':10s} {'gamma':>7s} {'n':>3s} {'mean':>10s} {'std':>10s}  seeds")
    for (kernel, gamma), vals in sorted(groups.items()):
        n = len(vals)
        mean = sum(vals) / n
        std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5 if n > 1 else 0.0
        print(f"{kernel:10s} {gamma:>7s} {n:>3d} {mean:>10.4f} {std:>10.4f}  "
              + " ".join(f"{v:.4f}" for v in vals))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/phase2_func.csv")
