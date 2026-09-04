"""Pivot the depth sweep: test AP (and final ||J||_2) by num_layers x kernel.

Usage:  python experiments/summarize_depth.py results/depth_func.csv
Reads the per-run *_spectral.csv files in results/depth/ for the final ||J|| too.
"""
import csv
import glob
import os
import sys
from collections import defaultdict


def pivot(rows, value_key, cast=float):
    kernels, layers = set(), set()
    cell = {}
    for r in rows:
        k, L = r["kernel"], int(r["num_layers"])
        kernels.add(k); layers.add(L)
        cell[(L, k)] = cast(r[value_key])
    kernels = sorted(kernels); layers = sorted(layers)
    print(f"    {'layers':>6s} " + "".join(f"{k:>14s}" for k in kernels))
    for L in layers:
        line = f"    {L:>6d} "
        for k in kernels:
            v = cell.get((L, k))
            line += f"{v:>14.4f}" if v is not None else f"{'-':>14s}"
        print(line)


def main(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print("empty summary"); return
    metric = "test_AP" if "test_AP" in rows[0] else "test_MAE"
    print(f"\n{metric} by depth x kernel  ({'higher' if metric=='test_AP' else 'lower'} better):")
    pivot(rows, metric)
    if "best_val_AP" in rows[0]:
        print(f"\nbest_val_AP by depth x kernel:")
        pivot(rows, "best_val_AP")

    # final ||J||_2 from the per-run spectral files, if present. Try a few layouts:
    # <summary_dir>/depth/ (default OUTDIR), <summary_dir>/, or an explicit argv[2].
    base = os.path.dirname(os.path.abspath(path))
    candidates = [sys.argv[2]] if len(sys.argv) > 2 else [os.path.join(base, "depth"), base]
    spec = []
    for d in candidates:
        spec = glob.glob(os.path.join(d, "*_spectral.csv"))
        if spec:
            break
    if spec:
        jrows = []
        for f in spec:
            srows = list(csv.DictReader(open(f)))
            if srows:
                last = srows[-1]  # final probed epoch
                jrows.append({"kernel": last["kernel"], "num_layers": _layers_from_name(f),
                              "maxJ": last["max_layer"]})
        if jrows and all(r["num_layers"] for r in jrows):
            print(f"\nfinal ||J||_2 (max over layers) by depth x kernel:")
            pivot(jrows, "maxJ")


def _layers_from_name(fname):
    base = os.path.basename(fname)
    if base.startswith("L"):
        try:
            return base[1:].split("_", 1)[0]
        except Exception:
            return ""
    return ""


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/depth_func.csv")
