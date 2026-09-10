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
    seen = {}          # (layers,kernel) -> list of (gamma, seed, value) for collision detection
    for r in rows:
        k, L = r["kernel"], int(r["num_layers"])
        kernels.add(k); layers.add(L)
        cell[(L, k)] = cast(r[value_key])
        seen.setdefault((L, k), []).append(
            (r.get("gamma", "?"), r.get("seed", "?"), r[value_key]))
    # a cell is only meaningful if ONE row maps to it; otherwise we'd silently
    # show the last row and hide the rest (they differ in gamma and/or seed).
    clashes = {c: v for c, v in seen.items() if len(v) > 1}
    if clashes:
        print(f"  !! WARNING: {len(clashes)} (layers,kernel) cell(s) have multiple rows "
              f"differing in gamma/seed; only the LAST is shown. Split the file or "
              f"summarize by gamma/seed instead:")
        for (L, k), v in sorted(clashes.items())[:6]:
            detail = ", ".join(f"g={g}/s={s}:{val}" for g, s, val in v)
            print(f"     L{L} {k}: {detail}")
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
    metric = next((m for m in ("test_AP", "test_acc", "test_MAE", "test_logMSE") if m in rows[0]), "test_AP")
    val_key = next((v for v in ("best_val_AP", "best_val_acc", "best_val_MAE", "best_val_logMSE") if v in rows[0]), None)
    higher = metric in ("test_AP", "test_acc")
    print(f"\n{metric} by depth x kernel  ({'higher' if higher else 'lower'} better):")
    pivot(rows, metric)
    if val_key:
        print(f"\n{val_key} by depth x kernel:")
        pivot(rows, val_key)

    # divergence map: did forward-Euler blow up to NaN, and at which epoch?
    if any(r.get("diverged", "") not in ("", "0") for r in rows):
        print("\ndiverged? (D=blew up to NaN / .=trained through) by depth x kernel:")
        kernels = sorted({r["kernel"] for r in rows})
        layers = sorted({int(r["num_layers"]) for r in rows})
        cell = {(int(r["num_layers"]), r["kernel"]):
                ("D@" + str(r.get("diverged_epoch", "?")) if str(r.get("diverged", "0")) == "1" else ".")
                for r in rows}
        print(f"    {'layers':>6s} " + "".join(f"{k:>14s}" for k in kernels))
        for L in layers:
            print(f"    {L:>6d} " + "".join(f"{cell.get((L,k),'-'):>14s}" for k in kernels))

    # final ||J||_2 from the per-run spectral files, if present. Try a few layouts:
    # <summary_dir>/depth/ (default OUTDIR), <summary_dir>/, or an explicit argv[2].
    # only report ||J|| when an explicit spectral dir is passed (argv[2]) -- avoids
    # accidentally globbing an unrelated sweep's spectral files.
    spec = glob.glob(os.path.join(sys.argv[2], "*_spectral.csv")) if len(sys.argv) > 2 else []
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
