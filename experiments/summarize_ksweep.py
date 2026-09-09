"""Summarize the K-sweep: does the dirichlet-vs-fejer gap open as the Dirichlet
damping kernel becomes more sign-changing?

For each K it prints, side by side:
  * the ANALYTIC severity of the sign-change on that ring's spectrum
      frac(s<0) and min s of the Dirichlet profile  s = sum_{k<K} T_k(lambda~)
  * the OBSERVED best test_acc per kernel (max over depth), the depth at which
    each kernel first solves (acc >= 0.95), and the gap  fejer - dirichlet.

Prediction: frac(s<0) = 0 and gap ~ 0 at K=2; both grow with K.

  python experiments/summarize_ksweep.py "results/ksweep_K*_g1.25_e2.0_s0.csv"
"""
import csv
import glob
import sys
from collections import defaultdict

import numpy as np


def cheb_T(K, x):
    """T_0..T_{K-1} at x (array) -> (K, n)."""
    x = np.asarray(x, dtype=float)
    T = np.empty((K, x.size))
    T[0] = 1.0
    if K > 1:
        T[1] = x
    for k in range(2, K):
        T[k] = 2 * x * T[k - 1] - T[k - 2]
    return T


def dirichlet_severity(K, P):
    """Sign-change severity of the Dirichlet profile on a P-node ring.
    Ring sym-Laplacian eigenvalues are 1 - cos(2*pi*j/P), so lambda~ = -cos(2*pi*j/P)."""
    lam = -np.cos(2 * np.pi * np.arange(P) / P)
    s = cheb_T(K, lam).sum(0)
    return float((s < 0).mean()), float(s.min())


def main(patterns):
    files = []
    for p in patterns:
        files += sorted(glob.glob(p)) or ([p] if p.endswith(".csv") else [])
    rows = []
    for f in files:
        if f.endswith("_spectral.csv"):
            continue
        rows += list(csv.DictReader(open(f)))
    if not rows:
        print("no rows found"); return

    # (K, kernel) -> {layers: acc}
    acc = defaultdict(dict)
    ring = {}
    for r in rows:
        K = int(r["K"]); k = r["kernel"]; L = int(r["num_layers"])
        acc[(K, k)][L] = float(r["test_acc"])
        ring[K] = int(r["ring"])
    Ks = sorted({K for K, _ in acc})
    kernels = ["dirichlet", "uniform", "fejer"]

    def best(K, k):
        d = acc.get((K, k), {})
        return max(d.values()) if d else float("nan")

    def solve_depth(K, k, thr=0.95):
        d = acc.get((K, k), {})
        ok = [L for L, a in sorted(d.items()) if a >= thr]
        return str(ok[0]) if ok else "-"

    print("\nK-sweep: sign-change severity (analytic) vs observed kernel gap  [matched gamma]\n")
    print(f"{'K':>3s} {'ring':>5s} {'frac(s<0)':>10s} {'min s':>7s} | "
          f"{'dir best':>9s} {'uni best':>9s} {'fej best':>9s} | "
          f"{'solve@ dir':>10s} {'uni':>5s} {'fej':>5s} | {'gap fej-dir':>11s}")
    for K in Ks:
        P = ring[K]
        frac, smin = dirichlet_severity(K, P)
        bd, bu, bf = best(K, "dirichlet"), best(K, "uniform"), best(K, "fejer")
        print(f"{K:>3d} {P:>5d} {frac:>10.3f} {smin:>7.2f} | "
              f"{bd:>9.3f} {bu:>9.3f} {bf:>9.3f} | "
              f"{solve_depth(K,'dirichlet'):>10s} {solve_depth(K,'uniform'):>5s} {solve_depth(K,'fejer'):>5s} | "
              f"{(bf - bd):>+11.3f}")
    print("\n(solve@ = shallowest depth reaching acc >= 0.95; '-' = never solved in the sweep)")
    print("(prediction: frac(s<0)=0 and gap~0 at K=2; both should grow with K)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: summarize_ksweep.py <csv-or-glob> [more ...]"); sys.exit(1)
    main(sys.argv[1:])
