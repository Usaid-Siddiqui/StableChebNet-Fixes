"""
Diagnostic A -- per-frequency gain profile of a TRAINED Stable-ChebNet layer.

Answers: at which graph frequencies does each damping kernel amplify vs contract?
Low lambda = smooth/global = long-range modes; high lambda (->2) = local oscillation.
The paper's non-dissipative ideal is gain ~ 1 everywhere.

For each eigenmode (u_i, lambda_i) of L~, the layer decouples (verified in
tools/check_damping.py to 1e-14) into the d x d block
    G_i = c*I + eps * sum_k T_k(lambda_i) * W_k      (||G_i|| = per-mode gain)
where W_k = lins[k].weight is the TRAINED, materialized per-order weight (which
already carries the -g_k*I damping for dirichlet/fejer), and c = 1 for
dirichlet/fejer, c = 1 - eps*gamma for uniform (gamma applied once to the signal).

Loads a checkpoint (rebuilt from its filename: ds_L{L}_{kernel}_g{gamma}_s{seed}.pth
or depth_L{L}_..., or pass the fields explicitly), evaluates ||G_i|| over the pooled
spectrum of a few real Peptides-func graphs for a chosen layer, and prints gain
binned by lambda. Point it at several kernels' checkpoints to compare.

Usage:
  python tools/diagnostic_a.py results/checkpoints/depth_L16_fejer_g2.2_s99.pth \
                               results/checkpoints/depth_L16_uniform_g2.2_s99.pth \
                               results/checkpoints/depth_L16_dirichlet_g0.1_s99.pth \
      --layer 0 --n_graphs 20 --out results/diagnostic_a.csv
"""
import argparse
import csv
import os
import re
import sys

import numpy as np
import torch

STABLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Peptides", "Stable"))
ORIG_CWD = os.getcwd()
sys.path.insert(0, STABLE_DIR)
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(STABLE_DIR)  # model_euler.py opens ./config_StableCheb.json at import time
from model_euler import EulerModel  # noqa: E402
from check_damping import normalized_laplacian_tilde, cheb_T  # noqa: E402

torch.set_default_dtype(torch.float64)


def _resolve(p):
    return p if os.path.isabs(p) else os.path.join(ORIG_CWD, p)


def parse_ckpt_name(path, config):
    """Infer (kernel, gamma, num_layers, seed) from a run-name-style filename."""
    base = os.path.basename(path)
    m = re.search(r"_L(\d+)_(dirichlet|uniform|fejer)_g([0-9.]+)_s(\d+)", base)
    if not m:
        raise ValueError(f"cannot parse kernel/layers/gamma/seed from '{base}'; "
                         "rename or pass --kernel/--gamma/--num_layers explicitly")
    L, kernel, gamma, seed = int(m.group(1)), m.group(2), float(m.group(3)), int(m.group(4))
    return kernel, gamma, L, seed


def load_model(path, config, kernel, gamma, num_layers):
    hidden = config["hidden"]; K = config["K"]; mlp = config["mlp_layers"]
    eps = config["step_size"]; num_classes = 10  # Peptides-func
    m = EulerModel(hidden, K, num_layers, mlp, num_classes, eps, gamma, damping_kernel=kernel)
    m.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    m.double().eval()
    return m, hidden, K, eps


def layer_gains(conv, kernel, gamma, eps, K, hidden, lam):
    """Per-mode gain ||G_i|| for every lambda in `lam`, for one conv layer."""
    # trained, materialized per-order weight blocks W_k (d x d)
    Wk = [conv.lins[k].weight.detach().double() for k in range(K)]
    I = torch.eye(hidden, dtype=torch.float64)
    c = (1.0 - eps * gamma) if kernel == "uniform" else 1.0
    Tv = cheb_T(K, lam)  # (K, len(lam))
    gains = np.empty(len(lam))
    for i in range(len(lam)):
        B = sum(float(Tv[k, i]) * Wk[k] for k in range(K))
        G = c * I + eps * B
        gains[i] = float(torch.linalg.svdvals(G).max())
    return gains


def peptides_eigs(n_graphs):
    from torch_geometric.datasets import LRGBDataset
    from torch_geometric.utils import to_dense_adj
    root = os.path.join(os.path.dirname(__file__), "..", "Peptides", "Stable")
    if not os.path.isdir(os.path.join(root, "peptides-func")):
        root = os.path.join(os.path.dirname(__file__), "..", "data", "LRGB")
    ds = LRGBDataset(root=root, name="Peptides-func", split="train")
    lams = []
    for i in range(min(n_graphs, len(ds))):
        d = ds[i]
        A = to_dense_adj(d.edge_index, max_num_nodes=d.num_nodes)[0].double()
        lams.append(torch.linalg.eigvalsh(normalized_laplacian_tilde(A)).numpy())
    return np.concatenate(lams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", help="checkpoint .pth files (one per kernel to compare)")
    ap.add_argument("--layer", type=int, default=0, help="which conv layer to probe (default 0)")
    ap.add_argument("--n_graphs", type=int, default=20)
    ap.add_argument("--nbins", type=int, default=10)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    import json
    config = json.load(open(os.path.join(STABLE_DIR, "config_StableCheb.json")))
    args.ckpts = [_resolve(p) for p in args.ckpts]
    if args.out:
        args.out = _resolve(args.out)

    lam = peptides_eigs(args.n_graphs)          # lambda-tilde in [-1, 1]
    lam_graph = lam + 1.0                        # graph frequency in [0, 2] (0=global/long-range)
    bins = np.linspace(0.0, 2.0, args.nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    results = {}  # kernel -> binned mean gain
    rows = []
    for path in args.ckpts:
        kernel, gamma, L, seed = parse_ckpt_name(path, config)
        model, hidden, K, eps = load_model(path, config, kernel, gamma, L)
        layer = min(args.layer, len(model.convs) - 1)
        gains = layer_gains(model.convs[layer], kernel, gamma, eps, K, hidden, lam)
        idx = np.clip(np.digitize(lam_graph, bins) - 1, 0, args.nbins - 1)
        binned = np.array([gains[idx == b].mean() if (idx == b).any() else np.nan
                           for b in range(args.nbins)])
        results[f"{kernel}@{gamma}"] = binned
        for b in range(args.nbins):
            rows.append({"kernel": kernel, "gamma": gamma, "layers": L, "seed": seed,
                         "layer_probed": layer, "lambda_center": round(float(centers[b]), 3),
                         "mean_gain": None if np.isnan(binned[b]) else round(float(binned[b]), 4)})
        print(f"[{kernel}@{gamma} L{L} s{seed}] layer {layer}: "
              f"gain range {np.nanmin(gains):.3f}-{np.nanmax(gains):.3f}, "
              f"frac(gain>1)={float((gains>1).mean()):.3f}")

    # comparison table: mean gain by graph-frequency bin x kernel
    print(f"\nper-frequency gain ||G_i|| by lambda bin (layer {args.layer}) "
          f"[lambda: 0=global/long-range, 2=local]:")
    kernels = list(results.keys())
    print(f"    {'lambda':>7s} " + "".join(f"{k:>16s}" for k in kernels))
    for b in range(args.nbins):
        line = f"    {centers[b]:>7.2f} "
        for k in kernels:
            v = results[k][b]
            line += f"{v:>16.4f}" if not np.isnan(v) else f"{'-':>16s}"
        print(line)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
