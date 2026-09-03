"""
Guard test for the Fejer/uniform damping patch (FEJER_FIX_README section 11 + Amendment 1).

Instantiates the REAL Euler_ChebConv from Peptides/Stable/EulerConv.py on a small graph
and checks the properties the three-way patch must satisfy:

  1. g=0  ->  dirichlet, uniform and fejer all give BIT-IDENTICAL output.
             If any differ, the filter W_k - W_k^T was reweighted (a no-op bug).
  2. uniform @ g=0  ==  dirichlet @ g=0   (explicit, per Amendment 1).
  3. g>0  ->  each arm differs from dirichlet AND from each other (the flag does something).
  4. The learnable antisymmetric filter blocks are identical across arms (only the
     gamma handling differs).

Run:  python tools/test_kernel_equiv.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Peptides", "Stable"))
from EulerConv import Euler_ChebConv  # noqa: E402

torch.set_default_dtype(torch.float64)

ARMS = ("dirichlet", "uniform", "fejer")


def small_graph(n=24, p=0.15, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = (torch.rand(n, n, generator=g) < p).triu(1)
    src, dst = A.nonzero(as_tuple=True)
    ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])  # undirected
    return ei, n


def build_arms(d, K, eps, g, arms=ARMS):
    """One conv per arm, all sharing identical raw antisymmetric weights (from the first)."""
    convs = {a: Euler_ChebConv(d, d, K=K, step_size=eps, dissipation_force=g, damping_kernel=a)
             for a in arms}
    ref = convs[arms[0]]
    for a in arms[1:]:
        for lr, la in zip(ref.lins, convs[a].lins):
            with torch.no_grad():
                la.parametrizations.weight.original.copy_(lr.parametrizations.weight.original)
    for c in convs.values():
        c.eval()
    return convs


def antisym_blocks(conv):
    out = []
    for lin in conv.lins:
        up = torch.triu(lin.weight, 1)
        out.append(up - up.T)  # strip any -g*diagonal, recover the pure filter
    return out


def fwd(conv, X, ei):
    with torch.no_grad():
        return conv(X, ei)


def main():
    d, K, eps = 8, 10, 0.45
    ei, n = small_graph()
    X = torch.randn(n, d)

    # ---- property 4: shared filter blocks across all arms ----
    convs = build_arms(d, K, eps, g=0.1)
    ref_blocks = antisym_blocks(convs["dirichlet"])
    for a in ARMS[1:]:
        for k, (bd, ba) in enumerate(zip(ref_blocks, antisym_blocks(convs[a]))):
            assert torch.equal(bd, ba), f"{a}: filter block {k} differs -> filter reweighted"
    print("[4] antisymmetric filter blocks identical across all arms  ... OK")

    # ---- property 1: g=0 all three bit-identical ----
    convs0 = build_arms(d, K, eps, g=0.0)
    ys = {a: fwd(convs0[a], X, ei) for a in ARMS}
    for a in ARMS[1:]:
        assert torch.equal(ys["dirichlet"], ys[a]), (
            f"g=0: {a} != dirichlet (max {(ys['dirichlet']-ys[a]).abs().max():.2e}) "
            "-> you reweighted the FILTER, not just the damping"
        )
    print("[1] g=0: dirichlet == uniform == fejer  (bit-identical)     ... OK")
    print("[2] uniform @ g=0 == dirichlet @ g=0                        ... OK")

    # ---- property 3: g>0 each arm differs ----
    convs = build_arms(d, K, eps, g=0.1)
    y = {a: fwd(convs[a], X, ei) for a in ARMS}
    pairs = [("uniform", "dirichlet"), ("fejer", "dirichlet"), ("uniform", "fejer")]
    for a, b in pairs:
        diff = float((y[a] - y[b]).abs().max())
        assert diff > 1e-9, f"g>0: {a} == {b} (diff {diff:.2e}) -> arm does nothing"
        print(f"[3] g>0: {a:9s} != {b:9s}  (max|.-.|={diff:.3e})        ... OK")

    print("\nALL CHECKS PASS -- all three arms touch the damping only.")


if __name__ == "__main__":
    main()
