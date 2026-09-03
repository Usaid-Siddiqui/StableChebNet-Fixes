"""
Phase 0 verification for the Fejer-damping hypothesis on Stable-ChebNet.

Checks, with NO GPU and NO training, the four claims in FEJER_FIX_README section 12:

  (A) s(lambda) = sum_k w_k T_k(lambda-tilde) is a Dirichlet kernel that changes
      sign for the paper's weights (w_k = 1), and a Fejer kernel that stays >= 1/2
      for w_k = 1 - k/K.  Report min s and fraction with s < 0.

  (B) The exact per-mode gain
          ||G_i||^2 = (1 - eps*gamma*s_i)^2 + eps^2 * sigma_max(B_i)^2
      matches the singular values of the TRUE Jacobian of a faithful reimplementation
      of the EulerConv forward loop.  This is the gate: if it disagrees to ~6 dp on a
      small graph, the derivation is wrong and everything else is void.

  (C) Sweeping gamma in [0, 4], the worst-case gain min_gamma max_i ||G_i|| is
      never <= 1 for Dirichlet, but reaches ~0.2 for Fejer with gamma* ~ 3.

The forward map is reimplemented here in dense torch, matching EulerConv.py exactly:
    Tx_0 = X;  Tx_1 = Ltil X;  Tx_2 = 2 Ltil Tx_1 - Tx_0        (Chebyshev recurrence)
    out  = sum_k Tx_k @ M_k.T,   M_k = triu(Wk,1) - triu(Wk,1).T - gamma*I
    F(X) = X + eps * out
Ltil is built exactly as EulerConv.__norm__: sym-normalized Laplacian, lambda_max=2,
then subtract 1 from the diagonal, so lambda-tilde in [-1, 1].
"""
import argparse
import numpy as np
import torch

torch.set_default_dtype(torch.float64)  # tight tolerances need float64


# ---------------------------------------------------------------------------
# Chebyshev damping profile s(lambda) for each kernel
# ---------------------------------------------------------------------------
def cheb_T(K, lam):
    """T_0..T_{K-1} evaluated at lam (array), via the recurrence. Returns (K, len(lam))."""
    lam = np.asarray(lam, dtype=np.float64)
    T = np.empty((K, lam.size))
    T[0] = 1.0
    if K > 1:
        T[1] = lam
    for k in range(2, K):
        T[k] = 2.0 * lam * T[k - 1] - T[k - 2]
    return T


def damping_weights(K, kernel):
    if kernel == "fejer":
        return np.array([1.0 - k / K for k in range(K)])
    return np.ones(K)  # dirichlet = paper


def s_profile(K, kernel, lam_tilde):
    lam_tilde = np.asarray(lam_tilde, dtype=np.float64)
    if kernel == "uniform":
        # gamma pulled out of the sum -> applied once to the signal -> flat damping s == 1.
        return np.ones_like(lam_tilde)
    w = damping_weights(K, kernel)
    return w @ cheb_T(K, lam_tilde)  # sum_k w_k T_k(lam_tilde)


# ---------------------------------------------------------------------------
# Faithful dense reimplementation of EulerConv, and its per-mode gain
# ---------------------------------------------------------------------------
def normalized_laplacian_tilde(A):
    """Sym-normalized Laplacian with EulerConv's rescaling: 2L/lambda_max - I, lambda_max=2.
    With lambda_max fixed at 2, this is exactly L_sym - I, eigenvalues in [-1, 1]."""
    A = A.clone()
    A.fill_diagonal_(0.0)
    deg = A.sum(1)
    dinv = torch.where(deg > 0, deg.pow(-0.5), torch.zeros_like(deg))
    L = torch.eye(A.shape[0]) - dinv[:, None] * A * dinv[None, :]  # L_sym
    return L - torch.eye(A.shape[0])  # 2L/2 - I


def make_weights(K, d, scale, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(d, d, generator=g) * scale for _ in range(K)]


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------
def random_graph(n, p, seed):
    g = torch.Generator().manual_seed(seed)
    A = (torch.rand(n, n, generator=g) < p).double()
    A = torch.triu(A, 1)
    A = A + A.T
    return A


def load_peptides_eigs(n_graphs):
    """Real Peptides-func graphs -> pooled lambda-tilde spectrum. Requires an LRGB download."""
    from torch_geometric.datasets import LRGBDataset
    from torch_geometric.utils import to_dense_adj

    ds = LRGBDataset(root="data/LRGB", name="Peptides-func", split="train")
    lams = []
    for i in range(min(n_graphs, len(ds))):
        data = ds[i]
        A = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0].double()
        Ltil = normalized_laplacian_tilde(A)
        lams.append(torch.linalg.eigvalsh(Ltil).numpy())
    return np.concatenate(lams)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--real", action="store_true", help="use real Peptides-func graphs (downloads LRGB)")
    ap.add_argument("--n_graphs", type=int, default=300)
    args = ap.parse_args()
    K, eps = args.K, args.eps

    print(f"=== Phase 0 verification | K={K} eps={eps} ===\n")

    # -----------------------------------------------------------------
    # (A) sign profile of the damping kernel
    # -----------------------------------------------------------------
    print("(A) Damping profile s(lambda) over the spectrum")
    if args.real:
        print(f"    loading {args.n_graphs} real Peptides-func graphs ...")
        lam = load_peptides_eigs(args.n_graphs)
        src = f"real Peptides-func ({lam.size} pooled eigenvalues)"
    else:
        lams = [torch.linalg.eigvalsh(normalized_laplacian_tilde(random_graph(60, 0.06, s))).numpy()
                for s in range(200)]
        lam = np.concatenate(lams)
        src = f"synthetic sparse graphs ({lam.size} pooled eigenvalues)"
    print(f"    source: {src}")
    for kernel in ("dirichlet", "uniform", "fejer"):
        s = s_profile(K, kernel, lam)
        frac_neg = float(np.mean(s < 0))
        print(f"    {kernel:9s}:  min s = {s.min():+.4f}   frac(s<0) = {frac_neg:.4f}")
    # dense grid for the exact analytic min of Fejer (should be +0.5)
    grid = np.linspace(-1, 1, 20001)
    print(f"    fejer min on dense grid = {s_profile(K, 'fejer', grid).min():+.6f}  (expect +0.5)")
    print()

    # -----------------------------------------------------------------
    # (B) THE GATE: closed-form gain vs true Jacobian
    # -----------------------------------------------------------------
    print("(B) Closed-form ||G_i|| vs true Jacobian singular values  [the gate]")
    n, d = 30, 8
    A = random_graph(n, 0.12, seed=7)
    Ltil = normalized_laplacian_tilde(A)
    Ws = make_weights(K, d, scale=0.1, seed=13)
    X0 = torch.randn(n, d)
    gamma = 0.7
    max_err = 0.0
    for kernel in ("dirichlet", "uniform", "fejer"):
        # Verify the closed form against the TRUE autograd Jacobian of the faithful
        # forward loop, for both kernels (damping-only reweighting -- Ws untouched).
        true_sv = true_jacobian_svals_kernel(A_ltil=Ltil, Ws=Ws, eps=eps, gamma=gamma, K=K, kernel=kernel, X=X0)
        pred_sv = predicted_mode_svals_kernel(Ltil, Ws, eps, gamma, kernel)
        true_sorted = torch.sort(true_sv, descending=True).values
        pred_sorted = torch.sort(pred_sv, descending=True).values
        err = float((true_sorted - pred_sorted).abs().max())
        max_err = max(max_err, err)
        print(f"    {kernel:9s}:  max|pred - true| over {n*d} svals = {err:.2e}   "
              f"(top sval pred={pred_sorted[0]:.6f} true={true_sorted[0]:.6f})")
    gate_ok = max_err < 1e-6
    print(f"    => GATE {'PASS' if gate_ok else 'FAIL'}  (threshold 1e-6, got {max_err:.2e})")
    if not gate_ok:
        print("    STOP: derivation disagrees with the true Jacobian.")
        return
    print()

    # -----------------------------------------------------------------
    # (C) gamma sweep: worst-case gain
    # -----------------------------------------------------------------
    print("(C) gamma sweep: argmin_gamma max_i ||G_i||   (K, eps as above, scale=0.1)")
    gammas = np.linspace(0, 4, 401)
    for kernel in ("dirichlet", "uniform", "fejer"):
        best_g, best_gain = None, np.inf
        for gm in gammas:
            sv = predicted_mode_svals_kernel(Ltil, Ws, eps, gm, kernel)
            worst = float(sv.max())
            if worst < best_gain:
                best_gain, best_g = worst, gm
        tag = "STABLE" if best_gain <= 1.0 else "never <= 1"
        print(f"    {kernel:9s}:  gamma* = {best_g:.3f}   worst gain = {best_gain:.4f}   [{tag}]")


# ---- kernel-aware wrappers (reweight the DAMPING term only) ----
def _damping_scale(K, kernel):
    return damping_weights(K, kernel)


def true_jacobian_svals_kernel(A_ltil, Ws, eps, gamma, K, kernel, X):
    """True Jacobian where the gamma*I term on order k is scaled by w_k (Fejer) or 1 (Dirichlet).
    The learnable antisymmetric blocks Wk are NOT reweighted -- only the damping."""
    Ltil = A_ltil
    n, d = X.shape
    # uniform: no per-order damping; gamma applied once to the signal (see below).
    wdamp = np.zeros(K) if kernel == "uniform" else _damping_scale(K, kernel)
    I = torch.eye(d)

    def Mk(W, k):
        up = torch.triu(W, diagonal=1)
        return up - up.T - gamma * float(wdamp[k]) * I

    def f(vec):
        Xr = vec.reshape(n, d)
        Tx0 = Xr
        out = Tx0 @ Mk(Ws[0], 0).T
        if K > 1:
            Tx1 = Ltil @ Xr
            out = out + Tx1 @ Mk(Ws[1], 1).T
        for k in range(2, K):
            Tx2 = 2.0 * (Ltil @ Tx1) - Tx0
            out = out + Tx2 @ Mk(Ws[k], k).T
            Tx0, Tx1 = Tx1, Tx2
        y = Xr + eps * out
        if kernel == "uniform":
            y = y - eps * gamma * Xr            # gamma applied ONCE to the signal
        return y.reshape(-1)

    x0 = X.reshape(-1).clone().requires_grad_(True)
    J = torch.autograd.functional.jacobian(f, x0)
    return torch.linalg.svdvals(J)


def predicted_mode_svals_kernel(Ltil, Ws, eps, gamma, kernel):
    n, d = Ltil.shape[0], Ws[0].shape[0]
    K = len(Ws)
    lam = torch.linalg.eigvalsh(Ltil).numpy()
    wdamp = _damping_scale(K, kernel)
    Tvals = cheb_T(K, lam)
    # antisymmetric block as the code builds it: strict upper triangle only.
    antis = [torch.triu(W, 1) - torch.triu(W, 1).T for W in Ws]
    I = torch.eye(d)
    out = []
    for i in range(n):
        s_i = 1.0 if kernel == "uniform" else float(wdamp @ Tvals[:, i])
        B_i = sum(float(Tvals[k, i]) * antis[k] for k in range(K))
        G_i = (1.0 - eps * gamma * s_i) * I + eps * B_i
        out.append(torch.linalg.svdvals(G_i))
    return torch.cat(out)


if __name__ == "__main__":
    main()
