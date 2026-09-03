"""
Shared helpers for the Fejer-damping experiments (imported by the entry scripts).

Provides:
  * get_wandb()        -- real wandb, or a no-op stub if unavailable / disabled.
  * conv_spectral_norm -- exact ||J||_2 of one EulerConv layer via power iteration
                          (the layer is linear in x, so the Jacobian is constant and
                           input-independent; this is the section-5 growth factor).
  * probe_model        -- ||J||_2 for every conv layer on a fixed batch graph.
  * probe_epochs       -- which epochs to probe at, given the total.
  * maybe_subset       -- shrink a dataset for a fast dry run.
  * append_csv         -- append one result row to a CSV (headers written once).
"""
import csv
import os

import torch


# --------------------------------------------------------------------------
# wandb: real, or a stub that swallows everything. Controlled by WANDB_MODE:
#   WANDB_MODE=disabled  -> use the stub (no account, no files, no network)
#   otherwise            -> real wandb if importable, else stub.
# --------------------------------------------------------------------------
class _WandbStub:
    class _Run:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    def init(self, *a, **k): return self._Run()
    def log(self, *a, **k): pass
    def finish(self, *a, **k): pass
    class config:
        @staticmethod
        def update(*a, **k): pass


def get_wandb():
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        return _WandbStub()
    try:
        import wandb
        return wandb
    except Exception:
        print("[experiment_utils] wandb unavailable -> using stub (no logging).")
        return _WandbStub()


# --------------------------------------------------------------------------
# Spectral norm of one EulerConv layer (exact; layer is linear in x).
# --------------------------------------------------------------------------
def conv_spectral_norm(conv, edge_index, num_nodes, dim, device, iters=100, seed=0):
    """Largest singular value of the linear map x -> conv(x, edge_index).

    The EulerConv layer is affine in x (x + eps*filter(x) + bias), so its Jacobian J is
    a constant linear operator. Two-sided power iteration on J converges v to the top
    right-singular vector and returns ||J v|| = sigma_max. Forward evals run under
    no_grad; the adjoint J^T u is one backward under enable_grad."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        zero = torch.zeros(num_nodes, dim, device=device)
        b = conv(zero, edge_index)                  # constant term conv(0)

    def Jv(v):                                       # forward:  J v = conv(v) - conv(0)
        with torch.no_grad():
            return conv(v, edge_index) - b

    def Jt(u):                                        # adjoint:  J^T u via one backward
        x = torch.zeros(num_nodes, dim, device=device, requires_grad=True)
        with torch.enable_grad():
            out = conv(x, edge_index)
            out.backward(u)
        return x.grad.detach()

    v = torch.randn(num_nodes, dim, generator=g).to(device)
    v = v / (v.norm() + 1e-12)
    for _ in range(iters):
        u = Jv(v)
        v = Jt(u / (u.norm() + 1e-12))
        v = v / (v.norm() + 1e-12)
    return float(Jv(v).norm())


def probe_model(model, batch, device, hidden_dim):
    """||J||_2 for each conv layer, evaluated on the batch's graph. Returns list of floats."""
    ei = batch.edge_index.to(device)
    n = batch.num_nodes
    norms = []
    for conv in model.convs:
        norms.append(conv_spectral_norm(conv, ei, n, hidden_dim, device))
    return norms


def probe_epochs(total_epochs, marks=(0, 50, 100, 200)):
    eps = {e for e in marks if e < total_epochs}
    eps.add(max(0, total_epochs - 1))               # always probe the last epoch
    return sorted(eps)


# --------------------------------------------------------------------------
# Dry-run dataset subsetting
# --------------------------------------------------------------------------
def maybe_subset(dataset, max_graphs):
    if max_graphs is None or max_graphs <= 0 or max_graphs >= len(dataset):
        return dataset
    return dataset[:max_graphs]


# --------------------------------------------------------------------------
# CSV result logging
# --------------------------------------------------------------------------
def append_csv(path, row):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
