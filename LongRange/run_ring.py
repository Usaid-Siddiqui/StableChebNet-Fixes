"""
Ring-transfer long-range benchmark for the Fejer-damping study.

A cycle of P nodes. Node 0 carries a one-hot class c in {0..C-1}; the antipodal
node P//2 is marked and must predict c. The payload therefore has to propagate
P//2 hops. With K=2 (one hop per Euler layer) the REQUIRED DEPTH is exactly P//2,
tunable via the ring size -- unlike Peptides/GraphProp/Barbell, this lets us push
the required depth INTO the forward-Euler instability regime and separate the arms.

Money question: is there a depth at which fejer SOLVES the task while dirichlet /
uniform cannot (too shallow -> can't reach; deep enough -> diverges)?

Metric: accuracy at the target node (higher better). Also records divergence.

Reuses the already-patched Euler_ChebConv (damping_kernel) from Peptides/Stable.
Run from repo root:
  python LongRange/run_ring.py --ring 32 --num_layers 16 --damping_kernel fejer \
      --gamma 1.4 --epochs 300 --results_csv results/ring_P32.csv
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Peptides", "Stable"))
from EulerConv import Euler_ChebConv  # noqa: E402  (already has damping_kernel)


# --------------------------------------------------------------------------
# Ring-transfer dataset
# --------------------------------------------------------------------------
def gen_ring(num_samples, P, C, gen):
    src = list(range(P)); nxt = [(i + 1) % P for i in range(P)]
    ei = torch.tensor([src + nxt, nxt + src], dtype=torch.long)  # undirected cycle
    target = P // 2
    out = []
    for _ in range(num_samples):
        c = int(torch.randint(C, (1,), generator=gen))
        x = torch.zeros(P, C + 1)
        x[0, c] = 1.0            # source carries the class payload
        x[target, C] = 1.0       # target marker (the reader)
        y = torch.full((P,), -100, dtype=torch.long)
        y[target] = c            # only the target node is scored (ignore_index -100)
        out.append(Data(x=x, edge_index=ei, y=y))
    return out


# --------------------------------------------------------------------------
# Model: embed -> num_layers x Euler_ChebConv -> node readout
# --------------------------------------------------------------------------
class RingModel(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers, K, eps, gamma, damping_kernel):
        super().__init__()
        self.emb = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([
            Euler_ChebConv(hidden, hidden, K=K, step_size=eps,
                           dissipation_force=gamma, damping_kernel=damping_kernel)
            for _ in range(num_layers)])
        self.readout = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index):
        x = self.emb(x)
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
        return self.readout(x)


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


def run_epoch(model, loader, criterion, optimizer, device, train_mode):
    model.train(train_mode)
    tot_correct, tot_targets, tot_loss, nb = 0, 0, 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        if train_mode:
            optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = criterion(out, batch.y)
        if train_mode:
            if not torch.isfinite(loss):
                return float("nan"), float("nan")
            loss.backward(); optimizer.step()
        mask = batch.y != -100
        tot_correct += (out[mask].argmax(1) == batch.y[mask]).sum().item()
        tot_targets += int(mask.sum().item())
        tot_loss += float(loss.detach().item()); nb += 1
    return tot_loss / max(nb, 1), tot_correct / max(tot_targets, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring", type=int, default=32, help="ring size P (required depth = P//2)")
    ap.add_argument("--num_classes", type=int, default=5)
    ap.add_argument("--num_layers", type=int, default=16)
    ap.add_argument("--K", type=int, default=2, help="Chebyshev order; K=2 -> 1 hop/layer")
    ap.add_argument("--epsilon", type=float, default=0.7)
    ap.add_argument("--gamma", type=float, default=0.01)
    ap.add_argument("--damping_kernel", default="dirichlet", choices=["dirichlet", "uniform", "fejer"])
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_eval", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results_csv", default=os.environ.get("RESULTS_CSV", ""))
    ap.add_argument("--ckpt_dir", default=os.environ.get("CKPT_DIR", "ring_checkpoints"))
    ap.add_argument("--run_name", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)

    C = args.num_classes
    train = gen_ring(args.n_train, args.ring, C, g)
    val = gen_ring(args.n_eval, args.ring, C, g)
    test = gen_ring(args.n_eval, args.ring, C, g)
    tl = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    vl = DataLoader(val, batch_size=args.batch_size)
    te = DataLoader(test, batch_size=args.batch_size)

    model = RingModel(C + 1, args.hidden, C, args.num_layers, args.K,
                      args.epsilon, args.gamma, args.damping_kernel).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    _run = args.run_name or f"ring{args.ring}_{args.damping_kernel}_g{args.gamma}_L{args.num_layers}_s{args.seed}"
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt = os.path.join(args.ckpt_dir, _run + ".pth")

    best_val, best_test, best_epoch = -1.0, -1.0, 0
    diverged, diverged_epoch = False, -1
    for epoch in range(args.epochs):
        tr_loss, tr_acc = run_epoch(model, tl, criterion, opt, device, True)
        if not np.isfinite(tr_loss):
            diverged, diverged_epoch = True, epoch
            print(f"[diverged] non-finite training loss at epoch {epoch} (best-val acc {best_val:.3f})")
            break
        with torch.no_grad():
            _, va = run_epoch(model, vl, criterion, opt, device, False)
            _, ta = run_epoch(model, te, criterion, opt, device, False)
        if va >= best_val:
            best_val, best_test, best_epoch = va, ta, epoch
            torch.save(model.state_dict(), ckpt)
        if epoch % 25 == 0:
            print(f"epoch {epoch:03d}  train_acc {tr_acc:.3f}  val_acc {va:.3f}  test_acc {ta:.3f}")
        if epoch - best_epoch > args.patience:
            print(f"early-stopped at epoch {epoch} (best {best_epoch})")
            break

    # one layer reaches K-1 hops, target is ring//2 hops away
    hops = args.ring // 2
    req_depth = -(-hops // max(args.K - 1, 1))      # ceil division
    chance = 1.0 / C
    print(f"[result] ring={args.ring} req_depth={req_depth} kernel={args.damping_kernel} "
          f"gamma={args.gamma} layers={args.num_layers} seed={args.seed}  "
          f"test_acc={best_test:.4f} best_val_acc={best_val:.4f} (chance {chance:.3f}) "
          f"@epoch {best_epoch}  diverged={diverged}" + (f" @epoch {diverged_epoch}" if diverged else ""))
    if args.results_csv:
        append_csv(args.results_csv, {
            "run": _run, "ring": args.ring, "req_depth": req_depth, "hops": hops,
            "kernel": args.damping_kernel, "gamma": args.gamma, "num_layers": args.num_layers,
            "K": args.K, "epsilon": args.epsilon, "hidden": args.hidden, "seed": args.seed,
            "test_acc": round(float(best_test), 4), "best_val_acc": round(float(best_val), 4),
            "chance": round(chance, 4), "best_epoch": int(best_epoch),
            "diverged": int(diverged), "diverged_epoch": int(diverged_epoch),
        })


if __name__ == "__main__":
    main()
