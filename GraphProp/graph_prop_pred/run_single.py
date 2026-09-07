"""
Lean single-config driver for the GraphProp long-range tasks (dist/ecc/diam).

Bypasses the ray + grid `model_selection` machinery: trains ONE fixed config of
DGN_GraphProp with a chosen damping_kernel, num_layers, etc. Adds the same
divergence guard, best-val checkpoint, and results-CSV row as the Peptides scripts.

Metric is log10(MSE) -- LOWER is better. Selection is by best val log10(MSE).

Run from GraphProp/graph_prop_pred/, e.g.:
  python run_single.py --task dist --damping_kernel fejer --dissipative_force 2.2 \
      --num_layers 10 --epochs 1500 --seed 41 --results_csv ../../results/graphprop_dist.csv
"""
import argparse
import csv
import os
import random

import sys

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool

# import the model module directly, bypassing models/__init__.py (which drags in
# torch_sparse via sibling models we don't use). Neither torch_scatter nor
# torch_sparse is needed for the Euler path under PyG 2.x.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
from dgn_GraphProp import DGN_GraphProp  # noqa: E402
from utils import get_dataset  # noqa: E402
from utils.pna_dataset import NODE_LVL_TASKS, GRAPH_LVL_TASKS  # noqa: E402


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["dist", "ecc", "diam"])
    ap.add_argument("--damping_kernel", default="dirichlet", choices=["dirichlet", "uniform", "fejer"])
    ap.add_argument("--dissipative_force", type=float, default=0.01)
    ap.add_argument("--num_layers", type=int, default=10)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--epsilon", type=float, default=0.8)
    ap.add_argument("--hidden", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--results_csv", default=os.environ.get("RESULTS_CSV", ""))
    ap.add_argument("--ckpt_dir", default=os.environ.get("CKPT_DIR", "gp_checkpoints"))
    ap.add_argument("--run_name", default="")
    ap.add_argument("--max_graphs", type=int, default=None, help="subset for a dry run")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # root MUST be absolute: GraphPropDataset.processed_file_names returns
    # join(root, ...), and PyG only finds the pre-extracted .pt files (skipping
    # process()) when that path is absolute.
    data_train, data_valid, data_test, num_features, num_classes = get_dataset(
        root=os.path.abspath(args.data_dir), name="GraphProp", task=args.task)
    if args.max_graphs:
        data_train, data_valid, data_test = (data_train[:args.max_graphs],
                                             data_valid[:args.max_graphs],
                                             data_test[:args.max_graphs])

    node_level = args.task in NODE_LVL_TASKS
    assert node_level or args.task in GRAPH_LVL_TASKS

    model = DGN_GraphProp(
        input_dim=num_features, output_dim=num_classes, K=args.K, epsilon=args.epsilon,
        dissipation_force=args.dissipative_force, hidden_dim=args.hidden,
        num_layers=args.num_layers, node_level_task=node_level, conv_layer="Euler",
        damping_kernel=args.damping_kernel).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def criterion(pred, label, batch):
        if node_level:
            nodes_in_graph = global_add_pool(torch.ones(batch.shape[0], 1, device=device), batch)
            nodes_loss = (pred - label.reshape(label.shape[0], 1)) ** 2
            return torch.mean(global_add_pool(nodes_loss, batch) / nodes_in_graph)
        return torch.mean((pred - label.reshape(label.shape[0], 1)) ** 2)

    train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(data_valid, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(data_test, batch_size=args.batch_size, shuffle=False)

    def run_epoch(loader, train_mode):
        model.train(train_mode)
        tot, n = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            if train_mode:
                optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y, batch.batch)
            if train_mode:
                if not torch.isfinite(loss):
                    return float("nan")
                loss.backward(); optimizer.step()
            tot += loss.detach().item(); n += 1
        mse = tot / max(n, 1)
        return np.log10(mse) if mse > 0 else float("-inf")   # log10(MSE), lower is better

    _run = args.run_name or f"{args.task}_{args.damping_kernel}_g{args.dissipative_force}_L{args.num_layers}_s{args.seed}"
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt = os.path.join(args.ckpt_dir, _run + ".pth")

    best_val, best_test, best_epoch = None, None, 0
    diverged, diverged_epoch = False, -1
    for epoch in range(args.epochs):
        tr = run_epoch(train_loader, True)
        if not np.isfinite(tr):
            diverged, diverged_epoch = True, epoch
            print(f"[diverged] non-finite training loss at epoch {epoch} "
                  f"(best-val epoch {best_epoch}, val {best_val})")
            break
        with torch.no_grad():
            vl = run_epoch(val_loader, False)
            te = run_epoch(test_loader, False)
        if best_val is None or vl <= best_val:
            best_val, best_test, best_epoch = vl, te, epoch
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, ckpt)
        if epoch % 100 == 0:
            print(f"epoch {epoch:04d}  train {tr:.4f}  val {vl:.4f}  test {te:.4f}  (log10 MSE)")
        if epoch - best_epoch > args.patience:
            print(f"early-stopped at epoch {epoch} (best epoch {best_epoch})")
            break

    print(f"[result] task={args.task} kernel={args.damping_kernel} gamma={args.dissipative_force} "
          f"layers={args.num_layers} seed={args.seed}  test_logMSE={best_test}  "
          f"best_val_logMSE={best_val} @epoch {best_epoch}  diverged={diverged}"
          + (f" @epoch {diverged_epoch}" if diverged else ""))
    if args.results_csv:
        append_csv(args.results_csv, {
            "run": _run, "task": args.task, "kernel": args.damping_kernel,
            "gamma": args.dissipative_force, "num_layers": args.num_layers, "K": args.K,
            "epsilon": args.epsilon, "hidden": args.hidden, "lr": args.lr, "seed": args.seed,
            "epochs": args.epochs,
            "test_logMSE": None if best_test is None else float(best_test),
            "best_val_logMSE": None if best_val is None else float(best_val),
            "best_epoch": int(best_epoch), "diverged": int(diverged),
            "diverged_epoch": int(diverged_epoch),
        })


if __name__ == "__main__":
    main()
