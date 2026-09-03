"""Download both LRGB datasets into Peptides/Stable/ (the path the entry scripts use).

Run this ONCE on a node with internet (e.g. the login node) before submitting jobs,
so compute nodes without internet can train. Safe to re-run (skips if present).

    cd Peptides/Stable && python ../../experiments/prefetch_data.py
    #   or from repo root:  python experiments/prefetch_data.py --root Peptides/Stable
"""
import argparse
import os

from torch_geometric.datasets import LRGBDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="dataset root (entry scripts use './' from Peptides/Stable)")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    for name in ("Peptides-func", "Peptides-struct"):
        for split in ("train", "val", "test"):
            ds = LRGBDataset(root=root, name=name, split=split)
            print(f"{name}/{split}: {len(ds)} graphs  ok")
    print(f"\nDatasets ready under {root}")


if __name__ == "__main__":
    main()
