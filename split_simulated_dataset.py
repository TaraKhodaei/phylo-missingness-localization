#!/usr/bin/env python3
"""
Step 1: Split observed trees by full_tree_id.

Why:
  Several observed trees are generated from the same full tree. To avoid leakage,
  all observed trees from the same full_tree_id must stay in the same split.

Input:
  --observed_manifest simulations/manifests/observed_tree_manifest.csv

Outputs:
  observed_tree_manifest_with_splits.csv
  observed_tree_manifest_train.csv
  observed_tree_manifest_val.csv
  observed_tree_manifest_test.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Split observed manifest by full_tree_id.")
    parser.add_argument("--observed_manifest", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--train_frac", type=float, default=0.60)
    parser.add_argument("--val_frac", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.observed_manifest)
    for col in ["full_tree_id", "obs_tree_id"]:
        if col not in df.columns:
            raise ValueError(f"Input manifest must contain column: {col}")

    full_ids = np.array(sorted(df["full_tree_id"].astype(str).unique()))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(full_ids)

    n = len(full_ids)
    n_train = int(round(args.train_frac * n))
    n_val = int(round(args.val_frac * n))

    train_ids = set(full_ids[:n_train])
    val_ids = set(full_ids[n_train:n_train + n_val])
    test_ids = set(full_ids[n_train + n_val:])

    def split_of(x: str) -> str:
        x = str(x)
        if x in train_ids:
            return "train"
        if x in val_ids:
            return "val"
        if x in test_ids:
            return "test"
        raise RuntimeError(f"Unassigned full_tree_id: {x}")

    df["split"] = df["full_tree_id"].astype(str).map(split_of)

    paths = {
        "all": outdir / "observed_tree_manifest_with_splits.csv",
        "train": outdir / "observed_tree_manifest_train.csv",
        "val": outdir / "observed_tree_manifest_val.csv",
        "test": outdir / "observed_tree_manifest_test.csv",
    }

    df.to_csv(paths["all"], index=False)
    for split in ["train", "val", "test"]:
        df[df["split"] == split].to_csv(paths[split], index=False)

    print("\n=========================================================")
    print("OBSERVED MANIFEST SPLIT SUMMARY")
    print("=========================================================")
    print(f"Input rows: {len(df)}")
    print(f"Full-tree groups: {n}")
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        print(
            f"{split:5s}: rows={len(sub):5d}, "
            f"full_tree_groups={sub['full_tree_id'].nunique():5d}, "
            f"observed_trees={sub['obs_tree_id'].nunique():5d}"
        )
    for p in paths.values():
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
