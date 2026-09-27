#!/usr/bin/env python3
"""
Step 2A: Build recursive_direction_dataset_selected.csv with targeted non-redundant features.

This version keeps the original Step-2 direction dataset design:

  - For each observed tree, use simulation truth to identify the target MRCA.
  - For every known-path decision node, create exactly two child-direction rows.
  - Correct child direction gets label=1.
  - Sibling direction gets label=0.

Main revision:
  Adds targeted features for the observed weaknesses:
    1. root / early-depth instability
    2. near-source instability
    3. smaller-child recovery

The added features are intentionally limited and non-redundant with the old feature set.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ete3 import Tree
from tqdm import tqdm


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def safe_div(num: float, den: float, default: float = 0.0) -> float:
    try:
        num = float(num)
        den = float(den)
        if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-15:
            return default
        return num / den
    except Exception:
        return default


def parse_tip_list(x: Any) -> list[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return [str(v) for v in x]

    text = str(x).strip()
    if text == "":
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            y = parser(text)
            if isinstance(y, (list, tuple, set)):
                return [str(v) for v in y]
        except Exception:
            pass

    for sep in [",", ";", "|"]:
        if sep in text:
            return [z.strip().strip("'\"") for z in text.split(sep) if z.strip()]

    return [text.strip("'\"")]


def load_tree(path: str | Path) -> Tree:
    return Tree(str(path), format=1)


def assign_node_ids(tree: Tree) -> None:
    for i, node in enumerate(tree.traverse("preorder")):
        node.add_feature("obs_node_id", f"obs_node_{i:05d}")


def tip_set(node: Tree) -> set[str]:
    return {leaf.name for leaf in node.iter_leaves()}


def node_depth(node: Tree) -> int:
    d = 0
    cur = node
    while cur.up is not None:
        d += 1
        cur = cur.up
    return d


def root_to_node_distance(node: Tree) -> float:
    dist = 0.0
    cur = node
    while cur.up is not None:
        dist += float(cur.dist)
        cur = cur.up
    return float(dist)


def subtree_height(node: Tree) -> float:
    leaves = list(node.iter_leaves())
    if not leaves:
        return 0.0
    return float(max(node.get_distance(leaf) for leaf in leaves))


def branch_stats(root: Tree) -> dict[str, float]:
    branch_lengths = []
    terminal_lengths = []
    internal_lengths = []

    for node in root.traverse():
        if node is root:
            continue
        d = float(node.dist)
        branch_lengths.append(d)
        if node.is_leaf():
            terminal_lengths.append(d)
        else:
            internal_lengths.append(d)

    total = float(np.sum(branch_lengths)) if branch_lengths else 0.0
    terminal_total = float(np.sum(terminal_lengths)) if terminal_lengths else 0.0
    internal_total = float(np.sum(internal_lengths)) if internal_lengths else 0.0

    return {
        "total_branch_length": total,
        "mean_branch_length": float(np.mean(branch_lengths)) if branch_lengths else 0.0,
        "var_branch_length": float(np.var(branch_lengths)) if branch_lengths else 0.0,
        "mean_terminal_branch_length": float(np.mean(terminal_lengths)) if terminal_lengths else 0.0,
        "mean_internal_branch_length": float(np.mean(internal_lengths)) if internal_lengths else 0.0,
        "max_terminal_branch_length": float(np.max(terminal_lengths)) if terminal_lengths else 0.0,
        "terminal_branch_length_total": terminal_total,
        "internal_branch_length_total": internal_total,
        "terminal_branch_fraction": safe_div(terminal_total, total),
    }


def child_balance(node: Tree) -> dict[str, float]:
    sizes = []
    heights = []
    totals = []

    for child in node.children:
        sizes.append(float(len(child.get_leaves())))
        heights.append(subtree_height(child))
        totals.append(branch_stats(child)["total_branch_length"])

    if len(sizes) < 2:
        return {
            "child_tip_size_balance_ratio": 0.0,
            "child_height_balance_ratio": 0.0,
            "child_total_branch_balance_ratio": 0.0,
        }

    return {
        "child_tip_size_balance_ratio": safe_div(min(sizes), max(sizes)),
        "child_height_balance_ratio": safe_div(min(heights), max(heights)),
        "child_total_branch_balance_ratio": safe_div(min(totals), max(totals)),
    }


# ---------------------------------------------------------------------
# Direction features
# ---------------------------------------------------------------------

def direction_features(child: Tree, decision_node: Tree, observed_root: Tree) -> dict[str, float]:
    """
    Original direction features + targeted non-redundant additions.

    New feature groups:
      A. depth / root-distance features
      B. smaller-child indicators
      C. sibling-pair shares and imbalance
      D. terminal-branch fraction
    """
    n_tips = float(len(child.get_leaves()))
    h = subtree_height(child)
    bs = branch_stats(child)
    total_bl = bs["total_branch_length"]

    dn_tips = float(len(decision_node.get_leaves()))
    dn_height = subtree_height(decision_node)
    dn_total_bl = branch_stats(decision_node)["total_branch_length"]

    root_tips = float(len(observed_root.get_leaves()))
    root_height = subtree_height(observed_root)
    root_total_bl = branch_stats(observed_root)["total_branch_length"]

    # Sibling quantities. In this dataset decisions should be binary.
    sibling = None
    for ch in decision_node.children:
        if ch is not child:
            sibling = ch
            break

    if sibling is not None:
        sib_tips = float(len(sibling.get_leaves()))
        sib_height = subtree_height(sibling)
        sib_bs = branch_stats(sibling)
        sib_total_bl = sib_bs["total_branch_length"]
        sib_terminal_frac = sib_bs["terminal_branch_fraction"]
    else:
        sib_tips = 0.0
        sib_height = 0.0
        sib_total_bl = 0.0
        sib_terminal_frac = 0.0

    pair_tips = n_tips + sib_tips
    pair_total_bl = total_bl + sib_total_bl
    pair_height = h + sib_height

    out = {
        # Original global / child / decision context
        "observed_root_n_tips": root_tips,
        "decision_node_n_tips": dn_tips,
        "n_tips": n_tips,
        "subtree_height": h,
        "total_branch_length": total_bl,
        "mean_branch_length": bs["mean_branch_length"],
        "var_branch_length": bs["var_branch_length"],
        "mean_terminal_branch_length": bs["mean_terminal_branch_length"],
        "max_terminal_branch_length": bs["max_terminal_branch_length"],
        "total_branch_length_per_tip": safe_div(total_bl, n_tips),
        "height_per_tip": safe_div(h, n_tips),
        "tip_density_per_height": safe_div(n_tips, h),
        "tip_density_per_branch_length": safe_div(n_tips, total_bl),
        "frac_of_decision_node_tips": safe_div(n_tips, dn_tips),
        "frac_of_decision_node_height": safe_div(h, dn_height),
        "frac_of_decision_node_total_branch_length": safe_div(total_bl, dn_total_bl),
        "frac_of_root_tips": safe_div(n_tips, root_tips),
        "frac_of_root_height": safe_div(h, root_height),
        "frac_of_root_total_branch_length": safe_div(total_bl, root_total_bl),

        # New A. Position features
        "decision_depth": float(node_depth(decision_node)),
        "child_depth": float(node_depth(child)),
        "decision_root_to_node_distance": root_to_node_distance(decision_node),
        "child_root_to_node_distance": root_to_node_distance(child),

        # New B. Smaller-child recovery features
        "child_is_smaller_by_tips": float(n_tips < sib_tips),
        "child_is_smaller_by_total_branch_length": float(total_bl < sib_total_bl),

        # New C. Sibling-pair share / imbalance features
        "child_tip_share_of_sibling_pair": safe_div(n_tips, pair_tips, default=0.5),
        "child_branch_share_of_sibling_pair": safe_div(total_bl, pair_total_bl, default=0.5),
        "sibling_pair_tip_imbalance": safe_div(abs(n_tips - sib_tips), pair_tips),
        "sibling_pair_branch_imbalance": safe_div(abs(total_bl - sib_total_bl), pair_total_bl),

        # New D. Terminal-branch composition
        "child_terminal_branch_fraction": bs["terminal_branch_fraction"],
        "sibling_terminal_branch_fraction": sib_terminal_frac,
    }

    out.update(child_balance(child))
    return out


def add_sibling_comparisons(row_a: dict[str, Any], row_b: dict[str, Any]) -> None:
    """
    Existing sibling contrast features.

    We keep these because they are already useful and not exactly equivalent to
    the new pair-share / imbalance features.
    """
    cols = [
        "n_tips",
        "subtree_height",
        "total_branch_length",
        "mean_terminal_branch_length",
        "max_terminal_branch_length",
        "total_branch_length_per_tip",
        "height_per_tip",
        "tip_density_per_height",
        "tip_density_per_branch_length",
    ]

    for col in cols:
        x = float(row_a.get(col, 0.0))
        y = float(row_b.get(col, 0.0))
        row_a[f"diff_vs_sibling_{col}"] = x - y
        row_a[f"ratio_vs_sibling_{col}"] = safe_div(x, y)


# ---------------------------------------------------------------------
# Path / truth helpers
# ---------------------------------------------------------------------

def is_ancestor_or_self(ancestor: Tree, node: Tree) -> bool:
    current = node
    while current is not None:
        if current is ancestor:
            return True
        current = current.up
    return False


def path_from_root_to_target(root: Tree, target: Tree) -> list[Tree]:
    rev = []
    cur = target

    while cur is not None:
        rev.append(cur)
        if cur is root:
            break
        cur = cur.up

    if not rev or rev[-1] is not root:
        return []

    return list(reversed(rev))


def child_on_path(decision: Tree, target: Tree) -> Tree | None:
    for child in decision.children:
        if is_ancestor_or_self(child, target):
            return child
    return None


def infer_col(df: pd.DataFrame, candidates: list[str], name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find required column for {name}. "
        f"Tried {candidates}. Available: {list(df.columns)}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build recursive-direction dataset with targeted Step-2 features."
    )
    parser.add_argument("--observed_manifest", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--write_debug_report", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_csv(args.observed_manifest)

    observed_tree_col = infer_col(
        manifest,
        ["observed_tree_file", "observed_tree_path", "obs_tree_path", "tree_path", "observed_path"],
        "observed tree file",
    )
    full_target_col = infer_col(
        manifest,
        ["full_descendant_tips_json", "target_full_descendant_tips_json"],
        "full descendant tips",
    )
    deleted_col = infer_col(
        manifest,
        ["deleted_tips_json", "deleted_tip_names_json"],
        "deleted tips",
    )

    for col in ["full_tree_id", "obs_tree_id"]:
        if col not in manifest.columns:
            raise ValueError(f"Manifest must contain {col}")

    rows = []
    skips: dict[str, int] = {}

    def add_skip(reason: str) -> None:
        skips[reason] = skips.get(reason, 0) + 1

    for _, rec in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc="Recursive direction rows",
        unit="tree",
    ):
        try:
            tree = load_tree(rec[observed_tree_col])
            assign_node_ids(tree)

            observed_tips = tip_set(tree)
            full_target_tips = set(parse_tip_list(rec[full_target_col]))
            deleted_tips = set(parse_tip_list(rec[deleted_col]))
            remaining_target_tips = sorted((full_target_tips - deleted_tips) & observed_tips)

            if len(remaining_target_tips) == 0:
                add_skip("no_remaining_target_tips")
                continue

            if len(remaining_target_tips) == 1:
                target_mrca = tree & remaining_target_tips[0]
            else:
                target_mrca = tree.get_common_ancestor(remaining_target_tips)

            if target_mrca.is_leaf():
                add_skip("target_mrca_is_leaf")
                continue

            path = path_from_root_to_target(tree, target_mrca)
            if len(path) < 2:
                add_skip("path_too_short")
                continue

            # Every internal node on root -> target_mrca path except target_mrca
            # is a decision node.
            for decision in path[:-1]:
                if decision.is_leaf():
                    continue

                if len(decision.children) != 2:
                    add_skip("non_binary_decision_node")
                    continue

                correct_child = child_on_path(decision, target_mrca)
                if correct_child is None:
                    add_skip("no_correct_child_found")
                    continue

                two_rows = []
                for child in decision.children:
                    row = {
                        "full_tree_id": str(rec["full_tree_id"]),
                        "obs_tree_id": str(rec["obs_tree_id"]),
                        "decision_node_id": decision.obs_node_id,
                        "child_node_id": child.obs_node_id,
                        "target_mrca_node_id": target_mrca.obs_node_id,
                        "label": int(child is correct_child),
                    }
                    row.update(direction_features(child, decision, tree))
                    two_rows.append(row)

                if len(two_rows) == 2:
                    add_sibling_comparisons(two_rows[0], two_rows[1])
                    add_sibling_comparisons(two_rows[1], two_rows[0])
                    rows.extend(two_rows)

        except Exception as e:
            add_skip(f"{type(e).__name__}: {e}")

    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    n_decisions = (
        out[["obs_tree_id", "decision_node_id"]].drop_duplicates().shape[0]
        if len(out)
        else 0
    )

    meta_cols = {
        "full_tree_id",
        "obs_tree_id",
        "decision_node_id",
        "child_node_id",
        "target_mrca_node_id",
        "label",
    }
    feature_cols = [
        c for c in out.columns
        if c not in meta_cols and pd.api.types.is_numeric_dtype(out[c])
    ]

    new_feature_cols = [
        "decision_depth",
        "child_depth",
        "decision_root_to_node_distance",
        "child_root_to_node_distance",
        "child_is_smaller_by_tips",
        "child_is_smaller_by_total_branch_length",
        "child_tip_share_of_sibling_pair",
        "child_branch_share_of_sibling_pair",
        "sibling_pair_tip_imbalance",
        "sibling_pair_branch_imbalance",
        "child_terminal_branch_fraction",
        "sibling_terminal_branch_fraction",
    ]

    print("\n=========================================================")
    print("RECURSIVE DIRECTION DATASET SUMMARY | TARGETED FEATURES")
    print("=========================================================")
    print(f"Input observed trees:        {len(manifest)}")
    print(f"Observed trees used:         {out['obs_tree_id'].nunique() if len(out) else 0}")
    print(f"Decision nodes used:         {n_decisions}")
    print(f"Rows written:                {len(out)}")
    print(f"Skipped events:              {sum(skips.values())}")
    print(f"Total columns:               {out.shape[1] if len(out) else 0}")
    print(f"Numeric feature columns:     {len(feature_cols)}")

    print("\nNew targeted feature columns:")
    for i, col in enumerate(new_feature_cols, start=1):
        status = "OK" if col in out.columns else "MISSING"
        print(f"  {i:02d}. {col:45s} {status}")

    if len(out):
        print("\nLabel distribution:")
        print(out["label"].value_counts().sort_index().to_string())
        print("\nRows per observed tree summary:")
        print(out.groupby("obs_tree_id").size().describe().to_string())

    print(f"\nSkip reasons: {skips}")
    print(f"Saved dataset to: {args.out_csv}")

    if args.write_debug_report:
        debug_path = str(Path(args.out_csv).with_suffix("")) + "_debug_report.json"
        debug = {
            "input_observed_trees": int(len(manifest)),
            "observed_trees_used": int(out["obs_tree_id"].nunique()) if len(out) else 0,
            "decision_nodes_used": int(n_decisions),
            "rows_written": int(len(out)),
            "skip_reasons": skips,
            "columns": list(out.columns),
            "numeric_feature_columns": feature_cols,
            "new_targeted_features": new_feature_cols,
            "feature_design_note": (
                "Original features retained. Added targeted features for "
                "early/root decisions, near-source decisions, and smaller-child recovery."
            ),
        }
        Path(debug_path).write_text(json.dumps(debug, indent=2))
        print(f"Saved debug report to: {debug_path}")


if __name__ == "__main__":
    main()
