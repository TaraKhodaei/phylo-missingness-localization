#!/usr/bin/env python3
"""
Generate beam-search paths and node support counts for observed phylogenies.

The pretrained direction model is used recursively to score child branches.
Beam paths are ranked by the geometric mean of their local direction
probabilities, and recurrence of nodes across retained paths is recorded as
beam support.

Main outputs:
- obs_tree_beam_paths.csv
- candidate_support_counts.csv
- beam_count_only_per_tree_summary.csv
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from ete3 import Tree
from tqdm import tqdm

# ---------------------------------------------------------------------
# FINAL MAIN CSV COLUMN POLICY
# ---------------------------------------------------------------------

ID_COLUMNS = [
    "split",
    "full_tree_id",
    "obs_tree_id",
    "candidate_node_id",
]

FINAL_FEATURE_COLUMNS = [
    # A. Path evidence
    "support_n_paths",
    "best_beam_geom_score",
    "best_beam_min_prob",
    "best_beam_path_length",
    "local_child_prob",

    # B. Candidate structure
    "candidate_obs_n_tips",
    "candidate_depth",
    "candidate_subtree_height",
    "candidate_total_branch_length",
    "candidate_mean_branch_length",
    "candidate_var_branch_length",
    "candidate_mean_terminal_branch_length",
    "candidate_max_terminal_branch_length",

    # C. Density / sparsity
    "candidate_total_branch_length_per_tip",
    "candidate_height_per_tip",
    "candidate_tip_density_per_height",
    "candidate_tip_density_per_branch_length",

    # D. Root context
    "candidate_frac_of_root_tips",
    "candidate_frac_of_root_height",
    "candidate_frac_of_root_total_branch_length",

    # E. Internal balance
    "candidate_child_tip_size_balance_ratio",
    "candidate_child_height_balance_ratio",
    "candidate_child_total_branch_balance_ratio",

    # F. Parent context
    "candidate_frac_of_parent_tips",
    "candidate_frac_of_parent_height",
    "candidate_frac_of_parent_total_branch_length",

    # G. Sibling contrast
    "candidate_diff_vs_sibling_n_tips",
    "candidate_ratio_vs_sibling_n_tips",
    "candidate_diff_vs_sibling_total_branch_length",
    "candidate_ratio_vs_sibling_total_branch_length",
    "candidate_diff_vs_sibling_mean_terminal_branch_length",
    "candidate_ratio_vs_sibling_mean_terminal_branch_length",
    "candidate_diff_vs_sibling_total_branch_length_per_tip",
    "candidate_ratio_vs_sibling_total_branch_length_per_tip",
    "candidate_diff_vs_sibling_tip_density_per_height",
    "candidate_ratio_vs_sibling_tip_density_per_height",

    # H. Within-tree scaled image-style context
    "candidate_scaled_support_n_paths",
    "candidate_scaled_best_beam_geom_score",
    "candidate_scaled_obs_n_tips",
    "candidate_scaled_subtree_height",
    "candidate_scaled_total_branch_length",
    "candidate_scaled_tip_density_per_height",
    "candidate_scaled_tip_density_per_branch_length",
    "candidate_scaled_total_branch_length_per_tip",
    "candidate_scaled_height_per_tip",
    "candidate_scaled_mean_terminal_branch_length",
    "candidate_scaled_max_terminal_branch_length",
    "candidate_scaled_ratio_vs_sibling_n_tips",
    "candidate_scaled_ratio_vs_sibling_total_branch_length",
    "candidate_scaled_ratio_vs_sibling_tip_density_per_height",
]

FINAL_LABEL_COLUMNS = [
    "severity_label",
    "coverage_label",
    "enrichment_label",
    "meaningful_label",
]

FINAL_MAIN_COLUMNS = ID_COLUMNS + FINAL_FEATURE_COLUMNS + FINAL_LABEL_COLUMNS



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


def infer_col(df: pd.DataFrame, candidates: list[str], name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find required column for {name}. Tried {candidates}. Available columns: {list(df.columns)}")


def load_tree(path: str | Path) -> Tree:
    return Tree(str(path), format=1)


def assign_node_ids(tree: Tree) -> None:
    for i, node in enumerate(tree.traverse("preorder")):
        node.add_feature("obs_node_id", f"obs_node_{i:05d}")


def node_depth(node: Tree) -> int:
    d = 0
    cur = node
    while cur.up is not None:
        d += 1
        cur = cur.up
    return d


def root_to_node_distance(node: Tree) -> float:
    """Compute root-to-node branch-length distance exactly as in training."""
    dist = 0.0
    cur = node
    while cur.up is not None:
        dist += float(cur.dist)
        cur = cur.up
    return float(dist)


def tip_set(node: Tree) -> set[str]:
    return {leaf.name for leaf in node.iter_leaves()}


def subtree_height(node: Tree) -> float:
    leaves = list(node.iter_leaves())
    if not leaves:
        return 0.0
    return float(max(node.get_distance(leaf) for leaf in leaves))


def branch_stats(root: Tree) -> dict[str, float]:
    """Branch summaries exactly matching the recursive-direction training builder."""
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
    sizes, heights, totals = [], [], []
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


# -------------------------
# Direction model features
# -------------------------

def direction_features(child: Tree, decision_node: Tree, observed_root: Tree) -> dict[str, float]:
    """
    Calculate the exact same 52-feature definitions used by
    build_direction_training_data.py
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

    sibling = None
    for ch in decision_node.children:
        if ch is not child:
            sibling = ch
            break

    if sibling is not None:
        sib_tips = float(len(sibling.get_leaves()))
        sib_bs = branch_stats(sibling)
        sib_total_bl = sib_bs["total_branch_length"]
        sib_terminal_frac = sib_bs["terminal_branch_fraction"]
    else:
        sib_tips = 0.0
        sib_total_bl = 0.0
        sib_terminal_frac = 0.0

    pair_tips = n_tips + sib_tips
    pair_total_bl = total_bl + sib_total_bl

    out = {
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

        "decision_depth": float(node_depth(decision_node)),
        "child_depth": float(node_depth(child)),
        "decision_root_to_node_distance": root_to_node_distance(decision_node),
        "child_root_to_node_distance": root_to_node_distance(child),

        "child_is_smaller_by_tips": float(n_tips < sib_tips),
        "child_is_smaller_by_total_branch_length": float(total_bl < sib_total_bl),

        "child_tip_share_of_sibling_pair": safe_div(n_tips, pair_tips, default=0.5),
        "child_branch_share_of_sibling_pair": safe_div(total_bl, pair_total_bl, default=0.5),
        "sibling_pair_tip_imbalance": safe_div(abs(n_tips - sib_tips), pair_tips),
        "sibling_pair_branch_imbalance": safe_div(abs(total_bl - sib_total_bl), pair_total_bl),

        "child_terminal_branch_fraction": bs["terminal_branch_fraction"],
        "sibling_terminal_branch_fraction": sib_terminal_frac,
    }

    out.update(child_balance(child))
    return out


def add_sibling_comparisons(row_a: dict[str, Any], row_b: dict[str, Any]) -> None:
    cols = [
        "n_tips", "subtree_height", "total_branch_length",
        "mean_terminal_branch_length", "max_terminal_branch_length",
        "total_branch_length_per_tip", "height_per_tip",
        "tip_density_per_height", "tip_density_per_branch_length",
    ]
    for col in cols:
        x = float(row_a.get(col, 0.0))
        y = float(row_b.get(col, 0.0))
        row_a[f"diff_vs_sibling_{col}"] = x - y
        row_a[f"ratio_vs_sibling_{col}"] = safe_div(x, y)


def child_scoring_rows(node: Tree, observed_root: Tree) -> tuple[pd.DataFrame, list[Tree]]:
    """
    Build exactly two sibling rows, as in training. Leaf children are included:
    they are valid direction alternatives and will simply terminate as beam paths.
    """
    if node.is_leaf() or len(node.children) != 2:
        return pd.DataFrame(), []

    rows, children = [], []
    for child in node.children:
        row = {"decision_node_id": node.obs_node_id, "child_node_id": child.obs_node_id}
        row.update(direction_features(child, node, observed_root))
        rows.append(row)
        children.append(child)

    add_sibling_comparisons(rows[0], rows[1])
    add_sibling_comparisons(rows[1], rows[0])
    return pd.DataFrame(rows), children


def score_children(direction_payload: dict[str, Any], node: Tree, observed_root: Tree) -> list[dict[str, Any]]:
    model = direction_payload["model"]
    feature_cols = direction_payload["feature_columns"]

    x, children = child_scoring_rows(node, observed_root)
    if x.empty or len(children) == 0:
        return []

    # Do not silently replace trained features with zeros. A mismatch must stop
    # the run so that training/inference inconsistency cannot go unnoticed.
    missing = [col for col in feature_cols if col not in x.columns]
    if missing:
        raise ValueError(
            "Inference feature mismatch. Missing model features: "
            + ", ".join(missing)
        )

    probs = model.predict_proba(x[feature_cols])[:, 1]
    return [{"child": ch, "prob": float(p)} for ch, p in zip(children, probs)]


# -------------------------
# Beam search
# -------------------------

def summarize_probs(log_sum: float, probs: list[float]) -> dict[str, float]:
    if len(probs) == 0:
        return {
            "beam_product_score": 1.0,
            "beam_geom_score": 1.0,
            "beam_mean_prob": 1.0,
            "beam_min_prob": 1.0,
            "beam_max_prob": 1.0,
        }
    return {
        "beam_product_score": float(math.exp(log_sum)),
        "beam_geom_score": float(math.exp(log_sum / len(probs))),
        "beam_mean_prob": float(np.mean(probs)),
        "beam_min_prob": float(np.min(probs)),
        "beam_max_prob": float(np.max(probs)),
    }


def finalize_path(path: dict[str, Any]) -> dict[str, Any]:
    out = path.copy()
    out.update(summarize_probs(out["log_sum"], out.get("probabilities", [])))
    out["beam_path_length"] = len(out["nodes"]) - 1
    out["beam_path"] = "->".join(out["node_ids"])
    out["beam_endpoint_node_id"] = out["node"].obs_node_id
    return out


def run_beam_search(
    tree: Tree,
    direction_payload: dict[str, Any],
    beam_width: int,
    max_depth: int,
    min_subtree_size: int,
    top_final_paths: int | None = None,
) -> list[dict[str, Any]]:
    """
    Clean rule:
      - beam_geom_score is used to keep/rank active paths.
      - beam_product_score is saved as diagnostic only.
      - with beam_width=B, final selected paths <= B.
    """
    if top_final_paths is None:
        top_final_paths = beam_width

    active = [{
        "node": tree,
        "nodes": [tree],
        "node_ids": [tree.obs_node_id],
        "local_child_probs": [],
        "probabilities": [],
        "log_sum": 0.0,
    }]

    for _ in range(max_depth):
        proposals = []
        expanded_any = False

        for path in active:
            node = path["node"]

            if node.is_leaf() or len(node.get_leaves()) <= min_subtree_size:
                proposals.append(finalize_path(path))
                continue

            scored = score_children(direction_payload, node, tree)
            if not scored:
                proposals.append(finalize_path(path))
                continue

            expanded_any = True
            for item in scored:
                child = item["child"]
                prob = max(float(item["prob"]), 1e-12)
                new_probs = path["probabilities"] + [prob]
                new_log_sum = path["log_sum"] + math.log(prob)
                summary = summarize_probs(new_log_sum, new_probs)

                proposal = {
                    "node": child,
                    "nodes": path["nodes"] + [child],
                    "node_ids": path["node_ids"] + [child.obs_node_id],

                    "local_child_probs":
                        path.get("local_child_probs", []) + [prob],

                    "probabilities": new_probs,
                    "log_sum": new_log_sum,
                    "beam_path_length": len(new_probs),
                    "beam_path": "->".join(path["node_ids"] + [child.obs_node_id]),
                    "beam_endpoint_node_id": child.obs_node_id,
                }
                proposal.update(summary)
                proposals.append(proposal)

        if not proposals:
            break

        proposals.sort(key=lambda z: z["beam_geom_score"], reverse=True)
        active = proposals[:beam_width]

        if not expanded_any:
            break

    ranked = [finalize_path(p) for p in active]
    ranked.sort(key=lambda z: z["beam_geom_score"], reverse=True)
    ranked = ranked[:top_final_paths]

    for i, path in enumerate(ranked, start=1):
        path["beam_path_rank"] = i
        path["beam_path_id"] = f"path_{i:03d}"

    return ranked


# -------------------------
# Path-only candidates
# -------------------------

def path_node_status(node: Tree, min_size: int) -> tuple[bool, str]:
    if node is None:
        return False, "none"
    if node.is_root():
        return False, "root_excluded"
    if node.is_leaf():
        return False, "leaf_excluded"
    if len(node.get_leaves()) < min_size:
        return False, "too_few_observed_tips"
    return True, "candidate"


def candidate_region_features(node: Tree, observed_root: Tree) -> dict[str, float]:
    """
    Compute candidate-region features.

    This function creates raw non-topology features:
      - candidate structure
      - density/sparsity
      - root context
      - internal child balance
      - parent context
      - finalized sibling contrasts

    It intentionally does NOT compute treeindices/topology descriptors.
    """
    n_tips = float(len(node.get_leaves()))
    h = subtree_height(node)
    bs = branch_stats(node)
    total_bl = bs["total_branch_length"]
    mean_terminal = bs["mean_terminal_branch_length"]
    max_terminal = bs["max_terminal_branch_length"]

    total_bl_per_tip = safe_div(total_bl, n_tips)
    height_per_tip = safe_div(h, n_tips)
    tip_density_height = safe_div(n_tips, h)
    tip_density_bl = safe_div(n_tips, total_bl)

    root_tips = float(len(observed_root.get_leaves()))
    root_height = subtree_height(observed_root)
    root_total_bl = branch_stats(observed_root)["total_branch_length"]

    out = {
        "candidate_obs_n_tips": n_tips,
        "candidate_depth": float(node_depth(node)),
        "candidate_subtree_height": h,
        "candidate_total_branch_length": total_bl,
        "candidate_mean_branch_length": bs["mean_branch_length"],
        "candidate_var_branch_length": bs["var_branch_length"],
        "candidate_mean_terminal_branch_length": mean_terminal,
        "candidate_max_terminal_branch_length": max_terminal,

        "candidate_total_branch_length_per_tip": total_bl_per_tip,
        "candidate_height_per_tip": height_per_tip,
        "candidate_tip_density_per_height": tip_density_height,
        "candidate_tip_density_per_branch_length": tip_density_bl,

        "candidate_frac_of_root_tips": safe_div(n_tips, root_tips),
        "candidate_frac_of_root_height": safe_div(h, root_height),
        "candidate_frac_of_root_total_branch_length": safe_div(total_bl, root_total_bl),
    }

    for k, v in child_balance(node).items():
        out[f"candidate_{k}"] = v

    parent = node.up
    if parent is not None:
        p_tips = float(len(parent.get_leaves()))
        p_height = subtree_height(parent)
        p_total_bl = branch_stats(parent)["total_branch_length"]

        out["candidate_frac_of_parent_tips"] = safe_div(n_tips, p_tips)
        out["candidate_frac_of_parent_height"] = safe_div(h, p_height)
        out["candidate_frac_of_parent_total_branch_length"] = safe_div(total_bl, p_total_bl)

        siblings = [ch for ch in parent.children if ch is not node]
        if siblings:
            sib = siblings[0]
            s_tips = float(len(sib.get_leaves()))
            s_h = subtree_height(sib)
            s_bs = branch_stats(sib)
            s_total_bl = s_bs["total_branch_length"]
            s_mean_terminal = s_bs["mean_terminal_branch_length"]

            s_total_bl_per_tip = safe_div(s_total_bl, s_tips)
            s_tip_density_height = safe_div(s_tips, s_h)

            out["candidate_diff_vs_sibling_n_tips"] = n_tips - s_tips
            out["candidate_ratio_vs_sibling_n_tips"] = safe_div(n_tips, s_tips)

            out["candidate_diff_vs_sibling_total_branch_length"] = total_bl - s_total_bl
            out["candidate_ratio_vs_sibling_total_branch_length"] = safe_div(total_bl, s_total_bl)

            out["candidate_diff_vs_sibling_mean_terminal_branch_length"] = mean_terminal - s_mean_terminal
            out["candidate_ratio_vs_sibling_mean_terminal_branch_length"] = safe_div(mean_terminal, s_mean_terminal)

            out["candidate_diff_vs_sibling_total_branch_length_per_tip"] = total_bl_per_tip - s_total_bl_per_tip
            out["candidate_ratio_vs_sibling_total_branch_length_per_tip"] = safe_div(total_bl_per_tip, s_total_bl_per_tip)

            out["candidate_diff_vs_sibling_tip_density_per_height"] = tip_density_height - s_tip_density_height
            out["candidate_ratio_vs_sibling_tip_density_per_height"] = safe_div(tip_density_height, s_tip_density_height)

    for col in [
        "candidate_frac_of_parent_tips",
        "candidate_frac_of_parent_height",
        "candidate_frac_of_parent_total_branch_length",

        "candidate_diff_vs_sibling_n_tips",
        "candidate_ratio_vs_sibling_n_tips",
        "candidate_diff_vs_sibling_total_branch_length",
        "candidate_ratio_vs_sibling_total_branch_length",
        "candidate_diff_vs_sibling_mean_terminal_branch_length",
        "candidate_ratio_vs_sibling_mean_terminal_branch_length",
        "candidate_diff_vs_sibling_total_branch_length_per_tip",
        "candidate_ratio_vs_sibling_total_branch_length_per_tip",
        "candidate_diff_vs_sibling_tip_density_per_height",
        "candidate_ratio_vs_sibling_tip_density_per_height",
    ]:
        out.setdefault(col, 0.0)

    return out


def mrca_or_none(full_tree: Tree, tips: set[str]) -> Tree | None:
    valid = sorted(tips & set(full_tree.get_leaf_names()))
    if not valid:
        return None
    try:
        return (full_tree & valid[0]) if len(valid) == 1 else full_tree.get_common_ancestor(valid)
    except Exception:
        return None


def compute_labels(full_tree: Tree, candidate_node: Tree, deleted_tips: set[str], total_deleted: int) -> dict[str, Any] | None:
    full_region = mrca_or_none(full_tree, tip_set(candidate_node))
    if full_region is None:
        return None

    full_region_tips = tip_set(full_region)
    n_full = len(full_region_tips)
    d = len(full_region_tips & deleted_tips)

    severity = safe_div(d, n_full)
    coverage = safe_div(d, total_deleted)
    enrichment = severity * coverage

    return {
        "deleted_tips_captured": int(d),
        "candidate_full_n_tips": int(n_full),
        "severity_label": float(severity),
        "coverage_label": float(coverage),
        "enrichment_label": float(enrichment),
        "nonzero_label": int(d > 0),
        "meaningful_label": int((severity >= 0.05) and (d >= 1)),
    }


def split_from_payload_or_manifest(full_tree_id: str, rec: pd.Series, payload: dict[str, Any]) -> str:
    groups = payload.get("split_groups", {})
    x = str(full_tree_id)
    for split in ["train", "val", "test"]:
        if x in set(groups.get(split, [])):
            return split
    if "split" in rec.index:
        return str(rec["split"])
    return "unknown"


def make_beam_path_rows(rec: pd.Series, split: str, paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        rows.append({
            "split": split,
            "full_tree_id": str(rec["full_tree_id"]),
            "obs_tree_id": str(rec["obs_tree_id"]),
            "beam_path_id": path["beam_path_id"],
            "beam_path_rank": int(path["beam_path_rank"]),
            "beam_endpoint_node_id": path["beam_endpoint_node_id"],
            "beam_path_length": int(path["beam_path_length"]),
            "beam_geom_score": float(path["beam_geom_score"]),
            "beam_product_score": float(path["beam_product_score"]),
            "beam_mean_prob": float(path["beam_mean_prob"]),
            "beam_min_prob": float(path["beam_min_prob"]),
            "beam_max_prob": float(path["beam_max_prob"]),
            "local_child_probs": json.dumps(path.get("local_child_probs", [])),
            "beam_path": path["beam_path"],
            "n_selected_paths_for_tree": int(len(paths)),
        })
    return rows



def make_obs_tree_beam_path_rows(
    rec: pd.Series,
    split: str,
    paths: list[dict[str, Any]],
    true_mrca_node_id: str,
) -> list[dict[str, Any]]:
    """
    Long format: one row per final beam path.

    For beam_width=50, each observed tree contributes up to 50 consecutive rows.
    The path_id/path_rank columns identify path_01, path_02, ..., path_50.

    path_contains_true_mrca is a diagnostic ground-truth label only:
      1 = the exact target MRCA node occurs anywhere in this beam path
      0 = it does not occur in this beam path

    This label is not used by beam generation or ranking.
    """
    rows: list[dict[str, Any]] = []

    for path in paths:
        rank = int(path["beam_path_rank"])
        path_node_ids = {str(node_id) for node_id in path["node_ids"]}
        contains_true_mrca = int(str(true_mrca_node_id) in path_node_ids)

        rows.append({
            "split": split,
            "full_tree_id": str(rec["full_tree_id"]),
            "obs_tree_id": str(rec["obs_tree_id"]),
            "path_id": f"path_{rank:02d}",
            "path_rank": rank,
            "path_nodes": str(path["beam_path"]),
            "path_length": int(path["beam_path_length"]),
            "path_geom_score": float(path["beam_geom_score"]),
            "path_product_score": float(path["beam_product_score"]),
            "path_mean_prob": float(path["beam_mean_prob"]),
            "path_min_prob": float(path["beam_min_prob"]),
            "path_max_prob": float(path["beam_max_prob"]),
            "path_local_probs": json.dumps(path.get("local_child_probs", [])),
            "path_endpoint_node_id": str(path["beam_endpoint_node_id"]),
            "path_contains_true_mrca": contains_true_mrca,
        })

    return rows


def add_candidate_support_from_paths(
    rec: pd.Series,
    split: str,
    paths: list[dict[str, Any]],
    min_size: int,
) -> list[dict[str, Any]]:
    """
    Count how many selected beam paths contain each internal candidate node.

    Output columns only:
      split
      full_tree_id
      obs_tree_id
      candidate_node_id
      support_n_paths
    """
    support: dict[str, int] = {}

    for path in paths:
        for node in path["nodes"]:
            ok, _status = path_node_status(node, min_size)
            if not ok:
                continue
            node_id = node.obs_node_id
            support[node_id] = support.get(node_id, 0) + 1

    rows = []
    for node_id, n in support.items():
        rows.append({
            "split": split,
            "full_tree_id": str(rec["full_tree_id"]),
            "obs_tree_id": str(rec["obs_tree_id"]),
            "candidate_node_id": node_id,
            "support_n_paths": int(n),
        })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate beam-search paths and node support counts for observed "
            "phylogenies using a pretrained direction model."
        )
    )
    parser.add_argument("--observed_manifest", required=True)
    parser.add_argument("--direction_model", required=True)
    parser.add_argument(
        "--recursive_direction_csv",
        required=True,
        help=(
            "Recursive-direction dataset containing one target_mrca_node_id "
            "for each observed tree. Used only to add the diagnostic "
            "path_contains_true_mrca label; it is not used by beam search."
        ),
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--top_final_paths", type=int, default=None)
    parser.add_argument("--max_depth", type=int, default=12)
    parser.add_argument("--min_candidate_obs_tips", type=int, default=4)
    parser.add_argument("--write_debug_report", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.observed_manifest)
    direction_payload = joblib.load(args.direction_model)

    recursive_direction_df = pd.read_csv(
        args.recursive_direction_csv,
        usecols=["obs_tree_id", "target_mrca_node_id"],
        dtype=str,
    ).dropna(subset=["obs_tree_id", "target_mrca_node_id"])

    mrca_counts = (
        recursive_direction_df.groupby("obs_tree_id")["target_mrca_node_id"]
        .nunique()
    )
    inconsistent_trees = mrca_counts[mrca_counts != 1]
    if not inconsistent_trees.empty:
        examples = inconsistent_trees.head(10).to_dict()
        raise ValueError(
            "Expected exactly one unique target_mrca_node_id per obs_tree_id. "
            f"Inconsistent examples: {examples}"
        )

    true_mrca_by_obs_tree = (
        recursive_direction_df.drop_duplicates(
            ["obs_tree_id", "target_mrca_node_id"]
        )
        .set_index("obs_tree_id")["target_mrca_node_id"]
        .to_dict()
    )

    obs_tree_col = infer_col(
        manifest,
        ["observed_tree_file", "observed_tree_path", "obs_tree_path", "tree_path", "observed_path"],
        "observed tree file",
    )

    for col in ["full_tree_id", "obs_tree_id"]:
        if col not in manifest.columns:
            raise ValueError(f"Manifest must contain {col}")

    candidate_rows = []
    beam_path_rows = []
    per_tree_rows = []
    skips: dict[str, int] = {}

    def add_skip(reason: str) -> None:
        skips[reason] = skips.get(reason, 0) + 1

    for _, rec in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc="Fast beam-count-only",
        unit="tree",
    ):
        try:
            observed_tree = load_tree(rec[obs_tree_col])
            assign_node_ids(observed_tree)

            split = split_from_payload_or_manifest(str(rec["full_tree_id"]), rec, direction_payload)

            paths = run_beam_search(
                tree=observed_tree,
                direction_payload=direction_payload,
                beam_width=args.beam_width,
                max_depth=args.max_depth,
                min_subtree_size=args.min_candidate_obs_tips,
                top_final_paths=args.top_final_paths,
            )

            candidate_rows.extend(
                add_candidate_support_from_paths(
                    rec=rec,
                    split=split,
                    paths=paths,
                    min_size=args.min_candidate_obs_tips,
                )
            )

            obs_tree_id = str(rec["obs_tree_id"])
            if obs_tree_id not in true_mrca_by_obs_tree:
                raise ValueError(
                    f"No target_mrca_node_id found for obs_tree_id={obs_tree_id}"
                )

            beam_path_rows.extend(
                make_obs_tree_beam_path_rows(
                    rec=rec,
                    split=split,
                    paths=paths,
                    true_mrca_node_id=true_mrca_by_obs_tree[obs_tree_id],
                )
            )

            per_tree_rows.append({
                "split": split,
                "full_tree_id": str(rec["full_tree_id"]),
                "obs_tree_id": str(rec["obs_tree_id"]),
                "n_selected_paths": int(len(paths)),
            })

        except Exception as e:
            add_skip(f"{type(e).__name__}: {e}")

    candidate_df = pd.DataFrame(candidate_rows)
    beam_paths_df = pd.DataFrame(beam_path_rows)
    per_tree_df = pd.DataFrame(per_tree_rows)

    candidate_cols = [
        "split",
        "full_tree_id",
        "obs_tree_id",
        "candidate_node_id",
        "support_n_paths",
    ]

    if candidate_df.empty:
        candidate_df = pd.DataFrame(columns=candidate_cols)
    else:
        candidate_df = candidate_df[candidate_cols].sort_values(
            ["split", "full_tree_id", "obs_tree_id", "support_n_paths", "candidate_node_id"],
            ascending=[True, True, True, False, True],
        )

    beam_path_cols = [
        "split",
        "full_tree_id",
        "obs_tree_id",
        "path_id",
        "path_rank",
        "path_nodes",
        "path_length",
        "path_geom_score",
        "path_product_score",
        "path_mean_prob",
        "path_min_prob",
        "path_max_prob",
        "path_local_probs",
        "path_endpoint_node_id",
        "path_contains_true_mrca",
    ]

    if beam_paths_df.empty:
        beam_paths_df = pd.DataFrame(columns=beam_path_cols)
    else:
        beam_paths_df = beam_paths_df[beam_path_cols].sort_values(
            ["split", "full_tree_id", "obs_tree_id", "path_rank"],
            ascending=[True, True, True, True],
            kind="stable",
        )

    candidate_csv = outdir / "candidate_support_counts.csv"
    beam_paths_csv = outdir / "obs_tree_beam_paths.csv"
    per_tree_csv = outdir / "beam_count_only_per_tree_summary.csv"

    candidate_df.to_csv(candidate_csv, index=False)
    beam_paths_df.to_csv(beam_paths_csv, index=False)
    per_tree_df.to_csv(per_tree_csv, index=False)

    print("\n=========================================================")
    print("FAST BEAM COUNT-ONLY SUMMARY")
    print("=========================================================")
    print(f"Input observed trees:          {len(manifest)}")
    print(f"Observed trees processed:      {per_tree_df['obs_tree_id'].nunique() if len(per_tree_df) else 0}")
    print(f"Skip reasons:                  {skips}")
    print(f"Beam width:                    {args.beam_width}")
    print(f"Top final paths saved:         {args.top_final_paths if args.top_final_paths is not None else args.beam_width}")
    print("Beam ranking score used:       beam_geom_score")
    print(f"Max depth:                     {args.max_depth}")
    print(f"Min candidate observed tips:   {args.min_candidate_obs_tips}")

    if len(per_tree_df):
        print("\nPath checks:")
        print(f"  mean selected paths/tree:    {per_tree_df['n_selected_paths'].mean():.2f}")
        print(f"  min selected paths/tree:     {per_tree_df['n_selected_paths'].min()}")
        print(f"  max selected paths/tree:     {per_tree_df['n_selected_paths'].max()}")

    if len(beam_paths_df):
        n_positive_paths = int(
            beam_paths_df["path_contains_true_mrca"].sum()
        )
        tree_has_mrca_path = (
            beam_paths_df.groupby("obs_tree_id")["path_contains_true_mrca"]
            .max()
        )
        print("\nTrue-MRCA diagnostic checks:")
        print(f"  beam paths containing MRCA:  {n_positive_paths}")
        print(
            "  trees with MRCA in >=1 beam: "
            f"{int(tree_has_mrca_path.sum())}/{len(tree_has_mrca_path)} "
            f"({100.0 * tree_has_mrca_path.mean():.2f}%)"
        )

    if len(candidate_df):
        print("\nCandidate checks:")
        print(f"  total candidate rows:        {len(candidate_df)}")
        print(f"  mean candidates/tree:        {candidate_df.groupby('obs_tree_id').size().mean():.2f}")
        print(f"  mean support_n_paths:        {candidate_df['support_n_paths'].mean():.3f}")
        print(f"  max support_n_paths:         {candidate_df['support_n_paths'].max()}")

    print("\nSaved files:")
    print(f"  candidate support counts:    {candidate_csv}")
    print(f"  observed-tree beam paths:    {beam_paths_csv}")
    print(f"  per-tree summary:            {per_tree_csv}")

    print("\nCandidate support columns:")
    for col in candidate_cols:
        print(f"  - {col}")

    print("\nBeam path output:")
    print("  - one row per beam path")
    print("  - rows are ordered by tree and path_rank")
    print("  - path_nodes, path_length, path_geom_score, path_product_score")
    print("  - path_mean_prob, path_min_prob, path_max_prob, path_local_probs")
    print("  - path_contains_true_mrca (diagnostic ground-truth label only)")

    if args.write_debug_report:
        debug_path = outdir / "beam_count_only_debug_report.json"
        debug = {
            "args": vars(args),
            "skip_reasons": skips,
            "n_candidate_rows": int(len(candidate_df)),
            "n_beam_path_rows": int(len(beam_paths_df)),
            "n_per_tree_rows": int(len(per_tree_df)),
            "output_files": {
                "candidate_support_counts": str(candidate_csv),
                "obs_tree_beam_paths": str(beam_paths_csv),
                "per_tree_summary": str(per_tree_csv),
            },
            "note": (
                "Beam search uses only the observed phylogeny and pretrained direction model. "
                "Ground-truth information is used only for optional diagnostic labels and does "
                "not affect path generation or ranking."
            ),
        }
        debug_path.write_text(json.dumps(debug, indent=2))
        print(f"  debug report:                {debug_path}")


if __name__ == "__main__":
    main()
