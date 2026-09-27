#!/usr/bin/env python3
import os
import json
import math
import argparse
import random
import pandas as pd
from ete3 import Tree

"""
Inject local missingness into simulated full phylogenetic trees.

This version enforces stricter, debugging-friendly simulation rules:

1. The target node cannot be the root.
2. The target node cannot be a leaf.
3. The target subtree must have at least --min_subtree_tips tips.
4. The target subtree must not be too small relative to the full tree.
5. The target subtree must not be too large relative to the full tree.
6. After deletion, at least --min_remaining_target_tips tips must remain inside
   the target subtree.
7. Within each full tree, each target node is used at most once.

The output manifest contains diagnostic columns useful for debugging, but these
columns should not be used directly as ML features later.
"""


def assign_internal_node_ids(tree: Tree) -> None:
    """
    Assign stable internal-node IDs to a tree using preorder traversal.
    IDs are stored as node.node_id features.
    """
    counter = 0
    for node in tree.traverse("preorder"):
        if node.is_leaf():
            node.add_feature("node_id", None)
        else:
            counter += 1
            node.add_feature("node_id", f"node_{counter}")


def get_descendant_tip_names(node: Tree) -> list[str]:
    """Return descendant tip names under a node."""
    return [leaf.name for leaf in node.iter_leaves()]


def choose_deletion_fraction(
    fraction_choices: list[float] | None,
    min_frac: float,
    max_frac: float,
    rng: random.Random,
) -> float:
    """Choose deletion fraction from fixed choices or uniformly from [min_frac, max_frac]."""
    if fraction_choices:
        return float(rng.choice(fraction_choices))
    return float(rng.uniform(min_frac, max_frac))


def compute_n_delete(target_subtree_n_tips: int, deletion_fraction: float) -> int:
    """
    Compute number of tips to delete.

    Uses floor just like the original script, enforces at least one deletion,
    and prevents deleting the entire target subtree.
    """
    n_delete = max(1, math.floor(deletion_fraction * target_subtree_n_tips))
    if n_delete >= target_subtree_n_tips:
        n_delete = target_subtree_n_tips - 1
    return n_delete


def get_valid_target_nodes(
    tree: Tree,
    full_n_tips: int,
    min_subtree_tips: int,
    min_target_frac: float,
    max_target_frac: float,
) -> list[Tree]:
    """
    Return all candidate target nodes satisfying structural constraints.

    Remaining-after-deletion is checked later because it depends on the sampled
    deletion fraction.
    """
    candidates = []
    for node in tree.traverse("preorder"):
        if node.is_root() or node.is_leaf():
            continue

        n_desc = len(node.get_leaves())
        frac = n_desc / full_n_tips if full_n_tips > 0 else 0.0

        if n_desc < min_subtree_tips:
            continue
        if frac < min_target_frac:
            continue
        if frac > max_target_frac:
            continue

        candidates.append(node)

    return candidates


def prune_tree_by_tip_deletion(tree: Tree, tips_to_delete: list[str]) -> Tree:
    """Return a pruned copy of the tree after deleting the specified tips."""
    delete_set = set(tips_to_delete)
    kept_tips = [leaf.name for leaf in tree.iter_leaves() if leaf.name not in delete_set]
    if len(kept_tips) < 2:
        raise ValueError("Pruning would leave fewer than 2 tips in the observed tree.")

    tr = tree.copy(method="deepcopy")
    tr.prune(kept_tips, preserve_branch_length=True)
    return tr


def find_full_tree_file(full_tree_dir: str, full_tree_id: str) -> str:
    """Find the Newick file corresponding to a full_tree_id."""
    tree_file = os.path.join(full_tree_dir, f"{full_tree_id}.nwk")
    if not os.path.exists(tree_file):
        raise FileNotFoundError(f"Full tree file not found: {tree_file}")
    return tree_file


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def parse_fraction_choices(text: str | None) -> list[float] | None:
    if text is None:
        return None
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        return None
    if any(f <= 0 or f >= 1 for f in values):
        raise ValueError("All fraction_choices must be strictly between 0 and 1.")
    return values


def validate_args(args: argparse.Namespace) -> None:
    if args.min_frac <= 0 or args.max_frac <= 0:
        raise ValueError("Deletion fractions must be positive.")
    if args.min_frac >= 1 or args.max_frac >= 1:
        raise ValueError("Deletion fractions must be strictly less than 1.")
    if args.min_frac > args.max_frac:
        raise ValueError("min_frac must be <= max_frac.")
    if args.n_obs_per_full < 1:
        raise ValueError("n_obs_per_full must be at least 1.")
    if args.min_subtree_tips < 2:
        raise ValueError("min_subtree_tips must be at least 2.")
    if args.min_remaining_target_tips < 1:
        raise ValueError("min_remaining_target_tips must be at least 1.")
    if args.min_target_frac < 0 or args.max_target_frac <= 0:
        raise ValueError("Target fractions must be positive or zero for min_target_frac.")
    if args.min_target_frac > args.max_target_frac:
        raise ValueError("min_target_frac must be <= max_target_frac.")
    if args.max_target_frac > 1:
        raise ValueError("max_target_frac cannot exceed 1.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inject local missingness into full phylogenetic trees."
    )

    parser.add_argument(
        "--full_tree_dir",
        type=str,
        required=True,
        help="Directory containing full tree .nwk files",
    )
    parser.add_argument(
        "--full_manifest",
        type=str,
        required=True,
        help="CSV manifest for full trees",
    )
    parser.add_argument(
        "--n_obs_per_full",
        type=int,
        default=5,
        help="Maximum number of observed trees to generate per full tree",
    )
    parser.add_argument(
        "--min_subtree_tips",
        type=int,
        default=8,
        help="Minimum descendant tips required for a target node",
    )
    parser.add_argument(
        "--min_remaining_target_tips",
        type=int,
        default=4,
        help="Minimum tips that must remain inside target subtree after deletion",
    )
    parser.add_argument(
        "--min_target_frac",
        type=float,
        default=0.05,
        help="Minimum target subtree size as fraction of full-tree tips",
    )
    parser.add_argument(
        "--max_target_frac",
        type=float,
        default=0.70,
        help="Maximum target subtree size as fraction of full-tree tips",
    )
    parser.add_argument(
        "--min_frac",
        type=float,
        default=0.2,
        help="Minimum deletion fraction if using uniform fraction sampling",
    )
    parser.add_argument(
        "--max_frac",
        type=float,
        default=0.6,
        help="Maximum deletion fraction if using uniform fraction sampling",
    )
    parser.add_argument(
        "--fraction_choices",
        type=str,
        default=None,
        help='Optional comma-separated fixed deletion fractions, e.g. "0.2,0.4,0.6"',
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory for observed trees",
    )
    parser.add_argument(
        "--manifest_out",
        type=str,
        default=None,
        help=(
            "Output CSV path for observed-tree manifest "
            "(default: <outdir>/../manifests/observed_tree_manifest.csv)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed logs during processing",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    rng = random.Random(args.seed)
    fraction_choices = parse_fraction_choices(args.fraction_choices)

    ensure_dir(args.outdir)

    if args.manifest_out is None:
        base_dir = os.path.dirname(args.outdir)
        manifest_dir = os.path.join(base_dir, "manifests")
        ensure_dir(manifest_dir)
        manifest_out = os.path.join(manifest_dir, "observed_tree_manifest.csv")
    else:
        manifest_out = args.manifest_out
        ensure_dir(os.path.dirname(os.path.abspath(manifest_out)))

    full_manifest = pd.read_csv(args.full_manifest)
    if "tree_id" not in full_manifest.columns:
        raise ValueError("Full-tree manifest must contain a 'tree_id' column.")

    records = []
    skipped_summary = []

    for _, row in full_manifest.iterrows():
        full_tree_id = str(row["tree_id"])
        full_tree_file = find_full_tree_file(args.full_tree_dir, full_tree_id)

        try:
            full_tree = Tree(full_tree_file, format=1)
        except Exception as e:
            print(f"Skipping {full_tree_id}: failed to read tree ({e})")
            skipped_summary.append({
                "full_tree_id": full_tree_id,
                "reason": "read_error",
                "n_valid_candidates": 0,
                "n_generated": 0,
            })
            continue

        assign_internal_node_ids(full_tree)
        full_n_tips = len(full_tree.get_leaves())

        candidates = get_valid_target_nodes(
            tree=full_tree,
            full_n_tips=full_n_tips,
            min_subtree_tips=args.min_subtree_tips,
            min_target_frac=args.min_target_frac,
            max_target_frac=args.max_target_frac,
        )
        rng.shuffle(candidates)

        if args.verbose:
            print(
                f"\nProcessing {full_tree_id}: "
                f"full_n_tips={full_n_tips}, valid_candidates={len(candidates)}"
            )

        generated_for_tree = 0
        attempted_candidates = 0

        # Use each candidate node at most once. Stop when enough observed trees are generated.
        for target_node in candidates:
            if generated_for_tree >= args.n_obs_per_full:
                break

            attempted_candidates += 1
            descendant_tips = get_descendant_tip_names(target_node)
            target_subtree_n_tips = len(descendant_tips)
            target_subtree_fraction = target_subtree_n_tips / full_n_tips

            deletion_fraction = choose_deletion_fraction(
                fraction_choices=fraction_choices,
                min_frac=args.min_frac,
                max_frac=args.max_frac,
                rng=rng,
            )
            n_delete = compute_n_delete(target_subtree_n_tips, deletion_fraction)
            remaining_target_n_tips = target_subtree_n_tips - n_delete

            if remaining_target_n_tips < args.min_remaining_target_tips:
                if args.verbose:
                    print(
                        f"  Skip {target_node.node_id}: "
                        f"target_tips={target_subtree_n_tips}, "
                        f"delete={n_delete}, remaining={remaining_target_n_tips} "
                        f"< min_remaining={args.min_remaining_target_tips}"
                    )
                continue

            tips_to_delete = rng.sample(descendant_tips, n_delete)

            try:
                observed_tree = prune_tree_by_tip_deletion(full_tree, tips_to_delete)
            except Exception as e:
                if args.verbose:
                    print(f"  Skip {target_node.node_id}: pruning failed ({e})")
                continue

            observed_n_tips = len(observed_tree.get_leaves())
            actual_deletion_fraction = n_delete / target_subtree_n_tips

            generated_for_tree += 1
            obs_tree_id = f"{full_tree_id}_obs_{generated_for_tree:03d}"
            obs_tree_file = os.path.join(args.outdir, f"{obs_tree_id}.nwk")
            observed_tree.write(outfile=obs_tree_file, format=1)

            records.append({
                "obs_tree_id": obs_tree_id,
                "full_tree_id": full_tree_id,
                "full_tree_file": full_tree_file,
                "observed_tree_file": obs_tree_file,
                "target_node_id": target_node.node_id,
                "target_is_root": bool(target_node.is_root()),
                "deletion_mechanism": "random_within_clade",
                "deletion_fraction": deletion_fraction,
                "actual_deletion_fraction": actual_deletion_fraction,
                "full_tree_n_tips": full_n_tips,
                "observed_n_tips": observed_n_tips,
                "target_subtree_n_tips": target_subtree_n_tips,
                "target_subtree_fraction_of_full": target_subtree_fraction,
                "n_deleted": n_delete,
                "remaining_target_n_tips": remaining_target_n_tips,
                "full_descendant_tips_json": json.dumps(descendant_tips),
                "deleted_tips_json": json.dumps(tips_to_delete),
            })

            if args.verbose:
                print(
                    f"  Created {obs_tree_id}: "
                    f"target={target_node.node_id}, "
                    f"target_tips={target_subtree_n_tips}, "
                    f"target_frac={target_subtree_fraction:.3f}, "
                    f"deleted={n_delete}, "
                    f"remaining_target={remaining_target_n_tips}, "
                    f"observed_tips={observed_n_tips}"
                )

        if generated_for_tree == 0:
            reason = "no_valid_observed_tree_after_remaining_tip_filter"
        elif generated_for_tree < args.n_obs_per_full:
            reason = "fewer_valid_nodes_than_requested"
        else:
            reason = "ok"

        skipped_summary.append({
            "full_tree_id": full_tree_id,
            "reason": reason,
            "full_tree_n_tips": full_n_tips,
            "n_valid_candidates_before_deletion": len(candidates),
            "n_candidates_attempted": attempted_candidates,
            "n_generated": generated_for_tree,
        })

    obs_manifest = pd.DataFrame(records)
    obs_manifest.to_csv(manifest_out, index=False)

    summary_path = os.path.join(os.path.dirname(manifest_out), "observed_tree_generation_summary.csv")
    pd.DataFrame(skipped_summary).to_csv(summary_path, index=False)

    print(f"\nDone. Observed-tree manifest saved to: {manifest_out}")
    print(f"Generation summary saved to: {summary_path}")
    print(f"Number of observed trees generated: {len(obs_manifest)}")

    if len(skipped_summary) > 0:
        summary_df = pd.DataFrame(skipped_summary)
        print("\nObserved trees generated per full tree:")
        print(summary_df["n_generated"].value_counts().sort_index().to_string())
        n_less = int((summary_df["n_generated"] < args.n_obs_per_full).sum())
        print(f"Full trees with fewer than requested observed trees: {n_less}")


if __name__ == "__main__":
    main()
