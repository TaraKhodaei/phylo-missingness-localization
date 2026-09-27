#!/usr/bin/env python3
"""
Evaluate predictions for a controlled full/observed tree pair without
hard-coded clade sizes or metadata files.

The evaluator:
1. computes deleted tips as full tips minus observed tips;
2. searches the full tree for all internal lineages that contain all deleted
   tips and retain descendants on both child branches;
3. identifies the intended controlled target by matching the known deletion
   fraction used to generate the observed tree;
4. maps the retained descendants to the observed tree to obtain the true MRCA;
5. compares the predicted MRCAs with the true MRCA and computes recovery.

The expected missing fraction is an experiment setting supplied at runtime,
not a hard-coded clade size.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
from ete3 import Tree


def load_module(module_path: str | Path, module_name: str) -> ModuleType:
    path = Path(module_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Required module not found: {path}")

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Python module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_tree(path: Path) -> Tree:
    if not path.exists():
        raise FileNotFoundError(f"Tree file not found: {path}")
    return Tree(str(path), format=1)


def tip_set(node: Tree) -> set[str]:
    return {leaf.name for leaf in node.iter_leaves()}


def root_to_node_path(node: Tree) -> list[Tree]:
    path: list[Tree] = []
    current = node
    while current is not None:
        path.append(current)
        current = current.up
    return list(reversed(path))


def node_depth(node: Tree) -> int:
    return len(root_to_node_path(node)) - 1


def identify_controlled_target(
    full_tree: Tree,
    observed_tips: set[str],
    deleted_tips: set[str],
    expected_missing_fraction: float,
    fraction_tolerance: float,
) -> tuple[Tree, pd.DataFrame]:
    """
    Identify the controlled target lineage generically.

    A valid target must:
      - be an internal binary node;
      - contain every deleted tip;
      - retain at least two descendants;
      - retain at least one descendant on each immediate child branch.

    Candidates are ranked by:
      1. absolute difference between their deleted fraction and the expected
         experimental missing fraction;
      2. deeper node;
      3. smaller clade.

    If the best fraction match exceeds the tolerance, stop rather than silently
    assigning an unrelated lineage.
    """
    candidate_rows: list[dict[str, Any]] = []
    candidate_nodes: list[Tree] = []

    for preorder_index, node in enumerate(full_tree.traverse("preorder")):
        if node.is_leaf() or node.is_root() or len(node.children) != 2:
            continue

        descendant_tips = tip_set(node)
        if not deleted_tips.issubset(descendant_tips):
            continue

        retained_tips = descendant_tips & observed_tips
        if len(retained_tips) < 2:
            continue

        retained_by_child = [
            tip_set(child) & observed_tips
            for child in node.children
        ]
        if not all(len(x) >= 1 for x in retained_by_child):
            continue

        deleted_within = descendant_tips & deleted_tips
        deleted_fraction = len(deleted_within) / len(descendant_tips)
        fraction_error = abs(deleted_fraction - expected_missing_fraction)

        candidate_nodes.append(node)
        candidate_rows.append(
            {
                "candidate_index": len(candidate_nodes) - 1,
                "full_tree_preorder_index": preorder_index,
                "candidate_depth": node_depth(node),
                "candidate_n_tips": len(descendant_tips),
                "candidate_n_deleted": len(deleted_within),
                "candidate_n_retained": len(retained_tips),
                "candidate_deleted_fraction": deleted_fraction,
                "fraction_error": fraction_error,
                "child_sizes": json.dumps(
                    [len(child.get_leaves()) for child in node.children]
                ),
                "retained_by_child": json.dumps(
                    [len(x) for x in retained_by_child]
                ),
            }
        )

    if not candidate_rows:
        raise ValueError(
            "No internal lineage contains all deleted tips while retaining "
            "descendants on both immediate child branches."
        )

    candidates = pd.DataFrame(candidate_rows)
    candidates = candidates.sort_values(
        ["fraction_error", "candidate_depth", "candidate_n_tips"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)

    best = candidates.iloc[0]

    if float(best["fraction_error"]) > fraction_tolerance:
        raise ValueError(
            "No candidate lineage matches the supplied expected missing fraction "
            f"within tolerance {fraction_tolerance}. Best candidate: "
            f"{best['candidate_n_deleted']}/{best['candidate_n_tips']} = "
            f"{best['candidate_deleted_fraction']:.6f}; expected "
            f"{expected_missing_fraction:.6f}."
        )

    # Detect a true ambiguity after applying all ranking criteria.
    tied = candidates[
        (candidates["fraction_error"] == best["fraction_error"])
        & (candidates["candidate_depth"] == best["candidate_depth"])
        & (candidates["candidate_n_tips"] == best["candidate_n_tips"])
    ]
    if len(tied) > 1:
        raise ValueError(
            "The controlled target remains ambiguous after matching the missing "
            "fraction. Inspect target_candidate_lineages.csv."
        )

    selected_index = int(best["candidate_index"])
    return candidate_nodes[selected_index], candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate independently generated missing-region predictions against "
            "a controlled full/observed tree pair without target_clade_size."
        )
    )
    parser.add_argument("--full_tree", required=True)
    parser.add_argument("--observed_tree", required=True)
    parser.add_argument("--predictions_csv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--expected_missing_fraction",
        type=float,
        required=True,
        help=(
            "Controlled deletion fraction used to generate the observed tree, "
            "for example 0.40 or 0.60."
        ),
    )
    parser.add_argument(
        "--fraction_tolerance",
        type=float,
        default=0.025,
        help=(
            "Allowed difference between the requested fraction and the realized "
            "whole-tip fraction. Default: 0.025."
        ),
    )
    args = parser.parse_args()

    if not 0.0 < args.expected_missing_fraction < 1.0:
        raise ValueError("--expected_missing_fraction must be between 0 and 1.")

    full_path = Path(args.full_tree).resolve()
    observed_path = Path(args.observed_tree).resolve()
    predictions_path = Path(args.predictions_csv).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    representative_module_path = Path(__file__).with_name(
        "select_representative_paths_and_mrcas.py"
    )
    representative = load_module(
        representative_module_path,
        "representative_paths_module_for_controlled_evaluation",
    )

    full_tree = load_tree(full_path)
    observed_tree = load_tree(observed_path)
    representative.assign_node_ids(observed_tree)

    full_tips = tip_set(full_tree)
    observed_tips = tip_set(observed_tree)

    unexpected_observed_tips = observed_tips - full_tips
    if unexpected_observed_tips:
        raise ValueError(
            "Observed tree contains tips absent from the full tree: "
            + ", ".join(sorted(unexpected_observed_tips))
        )

    deleted_tips = full_tips - observed_tips
    if not deleted_tips:
        raise ValueError("No deleted tips were found.")

    target_lineage, target_candidates = identify_controlled_target(
        full_tree=full_tree,
        observed_tips=observed_tips,
        deleted_tips=deleted_tips,
        expected_missing_fraction=args.expected_missing_fraction,
        fraction_tolerance=args.fraction_tolerance,
    )

    target_candidates_csv = outdir / "target_candidate_lineages.csv"
    target_candidates.to_csv(target_candidates_csv, index=False)

    target_tips = tip_set(target_lineage)
    retained_target_tips = sorted(target_tips & observed_tips)

    true_mrca = observed_tree.get_common_ancestor(retained_target_tips)
    true_path_nodes = root_to_node_path(true_mrca)
    true_path_ids = [str(node.obs_node_id) for node in true_path_nodes]

    predictions = pd.read_csv(predictions_path)
    if "predicted_mrca_node_id" not in predictions.columns:
        raise ValueError(
            "Predictions CSV must contain predicted_mrca_node_id."
        )

    output_rows: list[dict[str, Any]] = []
    combined_recovered_tips: set[str] = set()

    for _, prediction in predictions.iterrows():
        predicted_node_id = str(prediction["predicted_mrca_node_id"])

        relation = representative.classify_mrca_relation(
            predicted_node_id=predicted_node_id,
            true_path=true_path_ids,
            observed_tree_path=str(observed_path),
        )

        recovered_tips, _ = representative.predicted_deleted_tips_and_fraction(
            observed_tree_path=str(observed_path),
            full_tree_path=str(full_path),
            pred_node_id=predicted_node_id,
            deleted_tips=deleted_tips,
            relation=relation,
        )

        combined_recovered_tips.update(recovered_tips)
        recovery_fraction = len(recovered_tips) / len(deleted_tips)

        output_row = prediction.to_dict()
        output_row.update(
            {
                "true_mrca_node_id": str(true_mrca.obs_node_id),
                "true_path_nodes": "->".join(true_path_ids),
                "mrca_relation": relation,
                "n_deleted_tips_recovered": len(recovered_tips),
                "n_deleted_tips_total": len(deleted_tips),
                "recovery_fraction": recovery_fraction,
                "recovery_percentage": f"{100.0 * recovery_fraction:.1f}%",
                "recovered_deleted_tips_json": json.dumps(
                    sorted(recovered_tips)
                ),
            }
        )
        output_rows.append(output_row)

    evaluation = pd.DataFrame(output_rows)
    evaluation_csv = outdir / "case_study_evaluation.csv"
    evaluation.to_csv(evaluation_csv, index=False)

    best_recovery = (
        float(evaluation["recovery_fraction"].max())
        if not evaluation.empty
        else 0.0
    )
    combined_recovery = (
        len(combined_recovered_tips) / len(deleted_tips)
    )

    summary = {
        "full_tree": str(full_path),
        "observed_tree": str(observed_path),
        "predictions_csv": str(predictions_path),
        "expected_missing_fraction": args.expected_missing_fraction,
        "fraction_tolerance": args.fraction_tolerance,
        "full_tree_n_tips": len(full_tips),
        "observed_tree_n_tips": len(observed_tips),
        "deleted_tips": sorted(deleted_tips),
        "n_deleted_tips": len(deleted_tips),
        "target_lineage_n_tips": len(target_tips),
        "realized_missing_fraction": len(deleted_tips) / len(target_tips),
        "target_child_sizes": [
            len(child.get_leaves())
            for child in target_lineage.children
        ],
        "retained_target_tips": retained_target_tips,
        "n_retained_target_tips": len(retained_target_tips),
        "true_mrca_node_id": str(true_mrca.obs_node_id),
        "true_path_nodes": true_path_ids,
        "best_prediction_recovery_fraction": best_recovery,
        "best_prediction_recovery_percentage": f"{100.0 * best_recovery:.1f}%",
        "combined_recovery_fraction": combined_recovery,
        "combined_recovery_percentage": f"{100.0 * combined_recovery:.1f}%",
        "combined_recovered_deleted_tips": sorted(combined_recovered_tips),
        "target_candidate_lineages_csv": str(target_candidates_csv),
        "evaluation_csv": str(evaluation_csv),
    }

    summary_path = outdir / "case_study_evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n=========================================================")
    print("CONTROLLED CASE-STUDY EVALUATION")
    print("=========================================================")
    print(f"Full-tree tips:                 {len(full_tips)}")
    print(f"Observed-tree tips:             {len(observed_tips)}")
    print(f"Deleted tips:                   {len(deleted_tips)}")
    print(f"Expected missing fraction:      {args.expected_missing_fraction:.3f}")
    print(f"Target-lineage tips:            {len(target_tips)}")
    print(
        "Realized missing fraction:      "
        f"{len(deleted_tips) / len(target_tips):.3f}"
    )
    print(f"Retained target tips:           {len(retained_target_tips)}")
    print(f"True observed MRCA:             {true_mrca.obs_node_id}")
    print(f"Best prediction recovery:       {100.0 * best_recovery:.1f}%")
    print(f"Combined prediction recovery:   {100.0 * combined_recovery:.1f}%")
    print(f"Saved target candidates:        {target_candidates_csv}")
    print(f"Saved evaluation:               {evaluation_csv}")
    print(f"Saved summary:                  {summary_path}")


if __name__ == "__main__":
    main()
