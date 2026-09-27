#!/usr/bin/env python3
"""
Reusable inference for a new rooted observed phylogeny.

This script applies the pretrained direction model, beam search, representative
path selection, and MRCA estimation. It does not use a full tree, deleted tips,
the true MRCA, truth labels, or retraining.

Required project files
----------------------
- generate_beam_paths.py
- select_representative_paths_and_mrcas.py
- trained direction_model.pkl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import joblib
import pandas as pd


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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Apply the pretrained direction model, beam search, representative-path "
            "selection, and MRCA estimation to one new rooted observed tree."
        )
    )
    ap.add_argument("--observed_tree", required=True)
    ap.add_argument("--direction_model", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument(
        "--beam_module",
        default="generate_beam_paths.py",
        help="Path to the final beam-search script.",
    )

    ap.add_argument("--tree_id", default="new_tree")
    ap.add_argument("--beam_width", type=int, default=50)
    ap.add_argument("--top_final_paths", type=int, default=50)
    ap.add_argument("--max_depth", type=int, default=20)
    ap.add_argument("--min_candidate_obs_tips", type=int, default=4)
    ap.add_argument("--top_k", type=int, default=3)
    args = ap.parse_args()

    observed_tree_path = Path(args.observed_tree).resolve()
    model_path = Path(args.direction_model).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    beam = load_module(args.beam_module, "final_beam_module")
    representative_module_path = Path(__file__).with_name(
        "select_representative_paths_and_mrcas.py"
    )
    representative = load_module(
        representative_module_path,
        "representative_paths_module",
    )

    tree = beam.load_tree(observed_tree_path)
    beam.assign_node_ids(tree)
    direction_payload = joblib.load(model_path)

    # Exact final beam-search implementation from the uploaded project code.
    paths = beam.run_beam_search(
        tree=tree,
        direction_payload=direction_payload,
        beam_width=args.beam_width,
        max_depth=args.max_depth,
        min_subtree_size=args.min_candidate_obs_tips,
        top_final_paths=args.top_final_paths,
    )
    if not paths:
        raise RuntimeError("Beam search returned no paths.")

    # Save beam paths in the same long format, without truth-only diagnostics.
    beam_rows = []
    for path in paths:
        beam_rows.append(
            {
                "full_tree_id": "",
                "obs_tree_id": args.tree_id,
                "path_id": path["beam_path_id"],
                "path_rank": int(path["beam_path_rank"]),
                "path_nodes": str(path["beam_path"]),
                "path_length": int(path["beam_path_length"]),
                "path_geom_score": float(path["beam_geom_score"]),
                "path_product_score": float(path["beam_product_score"]),
                "path_mean_prob": float(path["beam_mean_prob"]),
                "path_min_prob": float(path["beam_min_prob"]),
                "path_max_prob": float(path["beam_max_prob"]),
                "path_local_probs": json.dumps(path.get("local_child_probs", [])),
                "path_endpoint_node_id": str(path["beam_endpoint_node_id"]),
            }
        )
    beam_df = pd.DataFrame(beam_rows)
    beam_csv = outdir / "obs_tree_beam_paths.csv"
    beam_df.to_csv(beam_csv, index=False)

    # Exact candidate support-count implementation from the beam script.
    rec = pd.Series({"full_tree_id": "", "obs_tree_id": args.tree_id})
    support_rows = beam.add_candidate_support_from_paths(
        rec=rec,
        split="inference",
        paths=paths,
        min_size=args.min_candidate_obs_tips,
    )
    support_df = pd.DataFrame(support_rows)
    support_csv = outdir / "candidate_support_counts.csv"
    support_df.to_csv(support_csv, index=False)

    # Representative-path selection and identity-preserving MRCA rule.
    path_node_lists = [list(map(str, p["node_ids"])) for p in paths]
    support_tree = representative.SupportTree(path_node_lists)
    selected_paths = representative.choose_top3_paths(support_tree, args.top_k)

    id_to_node = {
        str(node.obs_node_id): node
        for node in tree.traverse("preorder")
        if hasattr(node, "obs_node_id")
    }

    prediction_rows = []
    for rank, selected_path in enumerate(selected_paths, start=1):
        mrca_path, drop_info = representative.mrca_by_drop_preserve_identity(
            support_tree,
            selected_path,
            selected_paths,
        )
        predicted_mrca = mrca_path[-1] if mrca_path else ""
        node = id_to_node.get(predicted_mrca)
        descendant_tips = sorted(node.get_leaf_names()) if node is not None else []

        prediction_rows.append(
            {
                "tree_id": args.tree_id,
                "prediction_rank": rank,
                "selected_path_nodes": "->".join(selected_path),
                "selected_path_counts": "->".join(
                    str(x) for x in representative.path_counts(support_tree, selected_path)
                ),
                "predicted_path_nodes": "->".join(mrca_path),
                "predicted_path_counts": "->".join(
                    str(x) for x in representative.path_counts(support_tree, mrca_path)
                ),
                "predicted_mrca_node_id": predicted_mrca,
                "predicted_mrca_n_observed_tips": len(descendant_tips),
                "predicted_mrca_descendant_tips_json": json.dumps(descendant_tips),
                "mrca_drop_from_count": drop_info.get("from", ""),
                "mrca_drop_to_count": drop_info.get("to", ""),
                "mrca_drop_size": drop_info.get("drop", ""),
            }
        )

    predictions_df = pd.DataFrame(prediction_rows)
    predictions_csv = outdir / "predicted_missing_regions.csv"
    predictions_df.to_csv(predictions_csv, index=False)

    node_map_rows = []
    for node in tree.traverse("preorder"):
        node_map_rows.append(
            {
                "node_id": str(node.obs_node_id),
                "is_leaf": int(node.is_leaf()),
                "node_name": str(node.name or ""),
                "n_descendant_tips": len(node.get_leaves()),
                "descendant_tips_json": json.dumps(sorted(node.get_leaf_names())),
            }
        )
    node_map_csv = outdir / "observed_tree_node_map.csv"
    pd.DataFrame(node_map_rows).to_csv(node_map_csv, index=False)

    summary = {
        "tree_id": args.tree_id,
        "observed_tree": str(observed_tree_path),
        "direction_model": str(model_path),
        "n_observed_tips": len(tree.get_leaves()),
        "beam_width": args.beam_width,
        "n_final_beam_paths": len(paths),
        "top_k": args.top_k,
        "n_predictions": len(prediction_rows),
        "outputs": {
            "beam_paths": str(beam_csv),
            "candidate_support_counts": str(support_csv),
            "predicted_missing_regions": str(predictions_csv),
            "observed_tree_node_map": str(node_map_csv),
        },
    }
    summary_path = outdir / "inference_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n=========================================================")
    print("MISSING-REGION INFERENCE")
    print("=========================================================")
    print(f"Observed tree:             {observed_tree_path}")
    print(f"Observed tips:             {len(tree.get_leaves())}")
    print(f"Final beam paths:          {len(paths)}")
    print(f"Representative paths:      {len(prediction_rows)}")
    print(f"Saved beam paths:          {beam_csv}")
    print(f"Saved support counts:      {support_csv}")
    print(f"Saved predictions:         {predictions_csv}")
    print(f"Saved node map:            {node_map_csv}")
    print(f"Saved summary:             {summary_path}")


if __name__ == "__main__":
    main()
