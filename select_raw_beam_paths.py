#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from ete3 import Tree
from tqdm import tqdm


OUTPUT_COLS = [
    "full_tree_id",
    "obs_tree_id",
    "true_path_nodes",
    "selected_path_nodes",
    "selected_path_counts",
    "supported_path_nodes_till MRCA",
    "supported_path_counts",
    "predicted_mrca_missingness_fraction",
    "predicted_mrca_deleted_tips_json",
    "supported_family_rank",
    "mrca_relation",
]


def natural_key(value: Any) -> tuple:
    text = str(value)
    nums = [int(v) for v in re.findall(r"\d+", text)]
    return tuple(nums) if nums else (10**18, text)


def short_node(value: str) -> str:
    match = re.search(r"obs_node_0*(\d+)$", str(value))
    return match.group(1) if match else str(value)


def path_to_short_str(nodes: list[str]) -> str:
    return "->".join(short_node(node) for node in nodes)


def parse_path_nodes(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    text = str(value).strip()
    if not text:
        return []

    if "->" in text:
        return [x.strip() for x in text.split("->") if x.strip()]

    for separator in ("|", ",", ";"):
        if separator in text:
            return [
                x.strip()
                for x in text.split(separator)
                if x.strip()
            ]

    return [text]


def parse_tip_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, list):
        return [str(x) for x in value]

    text = str(value).strip()
    if not text:
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(x) for x in parsed]
        except Exception:
            pass

    for separator in (",", ";", "|"):
        if separator in text:
            return [
                x.strip().strip("'\"")
                for x in text.split(separator)
                if x.strip()
            ]

    return [text.strip("'\"")]


def infer_col(
    df: pd.DataFrame,
    candidates: list[str],
    label: str,
) -> str:
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Could not find column for {label}. Tried: {candidates}"
    )


def assign_node_ids(tree: Tree) -> None:
    for index, node in enumerate(tree.traverse("preorder")):
        node.add_feature("obs_node_id", f"obs_node_{index:05d}")


def tip_set(node: Tree) -> set[str]:
    return {leaf.name for leaf in node.iter_leaves()}


def mrca_or_none(full_tree: Tree, tips: set[str]) -> Tree | None:
    valid = sorted(tips & set(full_tree.get_leaf_names()))
    if not valid:
        return None

    try:
        if len(valid) == 1:
            return full_tree & valid[0]
        return full_tree.get_common_ancestor(valid)
    except Exception:
        return None


def load_beam_paths_any_format(path_csv: str) -> pd.DataFrame:
    """
    Return one row per beam path with columns:
    split, full_tree_id, obs_tree_id, beam_path_index, path_nodes
    """
    df = pd.read_csv(path_csv)

    for column in ("split", "full_tree_id", "obs_tree_id"):
        if column not in df.columns:
            raise ValueError(
                f"Beam-path CSV missing required column: {column}"
            )

    # Long format: one row per path.
    if "path_nodes" in df.columns:
        out = df[
            ["split", "full_tree_id", "obs_tree_id", "path_nodes"]
        ].copy()

        if "path_rank" in df.columns:
            rank_values = df["path_rank"]
        else:
            rank_values = (
                out.groupby("obs_tree_id", sort=False).cumcount() + 1
            )

        out["beam_path_index"] = (
            pd.to_numeric(rank_values, errors="coerce")
            .fillna(999999)
            .astype(int)
        )

        return out[
            [
                "split",
                "full_tree_id",
                "obs_tree_id",
                "beam_path_index",
                "path_nodes",
            ]
        ]

    # Wide format: path_01_nodes, path_02_nodes, ...
    path_columns = sorted(
        [
            column
            for column in df.columns
            if re.match(r"^path_\d+_nodes$", column)
        ],
        key=lambda column: int(re.findall(r"\d+", column)[0]),
    )

    if not path_columns:
        raise ValueError(
            "No path_nodes column or path_XX_nodes columns were found."
        )

    rows: list[dict[str, Any]] = []

    for _, record in df.iterrows():
        for column in path_columns:
            nodes = parse_path_nodes(record.get(column, ""))
            if not nodes:
                continue

            rows.append(
                {
                    "split": record["split"],
                    "full_tree_id": record["full_tree_id"],
                    "obs_tree_id": record["obs_tree_id"],
                    "beam_path_index": int(
                        re.findall(r"\d+", column)[0]
                    ),
                    "path_nodes": "->".join(nodes),
                }
            )

    return pd.DataFrame(rows)


def reconstruct_true_paths_from_recursive_direction(
    recursive_csv: str,
) -> dict[str, list[str]]:
    direction = pd.read_csv(recursive_csv)

    required = [
        "obs_tree_id",
        "decision_node_id",
        "child_node_id",
        "target_mrca_node_id",
        "label",
    ]

    for column in required:
        if column not in direction.columns:
            raise ValueError(
                f"Recursive-direction CSV missing required column: {column}"
            )

    true_paths: dict[str, list[str]] = {}

    for obs_tree_id, group in direction.groupby(
        "obs_tree_id",
        sort=False,
    ):
        target_values = (
            group["target_mrca_node_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if len(target_values) != 1:
            raise ValueError(
                f"Expected one target MRCA for {obs_tree_id}; "
                f"found {target_values}"
            )

        target = target_values[0]

        true_edges = group[
            pd.to_numeric(group["label"], errors="coerce") == 1
        ][["decision_node_id", "child_node_id"]]

        edge_map: dict[str, str] = {}

        for _, row in true_edges.iterrows():
            parent = str(row["decision_node_id"])
            child = str(row["child_node_id"])
            edge_map.setdefault(parent, child)

        root = "obs_node_00000"
        path = [root]
        current = root
        visited = {root}

        for _ in range(len(edge_map) + 10):
            if current == target or current not in edge_map:
                break

            nxt = edge_map[current]

            if nxt in visited:
                break

            path.append(nxt)
            visited.add(nxt)
            current = nxt

        true_paths[str(obs_tree_id)] = path

    return true_paths


class SupportTree:
    """
    Prefix-support structure built from all saved beam paths for one tree.
    """

    def __init__(self, paths: list[list[str]]):
        self.paths = [path for path in paths if path]
        self.counts: dict[tuple[str, ...], int] = {}

        for path in self.paths:
            for end in range(1, len(path) + 1):
                prefix = tuple(path[:end])
                self.counts[prefix] = self.counts.get(prefix, 0) + 1

    def count(
        self,
        prefix: list[str] | tuple[str, ...],
    ) -> int:
        return self.counts.get(tuple(prefix), 0)


def path_counts(
    support_tree: SupportTree,
    path: list[str],
) -> list[int]:
    """
    Support counts excluding the root, matching the existing report format.
    """
    return [
        support_tree.count(path[: index + 1])
        for index in range(1, len(path))
    ]


def counts_str(values: list[int]) -> str:
    return "->".join(str(int(value)) for value in values)


def estimate_mrca_from_raw_beam_path(
    support_tree: SupportTree,
    path: list[str],
) -> tuple[list[str], dict[str, Any]]:
    """
    Estimate one MRCA independently for one raw beam path.

    Fresh baseline rule
    -------------------
    1. Compute support counts along every edge of the path using all saved
       beam paths for the tree.
    2. Find the largest decrease in support from parent to child.
    3. Select the child node at that decrease as the predicted MRCA.
    4. If several edges have the same largest decrease, choose the deeper edge.

    This rule does not use representative-family identity, uniqueness
    constraints, or relationships among the selected top-three paths.
    """
    if len(path) <= 1:
        return path, {
            "from": "",
            "to": "",
            "drop": "",
        }

    counts_with_root = [
        support_tree.count(path[: index + 1])
        for index in range(len(path))
    ]

    candidates: list[tuple[int, int, int, int]] = []

    for end_pos in range(1, len(path)):
        from_count = counts_with_root[end_pos - 1]
        to_count = counts_with_root[end_pos]
        drop = from_count - to_count

        candidates.append(
            (
                drop,
                end_pos,
                from_count,
                to_count,
            )
        )

    # Largest drop first; deeper position wins ties.
    candidates.sort(
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )

    drop, end_pos, from_count, to_count = candidates[0]

    return path[: end_pos + 1], {
        "from": from_count,
        "to": to_count,
        "drop": drop,
    }


def classify_mrca_relation(
    predicted_node_id: str,
    true_path: list[str],
    observed_tree_path: str,
) -> str:
    predicted = str(predicted_node_id)

    if not predicted or predicted == "nan" or not true_path:
        return "unknown"

    true_mrca = true_path[-1]

    if predicted == true_mrca:
        return "exact_MRCA"

    if predicted in true_path:
        distance = (
            (len(true_path) - 1)
            - true_path.index(predicted)
        )

        if distance == 1:
            return "parent"
        if distance == 2:
            return "grandparent"

        return f"{distance}parent"

    try:
        observed_tree = Tree(str(observed_tree_path), format=1)
        assign_node_ids(observed_tree)

        id_to_node = {
            str(node.obs_node_id): node
            for node in observed_tree.traverse()
            if hasattr(node, "obs_node_id")
        }

        if (
            true_mrca not in id_to_node
            or predicted not in id_to_node
        ):
            return "unknown"

        current = id_to_node[predicted]
        distance = 0

        while current is not None:
            if str(current.obs_node_id) == true_mrca:
                if distance == 1:
                    return "first_child"
                if distance == 2:
                    return "grandchild"

                return f"{distance}child"

            current = current.up
            distance += 1

        return "wrong_branch"

    except Exception:
        return "unknown"


def predicted_deleted_tips_and_fraction(
    observed_tree_path: str,
    full_tree_path: str,
    predicted_node_id: str,
    deleted_tips: set[str],
    relation: str,
) -> tuple[list[str], str]:
    denominator = len(deleted_tips)

    if denominator == 0 or not predicted_node_id:
        return [], ""

    # Same recovery convention as the current pipeline.
    if (
        relation == "exact_MRCA"
        or relation == "parent"
        or relation == "grandparent"
        or relation.endswith("parent")
    ):
        captured = sorted(deleted_tips)
        return captured, f"{len(captured)}/{denominator}"

    try:
        observed_tree = Tree(str(observed_tree_path), format=1)
        assign_node_ids(observed_tree)

        id_to_node = {
            str(node.obs_node_id): node
            for node in observed_tree.traverse()
            if hasattr(node, "obs_node_id")
        }

        if predicted_node_id not in id_to_node:
            return [], f"0/{denominator}"

        observed_descendant_tips = tip_set(
            id_to_node[predicted_node_id]
        )

        full_tree = Tree(str(full_tree_path), format=1)

        full_region = mrca_or_none(
            full_tree,
            observed_descendant_tips,
        )

        if full_region is None:
            return [], f"0/{denominator}"

        captured = sorted(
            tip_set(full_region) & deleted_tips
        )

        return captured, f"{len(captured)}/{denominator}"

    except Exception:
        return [], f"0/{denominator}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fresh raw-top-K beam baseline: select the highest-ranked "
            "beam paths and estimate one MRCA independently on each path "
            "using the largest support-count drop."
        )
    )

    parser.add_argument(
        "--beam_paths_csv",
        required=True,
    )
    parser.add_argument(
        "--observed_manifest",
        required=True,
    )
    parser.add_argument(
        "--recursive_direction_csv",
        required=True,
    )
    parser.add_argument(
        "--out_csv",
        required=True,
    )
    parser.add_argument(
        "--split_filter",
        default="test",
        choices=["test", "val", "train", "all"],
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top_k must be at least 1.")

    beam_long = load_beam_paths_any_format(
        args.beam_paths_csv
    )
    manifest = pd.read_csv(
        args.observed_manifest
    )
    true_paths_by_tree = (
        reconstruct_true_paths_from_recursive_direction(
            args.recursive_direction_csv
        )
    )

    for frame in (beam_long, manifest):
        frame["full_tree_id"] = (
            frame["full_tree_id"].astype(str)
        )
        frame["obs_tree_id"] = (
            frame["obs_tree_id"].astype(str)
        )

    if args.split_filter != "all":
        beam_long = beam_long[
            beam_long["split"].astype(str)
            == args.split_filter
        ].copy()

        manifest = manifest[
            manifest["split"].astype(str)
            == args.split_filter
        ].copy()

    observed_tree_col = infer_col(
        manifest,
        [
            "observed_tree_file",
            "observed_tree_path",
            "obs_tree_path",
            "tree_path",
            "observed_path",
        ],
        "observed tree path",
    )

    full_tree_col = infer_col(
        manifest,
        [
            "full_tree_file",
            "full_tree_path",
            "full_tree",
            "full_path",
        ],
        "full tree path",
    )

    deleted_tips_col = infer_col(
        manifest,
        [
            "deleted_tips_json",
            "deleted_tip_names_json",
            "deleted_tips",
        ],
        "deleted tips",
    )

    manifest_small = manifest[
        [
            "full_tree_id",
            "obs_tree_id",
            observed_tree_col,
            full_tree_col,
            deleted_tips_col,
        ]
    ].copy()

    tree_keys = (
        beam_long[
            ["full_tree_id", "obs_tree_id"]
        ]
        .drop_duplicates()
        .merge(
            manifest_small,
            on=["full_tree_id", "obs_tree_id"],
            how="left",
        )
    )

    tree_keys["_full_sort_key"] = (
        tree_keys["full_tree_id"].map(natural_key)
    )
    tree_keys["_obs_sort_key"] = (
        tree_keys["obs_tree_id"].map(natural_key)
    )

    tree_keys = (
        tree_keys.sort_values(
            ["_full_sort_key", "_obs_sort_key"]
        )
        .drop(
            columns=[
                "_full_sort_key",
                "_obs_sort_key",
            ]
        )
    )

    rows: list[dict[str, Any]] = []
    skip: dict[str, int] = {}

    for _, tree_record in tqdm(
        tree_keys.iterrows(),
        total=len(tree_keys),
        desc="Raw top-K beam baseline",
        unit="tree",
    ):
        full_tree_id = str(
            tree_record["full_tree_id"]
        )
        obs_tree_id = str(
            tree_record["obs_tree_id"]
        )

        try:
            tree_beams = (
                beam_long[
                    beam_long["obs_tree_id"].astype(str)
                    == obs_tree_id
                ]
                .sort_values("beam_path_index")
            )

            all_paths = [
                parse_path_nodes(value)
                for value in tree_beams["path_nodes"].tolist()
            ]
            all_paths = [
                path
                for path in all_paths
                if path
            ]

            if not all_paths:
                raise ValueError("No valid beam paths.")

            # Fresh experiment: select raw top K only by original beam rank.
            selected_paths = all_paths[: args.top_k]

            support_tree = SupportTree(all_paths)

            true_path = true_paths_by_tree.get(
                obs_tree_id,
                [],
            )

            deleted_tips = set(
                parse_tip_list(
                    tree_record[deleted_tips_col]
                )
            )

            for rank, selected_path in enumerate(
                selected_paths,
                start=1,
            ):
                final_path, _drop_info = (
                    estimate_mrca_from_raw_beam_path(
                        support_tree,
                        selected_path,
                    )
                )

                predicted_mrca = (
                    final_path[-1]
                    if final_path
                    else ""
                )

                relation = classify_mrca_relation(
                    predicted_mrca,
                    true_path,
                    str(tree_record[observed_tree_col]),
                )

                captured_tips, fraction = (
                    predicted_deleted_tips_and_fraction(
                        str(tree_record[observed_tree_col]),
                        str(tree_record[full_tree_col]),
                        predicted_mrca,
                        deleted_tips,
                        relation,
                    )
                )

                rows.append(
                    {
                        "full_tree_id": full_tree_id,
                        "obs_tree_id": obs_tree_id,
                        "true_path_nodes": path_to_short_str(
                            true_path
                        ),
                        "selected_path_nodes": path_to_short_str(
                            selected_path
                        ),
                        "selected_path_counts": counts_str(
                            path_counts(
                                support_tree,
                                selected_path,
                            )
                        ),
                        "supported_path_nodes_till MRCA": (
                            path_to_short_str(final_path)
                        ),
                        "supported_path_counts": counts_str(
                            path_counts(
                                support_tree,
                                final_path,
                            )
                        ),
                        "predicted_mrca_missingness_fraction": (
                            fraction
                        ),
                        "predicted_mrca_deleted_tips_json": (
                            json.dumps(captured_tips)
                        ),
                        "supported_family_rank": rank,
                        "mrca_relation": relation,
                    }
                )

            for rank in range(
                len(selected_paths) + 1,
                args.top_k + 1,
            ):
                rows.append(
                    {
                        "full_tree_id": full_tree_id,
                        "obs_tree_id": obs_tree_id,
                        "true_path_nodes": path_to_short_str(
                            true_path
                        ),
                        "supported_family_rank": rank,
                    }
                )

        except Exception as error:
            key = (
                type(error).__name__
                + ": "
                + str(error)
            )
            skip[key] = skip.get(key, 0) + 1

    output = pd.DataFrame(rows)

    for column in OUTPUT_COLS:
        if column not in output.columns:
            output[column] = ""

    output = output[OUTPUT_COLS]

    output["_full_sort_key"] = (
        output["full_tree_id"].map(natural_key)
    )
    output["_obs_sort_key"] = (
        output["obs_tree_id"].map(natural_key)
    )

    output = (
        output.sort_values(
            [
                "_full_sort_key",
                "_obs_sort_key",
                "supported_family_rank",
            ]
        )
        .drop(
            columns=[
                "_full_sort_key",
                "_obs_sort_key",
            ]
        )
    )

    output_path = Path(args.out_csv)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.to_csv(
        output_path,
        index=False,
    )

    print("\n=========================================================")
    print("FRESH RAW TOP-K BEAM BASELINE")
    print("=========================================================")
    print(
        f"Trees processed:        "
        f"{tree_keys['obs_tree_id'].nunique()}"
    )
    print(
        f"Rows written:           {len(output)}"
    )
    print(
        f"Top K/tree:             {args.top_k}"
    )
    print(
        "Path selection:         original beam ranks 1..K"
    )
    print(
        "MRCA rule:              largest support-count drop "
        "along each raw path; ties choose deeper"
    )
    print(
        "Representative logic:   none"
    )
    print(
        f"Saved:                  {output_path}"
    )
    print(
        f"Skip:                   {skip}"
    )


if __name__ == "__main__":
    main()
