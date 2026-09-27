#!/usr/bin/env python3
from __future__ import annotations

import argparse, ast, json, re
from pathlib import Path
from typing import Any
import pandas as pd
from ete3 import Tree
from tqdm import tqdm

OUTPUT_COLS = [
    "full_tree_id", "obs_tree_id", "true_path_nodes",
    "selected_path_nodes", "selected_path_counts",
    "supported_path_nodes_till MRCA", "supported_path_counts",
    "predicted_mrca_missingness_fraction", "predicted_mrca_deleted_tips_json",
    "supported_family_rank", "mrca_relation",
]

def natural_key(x: Any) -> tuple:
    text = str(x)
    nums = [int(v) for v in re.findall(r"\d+", text)]
    return tuple(nums) if nums else (10**18, text)

def short_node(x: str) -> str:
    m = re.search(r"obs_node_0*(\d+)$", str(x))
    return m.group(1) if m else str(x)

def path_to_short_str(nodes: list[str]) -> str:
    return "->".join(short_node(x) for x in nodes)

def parse_path_nodes(x: Any) -> list[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    text = str(x).strip()
    if not text:
        return []
    if "->" in text:
        return [z.strip() for z in text.split("->") if z.strip()]
    for sep in ["|", ",", ";"]:
        if sep in text:
            return [z.strip() for z in text.split(sep) if z.strip()]
    return [text]

def parse_tip_list(x: Any) -> list[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    text = str(x).strip()
    if not text:
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
    raise ValueError(f"Could not find column for {name}. Tried {candidates}.")

def assign_node_ids(tree: Tree) -> None:
    for i, node in enumerate(tree.traverse("preorder")):
        node.add_feature("obs_node_id", f"obs_node_{i:05d}")

def tip_set(node: Tree) -> set[str]:
    return {leaf.name for leaf in node.iter_leaves()}

def mrca_or_none(full_tree: Tree, tips: set[str]) -> Tree | None:
    valid = sorted(tips & set(full_tree.get_leaf_names()))
    if not valid:
        return None
    try:
        return (full_tree & valid[0]) if len(valid) == 1 else full_tree.get_common_ancestor(valid)
    except Exception:
        return None

def load_beam_paths_any_format(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    for c in ["split", "full_tree_id", "obs_tree_id"]:
        if c not in df.columns:
            raise ValueError(f"Beam paths CSV missing required column: {c}")
    if "path_nodes" in df.columns:
        out = df[["split", "full_tree_id", "obs_tree_id", "path_nodes"]].copy()
        out["beam_path_index"] = pd.to_numeric(df.get("path_rank", out.groupby("obs_tree_id").cumcount() + 1), errors="coerce").fillna(999999).astype(int)
        return out[["split", "full_tree_id", "obs_tree_id", "beam_path_index", "path_nodes"]]
    path_cols = sorted([c for c in df.columns if re.match(r"^path_\d+_nodes$", c)], key=lambda c: int(re.findall(r"\d+", c)[0]))
    if not path_cols:
        raise ValueError("No path_nodes or path_XX_nodes columns found.")
    rows = []
    for _, r in df.iterrows():
        for c in path_cols:
            nodes = parse_path_nodes(r.get(c, ""))
            if nodes:
                rows.append({"split": r["split"], "full_tree_id": r["full_tree_id"], "obs_tree_id": r["obs_tree_id"],
                             "beam_path_index": int(re.findall(r"\d+", c)[0]), "path_nodes": "->".join(nodes)})
    return pd.DataFrame(rows)

def reconstruct_true_paths_from_recursive_direction(recursive_csv: str) -> dict[str, list[str]]:
    direction = pd.read_csv(recursive_csv)
    required = ["obs_tree_id", "decision_node_id", "child_node_id", "target_mrca_node_id", "label"]
    for c in required:
        if c not in direction.columns:
            raise ValueError(f"Recursive direction CSV missing required column: {c}")
    true_paths: dict[str, list[str]] = {}
    for obs_tree_id, g in direction.groupby("obs_tree_id", sort=False):
        target = str(g["target_mrca_node_id"].iloc[0])
        true_edges = g[g["label"].astype(int) == 1][["decision_node_id", "child_node_id"]]
        edge_map: dict[str, str] = {}
        for _, r in true_edges.iterrows():
            edge_map.setdefault(str(r["decision_node_id"]), str(r["child_node_id"]))
        root = "obs_node_00000"
        path, cur, visited = [root], root, {root}
        for _ in range(len(edge_map) + 10):
            if cur == target or cur not in edge_map:
                break
            nxt = edge_map[cur]
            path.append(nxt)
            if nxt in visited:
                break
            visited.add(nxt)
            cur = nxt
        true_paths[str(obs_tree_id)] = path
    return true_paths

class SupportTree:
    def __init__(self, paths: list[list[str]]):
        self.paths = [p for p in paths if p]
        self.counts: dict[tuple[str, ...], int] = {}
        self.children: dict[tuple[str, ...], set[str]] = {}
        for p in self.paths:
            for i in range(1, len(p) + 1):
                pref = tuple(p[:i])
                self.counts[pref] = self.counts.get(pref, 0) + 1
            for i in range(len(p) - 1):
                self.children.setdefault(tuple(p[:i+1]), set()).add(p[i+1])
    def count(self, prefix: list[str] | tuple[str, ...]) -> int:
        return self.counts.get(tuple(prefix), 0)
    def child_items(self, prefix: list[str] | tuple[str, ...]) -> list[tuple[str, int]]:
        pref = tuple(prefix)
        items = [(ch, self.count(pref + (ch,))) for ch in self.children.get(pref, set())]
        items.sort(key=lambda x: (-x[1], natural_key(x[0])))
        return items
    def has_count_in_subtree(self, prefix: list[str], target: int) -> bool:
        pref = tuple(prefix)
        return any(len(k) >= len(pref) and k[:len(pref)] == pref and v == target for k, v in self.counts.items())

def extend_path(st: SupportTree, start_prefix: list[str]) -> list[str]:
    """Extend by larger-supported child until first 2; if no 2 in this subtree, first 1.
    If both children have equal target count (2 vs 2 or 1 vs 1), stop at parent.
    If repeated target count appears, stopping at current prefix gives first target.
    """
    path = list(start_prefix)
    target = 2 if st.has_count_in_subtree(path, 2) else 1
    # If start already has target count, stop at first target.
    if len(path) > 1 and st.count(path) == target:
        return path
    for _ in range(1000):
        kids = st.child_items(path)
        if not kids:
            return path
        # uncertain equal target split: stop one before.
        if len(kids) >= 2 and kids[0][1] == target and kids[1][1] == target:
            return path
        # follow larger supported child.
        child, _cnt = kids[0]
        new_path = path + [child]
        path = new_path
        if st.count(path) == target:
            return path
    return path

def is_prefix(a: list[str], b: list[str]) -> bool:
    return len(a) <= len(b) and b[:len(a)] == a

def distinct_path(p: list[str], selected: list[list[str]]) -> bool:
    return not any(p == q for q in selected)

def branch_candidates_in_subtree(st: SupportTree, subtree_prefix: list[str], exclude_prefixes: list[list[str]]) -> list[tuple[int, int, list[str], list[str]]]:
    """Return candidate extended paths from every branch start in subtree, excluding exact represented prefixes.
    Sort key later: support desc, depth desc.
    """
    out = []
    sub = tuple(subtree_prefix)
    for pref_tup, cnt in st.counts.items():
        if len(pref_tup) <= len(sub):
            continue
        if pref_tup[:len(sub)] != sub:
            continue
        start = list(pref_tup)
        # A candidate branch start is any prefix; but skip prefixes that are just ancestors of an excluded selected path
        # when they would extend to the same selected path.
        ext = extend_path(st, start)
        if not ext or any(ext == ex for ex in exclude_prefixes):
            continue
        # avoid proposing a path that is a prefix of an already selected path or vice versa only if exactly same endpoint after extension
        out.append((cnt, len(start), start, ext))
    # Deduplicate by extended path, keep strongest/deepest start for it.
    best: dict[tuple[str, ...], tuple[int, int, list[str], list[str]]] = {}
    for item in out:
        key = tuple(item[3])
        if key not in best or (item[0], item[1]) > (best[key][0], best[key][1]):
            best[key] = item
    vals = list(best.values())
    vals.sort(key=lambda x: (-x[0], -x[1], natural_key("->".join(x[2]))))
    return vals

def choose_top3_paths(st: SupportTree, top_k: int = 3) -> list[list[str]]:
    root = st.paths[0][0] if st.paths else "obs_node_00000"
    root_kids = st.child_items([root])
    if not root_kids:
        return []
    X, n = root_kids[0]
    Y, m = (root_kids[1] if len(root_kids) > 1 else (None, 0))
    x_start = [root, X]
    y_start = [root, Y] if Y is not None else None
    x_best = extend_path(st, x_start)
    y_best = extend_path(st, y_start) if y_start else None
    x_alts_all = branch_candidates_in_subtree(st, x_start, [x_best])
    # Remove x_best and duplicates as we add.
    def next_x_alt(selected: list[list[str]]) -> list[str] | None:
        for _cnt, _depth, _start, ext in x_alts_all:
            if ext in selected:
                continue
            # avoid exact duplicates only; prefixes can be meaningful selected paths in this algorithm.
            return ext
        return None
    selected: list[list[str]] = []
    selected.append(x_best)
    if y_best is not None and m >= n / 2:
        if distinct_path(y_best, selected):
            selected.append(y_best)
        alt = next_x_alt(selected)
        if alt is not None:
            selected.append(alt)
    elif y_best is not None and m >= 3:
        alt = next_x_alt(selected)
        if alt is not None:
            selected.append(alt)
        if distinct_path(y_best, selected):
            selected.append(y_best)
    else:
        while len(selected) < top_k:
            alt = next_x_alt(selected)
            if alt is None:
                break
            selected.append(alt)
    # Flexible fallback: if fewer than top_k different paths and root Y has support >=2, use it as last path.
    if len(selected) < top_k and y_best is not None and m >= 2 and distinct_path(y_best, selected):
        selected.append(y_best)
    # More fallback: any remaining global candidate.
    if len(selected) < top_k:
        global_cands = branch_candidates_in_subtree(st, [root], selected)
        for _cnt, _depth, _start, ext in global_cands:
            if distinct_path(ext, selected):
                selected.append(ext)
            if len(selected) >= top_k:
                break
    return selected[:top_k]

def path_counts(st: SupportTree, path: list[str]) -> list[int]:
    # Counts exclude root, matching reports like 8->8->6.
    return [st.count(path[:i+1]) for i in range(1, len(path))]

def counts_str(vals: list[int]) -> str:
    return "->".join(str(int(v)) for v in vals)

def min_unique_pos(path: list[str], all_paths: list[list[str]]) -> int:
    # Node position in path (root=0). Find first prefix not shared as prefix by any other selected path.
    for pos in range(1, len(path)):
        pref = path[:pos+1]
        shared = any(other is not path and is_prefix(pref, other) for other in all_paths)
        if not shared:
            return pos
    return len(path) - 1

def mrca_by_drop_preserve_identity(
    st: SupportTree,
    path: list[str],
    selected_paths: list[list[str]],
) -> tuple[list[str], dict[str, Any]]:
    """
    Choose the MRCA by the largest support-count decrease while preserving
    representative-path identity.

    Support counts include the root so that a root-to-child decrease can be
    evaluated when it is the first eligible change along a selected path.
    """
    if len(path) <= 1:
        return path, {"from": "", "to": "", "drop": ""}

    unique_pos = min_unique_pos(path, selected_paths)

    counts_with_root = [
        st.count(path[:i + 1])
        for i in range(len(path))
    ]

    start_end_pos = unique_pos
    candidates = []

    for end_pos in range(start_end_pos, len(path)):
        from_count = counts_with_root[end_pos - 1]
        to_count = counts_with_root[end_pos]
        drop = from_count - to_count
        candidates.append((drop, end_pos, from_count, to_count))

    if not candidates:
        return path, {"from": "", "to": "", "drop": ""}

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    drop, end_pos, from_count, to_count = candidates[0]

    return path[:end_pos + 1], {
        "from": from_count,
        "to": to_count,
        "drop": drop,
    }

def classify_mrca_relation(predicted_node_id: str, true_path: list[str], observed_tree_path: str) -> str:
    pred = str(predicted_node_id)
    if not pred or pred == "nan" or not true_path:
        return "unknown"
    true_mrca = true_path[-1]
    if pred == true_mrca:
        return "exact_MRCA"
    if pred in true_path:
        dist = (len(true_path) - 1) - true_path.index(pred)
        if dist == 1:
            return "parent"
        if dist == 2:
            return "grandparent"
        return f"{dist}parent"
    try:
        obs_tree = Tree(str(observed_tree_path), format=1)
        assign_node_ids(obs_tree)
        id_to_node = {str(n.obs_node_id): n for n in obs_tree.traverse() if hasattr(n, "obs_node_id")}
        if true_mrca not in id_to_node or pred not in id_to_node:
            return "unknown"
        cur, dist = id_to_node[pred], 0
        while cur is not None:
            if str(cur.obs_node_id) == true_mrca:
                if dist == 1:
                    return "first_child"
                if dist == 2:
                    return "grandchild"
                return f"{dist}child"
            cur = cur.up
            dist += 1
        return "wrong_branch"
    except Exception:
        return "unknown"

def predicted_deleted_tips_and_fraction(observed_tree_path: str, full_tree_path: str, pred_node_id: str,
                                        deleted_tips: set[str], relation: str) -> tuple[list[str], str]:
    denom = len(deleted_tips)
    if denom == 0 or not pred_node_id:
        return [], ""
    # Observed-truth convention: exact or ancestor of true observed MRCA covers all missingness.
    if relation == "exact_MRCA" or relation == "parent" or relation == "grandparent" or relation.endswith("parent"):
        captured = sorted(deleted_tips)
        return captured, f"{len(captured)}/{denom}"
    try:
        obs_tree = Tree(str(observed_tree_path), format=1)
        assign_node_ids(obs_tree)
        id_to_node = {str(n.obs_node_id): n for n in obs_tree.traverse() if hasattr(n, "obs_node_id")}
        if pred_node_id not in id_to_node:
            return [], f"0/{denom}"
        obs_desc_tips = tip_set(id_to_node[pred_node_id])
        full_tree = Tree(str(full_tree_path), format=1)
        full_region = mrca_or_none(full_tree, obs_desc_tips)
        if full_region is None:
            return [], f"0/{denom}"
        captured = sorted(tip_set(full_region) & deleted_tips)
        return captured, f"{len(captured)}/{denom}"
    except Exception:
        return [], f"0/{denom}"

def main() -> None:
    ap = argparse.ArgumentParser(description="Select top 3 paths using support-tree rules, then choose MRCA by identity-preserving support drop.")
    ap.add_argument("--beam_paths_csv", required=True)
    ap.add_argument("--candidate_support_csv", required=False, help="Kept for compatibility; not used by this algorithm.")
    ap.add_argument("--observed_manifest", required=True)
    ap.add_argument("--recursive_direction_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--split_filter", default="test", choices=["test", "val", "train", "all"])
    ap.add_argument("--top_k", type=int, default=3)
    args = ap.parse_args()

    beam_long = load_beam_paths_any_format(args.beam_paths_csv)
    manifest = pd.read_csv(args.observed_manifest)
    true_paths_by_tree = reconstruct_true_paths_from_recursive_direction(args.recursive_direction_csv)

    for df in [beam_long, manifest]:
        df["full_tree_id"] = df["full_tree_id"].astype(str)
        df["obs_tree_id"] = df["obs_tree_id"].astype(str)
    if args.split_filter != "all":
        beam_long = beam_long[beam_long["split"].astype(str) == args.split_filter].copy()
        manifest = manifest[manifest["split"].astype(str) == args.split_filter].copy()

    obs_tree_col = infer_col(manifest, ["observed_tree_file", "observed_tree_path", "obs_tree_path", "tree_path", "observed_path"], "observed tree path")
    full_tree_col = infer_col(manifest, ["full_tree_file", "full_tree_path", "full_tree", "full_path"], "full tree path")
    deleted_col = infer_col(manifest, ["deleted_tips_json", "deleted_tip_names_json", "deleted_tips"], "deleted tips")
    manifest_small = manifest[["full_tree_id", "obs_tree_id", obs_tree_col, full_tree_col, deleted_col]].copy()
    tree_keys = beam_long[["full_tree_id", "obs_tree_id"]].drop_duplicates().merge(manifest_small, on=["full_tree_id", "obs_tree_id"], how="left")
    tree_keys["_full_sort_key"] = tree_keys["full_tree_id"].map(natural_key)
    tree_keys["_obs_sort_key"] = tree_keys["obs_tree_id"].map(natural_key)
    tree_keys = tree_keys.sort_values(["_full_sort_key", "_obs_sort_key"]).drop(columns=["_full_sort_key", "_obs_sort_key"])

    rows, skip = [], {}
    for _, tr in tqdm(tree_keys.iterrows(), total=len(tree_keys), desc="Top3 paths + drop MRCA", unit="tree"):
        full_tree_id, obs_tree_id = str(tr["full_tree_id"]), str(tr["obs_tree_id"])
        try:
            gpaths = beam_long[beam_long["obs_tree_id"].astype(str) == obs_tree_id].sort_values("beam_path_index")
            paths = [parse_path_nodes(x) for x in gpaths["path_nodes"].tolist()]
            paths = [p for p in paths if p]
            st = SupportTree(paths)
            selected = choose_top3_paths(st, args.top_k)
            true_path = true_paths_by_tree.get(obs_tree_id, [])
            deleted_tips = set(parse_tip_list(tr[deleted_col]))
            for rank, sel in enumerate(selected, start=1):
                final_path, _drop_info = mrca_by_drop_preserve_identity(st, sel, selected)
                pred = final_path[-1] if final_path else ""
                relation = classify_mrca_relation(pred, true_path, str(tr[obs_tree_col]))
                cap, frac = predicted_deleted_tips_and_fraction(str(tr[obs_tree_col]), str(tr[full_tree_col]), pred, deleted_tips, relation)
                rows.append({
                    "full_tree_id": full_tree_id,
                    "obs_tree_id": obs_tree_id,
                    "true_path_nodes": path_to_short_str(true_path),
                    "selected_path_nodes": path_to_short_str(sel),
                    "selected_path_counts": counts_str(path_counts(st, sel)),
                    "supported_path_nodes_till MRCA": path_to_short_str(final_path),
                    "supported_path_counts": counts_str(path_counts(st, final_path)),
                    "predicted_mrca_missingness_fraction": frac,
                    "predicted_mrca_deleted_tips_json": json.dumps(cap),
                    "supported_family_rank": rank,
                    "mrca_relation": relation,
                })
            for rank in range(len(selected)+1, args.top_k+1):
                rows.append({"full_tree_id": full_tree_id, "obs_tree_id": obs_tree_id,
                             "true_path_nodes": path_to_short_str(true_path), "supported_family_rank": rank})
        except Exception as e:
            skip[type(e).__name__ + ": " + str(e)] = skip.get(type(e).__name__ + ": " + str(e), 0) + 1

    out = pd.DataFrame(rows)
    for c in OUTPUT_COLS:
        if c not in out.columns:
            out[c] = ""
    out = out[OUTPUT_COLS]
    out["_full_sort_key"] = out["full_tree_id"].map(natural_key)
    out["_obs_sort_key"] = out["obs_tree_id"].map(natural_key)
    out = out.sort_values(["_full_sort_key", "_obs_sort_key", "supported_family_rank"]).drop(columns=["_full_sort_key", "_obs_sort_key"])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print("\n=========================================================")
    print("REPRESENTATIVE PATH AND MRCA SELECTION")
    print("=========================================================")
    print(f"Trees processed:        {tree_keys['obs_tree_id'].nunique()}")
    print(f"Rows written:           {len(out)}")
    print(f"Top K/tree:             {args.top_k}")
    print("Selection target:       first support count 2; if no 2, first 1")
    print("Root rule:              Y rank2 if m>=n/2; Y rank3 if m>=3; flexible Y if m>=2 only if needed")
    print("MRCA rule:              preserve selected-path identity; include root drop; largest drop; ties choose deeper")
    print(f"Saved:                  {args.out_csv}")
    print(f"Skip:                   {skip}")

if __name__ == "__main__":
    main()
