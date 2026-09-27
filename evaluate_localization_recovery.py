#!/usr/bin/env python3
from __future__ import annotations

import argparse, ast, json, re
from pathlib import Path
from typing import Any
import pandas as pd

MONTH = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"sept":9,"oct":10,"nov":11,"dec":12}

def parse_fraction(x: Any) -> float:
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return 0.0
    m = re.fullmatch(r"(\d+)\s*/\s*(\d+)", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return a / b if b else 0.0
    m = re.fullmatch(r"(\d+)-([A-Za-z]{3,5})", s)
    if m and m.group(2).lower() in MONTH:
        den = int(m.group(1)); num = MONTH[m.group(2).lower()]
        return num / den if den else 0.0
    try:
        clean = s.replace("%", "")
        v = float(clean)
        if "%" in s:
            return v / 100.0
        return v if v <= 1 else v / 100.0
    except Exception:
        return 0.0

def status_from_fraction(f: float) -> str:
    if f >= 0.999999:
        return "Y"
    if f > 0:
        return "Partial"
    return "N"

def percent_text(f: float) -> str:
    return f"{100*f:.1f}%"

def parse_tip_list(x: Any) -> list[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            y = parser(s)
            if isinstance(y, (list, tuple, set)):
                return [str(v) for v in y]
        except Exception:
            pass
    for sep in [",", ";", "|"]:
        if sep in s:
            return [z.strip().strip("'\"") for z in s.split(sep) if z.strip()]
    return [s.strip("'\"")]

def first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def insert_after_column(df: pd.DataFrame, after_col: str, new_cols: list[str]) -> pd.DataFrame:
    cols = list(df.columns)
    for c in new_cols:
        if c in cols:
            cols.remove(c)
    if after_col not in cols:
        return df[cols + new_cols]
    idx = cols.index(after_col) + 1
    return df[cols[:idx] + new_cols + cols[idx:]]

def add_tip_count_columns_first_row_only(df: pd.DataFrame, manifest_csv: str) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_csv)
    if "obs_tree_id" not in manifest.columns:
        raise ValueError("observed_manifest must contain obs_tree_id")
    deleted_col = first_existing_col(manifest, ["deleted_tips_json", "deleted_tip_names_json", "deleted_tips", "deleted_tip_names", "removed_tips_json"])
    if deleted_col is None:
        raise ValueError("Could not find deleted tips column in observed_manifest")
    manifest["_n_deleted_tips_for_obs_tree"] = manifest[deleted_col].apply(lambda x: len(parse_tip_list(x)))
    full_tip_col = first_existing_col(manifest, ["full_tree_n_tips", "full_n_tips", "n_full_tips", "full_num_tips", "generated_n_tips", "n_tips_full"])
    if full_tip_col is not None:
        manifest["_full_tree_tip_count"] = pd.to_numeric(manifest[full_tip_col], errors="coerce")
    else:
        obs_tip_col = first_existing_col(manifest, ["observed_tree_n_tips", "observed_n_tips", "obs_n_tips", "n_observed_tips", "n_tips_observed", "remaining_tips"])
        if obs_tip_col is None:
            raise ValueError("Could not infer full tree tip count")
        manifest["_full_tree_tip_count"] = pd.to_numeric(manifest[obs_tip_col], errors="coerce") + manifest["_n_deleted_tips_for_obs_tree"]
    keep = manifest[["obs_tree_id", "_full_tree_tip_count", "_n_deleted_tips_for_obs_tree"]].drop_duplicates("obs_tree_id")
    tip_map = keep.set_index("obs_tree_id").to_dict("index")
    df["full_tree_tip_number"] = ""
    df["obs_tree_deleted_tip_number"] = ""
    for obs_tree_id, idxs in df.groupby("obs_tree_id", sort=False).groups.items():
        first_idx = list(idxs)[0]
        if obs_tree_id in tip_map:
            full_tip = tip_map[obs_tree_id]["_full_tree_tip_count"]
            del_tip = tip_map[obs_tree_id]["_n_deleted_tips_for_obs_tree"]
            df.loc[first_idx, "full_tree_tip_number"] = "" if pd.isna(full_tip) else int(full_tip)
            df.loc[first_idx, "obs_tree_deleted_tip_number"] = "" if pd.isna(del_tip) else int(del_tip)
    return insert_after_column(df, "obs_tree_id", ["full_tree_tip_number", "obs_tree_deleted_tip_number"])

def union_fraction_from_rows(g_top: pd.DataFrame, denominator: int | None = None) -> float:
    tips: set[str] = set()
    if "predicted_mrca_deleted_tips_json" in g_top.columns:
        for x in g_top["predicted_mrca_deleted_tips_json"]:
            tips.update(parse_tip_list(x))
    if denominator is None or denominator <= 0:
        # infer denominator from fractions if possible
        denoms = []
        for x in g_top.get("predicted_mrca_missingness_fraction", []):
            m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", str(x))
            if m:
                denoms.append(int(m.group(2)))
        denominator = max(denoms) if denoms else 0
    return (len(tips) / denominator) if denominator else 0.0

def main() -> None:
    ap = argparse.ArgumentParser(description="Add row-specific recovery columns with union partial recovery; move deleted-tip JSON to last column.")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--add_summary_row", action="store_true")
    ap.add_argument("--observed_manifest", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)

    # These diagnostic drop columns were useful for debugging but should not appear in the final report.
    df = df.drop(columns=["mrca_drop_from_count", "mrca_drop_to_count", "mrca_drop_size"], errors="ignore")

    needed = ["obs_tree_id", "supported_family_rank", "predicted_mrca_missingness_fraction"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["supported_family_rank"] = pd.to_numeric(df["supported_family_rank"], errors="coerce")
    # If the input is accidentally an already summarized file, remove the old summary row before recomputing.
    df = df[df["supported_family_rank"].notna()].copy()
    df["_frac"] = df["predicted_mrca_missingness_fraction"].apply(parse_fraction)

    status_col = "Recovery status out (Y/N/Partial)"
    recovery_col = "Recovery fraction/Percentage"
    partial_union_col = "Partial recovery union/Percentage"
    df[status_col] = ""
    df[recovery_col] = ""
    df[partial_union_col] = ""

    tree_best_values = []
    tree_union_values = []

    # manifest deleted-tip denominators for union, if available
    denom_map: dict[str, int] = {}
    if args.observed_manifest is not None:
        manifest = pd.read_csv(args.observed_manifest)
        deleted_col = first_existing_col(manifest, ["deleted_tips_json", "deleted_tip_names_json", "deleted_tips", "deleted_tip_names", "removed_tips_json"])
        if deleted_col is not None and "obs_tree_id" in manifest.columns:
            denom_map = manifest.drop_duplicates("obs_tree_id").set_index("obs_tree_id")[deleted_col].apply(lambda x: len(parse_tip_list(x))).to_dict()

    for obs_tree_id, g in df.groupby("obs_tree_id", sort=False):
        g_top = g[g["supported_family_rank"] <= args.top_k].copy()
        if g_top.empty:
            g_top = g.copy()
        first_idx = df.index[df["obs_tree_id"] == obs_tree_id][0]

        best_frac = float(g_top["_frac"].max()) if len(g_top) else 0.0
        best_status = status_from_fraction(best_frac)
        tree_best_values.append(best_frac)

        # Row-specific recovery display:
        # - each positive row shows its own Y/Partial percentage
        # - if all top-k rows are zero, only the first row shows N / 0.0%
        for idx in g_top.index:
            frac = float(df.loc[idx, "_frac"])
            if frac >= 0.999999:
                df.loc[idx, status_col] = "Y"
                df.loc[idx, recovery_col] = percent_text(frac)
            elif frac > 0:
                df.loc[idx, status_col] = "Partial"
                df.loc[idx, recovery_col] = percent_text(frac)

        if best_status == "N":
            df.loc[first_idx, status_col] = "N"
            df.loc[first_idx, recovery_col] = percent_text(0.0)

        if best_status == "Y":
            union_value = 1.0
        elif best_status == "N":
            union_value = 0.0
        else:
            union_value = union_fraction_from_rows(g_top, denom_map.get(obs_tree_id))
        tree_union_values.append(union_value)
        df.loc[first_idx, partial_union_col] = percent_text(union_value)

    mean_best = sum(tree_best_values) / len(tree_best_values) if tree_best_values else 0.0
    mean_union = sum(tree_union_values) / len(tree_union_values) if tree_union_values else 0.0
    df = df.drop(columns=["_frac"])
    df = insert_after_column(df, "predicted_mrca_missingness_fraction", [status_col, recovery_col, partial_union_col])

    if args.observed_manifest is not None:
        df = add_tip_count_columns_first_row_only(df, args.observed_manifest)

    if args.add_summary_row:
        summary_row = {col: "" for col in df.columns}
        if "full_tree_id" in summary_row:
            summary_row["full_tree_id"] = "MEAN_PERCENTAGE_COVERAGE"
        if "obs_tree_id" in summary_row:
            summary_row["obs_tree_id"] = f"n_trees={len(tree_best_values)}"
        if "predicted_mrca_missingness_fraction" in summary_row:
            summary_row["predicted_mrca_missingness_fraction"] = "mean"
        summary_row[status_col] = "Mean"
        summary_row[recovery_col] = percent_text(mean_best)
        summary_row[partial_union_col] = percent_text(mean_union)
        df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

    # Move the detailed deleted-tip list to the final column for readability.
    if "predicted_mrca_deleted_tips_json" in df.columns:
        cols = [c for c in df.columns if c != "predicted_mrca_deleted_tips_json"] + ["predicted_mrca_deleted_tips_json"]
        df = df[cols]

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print("Saved:", args.out_csv)
    print("Tip/deleted counts are shown only on the first row of each observed tree.")
    print(f"Mean Recovery fraction/Percentage = {percent_text(mean_best)}")
    print(f"Mean Partial recovery union/Percentage = {percent_text(mean_union)}")

if __name__ == "__main__":
    main()
