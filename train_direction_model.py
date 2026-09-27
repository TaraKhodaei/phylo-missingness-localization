#!/usr/bin/env python3
"""
Final Step-2 recursive direction trainer.

Purpose
-------
Train the best practical left/right direction model from the experiments so far.
This script is conservative and reproducible:
  1. uses the original recursive-direction CSV format,
  2. compares a small set of strong candidates only,
  3. prints train/val/test direction accuracy plus true-smaller/true-larger diagnostics,
  4. saves a Step-3-compatible direction_model.pkl.

Important
---------
By default, the model is selected by validation direction accuracy.
For retrospective analysis only, you can select by test accuracy using:
  --select_by test
Do not use --select_by test for a final unbiased report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

META_COLS = {
    "full_tree_id", "obs_tree_id", "decision_node_id", "child_node_id",
    "target_mrca_node_id", "label", "split"
}

RAW_ABSOLUTE_FEATURES = {
    "observed_root_n_tips",
    "decision_node_n_tips",
    "n_tips",
    "subtree_height",
    "total_branch_length",
    "mean_branch_length",
    "var_branch_length",
    "mean_terminal_branch_length",
    "max_terminal_branch_length",
}

RAW_RATIO_PREFIXES = ("ratio_vs_sibling_",)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    features = []
    for c in df.columns:
        if c in META_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            features.append(c)
    return features


def split_by_full_tree(df: pd.DataFrame, group_col: str, train_frac: float, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    groups = np.array(sorted(df[group_col].astype(str).unique()))
    rng.shuffle(groups)
    n = len(groups)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    train_g = set(groups[:n_train])
    val_g = set(groups[n_train:n_train + n_val])
    test_g = set(groups[n_train + n_val:])

    def assign(g: Any) -> str:
        g = str(g)
        if g in train_g:
            return "train"
        if g in val_g:
            return "val"
        return "test"

    out = df.copy()
    out["split"] = out[group_col].astype(str).map(assign)
    split_groups = {"train": sorted(train_g), "val": sorted(val_g), "test": sorted(test_g)}
    return out, split_groups


def build_pair_truth(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    skipped = {"bad_pair_size": 0, "bad_label_sum": 0, "missing_n_tips": 0, "equal_tip_size": 0}

    for (obs_id, dec_id), g in df.groupby(["obs_tree_id", "decision_node_id"], sort=False):
        if len(g) != 2:
            skipped["bad_pair_size"] += 1
            continue
        if int(g["label"].sum()) != 1:
            skipped["bad_label_sum"] += 1
            continue
        if "n_tips" not in g.columns or g["n_tips"].isna().any():
            skipped["missing_n_tips"] += 1
            continue

        gg = g.sort_values("n_tips", ascending=True)
        small = gg.iloc[0]
        large = gg.iloc[1]
        if float(small["n_tips"]) == float(large["n_tips"]):
            skipped["equal_tip_size"] += 1
            # equal size is not smaller; keep as true_smaller=False diagnostic

        true_row = g[g["label"].astype(int) == 1].iloc[0]
        true_smaller = bool(float(true_row["n_tips"]) < float(g[g["label"].astype(int) == 0].iloc[0]["n_tips"]))

        rows.append({
            "obs_tree_id": obs_id,
            "decision_node_id": dec_id,
            "split": str(g["split"].iloc[0]),
            "true_smaller": int(true_smaller),
        })

    out = pd.DataFrame(rows)
    print(f"Pair truth built: {len(out)} decision pairs | skipped={skipped}")
    for s in ["train", "val", "test"]:
        sub = out[out["split"] == s]
        if len(sub):
            print(f"  {s:5s}: pairs={len(sub):5d}, true_smaller_rate={sub['true_smaller'].mean():.3f}")
    return out


def pair_weight_for_rows(df: pd.DataFrame, pair_truth: pd.DataFrame, true_smaller_weight: float) -> np.ndarray:
    key_to_small = {
        (r.obs_tree_id, r.decision_node_id): int(r.true_smaller)
        for r in pair_truth.itertuples(index=False)
    }
    weights = []
    for r in df[["obs_tree_id", "decision_node_id"]].itertuples(index=False):
        is_small_decision = key_to_small.get((r.obs_tree_id, r.decision_node_id), 0)
        weights.append(float(true_smaller_weight) if is_small_decision else 1.0)
    return np.asarray(weights, dtype=float)


def make_model(model_name: str, seed: int) -> Pipeline:
    if model_name == "histgb_smooth":
        clf = HistGradientBoostingClassifier(
            max_iter=220, learning_rate=0.035, max_leaf_nodes=15,
            min_samples_leaf=45, l2_regularization=0.8, random_state=seed
        )
    elif model_name == "histgb_light":
        clf = HistGradientBoostingClassifier(
            max_iter=180, learning_rate=0.035, max_leaf_nodes=12,
            min_samples_leaf=55, l2_regularization=1.0, random_state=seed
        )
    elif model_name == "rf_leaf5":
        clf = RandomForestClassifier(
            n_estimators=700, max_features="sqrt", min_samples_leaf=5,
            min_samples_split=10, class_weight="balanced_subsample",
            random_state=seed, n_jobs=-1
        )
    elif model_name == "rf_leaf10":
        clf = RandomForestClassifier(
            n_estimators=700, max_features="sqrt", min_samples_leaf=10,
            min_samples_split=20, class_weight="balanced_subsample",
            random_state=seed, n_jobs=-1
        )
    elif model_name == "extra_leaf10":
        clf = ExtraTreesClassifier(
            n_estimators=900, max_features="sqrt", min_samples_leaf=10,
            min_samples_split=20, class_weight="balanced", random_state=seed, n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", clf),
    ])


def evaluate_direction(df: pd.DataFrame, probs: np.ndarray, pair_truth: pd.DataFrame) -> dict[str, Any]:
    y = df["label"].astype(int).to_numpy()
    pred = (probs >= 0.5).astype(int)
    out: dict[str, Any] = {
        "row_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "row_precision": float(precision_score(y, pred, zero_division=0)),
        "row_recall": float(recall_score(y, pred, zero_division=0)),
        "row_f1": float(f1_score(y, pred, zero_division=0)),
        "row_roc_auc": float(roc_auc_score(y, probs)) if len(np.unique(y)) == 2 else None,
        "row_average_precision": float(average_precision_score(y, probs)) if len(np.unique(y)) == 2 else None,
    }

    tmp = df[["obs_tree_id", "decision_node_id", "label", "n_tips"]].copy()
    tmp["prob"] = probs
    small_map = {
        (r.obs_tree_id, r.decision_node_id): int(r.true_smaller)
        for r in pair_truth.itertuples(index=False)
    }

    total = correct = 0
    small_total = small_correct = 0
    large_total = large_correct = 0
    margins = []

    for (obs_id, dec_id), g in tmp.groupby(["obs_tree_id", "decision_node_id"], sort=False):
        if len(g) != 2 or int(g["label"].sum()) != 1:
            continue
        p1 = float(g.loc[g["label"] == 1, "prob"].iloc[0])
        p0 = float(g.loc[g["label"] == 0, "prob"].iloc[0])
        is_correct = int(p1 > p0)
        is_true_smaller = int(small_map.get((obs_id, dec_id), 0))
        total += 1
        correct += is_correct
        margins.append(p1 - p0)
        if is_true_smaller:
            small_total += 1
            small_correct += is_correct
        else:
            large_total += 1
            large_correct += is_correct

    out.update({
        "direction_choice_accuracy": float(correct / total) if total else np.nan,
        "true_smaller_accuracy": float(small_correct / small_total) if small_total else np.nan,
        "true_larger_accuracy": float(large_correct / large_total) if large_total else np.nan,
        "n_decisions": int(total),
        "n_true_smaller": int(small_total),
        "n_true_larger": int(large_total),
        "mean_prob_margin": float(np.mean(margins)) if margins else np.nan,
        "median_prob_margin": float(np.median(margins)) if margins else np.nan,
    })
    return out


def make_feature_sets(train: pd.DataFrame, all_features: list[str], seed: int) -> dict[str, list[str]]:
    contrast_norm = [
        f for f in all_features
        if f not in RAW_ABSOLUTE_FEATURES and not f.startswith(RAW_RATIO_PREFIXES)
    ]

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(train[all_features])
    y = train["label"].astype(int).to_numpy()
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=seed)
    top_idx = np.argsort(mi)[::-1]
    top12 = [all_features[i] for i in top_idx[:12]]
    top20 = [all_features[i] for i in top_idx[:20]]

    return {
        "all_original52": all_features,
        "contrast_normalized_no_raw_ratios": contrast_norm,
        "train_mi_top12": top12,
        "train_mi_top20": top20,
    }


def flatten(candidate_name: str, split_name: str, metrics: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    row = {"candidate": candidate_name, "split": split_name}
    row.update(meta)
    row.update(metrics)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--group_col", default="full_tree_id")
    ap.add_argument("--train_frac", type=float, default=0.60)
    ap.add_argument("--val_frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--select_by", choices=["val", "test"], default="val",
                    help="Select final model by validation accuracy (recommended) or test accuracy (retrospective only).")
    ap.add_argument("--force_candidate", default=None,
                    help="Optional exact candidate name to save, e.g. histgb_smooth_all_w125.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    for c in [args.group_col, "obs_tree_id", "decision_node_id", "child_node_id", "label", "n_tips"]:
        if c not in df.columns:
            raise ValueError(f"Required column missing: {c}")

    df, split_groups = split_by_full_tree(df, args.group_col, args.train_frac, args.val_frac, args.seed)
    all_features = get_feature_columns(df)
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()
    pair_truth = build_pair_truth(df)
    feature_sets = make_feature_sets(train, all_features, args.seed)

    candidates = [
        # Best stable candidates from prior runs
        {"name": "histgb_smooth_all_w1",   "model": "histgb_smooth", "feature_set": "all_original52", "true_smaller_weight": 1.0},
        {"name": "histgb_smooth_all_w125", "model": "histgb_smooth", "feature_set": "all_original52", "true_smaller_weight": 1.25},
        {"name": "histgb_light_all_w125",  "model": "histgb_light",  "feature_set": "all_original52", "true_smaller_weight": 1.25},
        {"name": "rf_leaf5_all_w1",        "model": "rf_leaf5",      "feature_set": "all_original52", "true_smaller_weight": 1.0},
        {"name": "rf_leaf5_contrast_w1",   "model": "rf_leaf5",      "feature_set": "contrast_normalized_no_raw_ratios", "true_smaller_weight": 1.0},
        {"name": "rf_leaf5_top12_w1",      "model": "rf_leaf5",      "feature_set": "train_mi_top12", "true_smaller_weight": 1.0},
        {"name": "rf_leaf5_top20_w1",      "model": "rf_leaf5",      "feature_set": "train_mi_top20", "true_smaller_weight": 1.0},
        {"name": "rf_leaf10_all_w125",     "model": "rf_leaf10",     "feature_set": "all_original52", "true_smaller_weight": 1.25},
        {"name": "extra_leaf10_all_w125",  "model": "extra_leaf10",  "feature_set": "all_original52", "true_smaller_weight": 1.25},
    ]

    print("\n=========================================================")
    print("FINAL RECURSIVE DIRECTION BEST-MODEL SELECTION V2")
    print("=========================================================")
    print(f"Input: {args.input_csv}")
    print(f"Rows: {len(df)} | trees: {df['obs_tree_id'].nunique()} | decisions: {df[['obs_tree_id','decision_node_id']].drop_duplicates().shape[0]}")
    print(f"Numeric features: {len(all_features)}")
    print(f"Rows train/val/test: {len(train)} / {len(val)} / {len(test)}")
    print(f"Selection mode: {args.select_by}{' (RETROSPECTIVE, NOT UNBIASED)' if args.select_by == 'test' else ''}")
    print("\nFeature sets:")
    for name, feats in feature_sets.items():
        print(f"  {name:36s} {len(feats):3d}")
    print("\nCandidates:")
    for c in candidates:
        print(f"  {c['name']:28s} model={c['model']}, features={c['feature_set']}, true_smaller_weight={c['true_smaller_weight']}")

    trained = {}
    rows = []

    for i, cand in enumerate(candidates, start=1):
        cname = cand["name"]
        feats = feature_sets[cand["feature_set"]]
        model = make_model(cand["model"], args.seed)
        w_train = pair_weight_for_rows(train, pair_truth, cand["true_smaller_weight"])
        model.fit(train[feats], train["label"].astype(int), classifier__sample_weight=w_train)

        probs = {
            "train": model.predict_proba(train[feats])[:, 1],
            "val": model.predict_proba(val[feats])[:, 1],
            "test": model.predict_proba(test[feats])[:, 1],
        }
        split_df = {"train": train, "val": val, "test": test}
        metrics = {s: evaluate_direction(split_df[s], probs[s], pair_truth) for s in ["train", "val", "test"]}
        trained[cname] = {"model": model, "features": feats, "candidate": cand, "metrics": metrics}
        for s in ["train", "val", "test"]:
            rows.append(flatten(cname, s, metrics[s], cand))

        print(f"\n[{i}] {cname} | n_features={len(feats)}")
        for s in ["train", "val", "test"]:
            m = metrics[s]
            print(
                f"  {s:5s}: overall={m['direction_choice_accuracy']:.4f} "
                f"small={m['true_smaller_accuracy']:.4f} large={m['true_larger_accuracy']:.4f}"
            )

    summary = pd.DataFrame(rows)
    summary_path = outdir / "final_direction_model_v2_summary.csv"
    summary.to_csv(summary_path, index=False)

    pivot = summary.pivot_table(index="candidate", columns="split", values="direction_choice_accuracy", aggfunc="first").reset_index()
    pivot = pivot.merge(summary[summary["split"] == "val"][["candidate", "model", "feature_set", "true_smaller_weight"]], on="candidate", how="left")
    pivot = pivot.sort_values([args.select_by, "val"], ascending=False)
    best_by_val = pivot.sort_values(["val", "test"], ascending=False)
    best_by_test = pivot.sort_values(["test", "val"], ascending=False)
    best_by_val.to_csv(outdir / "final_direction_model_v2_best_by_val.csv", index=False)
    best_by_test.to_csv(outdir / "final_direction_model_v2_best_by_test.csv", index=False)

    if args.force_candidate:
        if args.force_candidate not in trained:
            raise ValueError(f"force_candidate not found: {args.force_candidate}. Available: {list(trained)}")
        selected_name = args.force_candidate
    else:
        selected_name = str(pivot.iloc[0]["candidate"])

    selected = trained[selected_name]
    selected_metrics = selected["metrics"]
    selected_cand = selected["candidate"]

    payload = {
        "model": selected["model"],
        "feature_columns": selected["features"],
        "model_name": selected_name,
        "selection_mode": args.select_by,
        "selected_candidate": selected_cand,
        "split_groups": split_groups,
        "metrics": selected_metrics,
        "all_candidate_summary_file": str(summary_path),
        "note": "Step-3-compatible payload: use payload['model'].predict_proba(X[payload['feature_columns']])[:,1].",
    }
    joblib.dump(payload, outdir / "direction_model.pkl")
    (outdir / "selected_model_diagnostics.json").write_text(json.dumps(selected_metrics, indent=2))
    (outdir / "selected_feature_columns.json").write_text(json.dumps(selected["features"], indent=2))

    print("\n=========================================================")
    print("SELECTED FINAL MODEL V2")
    print("=========================================================")
    print(f"Selected candidate      : {selected_name}")
    print(f"Model type              : {selected_cand['model']}")
    print(f"Feature set             : {selected_cand['feature_set']} ({len(selected['features'])} features)")
    print(f"True-smaller weight     : {selected_cand['true_smaller_weight']}")
    print(f"Selection mode          : {args.select_by}")
    for s in ["train", "val", "test"]:
        m = selected_metrics[s]
        print(
            f"{s:5s}: overall={m['direction_choice_accuracy']:.4f} | "
            f"true_smaller={m['true_smaller_accuracy']:.4f} (n={m['n_true_smaller']}) | "
            f"true_larger={m['true_larger_accuracy']:.4f} (n={m['n_true_larger']})"
        )

    print("\nBest by validation:")
    print(best_by_val[["candidate", "train", "val", "test", "model", "feature_set", "true_smaller_weight"]].head(5).to_string(index=False))
    print("\nBest observed test (retrospective only):")
    print(best_by_test[["candidate", "train", "val", "test", "model", "feature_set", "true_smaller_weight"]].head(5).to_string(index=False))

    print("\nSaved:")
    print(f"  {outdir / 'direction_model.pkl'}")
    print(f"  {summary_path}")
    print(f"  {outdir / 'final_direction_model_v2_best_by_val.csv'}")
    print(f"  {outdir / 'final_direction_model_v2_best_by_test.csv'}")


if __name__ == "__main__":
    main()
