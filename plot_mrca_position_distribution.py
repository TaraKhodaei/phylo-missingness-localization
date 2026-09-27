#!/usr/bin/env python3
"""
Plot the topological position of the predicted MRCA relative to the true MRCA.

For each test phylogeny, the best candidate among the top-k predictions is used.
Predictions on the same ancestral lineage are summarized by their topological
distance from the true MRCA.

Output:
    MRCA_distribution.pdf
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


TREE_COL = "obs_tree_id"
RANK_COL = "supported_family_rank"
FRACTION_COL = "predicted_mrca_missingness_fraction"
RELATION_COL = "mrca_relation"


def parse_fraction(x: Any) -> float:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return 0.0

    s = str(x).strip()
    if not s or s.lower() == "nan":
        return 0.0

    m = re.fullmatch(r"(\d+)\s*/\s*(\d+)", s)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        return num / den if den else 0.0

    try:
        if s.endswith("%"):
            return float(s[:-1]) / 100.0
        v = float(s)
        return v if v <= 1 else v / 100.0
    except Exception:
        return 0.0


def signed_distance_from_relation(x: Any) -> float:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return np.nan

    s = str(x).strip()

    mapping = {
        "exact_MRCA": 0.0,
        "parent": 1.0,
        "grandparent": 2.0,
        "first_child": -1.0,
        "grandchild": -2.0,
    }

    if s in mapping:
        return mapping[s]

    m = re.fullmatch(r"(\d+)parent", s)
    if m:
        return float(int(m.group(1)))

    m = re.fullmatch(r"(\d+)child", s)
    if m:
        return -float(int(m.group(1)))

    return np.nan


def load_best_prediction_per_tree(path: str, top_k: int) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = [TREE_COL, FRACTION_COL, RELATION_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    if RANK_COL in df.columns:
        df[RANK_COL] = pd.to_numeric(df[RANK_COL], errors="coerce")
        df = df[df[RANK_COL].notna()].copy()
        df = df[df[RANK_COL] <= top_k].copy()
    else:
        df[RANK_COL] = 999999

    df["coverage"] = df[FRACTION_COL].apply(parse_fraction)

    # Same tree-level choice used previously:
    # highest recovery, then lowest family rank.
    df = df.sort_values(
        [TREE_COL, "coverage", RANK_COL],
        ascending=[True, False, True],
    )

    best = df.drop_duplicates(TREE_COL, keep="first").copy()
    best["signed_distance"] = best[RELATION_COL].apply(
        signed_distance_from_relation
    )

    return best


def label_for_distance(d: int) -> str:
    if d == 0:
        return "Exact MRCA"

    if d < 0:
        k = abs(d)
        if k == 1:
            return "Child"
        if k == 2:
            return "Grandchild"
        if k == 3:
            return "3rd descendant"
        return f"{k}th descendant"

    if d == 1:
        return "Parent"
    if d == 2:
        return "Grandparent"
    if d == 3:
        return "3rd ancestor"
    return f"{d}th ancestor"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot the topological position of the predicted MRCA relative to the true MRCA."
    )
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--top_k", type=int, default=3)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    best = load_best_prediction_per_tree(args.input_csv, args.top_k)
    n_total = len(best)

    same = best[best["signed_distance"].notna()].copy()
    same["distance"] = same["signed_distance"].round().astype(int)

    wrong_n = n_total - len(same)

    # Exact category range as present in the data.
    min_d = int(same["distance"].min())
    max_d = int(same["distance"].max())

    rows = []

    # Group 5th descendant and deeper.
    n_lt5_desc = int((same["distance"] <= -5).sum())
    if n_lt5_desc > 0:
        rows.append({"distance": -5, "label": "<5th descendant", "n_trees": n_lt5_desc,
                     "percentage": 100.0 * n_lt5_desc / n_total, "side": "descendant"})

    for d in range(-4, 1):
        n = int((same["distance"] == d).sum())
        if n == 0:
            continue
        side = "descendant" if d < 0 else "exact"
        rows.append({"distance": d, "label": label_for_distance(d), "n_trees": n,
                     "percentage": 100.0 * n / n_total, "side": side})

    for d in range(1, 7):
        n = int((same["distance"] == d).sum())
        if n == 0:
            continue
        rows.append({"distance": d, "label": label_for_distance(d), "n_trees": n,
                     "percentage": 100.0 * n / n_total, "side": "ancestor"})

    # Group ancestors beyond the 6th.
    n_gt6_anc = int((same["distance"] > 6).sum())
    if n_gt6_anc > 0:
        rows.append({"distance": 7, "label": ">6th ancestor", "n_trees": n_gt6_anc,
                     "percentage": 100.0 * n_gt6_anc / n_total, "side": "ancestor"})

    plot_df = pd.DataFrame(rows)

    colors = {
        "descendant": "orange",
        "exact": "green",
        "ancestor": "gray",
    }

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig_width = max(10.0, 0.78 * len(plot_df) + 2.8)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))

    x = np.arange(len(plot_df))
    bar_colors = [colors[s] for s in plot_df["side"]]

    bars = ax.bar(
        x,
        plot_df["percentage"],
        width=0.72,
        color=bar_colors,
        edgecolor="none",
    )

    ax.set_title("Distribution of predicted MRCA", pad=12)
    ax.set_ylabel("Percentage of test phylogenies")

    ax.set_xticks(x)
    ax.set_xticklabels(
        plot_df["label"],
        rotation=38,
        ha="right",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18, linewidth=0.7)
    ax.set_axisbelow(True)

    ymax = max(float(plot_df["percentage"].max()), 1.0)
    ax.set_ylim(0, ymax * 1.20)

    for bar, pct in zip(bars, plot_df["percentage"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.02,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    legend_handles = [
        Patch(facecolor=colors["descendant"], label="Below true MRCA"),
        Patch(facecolor=colors["exact"], label="Exact MRCA"),
        Patch(facecolor=colors["ancestor"], label="Above true MRCA"),
    ]

    ax.legend(
        handles=legend_handles,
        frameon=False,
        ncol=3,
        loc="upper right",
    )

#    fig.text(
#        0.5,
#        0.018,
#        "Wrong-branch predictions are excluded because they do not have a defined "
#        "ancestor-descendant distance from the true MRCA.",
#        ha="center",
#        fontsize=8,
#    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    output = outdir / "MRCA_distribution.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
