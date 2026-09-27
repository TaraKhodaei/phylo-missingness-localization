import os
import numpy as np
import pandas as pd
from ete3 import Tree
import argparse

"""
Compute empirical summary statistics from real county-level phylogenetic trees.

Description
-----------
This script reads all phylogenetic trees (.nwk files) stored in subdirectories
of the "data/" folder, where each subdirectory corresponds to a county.

For each tree, it computes a set of global structural summaries that characterize
the tree’s size, branch-length distribution, and depth. These summaries define
the empirical distributions used later to guide the simulation of realistic
phylogenetic trees.

The computed statistics include:
    - Number of tips (leaves)
    - Tree height (maximum root-to-tip distance)
    - Total branch length
    - Mean branch length
    - Variance of branch lengths
    - Mean terminal branch length (leaf edges)
    - Mean internal branch length (internal edges)

Output
------
A CSV file named "empirical_tree_summaries.csv" containing one row per county tree,
with all computed statistics.

Notes
-----
- FASTA files are ignored; only .nwk tree files are used.
- Trees are assumed to be rooted (as in dataset).
- This step corresponds to Section 1.1 (Empirical Characterization of Real Trees)
  in the simulation design document.
"""

# -----------------------------
# Argument 
# -----------------------------
parser = argparse.ArgumentParser(
    description="Compute empirical tree summaries from real trees"
)

parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="Path to data directory containing county subfolders"
)

args = parser.parse_args()


DATA_DIR = args.data_dir


results = []

def get_tree_height(tree):
    """
    Compute tree height as the maximum distance from root to any leaf.
    """
    return max(tree.get_distance(leaf) for leaf in tree.iter_leaves())

for county in os.listdir(DATA_DIR):
    county_path = os.path.join(DATA_DIR, county)

    if not os.path.isdir(county_path):
        continue

    # Find the .nwk file in the county folder
    tree_file = None
    for f in os.listdir(county_path):
        if f.endswith(".nwk"):
            tree_file = os.path.join(county_path, f)
            break

    if tree_file is None:
        print(f"⚠️ No tree file in {county}")
        continue

    try:
        tree = Tree(tree_file, format=1)
    except Exception as e:
        print(f"❌ Error reading {tree_file}: {e}")
        continue

    # --- Compute statistics ---
    n_tips = len(tree.get_leaves())

    branch_lengths = []
    terminal_branches = []
    internal_branches = []

    for node in tree.traverse():
        if node.is_root():
            continue

        bl = node.dist
        branch_lengths.append(bl)

        if node.is_leaf():
            terminal_branches.append(bl)
        else:
            internal_branches.append(bl)

    tree_height = get_tree_height(tree)

    results.append({
        "county": county,
        "n_tips": n_tips,
        "tree_height": tree_height,
        "total_branch_length": np.sum(branch_lengths),
        "mean_branch_length": np.mean(branch_lengths),
        "var_branch_length": np.var(branch_lengths),
        "mean_terminal_bl": np.mean(terminal_branches) if terminal_branches else np.nan,
        "mean_internal_bl": np.mean(internal_branches) if internal_branches else np.nan,
    })

# Convert to DataFrame
df = pd.DataFrame(results)

# Save results
df.to_csv("empirical_tree_summaries.csv", index=False)

print("\n✅ Done! Saved as empirical_tree_summaries.csv")
print(df.head())
