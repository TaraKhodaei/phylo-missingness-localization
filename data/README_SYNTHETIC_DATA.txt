SYNTHETIC TEST DATA — NOT GISAID DATA

This folder contains eight fully artificial rooted binary phylogenetic trees in Newick (.nwk) format. The trees were created for testing the repository's workflow and contain no GISAID sequences, reconstructed GISAID-derived trees, or empirical pathogen data.

Subdirectories: Test_Tree_01 through Test_Tree_08.
Tip labels: tip_001, tip_002, etc. (unique within each tree).
Number of tips: 55, 65, 78, 92, 108, 123, 137, 149.

From the repository root, run Step 1:
  python compute_empirical.py --data_dir data

This generates empirical_tree_summaries.csv. Then run Step 2:
  Rscript simulate_full_trees.R empirical_tree_summaries.csv 5000 50 150 70

IMPORTANT: These test trees demonstrate how to run the pipeline, but using them in place of the original empirical trees will not reproduce the exact quantitative results in the manuscript.
