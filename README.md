# Phylogenetic Missingness Localization

Machine-learning framework for localizing phylogenetically localized unsampled regions in rooted pathogen phylogenies.

This repository contains the code associated with the manuscript:

**Localizing Unsampled Regions in Pathogen Phylogenies Using Machine Learning**

**Authors:** Marzieh Khodaei, Nathaniel Hupert, and Arindam RoyChoudhury

## Overview

This repository contains the code for a machine-learning framework that localizes regions of missingness in rooted pathogen phylogenies. The method combines a recursive direction classifier with beam search to identify candidate paths and localize the missing region. Training and evaluation use simulated trees with controlled tip deletion, while inference requires only the observed phylogeny.

## Repository contents

| File | Purpose |
|---|---|
| `compute_empirical.py` | Summarize empirical rooted phylogenies used to parameterize simulations. |
| `hybrid_sampling.py` | Helper module used during empirical-matched simulation. |
| `simulate_full_trees.R` | Simulate complete rooted phylogenies over the target tree-size range. |
| `inject_local_missingness.py` | Introduce localized missingness by deleting descendants from an eligible internal lineage. |
| `split_simulated_dataset.py` | Split simulated trees into training, validation, and test sets while preventing leakage across trees. |
| `build_direction_training_data.py` | Build the recursive direction-classification dataset from observed phylogenies. |
| `train_direction_model.py` | Train and select the recursive direction classifier. |
| `generate_beam_paths.py` | Run beam search and calculate node recurrence/support across retained paths. |
| `select_representative_paths_and_mrcas.py` | Select representative localization paths and estimate candidate MRCAs. |
| `evaluate_localization_recovery.py` | Evaluate localization recovery for the selected candidate paths. |
| `select_raw_beam_paths.py` | Construct a raw top-beam baseline for comparison. |
| `plot_mrca_position_distribution.py` | Plot the topological relationship between predicted and true MRCAs. |
| `infer_missing_regions.py` | Apply the trained framework to an observed rooted phylogeny. |
| `evaluate_controlled_deletion.py` | Evaluate predictions when a complete tree and controlled deletion are available. |

## Main workflow

Run commands from the repository root so that the scripts can locate neighboring modules correctly.

### 1. Summarize empirical phylogenies

```bash
python compute_empirical.py --data_dir data
```

### 2. Simulate complete phylogenies

```bash
Rscript simulate_full_trees.R empirical_tree_summaries.csv 5000 50 150 70
```

### 3. Introduce localized missingness

```bash
python inject_local_missingness.py \
  --full_tree_dir simulations/full_trees \
  --full_manifest simulations/manifests/full_tree_manifest.csv \
  --n_obs_per_full 1 \
  --fraction_choices 0.2,0.4,0.6 \
  --min_subtree_tips 8 \
  --min_remaining_target_tips 4 \
  --min_target_frac 0.05 \
  --max_target_frac 0.7 \
  --outdir simulations/observed_trees
```

### 4. Create train/validation/test splits

```bash
python split_simulated_dataset.py \
  --observed_manifest simulations/manifests/observed_tree_manifest.csv \
  --outdir simulations/splits \
  --train_frac 0.60 \
  --val_frac 0.20 \
  --seed 1
```

### 5. Build the recursive direction dataset

```bash
python build_direction_training_data.py \
  --observed_manifest simulations/splits/observed_tree_manifest_with_splits.csv \
  --out_csv simulations/recursive_direction_dataset/recursive_direction_dataset_selected_targeted_features.csv \
  --write_debug_report
```

### 6. Train the direction classifier

```bash
python train_direction_model.py \
  --input_csv simulations/recursive_direction_dataset/recursive_direction_dataset_selected_targeted_features.csv \
  --outdir simulations/recursive_direction_training_outputs_best_model \
  --select_by val
```

### 7. Generate beam-search paths

```bash
python generate_beam_paths.py \
  --observed_manifest simulations/splits/observed_tree_manifest_with_splits.csv \
  --direction_model simulations/recursive_direction_training_outputs_best_model/direction_model.pkl \
  --recursive_direction_csv simulations/recursive_direction_dataset/recursive_direction_dataset_selected_targeted_features.csv \
  --outdir simulations/beam_search \
  --beam_width 50 \
  --top_final_paths 50 \
  --max_depth 20 \
  --min_candidate_obs_tips 4 \
  --write_debug_report
```

### 8. Select representative paths and candidate MRCAs

```bash
python select_representative_paths_and_mrcas.py \
  --beam_paths_csv simulations/beam_search/obs_tree_beam_paths.csv \
  --candidate_support_csv simulations/beam_search/candidate_support_counts.csv \
  --observed_manifest simulations/splits/observed_tree_manifest_with_splits.csv \
  --recursive_direction_csv simulations/recursive_direction_dataset/recursive_direction_dataset_selected_targeted_features.csv \
  --out_csv simulations/beam_search/representative_paths_test.csv \
  --split_filter test \
  --top_k 3
```

### 9. Evaluate localization recovery

```bash
python evaluate_localization_recovery.py \
  --input_csv simulations/beam_search/representative_paths_test.csv \
  --out_csv simulations/beam_search/representative_paths_test_with_recovery.csv \
  --observed_manifest simulations/splits/observed_tree_manifest_with_splits.csv \
  --add_summary_row
```

## Baseline comparison

For comparison with the representative-path selection procedure, the top three beam-search paths can be selected directly from the raw beam output. This provides the baseline used to assess whether representative-path selection improves localization over simply taking the highest-ranked beam paths.

```bash
python select_raw_beam_paths.py \
  --beam_paths_csv simulations/beam_search/obs_tree_beam_paths.csv \
  --observed_manifest simulations/splits/observed_tree_manifest_with_splits.csv \
  --recursive_direction_csv simulations/recursive_direction_dataset/recursive_direction_dataset_selected_targeted_features.csv \
  --out_csv simulations/beam_search/raw_beam_baseline_test.csv \
  --split_filter test \
  --top_k 3
  ```
The resulting file can be evaluated with `evaluate_localization_recovery.py`.

## MRCA localization plot

The following command summarizes the topological position of the predicted MRCA relative to the true MRCA across the test trees and generates the corresponding distribution plot.

```bash
python plot_mrca_position_distribution.py \
  --input_csv simulations/beam_search/representative_paths_test_with_recovery.csv \
  --outdir simulations/beam_search/final_mrca_plot
```

The generated figure is saved as `MRCA_distribution.pdf` in the specified output directory.

## Applying the trained framework to an observed phylogeny

Once the direction model has been trained, the framework can be applied to a rooted observed phylogeny to localize the missing region. The example below uses the Santa Cruz controlled-deletion case included in the study.

```bash
python infer_missing_regions.py \
  --observed_tree simulations/case_studies/SantaCruz/obs_SantaCruz_case10_node28_20pct.nwk \
  --direction_model simulations/recursive_direction_training_outputs_best_model/direction_model.pkl \
  --beam_module generate_beam_paths.py \
  --outdir simulations/case_studies/SantaCruz/results_case10_node28_20pct/inference \
  --tree_id SantaCruz_case10_node28_20pct \
  --beam_width 50 \
  --top_final_paths 50 \
  --max_depth 20 \
  --min_candidate_obs_tips 4 \
  --top_k 3
```



If the complete phylogeny is also available, as in a controlled-deletion experiment, the inferred missing region can be evaluated against the known deleted region:

```bash
python evaluate_controlled_deletion.py \
  --full_tree simulations/case_studies/SantaCruz/full_SantaCruz.nwk \
  --observed_tree simulations/case_studies/SantaCruz/obs_SantaCruz_case10_node28_20pct.nwk \
  --predictions_csv simulations/case_studies/SantaCruz/results_case10_node28_20pct/inference/predicted_missing_regions.csv \
  --expected_missing_fraction 0.2173913043 \
  --outdir simulations/case_studies/SantaCruz/results_case10_node28_20pct/evaluation
```



## Installation

The workflow requires Python 3.10 or later and R.

Clone the repository and move to the project directory:

```bash
git clone https://github.com/TaraKhodaei/phylo-missingness-localization.git
cd phylo-missingness-localization
```

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The tree-simulation step additionally requires the R package `ape`:

```r
install.packages("ape")
```

Run all analysis commands from the repository root.


## Citation

Citation information will be updated when the associated manuscript is publicly available.



## Contact

For questions about the method, code, or reproducibility, please open an issue in this repository.
