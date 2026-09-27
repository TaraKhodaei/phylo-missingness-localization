library(ape)

# =========================================================
# Simulate full phylogenetic trees from empirical summaries
# =========================================================
# Usage:
#   Rscript simulate_full_trees.R empirical_tree_summaries.csv 100
#   Rscript simulate_full_trees.R empirical_tree_summaries.csv 100 50
#   Rscript simulate_full_trees.R empirical_tree_summaries.csv 100 50 150
#   Rscript simulate_full_trees.R empirical_tree_summaries.csv 100 50 150 70
#
# Arguments:
#   1. empirical summary CSV
#   2. number of full trees to simulate
#   3. optional minimum number of tips in generated full trees
#   4. optional maximum number of tips in generated full trees
#   5. optional uniform_percent for tip-count sampling
#      Example: 70 means 70% uniform sampling and 30% empirical+jitter sampling.
#      Default: 50.
#
# Example forcing 50 to 150 tips with 70% uniform sampling:
#   Rscript simulate_full_trees.R empirical_tree_summaries.csv 100 50 150 70

# -----------------------------
# User settings
# -----------------------------
args <- commandArgs(trailingOnly = TRUE)

summary_csv <- if (length(args) >= 1) args[1] else "empirical_tree_summaries.csv"
n_reps <- if (length(args) >= 2) as.integer(args[2]) else 100
tip_min_floor <- if (length(args) >= 3) as.integer(args[3]) else NA_integer_
tip_max_ceiling <- if (length(args) >= 4) as.integer(args[4]) else NA_integer_
uniform_percent <- if (length(args) >= 5) as.numeric(args[5]) else 50

if (is.na(n_reps) || n_reps < 1) {
  stop("n_reps must be a positive integer.")
}
if (!is.na(tip_min_floor) && tip_min_floor < 2) {
  stop("tip_min_floor must be at least 2.")
}
if (!is.na(tip_max_ceiling) && tip_max_ceiling < 2) {
  stop("tip_max_ceiling must be at least 2.")
}
if (!is.na(tip_min_floor) && !is.na(tip_max_ceiling) && tip_min_floor > tip_max_ceiling) {
  stop("tip_min_floor cannot be greater than tip_max_ceiling.")
}
if (is.na(uniform_percent) || uniform_percent < 0 || uniform_percent > 100) {
  stop("uniform_percent must be a number between 0 and 100. Example: 70 means 70% uniform and 30% empirical.")
}

summary_csv_abs <- normalizePath(summary_csv, winslash = "/", mustWork = TRUE)
project_dir <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)

base_dir <- "simulations"
outdir <- file.path(base_dir, "full_trees")
manifest_dir <- file.path(base_dir, "manifests")
manifest_file <- file.path(manifest_dir, "full_tree_manifest.csv")

dir.create(base_dir, showWarnings = FALSE)
dir.create(manifest_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

cat("Using empirical summary file:", summary_csv_abs, "\n")
cat("Number of replicates:", n_reps, "\n")
cat("Minimum generated tips:", ifelse(is.na(tip_min_floor), "not forced", tip_min_floor), "\n")
cat("Maximum generated tips:", ifelse(is.na(tip_max_ceiling), "not forced", tip_max_ceiling), "\n")
cat("Output full trees directory:", outdir, "\n")
cat("Manifest file:", manifest_file, "\n")
cat("Tip sampling strategy:\n")
cat(" ", uniform_percent, "% uniform sampling\n", sep = "")
cat(" ", 100 - uniform_percent, "% filtered empirical+jitter sampling\n", sep = "")
cat("  No hard clipping to min/max. Invalid empirical/jittered values are rejected.\n")

# -----------------------------
# Regime library
# -----------------------------
sample_regime <- function() {
  regime_id <- sample(1:4, 1)

  if (regime_id == 1) {
    list(regime_name = "const_moderate", birth = 1.0, death = 0.3)
  } else if (regime_id == 2) {
    list(regime_name = "const_high_growth", birth = 1.5, death = 0.4)
  } else if (regime_id == 3) {
    list(regime_name = "const_high_turnover", birth = 1.4, death = 0.9)
  } else {
    list(regime_name = "const_low_turnover", birth = 0.8, death = 0.2)
  }
}

# -----------------------------
# Call Python hybrid sampler
# -----------------------------
sample_targets_from_python <- function(summary_csv_path, project_dir_path, tip_min_floor, tip_max_ceiling, uniform_percent) {
  tmp_py <- tempfile(fileext = ".py")

  tip_min_py <- if (is.na(tip_min_floor)) "None" else as.character(tip_min_floor)
  tip_max_py <- if (is.na(tip_max_ceiling)) "None" else as.character(tip_max_ceiling)
  uniform_percent_py <- as.character(uniform_percent)

  py_lines <- c(
    "import sys",
    sprintf("sys.path.insert(0, %s)", shQuote(project_dir_path, type = "cmd")),
    "from hybrid_sampling import HybridSampler",
    sprintf(
      "s = HybridSampler(summary_csv=%s, tip_min_floor=%s, tip_max_ceiling=%s, uniform_percent=%s)",
      shQuote(summary_csv_path, type = "cmd"),
      tip_min_py,
      tip_max_py,
      uniform_percent_py
    ),
    "n_target, tip_source = s.sample_n_tips_with_source()",
    "print(str(n_target) + ',' + str(s.sample_tree_height()) + ',' + str(tip_source))"
  )

  writeLines(py_lines, tmp_py)

  result <- system2(
    command = "python",
    args = c(tmp_py),
    stdout = TRUE,
    stderr = TRUE
  )

  unlink(tmp_py)

  if (length(result) == 0) {
    stop("Python sampler returned no output.")
  }

  last_line <- tail(result, 1)
  parts <- strsplit(last_line, ",")[[1]]

  if (length(parts) != 3) {
    stop("Unexpected sampler output:\n", paste(result, collapse = "\n"))
  }

  n_target <- as.integer(parts[1])
  h_target <- as.numeric(parts[2])
  tip_source <- trimws(parts[3])

  if (is.na(n_target) || is.na(h_target) || !(tip_source %in% c("uniform", "empirical_jitter"))) {
    stop("Could not parse sampler output:\n", paste(result, collapse = "\n"))
  }
  if (!is.na(tip_min_floor) && n_target < tip_min_floor) {
    stop("Sampler returned n_target below tip_min_floor: ", n_target, " < ", tip_min_floor)
  }
  if (!is.na(tip_max_ceiling) && n_target > tip_max_ceiling) {
    stop("Sampler returned n_target above tip_max_ceiling: ", n_target, " > ", tip_max_ceiling)
  }

  list(n_target = n_target, h_target = h_target, tip_source = tip_source)
}

# -----------------------------
# Tree height computation
# -----------------------------
get_tree_height <- function(tr) {
  max(node.depth.edgelength(tr))
}

# -----------------------------
# Manifest storage
# -----------------------------
manifest <- data.frame(
  tree_id = character(),
  tree_file = character(),
  summary_csv = character(),
  n_target = integer(),
  h_target = numeric(),
  tip_min_floor = integer(),
  tip_max_ceiling = integer(),
  uniform_percent = numeric(),
  empirical_percent = numeric(),
  tip_sampling_source = character(),
  regime_name = character(),
  birth = numeric(),
  death = numeric(),
  h_sim_before_scaling = numeric(),
  scale_factor = numeric(),
  n_tips_sim = integer(),
  h_final = numeric(),
  stringsAsFactors = FALSE
)

# -----------------------------
# Main simulation loop
# -----------------------------
pb <- txtProgressBar(min = 0, max = n_reps, style = 3)

for (i in seq_len(n_reps)) {

  targets <- sample_targets_from_python(summary_csv_abs, project_dir, tip_min_floor, tip_max_ceiling, uniform_percent)
  n_target <- targets$n_target
  h_target <- targets$h_target
  tip_source <- targets$tip_source

  regime <- sample_regime()

  tr <- rphylo(
    n = n_target,
    birth = regime$birth,
    death = regime$death
  )

  h_sim <- get_tree_height(tr)

  if (is.na(h_sim) || h_sim <= 1e-12) {
    setTxtProgressBar(pb, i)
    warning("Skipping tree ", i, " due to invalid simulated height.")
    next
  }

  scale_factor <- h_target / h_sim
  tr$edge.length <- tr$edge.length * scale_factor

  h_final <- get_tree_height(tr)
  n_tips_sim <- length(tr$tip.label)

  if (!is.na(tip_min_floor) && n_tips_sim < tip_min_floor) {
    stop("Generated tree has fewer tips than requested: ", n_tips_sim, " < ", tip_min_floor)
  }
  if (!is.na(tip_max_ceiling) && n_tips_sim > tip_max_ceiling) {
    stop("Generated tree has more tips than requested: ", n_tips_sim, " > ", tip_max_ceiling)
  }

  tree_id <- sprintf("full_tree_%05d", i)
  tree_file <- file.path(outdir, paste0(tree_id, ".nwk"))

  write.tree(tr, file = tree_file)

  manifest <- rbind(
    manifest,
    data.frame(
      tree_id = tree_id,
      tree_file = tree_file,
      summary_csv = summary_csv_abs,
      n_target = n_target,
      h_target = h_target,
      tip_min_floor = ifelse(is.na(tip_min_floor), NA_integer_, tip_min_floor),
      tip_max_ceiling = ifelse(is.na(tip_max_ceiling), NA_integer_, tip_max_ceiling),
      uniform_percent = uniform_percent,
      empirical_percent = 100 - uniform_percent,
      tip_sampling_source = tip_source,
      regime_name = regime$regime_name,
      birth = regime$birth,
      death = regime$death,
      h_sim_before_scaling = h_sim,
      scale_factor = scale_factor,
      n_tips_sim = n_tips_sim,
      h_final = h_final,
      stringsAsFactors = FALSE
    )
  )

  setTxtProgressBar(pb, i)
}

close(pb)
cat("\n")

# -----------------------------
# Save manifest
# -----------------------------
write.csv(manifest, manifest_file, row.names = FALSE)

cat("\nDone. Manifest saved to:", manifest_file, "\n")
cat("Number of trees written:", nrow(manifest), "\n")
if (nrow(manifest) > 0) {
  cat("Generated tip range:", min(manifest$n_tips_sim), "to", max(manifest$n_tips_sim), "\n")
  cat("Generated height range:", min(manifest$h_final), "to", max(manifest$h_final), "\n")
}
