"""
Hybrid sampling utilities for phylogenetic tree simulation.

Corrected tip-count behavior
----------------------------
The old version sampled from the full empirical n_tips distribution and then
hard-clipped values into [tip_min_floor, tip_max_ceiling]. That caused many
small empirical trees, for example 13, 20, 42, 48, to collapse to exactly 50.

This corrected version does NOT hard-clip tip counts after sampling.
Instead, it first restricts the empirical tip-count distribution to the allowed
range and then samples/jitters only values that remain inside the requested
range. Therefore, asking for 50 to 150 tips no longer creates an artificial
spike at 50.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd


class HybridSampler:
    def __init__(
        self,
        summary_csv: str = "empirical_tree_summaries.csv",
        p_exact: float = 0.5,
        tip_jitter: int = 3,
        tip_min_floor: int | None = None,
        tip_max_ceiling: int | None = None,
        height_noise_frac: float = 0.08,
        height_lower_mult: float = 0.9,
        height_upper_mult: float = 1.1,
        random_seed: int | None = None,
        max_resample_attempts: int = 1000,
        uniform_percent: float = 50.0,
    ) -> None:
        self.df = pd.read_csv(summary_csv)

        if "n_tips" not in self.df.columns or "tree_height" not in self.df.columns:
            raise ValueError("CSV must contain 'n_tips' and 'tree_height' columns.")

        raw_tip_counts = self.df["n_tips"].dropna().astype(int).to_numpy()
        self.tree_heights = self.df["tree_height"].dropna().astype(float).to_numpy()

        if len(raw_tip_counts) == 0 or len(self.tree_heights) == 0:
            raise ValueError("Empirical distributions are empty.")

        if not (0.0 <= p_exact <= 1.0):
            raise ValueError("p_exact must be between 0 and 1.")
        if tip_jitter < 0:
            raise ValueError("tip_jitter must be >= 0.")
        if max_resample_attempts < 1:
            raise ValueError("max_resample_attempts must be >= 1.")
        if not (0.0 <= float(uniform_percent) <= 100.0):
            raise ValueError("uniform_percent must be between 0 and 100.")

        self.p_exact = float(p_exact)
        self.tip_jitter = int(tip_jitter)
        self.height_noise_frac = float(height_noise_frac)
        self.max_resample_attempts = int(max_resample_attempts)
        self.uniform_percent = float(uniform_percent)
        self.p_uniform = self.uniform_percent / 100.0

        emp_tip_min = int(np.min(raw_tip_counts))
        emp_tip_max = int(np.max(raw_tip_counts))

        self.tip_min = int(tip_min_floor) if tip_min_floor is not None else max(2, emp_tip_min - self.tip_jitter)
        self.tip_max = int(tip_max_ceiling) if tip_max_ceiling is not None else emp_tip_max + self.tip_jitter

        if self.tip_min < 2:
            self.tip_min = 2
        if self.tip_min > self.tip_max:
            raise ValueError(
                f"Invalid tip bounds: tip_min={self.tip_min} is greater than tip_max={self.tip_max}."
            )

        # Key correction: filter first, do not sample from too-small trees and clip them to 50.
        eligible = raw_tip_counts[(raw_tip_counts >= self.tip_min) & (raw_tip_counts <= self.tip_max)]

        if len(eligible) == 0:
            raise ValueError(
                "No empirical tip counts fall inside the requested range "
                f"[{self.tip_min}, {self.tip_max}]. Use a wider range or a different empirical CSV."
            )

        self.tip_counts = eligible.astype(int)

        emp_h_min = float(np.min(self.tree_heights))
        emp_h_max = float(np.max(self.tree_heights))
        self.height_min = float(height_lower_mult) * emp_h_min
        self.height_max = float(height_upper_mult) * emp_h_max

        self.rng = np.random.default_rng(random_seed)

    def _in_tip_bounds(self, n: int) -> bool:
        return self.tip_min <= int(n) <= self.tip_max

    def _sample_empirical_jittered_n_tips(self) -> int:
        """
        Sample from filtered empirical counts plus bounded jitter.

        Values outside the requested range are rejected/resampled, not clipped.
        """
        for _ in range(self.max_resample_attempts):
            base = int(self.rng.choice(self.tip_counts))

            if self.rng.random() < self.p_exact or self.tip_jitter == 0:
                n = base
            else:
                eps = int(self.rng.integers(-self.tip_jitter, self.tip_jitter + 1))
                n = base + eps

            if self._in_tip_bounds(n):
                return int(n)

        # Very rare fallback. This preserves validity without boundary clipping of invalid empirical draws.
        return int(self.rng.choice(self.tip_counts))

    def _sample_uniform_n_tips(self) -> int:
        """
        Sample uniformly across all integer tip counts in [tip_min, tip_max].
        """
        return int(self.rng.integers(self.tip_min, self.tip_max + 1))

    def sample_n_tips_with_source(self) -> tuple[int, str]:
        """
        Sample target tip count from a user-controlled mixture.

        With probability uniform_percent/100, sample uniformly from the full
        requested range. Otherwise, sample from the filtered empirical+jitter
        distribution. This avoids artificial boundary spikes while improving
        coverage across the requested tip-count range.
        """
        if self.rng.random() < self.p_uniform:
            return self._sample_uniform_n_tips(), "uniform"
        return self._sample_empirical_jittered_n_tips(), "empirical_jitter"

    def sample_n_tips(self) -> int:
        """
        Backward-compatible method returning only the sampled tip count.
        """
        n, _ = self.sample_n_tips_with_source()
        return n

    def sample_tree_height(self) -> float:
        """
        Sample target tree height using hybrid exact + smoothed empirical sampling.
        """
        base = float(self.rng.choice(self.tree_heights))

        if self.rng.random() < self.p_exact:
            return float(max(self.height_min, min(self.height_max, base)))

        noise_sd = self.height_noise_frac * base
        eta = float(self.rng.normal(0.0, noise_sd))
        h_target = base + eta
        h_target = max(self.height_min, min(self.height_max, h_target))
        return float(h_target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid sampling with corrected bounded tip sampling")
    parser.add_argument("--summary_csv", type=str, default="empirical_tree_summaries.csv")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tip_min_floor", type=int, default=None)
    parser.add_argument("--tip_max_ceiling", type=int, default=None)
    parser.add_argument("--n", type=int, default=10, help="Number of samples to print")
    parser.add_argument(
        "--uniform_percent",
        type=float,
        default=50.0,
        help="Percent of tip-count samples drawn uniformly from the requested range. Example: 70 means 70%% uniform and 30%% empirical+jitter.",
    )

    args = parser.parse_args()

    sampler = HybridSampler(
        summary_csv=args.summary_csv,
        random_seed=args.seed,
        tip_min_floor=args.tip_min_floor,
        tip_max_ceiling=args.tip_max_ceiling,
        uniform_percent=args.uniform_percent,
    )

    print("Sampled tip counts:")
    print([sampler.sample_n_tips_with_source() for _ in range(args.n)])

    print("\nSampled tree heights:")
    print([sampler.sample_tree_height() for _ in range(args.n)])
