"""Bootstrap confidence intervals and model comparison utilities."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_seed: int = 42,
) -> tuple[float, float, float]:
    """Estimate the mean and a percentile bootstrap confidence interval.

    Parameters
    ----------
    scores:
        A list of per-task scores (typically 0.0 or 1.0).
    n_bootstrap:
        Number of bootstrap resamples to draw.
    ci:
        Confidence level, e.g. 0.95 for a 95% CI.
    random_seed:
        Seed for the random number generator for reproducibility.

    Returns
    -------
    tuple[float, float, float]
        ``(mean, lower_bound, upper_bound)`` where bounds are the percentile
        bootstrap interval at the requested confidence level.
    """
    rng = np.random.default_rng(random_seed)
    arr = np.asarray(scores, dtype=float)
    mean: float = float(arr.mean())

    boot_means = np.empty(n_bootstrap)
    n = len(arr)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = 1.0 - ci
    lower: float = float(np.percentile(boot_means, 100 * alpha / 2))
    upper: float = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return mean, lower, upper


def bootstrap_comparison(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 1000,
    random_seed: int = 42,
) -> dict:
    """Test whether model A significantly outperforms model B.

    Uses a paired bootstrap: for each resample, draw the *same* indices from
    both score lists so that task-level correlation is preserved.

    Parameters
    ----------
    scores_a:
        Per-task scores for model A (0.0 or 1.0).
    scores_b:
        Per-task scores for model B (0.0 or 1.0).  Must be the same length as
        *scores_a* and aligned by task.
    n_bootstrap:
        Number of bootstrap resamples.
    random_seed:
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        - ``mean_a``  : float — observed mean of scores_a
        - ``mean_b``  : float — observed mean of scores_b
        - ``diff``    : float — observed difference mean_a - mean_b
        - ``p_value`` : float — fraction of bootstrap samples where diff ≤ 0
          (one-sided p-value for H0: A is not better than B)

    Raises
    ------
    ValueError
        If the two score lists have different lengths or are empty.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"scores_a and scores_b must have the same length "
            f"({len(scores_a)} vs {len(scores_b)})"
        )
    if not scores_a:
        raise ValueError("Score lists must not be empty.")

    rng = np.random.default_rng(random_seed)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)

    mean_a: float = float(a.mean())
    mean_b: float = float(b.mean())
    observed_diff: float = mean_a - mean_b

    n = len(a)
    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = a[idx].mean() - b[idx].mean()

    # p-value: fraction of bootstrap samples where the difference is ≤ 0
    p_value: float = float((boot_diffs <= 0).mean())

    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "diff": observed_diff,
        "p_value": p_value,
    }
