"""
stats.py — Statistical utilities.

Functions exactly as specified in implementation plan §8:
  normalised_recovery   — with min_spread guard
  paired_bootstrap      — paired per-episode binary outcomes
  mcnemar_paired        — exact McNemar test (statsmodels)
"""

from __future__ import annotations

import numpy as np


def normalised_recovery(
    sr_fit: float,
    sr_oracle: float,
    sr_random: float,
    min_spread: float = 0.10,
) -> float | None:
    """
    Normalised recovery of *sr_fit* relative to oracle and random.

    Args:
        sr_fit:      Success rate of the method being evaluated.
        sr_oracle:   Success rate of oracle routing.
        sr_random:   Success rate of random routing.
        min_spread:  Minimum oracle–random spread to report a value.
                     Returns None if spread < min_spread (denominator too small).

    Returns:
        Normalised recovery in [0, 1] (may exceed 1 if sr_fit > sr_oracle),
        or None if the denominator is too small to be meaningful.
    """
    spread = sr_oracle - sr_random
    if spread < min_spread:
        return None
    return (sr_fit - sr_random) / spread


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    n: int = 10_000,
    seed: int = 0,
    ci: float = 95.0,
) -> tuple[float, tuple[float, float]]:
    """
    Paired bootstrap confidence interval for the mean difference a − b.

    Args:
        a:    Binary outcomes for arm A, shape [N_episodes].
        b:    Binary outcomes for arm B, in the SAME episode order as *a*.
        n:    Number of bootstrap resamples.
        seed: RNG seed for reproducibility.
        ci:   Confidence level (default 95 → [2.5, 97.5] percentiles).

    Returns:
        (mean_diff, (lower_ci, upper_ci))

    Raises:
        ValueError: if a and b have different lengths.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape; got {a.shape} vs {b.shape}")
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    bootstrap_means = d[idx].mean(axis=1)
    alpha = (100.0 - ci) / 2.0
    lo, hi = float(np.percentile(bootstrap_means, alpha)), float(np.percentile(bootstrap_means, 100.0 - alpha))
    return float(d.mean()), (lo, hi)


def mcnemar_paired(a: np.ndarray, b: np.ndarray) -> float:
    """
    Exact McNemar test for paired binary outcomes.

    Args:
        a: Binary outcomes for arm A, shape [N_episodes].
        b: Binary outcomes for arm B, in the SAME episode order.

    Returns:
        Two-sided p-value.

    Raises:
        ImportError: if statsmodels is not installed.
        ValueError:  if a and b have different lengths or are not binary.
    """
    try:
        from statsmodels.stats.contingency_tables import mcnemar
    except ImportError as e:
        raise ImportError(
            "statsmodels is required for McNemar test. "
            "Install it with: uv pip install statsmodels"
        ) from e

    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape; got {a.shape} vs {b.shape}")

    table = [
        [(( a) & ( b)).sum(), (( a) & (~b)).sum()],
        [((~a) & ( b)).sum(), ((~a) & (~b)).sum()],
    ]
    result = mcnemar(table, exact=True)
    return float(result.pvalue)


def oracle_gap_permutation(
    per_chart_successes,
    n: int = 10_000,
    seed: int = 0,
):
    """Permutation test for the oracle-minus-random success gap of a chart
    library (FIX_SPEC.md A6).

    The observed gap is
        gap = mean_i [ max_c X[c, i] ]  -  mean_{c,i} X[c, i]
    where X is `per_chart_successes` with shape [n_charts, n_episodes] of binary
    outcomes. `d_i = oracle_i - random_i >= 0` at every episode by construction,
    so a paired bootstrap CI can never contain zero and tests nothing. This
    permutation test instead asks: is this library's oracle gap larger than
    what independent charts with the same per-chart success rates would produce
    by chance?

    Null model: each chart's outcome vector is shuffled across episodes
    INDEPENDENTLY (destroying any cross-chart specialisation / complementarity
    while preserving every chart's marginal success count, hence the random SR).
    The oracle SR is recomputed under each shuffle; p = fraction of shuffles
    whose gap is >= the observed gap (one-sided).

    NOTE ON SPEC WORDING: FIX_SPEC.md A6 literally says "permute chart labels
    within each episode". That operation leaves both max_c and mean_c over a
    column unchanged (it only relabels values already present), so it is a
    no-op for this statistic. The independent across-episode shuffle above is
    the non-degenerate test that satisfies A6's stated assertion (identical
    charts => gap 0, p ~ 1). Flagged in the Stage 1 report.

    Returns:
        (observed_gap, p_value, null_distribution_summary)
        null_distribution_summary: dict with mean/std/p05/p95 of the null gaps.
    """
    X = np.asarray(per_chart_successes, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"per_chart_successes must be 2-D [n_charts, n_episodes]; got {X.shape}")
    n_charts, n_ep = X.shape

    sr_random = X.mean()
    observed_gap = float(X.max(axis=0).mean() - sr_random)

    rng = np.random.default_rng(seed)
    null_gaps = np.empty(n, dtype=float)
    for k in range(n):
        Xp = np.empty_like(X)
        for c in range(n_charts):
            Xp[c] = X[c, rng.permutation(n_ep)]
        null_gaps[k] = Xp.max(axis=0).mean() - Xp.mean()

    p_value = float((null_gaps >= observed_gap - 1e-12).mean())
    summary = {
        "n_permutations": n,
        "null_mean": float(null_gaps.mean()),
        "null_std": float(null_gaps.std()),
        "null_p05": float(np.percentile(null_gaps, 5)),
        "null_p95": float(np.percentile(null_gaps, 95)),
    }
    return observed_gap, p_value, summary


def success_rate_ci(
    outcomes: np.ndarray,
    ci: float = 95.0,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> tuple[float, tuple[float, float]]:
    """
    Bootstrap confidence interval for a success rate.

    Args:
        outcomes:    Binary success/failure array, shape [N].
        ci:          Confidence level.
        n_bootstrap: Number of resamples.
        seed:        RNG seed.

    Returns:
        (mean, (lower, upper))
    """
    outcomes = np.asarray(outcomes, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(outcomes), (n_bootstrap, len(outcomes)))
    bootstrap_means = outcomes[idx].mean(axis=1)
    alpha = (100.0 - ci) / 2.0
    lo = float(np.percentile(bootstrap_means, alpha))
    hi = float(np.percentile(bootstrap_means, 100.0 - alpha))
    return float(outcomes.mean()), (lo, hi)
