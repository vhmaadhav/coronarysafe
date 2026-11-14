"""Patient-clustered bootstrap. Each row belongs to a cluster (patient); clusters are
resampled with replacement and all their rows (one per segmenter) come along."""

from __future__ import annotations

import numpy as np


def cluster_bootstrap(clusters, statistic, n_boot: int = 2000, seed: int = 0, ci: float = 0.95):
    """Bootstrap a statistic over clusters.

    statistic: callable taking an array of row indices and returning a float.
    Returns dict with estimate (full sample), mean (bootstrap mean), lo, hi
    (percentile CI) and samples.
    """
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    estimate = float(statistic(np.arange(len(clusters))))
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        rows = np.concatenate([members[k] for k in pick])
        vals[b] = statistic(rows)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.nanpercentile(vals, [100 * alpha, 100 * (1 - alpha)])
    return {
        "estimate": estimate,
        "mean": float(np.nanmean(vals)),
        "lo": float(lo),
        "hi": float(hi),
        "samples": vals,
    }


def paired_bootstrap_difference(
    clusters, stat_a, stat_b, n_boot: int = 2000, seed: int = 0, ci: float = 0.95
):
    """Bootstrap of stat_a(rows) - stat_b(rows) with the same resampled patients for both."""
    return cluster_bootstrap(
        clusters,
        lambda rows: stat_a(rows) - stat_b(rows),
        n_boot=n_boot,
        seed=seed,
        ci=ci,
    )
