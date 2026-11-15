"""Conformal risk control for a review threshold (fixed-sequence Learn-then-Test).

Convention (as in eval/triage.py): higher score = riskier = reviewed first. A unit is flagged when
score >= threshold. The risk is the miss rate of critical units among NON-flagged units.

Procedure. Order the development units by score (highest first). For k = n, n-1, ..., 0 the
non-flagged set is the suffix after the top k. Each k is tested with a one-sided Clopper-Pearson
upper confidence bound (level 1 - delta) on the miss rate of that suffix. The sequence stops at the
first k whose bound exceeds alpha; the smallest accepted k gives the threshold. Fixed-sequence
testing controls the probability of accepting a too-risky threshold at delta.

Units are (segmenter, patient), and patients contribute several units. The binomial counts are
therefore deflated by the design effect n / n_patients (units per patient), a heuristic cluster
correction rather than a theorem. The guarantee holds under exchangeability of development and
deployment units. An unseen segmenter violates this, so test-set miss rates are reported as the
observed result.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import beta


def clopper_pearson_upper(k, n, delta):
    """One-sided (1 - delta) upper confidence bound for a binomial proportion k / n."""
    if n <= 0 or k >= n:
        return 1.0
    return float(beta.ppf(1.0 - delta, k + 1.0, n - k))


def _prepare(scores, critical, groups):
    s = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(critical).astype(bool).ravel()
    g = np.asarray(groups).ravel()
    if not (len(s) == len(y) == len(g)):
        raise ValueError("scores, critical and groups must have the same length")
    if len(s) == 0 or not np.isfinite(s).all():
        raise ValueError("scores must be non-empty and finite")
    return s, y, g


def calibrate_review_threshold(scores, critical, groups, alpha, delta=0.1):
    """Choose the review threshold on development data with a finite-sample miss-rate bound.

    Returns a dict with threshold (flag when score >= threshold), k_flagged, review_fraction,
    the development miss rate among non-flagged units and its upper bound, and the design effect.
    """
    if not 0 < alpha < 1 or not 0 < delta < 1:
        raise ValueError("alpha and delta must lie in (0, 1)")
    s, y, g = _prepare(scores, critical, groups)
    n = len(s)
    n_patients = len(np.unique(g))
    design = n / n_patients

    order = np.argsort(-s, kind="stable")
    ys = y[order]
    ss = s[order]
    suffix_miss = np.concatenate([np.cumsum(ys[::-1])[::-1], [0]])
    suffix_n = n - np.arange(n + 1)

    # Only thresholds whose non-flagged set is large enough for the bound to certify anything are
    # tested: a zero-miss Clopper-Pearson bound is <= alpha only for effective n >= log(delta) /
    # log(1 - alpha). The sequence depends on (n, alpha, delta, design) only, not on the labels,
    # so the fixed-sequence guarantee still applies. Without this, tiny non-flagged sets always
    # fail and the procedure reviews everything.
    n_min_eff = np.log(delta) / np.log(1.0 - alpha)
    accepted_k, accepted_ub = n, 0.0
    for k in range(n, -1, -1):
        n_nf = int(suffix_n[k])
        if n_nf / design < n_min_eff:
            continue
        ub = clopper_pearson_upper(suffix_miss[k] / design, n_nf / design, delta)
        if ub <= alpha:
            accepted_k, accepted_ub = k, ub
        else:
            break

    threshold = float(ss[accepted_k - 1]) if accepted_k > 0 else float("inf")
    n_nf = int(suffix_n[accepted_k])
    miss = int(suffix_miss[accepted_k])
    return {
        "threshold": threshold,
        "k_flagged": int(accepted_k),
        "review_fraction": float(accepted_k / n),
        "alpha": float(alpha),
        "delta": float(delta),
        "n_dev": int(n),
        "n_patients": int(n_patients),
        "design_effect": float(design),
        "dev_nonflagged": n_nf,
        "dev_missed": miss,
        "dev_miss_rate": float(miss / n_nf) if n_nf else 0.0,
        "dev_miss_upper_bound": float(accepted_ub),
        "dev_prevalence": float(y.mean()),
    }


def apply_threshold(scores, critical, threshold):
    """Flag units with score >= threshold and report misses among the non-flagged units."""
    s = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(critical).astype(bool).ravel()
    if len(s) != len(y):
        raise ValueError("scores and critical must have the same length")
    flagged = s >= threshold
    nf = ~flagged
    n_nf = int(nf.sum())
    missed = int((y & nf).sum())
    total_critical = int(y.sum())
    return {
        "threshold": float(threshold),
        "n": int(len(s)),
        "review_fraction": float(flagged.mean()),
        "n_nonflagged": n_nf,
        "n_missed": missed,
        "miss_rate": float(missed / n_nf) if n_nf else 0.0,
        "critical": total_critical,
        "critical_recall": float((y & flagged).sum() / total_critical) if total_critical else float("nan"),
    }
