"""Anatomical priors: per-feature mean and std from train-split GT feature dicts.

The caller is responsible for passing only train-split ground-truth features.
This module never reads label files.
"""

from __future__ import annotations

import numpy as np

_MIN_STD = 1e-6


def fit_priors(feature_dicts) -> dict:
    """Return {feature: {"mean": m, "std": s}} using NaN-aware statistics."""
    dicts = list(feature_dicts)
    if not dicts:
        raise ValueError("fit_priors needs at least one feature dict")
    keys = sorted(set().union(*[d.keys() for d in dicts]))
    priors = {}
    for k in keys:
        vals = np.array([d.get(k, np.nan) for d in dicts], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        priors[k] = {"mean": mean, "std": max(std, _MIN_STD)}
    return priors


def zscore(features, priors) -> dict:
    """z = (x - mean) / std for every feature present in priors; NaN when missing."""
    out = {}
    for k, pr in priors.items():
        x = features.get(k, np.nan)
        x = float(x) if x is not None else np.nan
        out[k] = (x - pr["mean"]) / pr["std"] if np.isfinite(x) else np.nan
    return out
