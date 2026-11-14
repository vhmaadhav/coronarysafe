"""Triage metrics. Convention: ``scores`` are QA risk (higher = more likely a failure, so
reviewed first); confidence = -score. ``losses`` are per-case risk (e.g. 1 - Dice or the
failure indicator) used by AURC."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def _arrays(scores, other):
    s = np.asarray(scores, dtype=float).ravel()
    o = np.asarray(other).ravel()
    if s.shape != o.shape:
        raise ValueError("scores and labels must have the same length")
    return s, o


def _review_count(n, budget):
    return int(math.ceil(budget * n - 1e-12))


def recall_at_budget(scores, is_failure, budget: float = 0.2) -> float:
    """Fraction of failures caught when reviewing the top ``budget`` fraction by score."""
    s, f = _arrays(scores, is_failure)
    f = f.astype(bool)
    n_fail = int(f.sum())
    if n_fail == 0:
        return float("nan")
    k = _review_count(len(s), budget)
    order = np.argsort(-s, kind="stable")
    return float(f[order[:k]].sum() / n_fail)


def oracle_recall_at_budget(is_failure, budget: float = 0.2) -> float:
    """Best possible recall at the budget (reviewing failures first)."""
    f = np.asarray(is_failure).ravel().astype(bool)
    n_fail = int(f.sum())
    if n_fail == 0:
        return float("nan")
    k = _review_count(len(f), budget)
    return float(min(k, n_fail) / n_fail)


def normalised_recall(scores, is_failure, budget: float = 0.2) -> float:
    """recall@budget divided by oracle recall@budget (primary metric)."""
    r = recall_at_budget(scores, is_failure, budget)
    o = oracle_recall_at_budget(is_failure, budget)
    if not np.isfinite(r) or not np.isfinite(o) or o == 0:
        return float("nan")
    return float(r / o)


def _aurc_from_ordered(losses_ordered):
    n = len(losses_ordered)
    cum = np.cumsum(losses_ordered) / np.arange(1, n + 1)
    return float(cum.mean())


def aurc(scores, losses) -> float:
    """Area under the risk-coverage curve. Most confident (lowest score) kept first."""
    s, loss = _arrays(scores, losses)
    order = np.argsort(s, kind="stable")
    return _aurc_from_ordered(loss[order].astype(float))


def eaurc(scores, losses) -> float:
    """Excess AURC: AURC minus the AURC of the oracle ordering by loss."""
    _, loss = _arrays(scores, losses)
    loss = loss.astype(float)
    return aurc(scores, loss) - _aurc_from_ordered(np.sort(loss))


def auroc(scores, is_failure) -> float:
    s, f = _arrays(scores, is_failure)
    f = f.astype(bool)
    if f.all() or (~f).all():
        return float("nan")
    return float(roc_auc_score(f, s))


def spearman(scores, values) -> float:
    s, v = _arrays(scores, values)
    if np.ptp(s) == 0 or np.ptp(v) == 0:
        return float("nan")
    return float(stats.spearmanr(s, v).correlation)
