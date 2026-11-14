"""Triage metrics with hand-computed values, plus deterministic clustered bootstrap."""

import numpy as np
import pytest

from coronarysafe.eval.bootstrap import cluster_bootstrap, paired_bootstrap_difference
from coronarysafe.eval.metrics import (
    aurc,
    eaurc,
    normalised_recall,
    oracle_recall_at_budget,
    recall_at_budget,
    spearman,
)


def test_recall_and_oracle_at_budget():
    # 10 cases, 2 failures; budget 0.2 reviews 2 cases.
    scores = np.array([0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.05, 0.15, 0.25])
    fail = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=bool)
    # top-2 by score: indices 0 (0.9) and 6 (0.6) -> catches 1 of 2 failures
    assert recall_at_budget(scores, fail, 0.2) == pytest.approx(0.5)
    assert oracle_recall_at_budget(fail, 0.2) == pytest.approx(1.0)
    assert normalised_recall(scores, fail, 0.2) == pytest.approx(0.5)


def test_aurc_and_eaurc_tiny_example():
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    losses = np.array([0.0, 1.0, 0.0, 1.0])
    # most confident first: cumulative risks 0, 1/2, 1/3, 1/2 -> mean 1/3
    assert aurc(scores, losses) == pytest.approx(1.0 / 3.0)
    # oracle order 0,0,1,1 -> cumulative 0, 0, 1/3, 1/2 -> mean 5/24
    assert eaurc(scores, losses) == pytest.approx(1.0 / 3.0 - 5.0 / 24.0)


def test_spearman_perfect_monotone():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_cluster_bootstrap_is_deterministic_and_covers_estimate():
    clusters = np.array(["a", "a", "b", "b", "c", "d"])
    values = np.array([0.0, 0.2, 0.4, 0.6, 1.0, 0.5])
    stat = lambda rows: values[rows].mean()  # noqa: E731
    r1 = cluster_bootstrap(clusters, stat, n_boot=300, seed=7)
    r2 = cluster_bootstrap(clusters, stat, n_boot=300, seed=7)
    assert r1["mean"] == r2["mean"] and r1["lo"] == r2["lo"] and r1["hi"] == r2["hi"]
    assert r1["lo"] <= r1["estimate"] <= r1["hi"]


def test_paired_difference_of_constant_scorers():
    clusters = np.array([1, 1, 2, 3])
    a = np.array([0.9, 0.9, 0.9, 0.9])
    b = np.array([0.8, 0.8, 0.8, 0.8])
    res = paired_bootstrap_difference(
        clusters, lambda r: a[r].mean(), lambda r: b[r].mean(), n_boot=100, seed=0
    )
    assert res["lo"] == pytest.approx(0.1) and res["hi"] == pytest.approx(0.1)
