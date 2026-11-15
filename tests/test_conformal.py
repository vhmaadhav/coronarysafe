"""Conformal risk control: finite-sample bound and threshold selection on synthetic dev data."""

import numpy as np

from coronarysafe.eval.conformal import (
    apply_threshold,
    calibrate_review_threshold,
    clopper_pearson_upper,
)


def test_clopper_pearson_zero_misses_matches_closed_form():
    n, delta = 100, 0.1
    assert abs(clopper_pearson_upper(0, n, delta) - (1 - delta ** (1.0 / n))) < 1e-6
    assert clopper_pearson_upper(5, 5, 0.1) == 1.0


def test_separable_scores_give_threshold_that_catches_all_critical():
    rng = np.random.default_rng(0)
    n_units = 400
    critical = np.zeros(n_units, dtype=bool)
    critical[:160] = True
    scores = np.where(critical, rng.uniform(0.6, 1.0, n_units), rng.uniform(0.0, 0.5, n_units))
    groups = np.arange(n_units) // 2  # two units per patient
    res = calibrate_review_threshold(scores, critical, groups, alpha=0.05, delta=0.1)
    assert res["dev_miss_upper_bound"] <= 0.05
    assert 0.38 <= res["review_fraction"] <= 0.5  # about the prevalence (0.40)
    applied = apply_threshold(scores, critical, res["threshold"])
    # The guarantee is on the miss RATE among non-flagged units, not on zero misses.
    assert applied["miss_rate"] <= 0.05
    assert applied["critical_recall"] >= 0.95


def test_uninformative_scores_force_near_full_review():
    rng = np.random.default_rng(1)
    n_units = 200
    critical = rng.random(n_units) < 0.3
    scores = rng.random(n_units)
    groups = np.arange(n_units) // 2
    res = calibrate_review_threshold(scores, critical, groups, alpha=0.05, delta=0.1)
    assert res["review_fraction"] >= 0.9
    assert res["dev_miss_upper_bound"] <= 0.05 or res["k_flagged"] == n_units


def test_threshold_is_monotone_in_alpha():
    rng = np.random.default_rng(2)
    n_units = 300
    critical = rng.random(n_units) < 0.25
    scores = np.where(critical, rng.normal(1.0, 1.0, n_units), rng.normal(0.0, 1.0, n_units))
    groups = np.arange(n_units) // 2
    loose = calibrate_review_threshold(scores, critical, groups, alpha=0.10)
    tight = calibrate_review_threshold(scores, critical, groups, alpha=0.05)
    assert tight["review_fraction"] >= loose["review_fraction"]


def test_apply_threshold_counts():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    critical = np.array([True, False, True, False])
    res = apply_threshold(scores, critical, 0.5)
    assert res["n_nonflagged"] == 2
    assert res["n_missed"] == 1
    assert abs(res["miss_rate"] - 0.5) < 1e-12
    assert abs(res["review_fraction"] - 0.5) < 1e-12
