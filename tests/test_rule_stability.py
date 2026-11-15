"""Small tests for Addendum N (rule_stability): flag cut-offs are monotone, the ablation refits on the
remaining inputs, and the frozen ranking comparator is unchanged."""

import numpy as np
import pandas as pd

from coronarysafe.analysis import rule_stability as rs

FEATS = ["g_total_length_mm", "g_max_radius_mm", "g_n_components_ge_10mm", "g_n_gaps", "g_share_top2"]


def _frozen():
    mean = {"g_total_length_mm": 500.0, "g_max_radius_mm": 2.0, "g_n_components_ge_10mm": 2.0,
            "g_n_gaps": 1.0, "g_share_top2": 0.8}
    std = {"g_total_length_mm": 100.0, "g_max_radius_mm": 0.5, "g_n_components_ge_10mm": 1.0,
           "g_n_gaps": 1.0, "g_share_top2": 0.1}
    return {
        "features": FEATS,
        "preprocessing": {"impute_median": dict(mean), "scale_mean": mean, "scale_std": std},
        "logistic": {"coef": {f: 0.3 for f in FEATS}, "intercept": -0.5, "C": 0.1},
    }


def _df(n=40, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "method": ["cas_net"] * n,
        "scan_id": [str(i) for i in range(n)],
        "critical": rng.random(n) < 0.4,
        "burden": rng.random(n),
        "g_total_length_mm": rng.normal(500, 100, n),
        "g_max_radius_mm": rng.normal(2.0, 0.5, n),
        "g_n_components_ge_10mm": rng.integers(0, 5, n).astype(float),
        "g_n_gaps": rng.integers(0, 3, n).astype(float),
        "g_share_top2": rng.uniform(0.5, 1.0, n),
    })
    d.loc[0, "g_n_gaps"] = np.nan  # a missing count is imputed to 0 in the ranking form
    return d


def test_flag_form_is_monotone_in_cut():
    df = _df()
    frozen = _frozen()
    t = rs.n1_table(df, frozen)
    pooled = t[(t.group == "pooled") & (t.form == "flag")]
    for rule, g in pooled.groupby("rule"):
        g = g.set_index("cut")
        lo, mid, hi = g.loc["x0.8", "flagged_fraction"], g.loc["x1.0", "flagged_fraction"], g.loc["x1.2", "flagged_fraction"]
        if rule == "exploratory_neg_share_top2":  # low share is the risk side: a higher cut flags more
            assert lo <= mid <= hi
        else:  # high count or z is the risk side: a higher cut flags fewer
            assert lo >= mid >= hi


def test_n1_has_ranking_and_three_cut_rows_per_rule():
    df = _df()
    t = rs.n1_table(df, _frozen())
    pooled = t[t.group == "pooled"]
    assert set(pooled.rule) == {"rule_n_components_ge_10mm", "rule_n_gaps", "rule_total_length_z",
                                "rule_max_radius_z", "exploratory_neg_share_top2"}
    assert (pooled.groupby("rule").size() == 4).all()


def test_n2_ablation_refits_on_remaining_inputs():
    df = _df(n=60)
    frozen = _frozen()
    t = rs.n2_table(df, frozen)
    assert len(t) == len(FEATS) + 1
    assert t.iloc[0]["ablated"].startswith("none")
    assert set(t.ablated[1:]) == set(FEATS)
    assert (t.n_inputs[1:] == len(FEATS) - 1).all()
    assert np.isfinite(t["aurc"]).all()
