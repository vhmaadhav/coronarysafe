"""Fallback 1D hemodynamic model (Addendum C). Runs on the droplet only."""

import numpy as np

from coronarysafe.analysis import hemo1d as h
from coronarysafe.analysis.harm import nearest_gt_labels
from phantoms import SPACING, build_gt


def test_tube_with_leaf_resistance_matches_analytic():
    r = np.full(40, 1.5e-3)
    L = np.full(40, 1e-3)
    net = h.chain_net(r, L)
    R_tube = 8 * h.MU * 0.040 / (np.pi * 1.5e-3 ** 4)
    net["Rout"][-1] = 2 * R_tube
    Q, _ = h.solve(net)
    q_ana = h.P_AORTA / (3 * R_tube)
    assert abs(Q[-1] - q_ana) / q_ana < 0.01


def test_stenosis_lowers_ffr_monotonically():
    n = 120
    x = (np.arange(n) + 0.5) * 0.5
    ffr = []
    for frac in (0.0, 0.3, 0.5, 0.8):
        r = np.full(n, 1.5e-3)
        inside = np.abs(x - 30) < 5
        rmin = 1.5e-3 * (1 - frac)
        r[inside] = rmin + (1.5e-3 - rmin) * (1 - np.cos(np.pi * (x[inside] - 30) / 5)) / 2
        net = h.chain_net(r, np.full(n, 0.5e-3))
        h.assign_outlets(net, 3e-6)
        _, xp = h.solve(net)
        ffr.append(float(np.clip(xp[-1] / h.P_AORTA, 0, 1)))
    assert all(0 < v <= 1 for v in ffr)
    assert ffr[0] > ffr[1] > ffr[2] > ffr[3]


def test_perfect_prediction_has_no_error():
    gt = build_gt()
    lab = nearest_gt_labels(gt, SPACING)
    res = h.analyse_mask(gt > 0, SPACING, lab)
    per = h.compare_units(res, res)
    for v in ("LAD", "LCx", "RCA"):
        if per[v]["eligible"]:
            assert abs(per[v]["d_ffr_distal"]) < 1e-12
            assert per[v]["flip"] is False
            assert not per[v]["unmeasurable"]


def test_removed_lad_is_unmeasurable():
    gt = build_gt()
    lab = nearest_gt_labels(gt, SPACING)
    gt_res = h.analyse_mask(gt > 0, SPACING, lab)
    pred = (gt > 0) & (gt != 2)
    pr_res = h.analyse_mask(pred, SPACING, lab)
    per = h.compare_units(gt_res, pr_res)
    if per["LAD"]["eligible"]:
        assert per["LAD"]["unmeasurable"]
        assert per["LAD"]["missing"]
