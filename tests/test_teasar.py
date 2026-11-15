"""TEASAR (kimimaro) centreline path of the 1D model (Addendum E). Droplet only."""

import numpy as np

from coronarysafe.analysis import hemo1d as h
from coronarysafe.analysis.harm import nearest_gt_labels
from phantoms import SPACING, build_gt


def test_teasar_tree_is_finite_and_self_consistent():
    gt = build_gt()
    lab = nearest_gt_labels(gt, SPACING)
    res = h.analyse_mask(gt > 0, SPACING, lab, graph="teasar")
    assert res["n_elements"] > 0
    for v in ("LAD", "LCx", "RCA"):
        m = res["vessels"][v]
        if m["length_mm"] > 0:
            assert 0.0 <= m["ffr_distal"] <= 1.0
    per = h.compare_units(res, res)
    for v in ("LAD", "LCx", "RCA"):
        if per[v]["eligible"]:
            assert abs(per[v]["d_ffr_distal"]) < 1e-12
