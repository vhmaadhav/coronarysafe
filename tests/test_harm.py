"""Downstream-harm analysis on the synthetic coronary phantom (GT labels from phantoms.build_gt)."""

import numpy as np

from coronarysafe.analysis.harm import analyse_unit, unit_errors
from phantoms import SPACING, build_gt


def test_perfect_prediction_has_zero_errors():
    gt = build_gt()
    per = analyse_unit(gt, gt > 0, SPACING)
    for name in ("LM", "LAD", "LCx", "RCA"):
        rows = {r["vessel"]: r for r in unit_errors(per)}[name]
        if rows["gt_present"]:
            assert rows["gt_length_mm"] > 0
            assert abs(rows["rel_length_err"]) < 1e-9
            assert abs(rows["diam_err_mm"]) < 1e-9
            assert abs(rows["log_resistance_ratio"]) < 1e-9
            assert not rows["missing"]


def test_removed_lad_is_reported_as_missing():
    gt = build_gt()
    pred = (gt > 0) & (gt != 2)
    rows = {r["vessel"]: r for r in unit_errors(analyse_unit(gt, pred, SPACING))}
    lad = rows["LAD"]
    assert lad["gt_present"]
    assert lad["pred_length_mm"] < 0.1 * lad["gt_length_mm"]
    assert lad["rel_length_err"] < -0.9
    assert not rows["LM"]["missing"]


def test_narrowed_lcx_gives_smaller_minimum_diameter():
    gt = build_gt()
    lumen = gt > 0
    # erode the LCx lumen by one voxel in x and y across its whole length
    from scipy import ndimage

    lcx = gt == 3
    thin = ndimage.binary_erosion(lcx, structure=np.ones((3, 3, 1)))
    pred = (lumen & ~lcx) | thin
    rows = {r["vessel"]: r for r in unit_errors(analyse_unit(gt, pred, SPACING))}
    assert rows["LCx"]["diam_err_mm"] < 0
