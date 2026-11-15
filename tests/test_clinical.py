"""Clinical measures (MLD, CAD-RADS band, SCCT coverage) on the synthetic coronary phantom."""

import numpy as np
from scipy import ndimage

from coronarysafe.analysis.clinical import cadrads_band, clinical_rows, clinical_unit
from phantoms import SPACING, build_gt


def _rows(gt, pred):
    res = clinical_unit(gt, pred, SPACING)
    return {r["vessel"]: r for r in clinical_rows(res)}, res


def test_band_edges():
    assert cadrads_band(0.0) == "0"
    assert cadrads_band(10.0) == "1-24"
    assert cadrads_band(30.0) == "25-49"
    assert cadrads_band(60.0) == "50-69"
    assert cadrads_band(85.0) == "70-99"
    assert cadrads_band(100.0) == "100"


def test_perfect_prediction_is_identical_and_fully_covered():
    gt = build_gt()
    rows, res = _rows(gt, gt > 0)
    for name in ("LM", "LAD", "LCx", "RCA"):
        if name not in rows:
            continue
        r = rows[name]
        assert abs(r["mld_err_mm"]) < 1e-9
        assert r["band_gt"] == r["band_pred"]
        assert not r["band_flip"]
        assert r["coverage"] >= 0.999


def test_focal_stenosis_flips_band_and_keeps_coverage():
    from scipy import ndimage as ndi

    gt = build_gt()
    lad = gt == 2
    eroded = ndi.binary_erosion(lad, structure=np.ones((5, 5, 1)))  # about 0.8 mm narrower
    lad_pred = lad.copy()
    lad_pred[:, :, 20:31] = eroded[:, :, 20:31]  # focal narrowing only (10 mm, mid-vessel)
    pred = ((gt > 0) & (gt != 2)) | lad_pred
    rows, _ = _rows(gt, pred)
    lad_r = rows["LAD"]
    # The unstenosed GT already reads 1-24% from sub-voxel noise and the bend, so compare relatively.
    assert lad_r["stenosis_gt_pct"] < 25.0
    assert lad_r["stenosis_pred_pct"] - lad_r["stenosis_gt_pct"] >= 25.0
    assert lad_r["stenosis_pred_pct"] <= 80.0
    assert lad_r["band_pred"] not in ("0", "1-24")
    assert lad_r["band_flip"]
    assert lad_r["mld_err_mm"] < 0
    assert lad_r["coverage"] >= 0.9  # the thinned lumen still lies within max(1.5 mm, r)


def test_occluded_vessel_is_reported_as_100_percent():
    gt = build_gt()
    pred = (gt > 0) & (gt != 3)  # remove the whole LCx
    rows, _ = _rows(gt, pred)
    lcx = rows["LCx"]
    assert lcx["coverage"] < 0.2
    assert lcx["band_pred"] == "100" or lcx["stenosis_pred_pct"] >= 70.0
    assert lcx["band_flip"]
