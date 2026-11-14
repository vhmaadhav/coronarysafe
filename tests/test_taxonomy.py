"""Failure taxonomy on synthetic phantoms (anisotropic spacing)."""

import numpy as np
import pytest

from coronarysafe.labels.taxonomy import label_failures, load_taxonomy_config
from phantoms import SPACING, build_gt, lad_gap, tube_mask

CFG = load_taxonomy_config()


@pytest.fixture(scope="module")
def gt():
    return build_gt()


def _run(gt, pred):
    return label_failures(gt, None, pred, SPACING, CFG)


def test_perfect_prediction_has_no_failures(gt):
    res = _run(gt, gt > 0)
    assert res["flags"] == []
    assert not res["severe"] and not res["minor"]
    assert res["centerline_source"] == "gt_skeleton"


def test_lad_chunk_15mm_is_f1_severe_and_f2(gt):
    res = _run(gt, lad_gap(gt > 0, 20.0, 35.0))
    assert "F1" in res["flags"]
    assert "F2" in res["flags"]
    assert res["severe"]
    f1 = [e for e in res["events"] if e["flag"] == "F1"]
    assert any(e["segment"] == "LAD" and e["severity"] == "severe" for e in f1)


def test_gap_straddling_lm_lad_junction_is_one_run(gt):
    i, j, k = np.indices(gt.shape)
    mm = np.stack([i * SPACING[0], j * SPACING[1], k * SPACING[2]])
    near = np.linalg.norm(mm - np.array([20.0, 20.0, 45.0])[:, None, None, None], axis=0) <= 7.0
    pred = (gt > 0) & ~(near & np.isin(gt, [1, 2, 3]))
    res = _run(gt, pred)
    f1 = [e for e in res["events"] if e["flag"] == "F1"]
    assert len(f1) == 1
    assert f1[0]["length_mm"] >= 10.0
    assert f1[0]["severity"] == "severe"


def test_lad_chunk_20mm_is_f1_severe_and_f2(gt):
    res = _run(gt, lad_gap(gt > 0, 18.0, 38.0))
    assert "F1" in res["flags"]
    assert "F2" in res["flags"]
    assert res["severe"]
    f1 = [e for e in res["events"] if e["flag"] == "F1"]
    assert any(e["segment"] == "LAD" and e["severity"] == "severe" for e in f1)


def test_lad_chunk_4mm_is_not_f1(gt):
    # Pieces on both sides are long, so F2 fires; the 4 mm hole is too short for F1.
    res = _run(gt, lad_gap(gt > 0, 26.0, 30.0))
    assert "F1" not in res["flags"]
    assert "F2" in res["flags"]


def test_stray_tube_is_f3_leak(gt):
    stray = tube_mask((4, 36, 30), (4, 36, 15))
    res = _run(gt, (gt > 0) | stray)
    assert "F3" in res["flags"]
    assert any(e["flag"] == "F3" and e["subtype"] == "leak" for e in res["events"])


def test_bridge_between_trees_is_f3_bridge(gt):
    bridge = tube_mask((29.6, 29.6, 42.0), (32.0, 8.0, 42.0))
    res = _run(gt, (gt > 0) | bridge)
    assert any(e["flag"] == "F3" and e["subtype"] == "bridge" for e in res["events"])
    assert res["severe"]


def test_side_branch_loss_is_minor(gt):
    pred = (gt > 0) & (gt != 4)
    res = _run(gt, pred)
    assert "F1" in res["flags"]
    assert not res["severe"]
    assert res["minor"]
    assert all(e["severity"] == "minor" for e in res["events"] if e["flag"] == "F1")
    assert all(e["severity"] == "degraded" for e in res["events"] if e["flag"] == "F1b")
    assert any(e["segment"] == "D1" for e in res["events"] if e["flag"] == "F1")


def test_sensitivity_variants_are_reported(gt):
    res = _run(gt, lad_gap(gt > 0, 20.0, 35.0))
    assert set(res["sensitivity"]) == {v["name"] for v in CFG["sensitivity"]}
    assert res["sensitivity"]["run_5mm"]["severe"]
    # the 15 mm hole leaves a 15 mm piece, below the 20 mm component threshold
    assert not res["sensitivity"]["run_20mm"]["severe"]
