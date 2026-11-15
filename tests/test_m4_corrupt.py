"""Smoke tests for M4 corruption operators and labelling on a synthetic GT tree (no real data)."""

import numpy as np

from coronarysafe.labels.taxonomy import label_failures, load_taxonomy_config
from coronarysafe.m4.build import GRID, to_grid
from coronarysafe.m4.corrupt import CLEAN_OPS, FAILURE_OPS, apply_op, corrupt, sample_plan

SP = np.array([0.5, 0.5, 0.5])
SHAPE = (120, 120, 120)


def _tube_lab(lab, p0, p1, r_vox, value):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    grid = np.indices(lab.shape).reshape(3, -1).T.astype(float)
    d = p1 - p0
    t = np.clip(((grid - p0) @ d) / (d @ d), 0, 1)
    dist = np.linalg.norm(grid - (p0 + t[:, None] * d), axis=1)
    sel = (dist <= r_vox).reshape(lab.shape)
    lab[sel] = value
    return lab


def _gt():
    """Left tree (LAD=2) and right tree (RCA=9) with a few branches, 15-class labels."""
    lab = np.zeros(SHAPE, np.int32)
    lab = _tube_lab(lab, (60, 30, 60), (60, 90, 60), 4, 2)    # LAD
    lab = _tube_lab(lab, (60, 90, 60), (60, 110, 80), 3, 2)   # LAD distal
    lab = _tube_lab(lab, (60, 60, 60), (30, 60, 40), 3, 3)    # LCx (left)
    lab = _tube_lab(lab, (60, 60, 60), (90, 60, 40), 3, 9)    # RCA (right)
    lab = _tube_lab(lab, (90, 60, 40), (90, 100, 20), 2, 9)   # RCA distal
    return lab


def test_every_op_runs_and_returns_bool_mask():
    rng = np.random.default_rng(0)
    gt = _gt()
    base = gt > 0
    for op in FAILURE_OPS + CLEAN_OPS:
        m, ok = apply_op(op, base, gt, SP, rng, left_labels=[2, 3], right_labels=[9])
        assert m.dtype == bool and m.shape == base.shape, op


def test_sample_plan_uses_known_ops():
    rng = np.random.default_rng(1)
    for _ in range(50):
        for op in sample_plan(rng) + sample_plan(rng, clean=True):
            assert op in FAILURE_OPS or op in CLEAN_OPS


def test_corruption_changes_mask_and_labels_run():
    rng = np.random.default_rng(2)
    gt = _gt()
    cfg = load_taxonomy_config()
    m, applied = corrupt(gt, SP, rng, ["delete_run"], left_labels=cfg["left_labels"],
                         right_labels=cfg["right_labels"])
    assert applied == ["delete_run"]
    assert (m != (gt > 0)).any()
    lab = label_failures(gt, None, m, SP, cfg)
    assert "critical" in lab and "burden" in lab
    assert lab["burden"] >= 0.0


def test_clean_dilate_is_not_a_failure_by_itself():
    rng = np.random.default_rng(3)
    gt = _gt()
    cfg = load_taxonomy_config()
    m, _ = corrupt(gt, SP, rng, ["clean_dilate"], left_labels=cfg["left_labels"],
                   right_labels=cfg["right_labels"])
    lab = label_failures(gt, None, m, SP, cfg)
    assert not lab["critical"], lab["critical_reasons"]


def test_to_grid_shape_and_content():
    gt = _gt()
    g = to_grid(gt > 0, SP)
    assert g.shape == (GRID, GRID, GRID)
    assert g.sum() > 0
