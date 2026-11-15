"""Synthetic-phantom tests for the ostium/aorta features (M3). No ground truth, no real data."""

import numpy as np
from scipy import ndimage

from coronarysafe.features.ostium import aorta_quality, ostium_features

SP = (1.0, 1.0, 1.0)  # mm per voxel
SHAPE = (160, 160, 160)


def _tube(mask, p0, p1, radius_vox=3):
    """Draw a straight tube of given radius (voxels) between two index points."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    grid = np.indices(mask.shape).reshape(3, -1).T.astype(float)
    d = p1 - p0
    t = np.clip(((grid - p0) @ d) / (d @ d), 0, 1)
    proj = p0 + t[:, None] * d
    dist = np.linalg.norm(grid - proj, axis=1)
    mask |= (dist <= radius_vox).reshape(mask.shape)
    return mask


def _aorta():
    """Vertical aortic root tube, a 1 cm radius-ish column in the middle of the volume."""
    a = np.zeros(SHAPE, bool)
    return _tube(a, (80, 80, 30), (80, 80, 70), radius_vox=20)  # r = 20 mm, 40 mm long, ~50 ml


def _ostia_pred(aorta, extra=None):
    """Left and right coronary-like branches leaving the aortic root, plus optional extras."""
    pred = np.zeros(SHAPE, bool)
    pred = _tube(pred, (80, 103, 66), (60, 103, 60), radius_vox=2)  # left branch (+y side), 1 mm from wall
    pred = _tube(pred, (80, 57, 66), (100, 57, 60), radius_vox=2)   # right branch (-y side)
    if extra is not None:
        pred |= extra
    return pred


def test_aorta_quality_ok_for_single_column():
    ok, info = aorta_quality(_aorta(), SP)
    assert ok, info
    assert info["n_objects"] == 1


def test_aorta_quality_fails_when_split_or_empty():
    a = _aorta()
    a2 = a.copy()
    a2[:, :, 50:55] = False  # cut the column into two pieces
    ok, info = aorta_quality(a2, SP)
    assert not ok and info["n_objects"] == 2
    ok0, _ = aorta_quality(np.zeros(SHAPE, bool), SP)
    assert not ok0


def test_two_attached_ostia_detected():
    aorta = _aorta()
    pred = _ostia_pred(aorta)
    f = ostium_features(pred, aorta, SP, lr_axis=1, lr_sign=1)
    assert f["n_ostia"] == 2, f
    assert f["ostia_deviation"] == 0
    assert f["left_attached"] and f["right_attached"]
    assert f["unattached_len_mm"] < 1.0
    assert f["aorta_quality_ok"]


def test_detached_tube_counts_as_unattached_length():
    aorta = _aorta()
    det = np.zeros(SHAPE, bool)
    det = _tube(det, (20, 20, 20), (20, 20, 60), radius_vox=2)  # ~40 mm detached tube
    pred = _ostia_pred(aorta, extra=det)
    f = ostium_features(pred, aorta, SP, lr_axis=1, lr_sign=1)
    assert f["n_ostia"] == 2
    assert 30.0 < f["unattached_len_mm"] < 50.0, f["unattached_len_mm"]
    assert 0.0 < f["unattached_share"] < 0.5
    assert max(f["attached_dist_mm"]) > 5.0  # the detached tube is far from the aorta


def test_missing_left_ostium_is_flagged():
    aorta = _aorta()
    pred = np.zeros(SHAPE, bool)
    pred = _tube(pred, (80, 57, 66), (100, 57, 60), radius_vox=2)  # only the right branch
    f = ostium_features(pred, aorta, SP, lr_axis=1, lr_sign=1)
    assert f["n_ostia"] == 1
    assert f["ostia_deviation"] == 1
    assert f["right_attached"] and not f["left_attached"]


def test_empty_prediction_is_handled():
    f = ostium_features(np.zeros(SHAPE, bool), _aorta(), SP)
    assert f["n_ostia"] == 0 and f["unattached_len_mm"] == 0.0
