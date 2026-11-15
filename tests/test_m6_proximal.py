"""M6 proximal features on synthetic phantoms (1 mm voxels, bright CT inside the lumen and aorta)."""

import numpy as np

from coronarysafe.features.proximal import proximal_features

SP = (1.0, 1.0, 1.0)
SHAPE = (120, 160, 120)


def _aorta():
    a = np.zeros(SHAPE, bool)
    yy, xx = np.meshgrid(np.arange(SHAPE[1]), np.arange(SHAPE[0]))
    disk = (xx - 80) ** 2 + (yy - 80) ** 2 <= 20 ** 2
    a[:, :, 30:71] = disk[:, :, None]
    return a


def _segment(mask, x0, x1, y, z, r=2):
    xs = np.arange(min(x0, x1), max(x0, x1) + 1)
    for x in xs:
        yy, zz = np.meshgrid(np.arange(SHAPE[1]), np.arange(SHAPE[2]), indexing="ij")
        mask[x] |= ((yy - y) ** 2 + (zz - z) ** 2) <= r * r
    return mask


def _phantom(gap_near=True, gap_far=False):
    m = np.zeros(SHAPE, bool)
    # left branch: runs from the ostium (x~80, y=103) along -x, at z=66
    m = _segment(m, 80, 66 if gap_near else 50, 103, 66)
    if gap_near:
        m = _segment(m, 62, 50, 103, 66)  # a 3 mm break at x=63..65
    if gap_far:
        m = _segment(m, 20, 49, 103, 66)  # ends beyond the 60 mm proximal radius
    # right branch, intact, at y=57
    m = _segment(m, 80, 100, 57, 60)
    return m


def _run(mask, gap_near=True):
    ao = _aorta()
    hu = np.where(mask | ao, 400.0, 0.0).astype(np.float32)
    return proximal_features(mask, hu, SP, ao, lr_axis=1, lr_sign=1)


def test_no_gap_gives_no_proximal_gap_and_both_trees_attached():
    f = _run(_phantom(gap_near=False))
    assert f["m6_L_attached"] == 1.0 and f["m6_R_attached"] == 1.0
    assert f["m6_L_n_prox_gaps"] == 0.0 and f["m6_R_n_prox_gaps"] == 0.0


def test_near_gap_is_counted_on_the_left_tree():
    f = _run(_phantom(gap_near=True))
    assert f["m6_L_n_prox_gaps"] == 1.0, f
    assert f["m6_R_n_prox_gaps"] == 0.0
    assert f["m6_sum_prox_gaps"] == 1.0
    assert np.isfinite(f["m6_L_prox_gap_evid"]) and f["m6_L_prox_gap_evid"] >= 0.0


def test_reach_is_truncated_at_the_gap():
    f = _run(_phantom(gap_near=True))
    # reachable from the ostium up to the break (x ~ 66): roughly 14 mm, far below 50 mm
    assert f["m6_L_reach50_mm"] < 25.0, f["m6_L_reach50_mm"]
    f_intact = _run(_phantom(gap_near=False))
    assert f_intact["m6_L_reach50_mm"] > 28.0, f_intact["m6_L_reach50_mm"]  # branch is 30 mm long


def test_far_gap_is_not_proximal():
    f = _run(_phantom(gap_near=False, gap_far=True))
    assert f["m6_L_n_prox_gaps"] == 0.0


def test_constant_radius_has_no_radius_drop():
    f = _run(_phantom(gap_near=False))
    # noise floor for an r=2 mm voxel tube (end effects) is ~0.8 mm; a real taper would be larger
    assert f["m6_L_radius_drop_60"] < 1.0, f["m6_L_radius_drop_60"]


def test_empty_prediction_is_zero():
    ao = _aorta()
    f = proximal_features(np.zeros(SHAPE, bool), np.zeros(SHAPE, np.float32), SP, ao, 1, 1)
    assert f["m6_worst_attached"] == 0.0 and f["m6_sum_prox_gaps"] == 0.0


def test_share_attached_is_fraction_of_total():
    f = _run(_phantom(gap_near=False))
    assert 0.0 < f["m6_L_share_attached"] <= 1.0
    assert f["m6_R_share_attached"] > 0.0
