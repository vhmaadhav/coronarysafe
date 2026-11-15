"""M5 image-evidence features on synthetic phantoms: bright vs dark gaps and tip continuation."""

import numpy as np

from coronarysafe.features.evidence import evidence_features

SP = (0.4, 0.4, 0.5)
SHAPE = (60, 60, 100)
TUBE_HU = 400.0
BG_HU = -100.0


def _tube_voxels(z_lo, z_hi, radius_mm=1.5):
    yy, xx = np.meshgrid(np.arange(SHAPE[1]) * SP[1], np.arange(SHAPE[0]) * SP[0], indexing="ij")
    disk = (xx - 30 * SP[0]) ** 2 + (yy - 30 * SP[1]) ** 2 <= radius_mm**2
    tube = np.zeros(SHAPE, dtype=bool)
    for k in range(z_lo, z_hi + 1):
        tube[:, :, k] = disk.T
    return tube


def _hu(tube):
    hu = np.full(SHAPE, BG_HU, dtype=np.float32)
    hu[tube] = TUBE_HU
    return hu


def test_bright_gap_scores_higher_than_dark_gap():
    full = _tube_voxels(5, 95)
    mask = full.copy()
    mask[:, :, 40:48] = False  # 4 mm break in the prediction
    bright = evidence_features(mask, _hu(full), SP)  # vessel is present in the CT across the break
    dark_tube = full.copy()
    dark_tube[:, :, 40:48] = False  # no vessel in the CT across the break either
    dark = evidence_features(mask, _hu(dark_tube), SP)

    assert bright["m5_n_gaps"] >= 1 and dark["m5_n_gaps"] >= 1
    assert bright["m5_gap_evidence_max"] > 0.5
    assert dark["m5_gap_evidence_max"] < 0.2
    assert bright["m5_gap_evidence_max"] > dark["m5_gap_evidence_max"] + 0.3
    assert bright["m5_gap_contrast_max"] > 0.8
    assert dark["m5_gap_contrast_max"] < 0.3


def test_tip_continuation_follows_bright_vessel_beyond_prediction():
    mask = _tube_voxels(5, 75)  # prediction stops at z = 75
    bright = evidence_features(mask, _hu(_tube_voxels(5, 95)), SP)  # CT vessel continues to z = 95
    dark = evidence_features(mask, _hu(_tube_voxels(5, 75)), SP)  # CT vessel stops with the prediction

    assert bright["m5_tip_evidence_max"] > 0.5
    assert 6.0 <= bright["m5_tip_cont_len_max_mm"] <= 12.0  # about 10 mm of CT vessel past the tip
    assert dark["m5_tip_evidence_max"] < 0.2
    assert dark["m5_tip_cont_len_max_mm"] == 0.0


def test_reference_hu_uses_aorta_when_given():
    mask = _tube_voxels(5, 95)
    hu = _hu(mask)
    aorta = np.zeros(SHAPE, dtype=bool)
    aorta[5:10, 5:10, 5:10] = True
    hu[aorta] = 300.0
    f = evidence_features(mask, hu, SP, aorta=aorta)
    assert f["m5_ref_fallback"] == 0.0
    assert abs(f["m5_ref_hu"] - 300.0) < 1e-3


def test_empty_mask_returns_zero_counts():
    f = evidence_features(np.zeros(SHAPE, dtype=bool), np.full(SHAPE, BG_HU, np.float32), SP)
    assert f["m5_n_gaps"] == 0.0 and f["m5_n_tips"] == 0.0
