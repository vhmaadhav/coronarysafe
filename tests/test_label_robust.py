"""Tests for the label-robustness analysis (Addendum I): transfer, agreement and restricted critical labels."""

from pathlib import Path

import numpy as np
import pytest

from coronarysafe.analysis.label_robust import OTHER, agreement_stats, restricted_critical, transfer_labels
from coronarysafe.labels.taxonomy import label_failures, load_taxonomy_config

REPO = Path(__file__).resolve().parents[1]
SP = (0.5, 0.5, 1.0)  # anisotropic voxels (mm)


def _phantom():
    gt = np.zeros((60, 40, 40), dtype=np.uint8)
    gt[:, 15:18, 20] = 2  # LAD-like tube along x, label 2
    gt[30, 5:35, 20] = 3  # LCx-like tube along y, label 3
    gt[20:24, 20:25, 20] = 3  # a short labelled blob, label 3
    return gt


def test_transfer_exact_labels_and_spacing_aware_cutoff():
    gt = _phantom()
    orig = gt > 0
    relab, st = transfer_labels(orig, gt, SP)
    # labelled voxels keep their own label
    assert np.array_equal(relab[gt > 0], gt[gt > 0])
    assert st["mapped_fraction"] == pytest.approx(1.0)

    # a single labelled voxel: distance along x is 0.5 mm per voxel, along z 1.0 mm per voxel
    single = np.zeros((30, 30, 30), dtype=np.uint8)
    single[10, 10, 15] = 4
    orig2 = np.zeros_like(single, dtype=bool)
    orig2[11, 10, 15] = True  # 0.5 mm on x -> mapped
    orig2[16, 10, 15] = True  # 3.0 mm on x -> mapped (boundary, <= 3 mm)
    orig2[17, 10, 15] = True  # 3.5 mm on x -> unmapped (Other)
    orig2[10, 10, 18] = True  # 3.0 mm on z (1.0 mm voxels) -> mapped
    orig2[10, 10, 19] = True  # 4.0 mm on z -> unmapped (Other)
    relab2, st2 = transfer_labels(orig2, single, SP)
    assert relab2[11, 10, 15] == 4
    assert relab2[16, 10, 15] == 4
    assert relab2[17, 10, 15] == OTHER
    assert relab2[10, 10, 18] == 4
    assert relab2[10, 10, 19] == OTHER
    assert st2["mapped_fraction"] == pytest.approx(3 / 5)


def test_relabelled_mask_is_the_original_lumen():
    gt = _phantom()
    orig = np.zeros_like(gt, dtype=float)
    orig[gt > 0] = 1.0
    orig[25, 20, 5] = 1.0  # extra original voxel 15 mm from any ImageCAS-X label (z=20) -> Other
    relab, _ = transfer_labels(orig, gt, SP)
    assert np.array_equal(relab > 0, orig > 0)
    assert relab[25, 20, 5] == OTHER


def test_frozen_label_function_runs_on_relabelled_phantom():
    cfg = load_taxonomy_config(REPO / "configs" / "taxonomy_v3.yaml")
    gt = _phantom()
    relab, _ = transfer_labels(gt > 0, gt, SP)
    out = label_failures(relab, None, gt > 0, SP, cfg)
    assert out["critical"] is False  # prediction equals the reference, nothing missed or leaked
    assert out["centerline_source"] == "gt_skeleton"


def test_agreement_hand_example():
    v3 = np.array([1, 1, 0, 0])
    alt = np.array([1, 0, 0, 0])
    s = agreement_stats(v3, alt)
    assert s["percent_agreement"] == pytest.approx(0.75)
    assert s["kappa"] == pytest.approx(0.5)
    assert s["pabak"] == pytest.approx(0.5)
    assert s["ac1"] == pytest.approx((0.75 - 0.46875) / (1 - 0.46875))


def _ev(flag, label, dist, radius=0.8, **extra):
    base = {"flag": flag, "label_ids": [label], "segment": str(label), "dist_from_ostium_mm": dist,
            "gt_radius_mm": radius}
    base.update(extra)
    return base


def test_restricted_keeps_only_proximal_missed_or_disconnected_major():
    cfg = load_taxonomy_config(REPO / "configs" / "taxonomy_v3.yaml")
    # proximal LAD (label 2) missed within 50 mm: counts
    assert restricted_critical([_ev("F1", 2, 20.0)], cfg)[0] is True
    # proximal LCx (label 3) disconnected within 50 mm: counts
    assert restricted_critical([_ev("F2", 3, 35.0)], cfg)[0] is True
    # distal, thin major-segment loss (radius clause dropped): does not count
    assert restricted_critical([_ev("F1", 2, 80.0, radius=0.9)], cfg)[0] is False
    # thick (GT radius >= 1.25) but distal loss: not counted in the restricted tier
    assert restricted_critical([_ev("F1", 9, 90.0, radius=2.0)], cfg)[0] is False
    # PDA (10) missed proximally is not a base major segment: does not count
    assert restricted_critical([_ev("F1", 10, 10.0)], cfg)[0] is False
    # thickness-based terms are dropped: leak, bridge and detached FP never count
    assert restricted_critical([_ev("F3c", 2, 5.0, subtype="leak_thick")], cfg)[0] is False
    assert restricted_critical([_ev("F3c", 2, 5.0, subtype="bridge")], cfg)[0] is False
    assert restricted_critical([_ev("F3b", 2, 5.0, radius_mm=3.0)], cfg)[0] is False
    # no events: not critical
    assert restricted_critical([], cfg) == (False, [])
