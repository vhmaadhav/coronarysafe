"""taxonomy_v3: ostium distance on events, critical tier, and burden on phantoms."""

from pathlib import Path

import pytest

from coronarysafe.labels.taxonomy import label_failures, load_taxonomy_config
from phantoms import SPACING, build_gt, lad_gap, tube_mask

CFG3 = load_taxonomy_config(Path(__file__).resolve().parents[1] / "configs" / "taxonomy_v3.yaml")


@pytest.fixture(scope="module")
def gt():
    return build_gt()


def _run(gt, pred):
    return label_failures(gt, None, pred, SPACING, CFG3)


def test_proximal_lad_gap_is_critical_with_ostium_distance(gt):
    res = _run(gt, lad_gap(gt > 0, 18.0, 38.0))
    f1 = [e for e in res["events"] if e["flag"] == "F1" and e["segment"] == "LAD"]
    assert f1, res["flags"]
    e = f1[0]
    # LM root (z 55) -> LM bifurcation (z 45) -> LAD corner -> down to z 38: about 33 mm
    assert 20.0 < e["dist_from_ostium_mm"] < 50.0
    assert 1.2 < e["gt_radius_mm"] < 1.9
    assert set(e["segments"]) >= {"LAD"}
    assert res["critical"]
    assert res["burden"] > 0.0
    assert res["missed_major_mm"] >= e["length_mm"] - 1e-9


def test_side_branch_loss_is_not_critical(gt):
    res = _run(gt, (gt > 0) & (gt != 4))
    assert not res["critical"]
    assert res["missed_major_mm"] == 0.0
    assert res["burden"] == 0.0


def test_detached_thin_stray_is_not_critical(gt):
    stray = tube_mask((4, 36, 30), (4, 36, 15))  # detached, radius 1.5 mm
    res = _run(gt, (gt > 0) | stray)
    assert any(e["flag"] == "F3b" for e in res["events"])
    assert not res["critical"]


def test_bridge_is_critical(gt):
    bridge = tube_mask((29.6, 29.6, 42.0), (32.0, 8.0, 42.0))
    res = _run(gt, (gt > 0) | bridge)
    assert res["critical"]
    assert any("F3c" in r for r in res["critical_reasons"])


def test_every_event_has_anchor_fields(gt):
    res = _run(gt, lad_gap(gt > 0, 18.0, 38.0))
    for e in res["events"]:
        if e["flag"] in ("F1", "F2", "F3a", "F3b", "F3c", "F3d", "F3h"):
            assert "dist_from_ostium_mm" in e and "gt_radius_mm" in e and "segments" in e
