"""taxonomy_v2 F3 rules: over-extension is reported, not severe; detached FPs are split by size."""

from pathlib import Path

import pytest

from coronarysafe.labels.taxonomy import label_failures, load_taxonomy_config
from phantoms import SPACING, build_gt, tube_mask

CFG2 = load_taxonomy_config(Path(__file__).resolve().parents[1] / "configs" / "taxonomy_v2.yaml")


@pytest.fixture(scope="module")
def gt():
    return build_gt()


def _run(gt, pred):
    return label_failures(gt, None, pred, SPACING, CFG2)


def test_distal_over_extension_is_reported_not_severe(gt):
    # thin (r = 1 mm) continuation from the LAD distal terminal, 15 mm long
    ext = tube_mask((20, 4, 5), (6, 0.5, 1.0), r=1.0)
    res = _run(gt, (gt > 0) | ext)
    flags = {e["flag"] for e in res["events"]}
    assert "F3a" in flags
    assert not res["severe"]
    assert all(e["severity"] == "reported" for e in res["events"] if e["flag"] == "F3a")


def test_detached_stray_tube_is_minor_under_v2(gt):
    stray = tube_mask((4, 36, 30), (4, 36, 15))  # 15 mm, r = 1.5 mm, detached
    res = _run(gt, (gt > 0) | stray)
    assert any(e["flag"] == "F3b" for e in res["events"])
    assert not res["severe"]
    assert res["minor"]


def test_bridge_remains_severe_under_v2(gt):
    bridge = tube_mask((29.6, 29.6, 42.0), (32.0, 8.0, 42.0))
    res = _run(gt, (gt > 0) | bridge)
    assert any(e["flag"] == "F3c" and e["subtype"] == "bridge" for e in res["events"])
    assert res["severe"]
