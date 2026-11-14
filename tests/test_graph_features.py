"""Graph and uncertainty features plus the no-label-leak guard."""

import re
from pathlib import Path

import numpy as np

from coronarysafe.features.graph_features import compute_graph_features
from coronarysafe.features.priors import fit_priors, zscore
from coronarysafe.features.uncertainty import pairwise_dice, uncertainty_features

SP = (0.4, 0.4, 0.5)
SHAPE = (60, 60, 100)
FEAT_DIR = Path(__file__).resolve().parents[1] / "src" / "coronarysafe" / "features"


def _tube(radius_mm=1.5, z_lo=5, z_hi=95):
    m = np.zeros(SHAPE, dtype=bool)
    yy, xx = np.meshgrid(np.arange(60) * SP[1], np.arange(60) * SP[0], indexing="ij")
    cx, cy = 30 * SP[0], 30 * SP[1]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius_mm**2
    for k in range(z_lo, z_hi + 1):
        m[:, :, k] = disk.T
    return m


def test_break_increases_components_and_gaps():
    tube = _tube()
    broken = tube.copy()
    broken[:, :, 40:48] = False  # 4 mm gap along z (8 slices * 0.5 mm)
    f0 = compute_graph_features(tube, SP)
    f1 = compute_graph_features(broken, SP)
    assert f0["n_components"] == 1
    assert f1["n_components"] == 2
    assert f1["n_gaps"] > f0["n_gaps"]
    assert f1["n_gaps"] >= 1


def test_large_blob_raises_leak_fraction():
    tube = _tube()
    i, j, k = np.indices(SHAPE)
    centre = np.array([30 * SP[0], 30 * SP[1], 50 * SP[2]])
    coords = np.stack([i * SP[0], j * SP[1], k * SP[2]])
    dist = np.linalg.norm(coords - centre[:, None, None, None], axis=0)
    blob = tube | (dist <= 6.0)
    f0 = compute_graph_features(tube, SP)
    f1 = compute_graph_features(blob, SP)
    assert f1["leak_fraction"] > f0["leak_fraction"]
    assert f1["max_radius_mm"] > f0["max_radius_mm"]


def test_features_are_finite_for_simple_tube():
    f = compute_graph_features(_tube(), SP)
    assert f["n_endpoints"] == 2
    assert np.isfinite(f["total_length_mm"]) and f["total_length_mm"] > 40  # 45 mm tube
    assert abs(f["mean_tortuosity"] - 1.0) < 0.05


def test_uncertainty_and_pairwise_dice():
    m = _tube()
    p = np.where(m, 0.9, 0.05)
    u = uncertainty_features(m, p, SP)
    assert u["mean_entropy_mask"] < 0.4
    assert pairwise_dice([m, m]) == 1.0
    other = np.roll(m, 3, axis=0)
    assert 0.0 < pairwise_dice([m, other]) < 1.0


def test_priors_zscore_round_trip():
    pri = fit_priors([{"a": 1.0, "b": np.nan}, {"a": 3.0, "b": 2.0}])
    assert pri["a"]["mean"] == 2.0
    z = zscore({"a": 4.0, "b": 1.0}, pri)
    assert np.isclose(z["a"], 2.0 / pri["a"]["std"])


def test_feature_code_does_not_touch_labels_or_gt_paths():
    forbidden = re.compile(
        r"(from\s+\.+labels|import\s+.*\blabels\b|coronarysafe\.labels|\bgt_path\b|\.nii|['\"]gt[/'\"])"
    )
    files = list(FEAT_DIR.glob("*.py"))
    assert files
    for path in files:
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            assert not forbidden.search(line), f"{path.name}:{lineno} touches labels/GT: {line}"
