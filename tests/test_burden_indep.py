"""Unit tests for Addendum L (burden_indep): label math, Euclidean dilation, NR top set, C2 refit,
patient-cluster bootstrap, exclusion, affine errors, decision rule, and the test-mode gate."""

import json

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from coronarysafe.analysis import burden_indep as bi
from coronarysafe.analysis.label_robust import ICX_GT_DIR
from coronarysafe.eval.bootstrap import cluster_bootstrap


def make_phantom(n=24):
    """L-shaped phantom: label-1 tube along z at (12,12), label-2 branch in +x from z=18."""
    gt = np.zeros((n, n, n), dtype=np.uint8)
    for z in range(4, 21):
        for x in range(11, 14):
            for y in range(11, 14):
                gt[x, y, z] = 1
    for z in range(18, 21):
        for x in range(12, 21):
            for y in range(11, 14):
                if gt[x, y, z] == 0:
                    gt[x, y, z] = 2
    return gt


def test_perfect_prediction_zero_burden():
    gt = make_phantom()
    orig = gt > 0
    r = bi.compute_balt(orig, gt, gt > 0, spacing=(1.0, 1.0, 1.0), pad_mm=2.0)
    assert r["n_orig_mj"] > 0
    assert r["b_alt"] == 0.0


def test_empty_prediction_full_burden():
    gt = make_phantom()
    r = bi.compute_balt(gt > 0, gt, np.zeros_like(gt, dtype=bool), (1.0, 1.0, 1.0), 2.0)
    assert r["b_alt"] == 1.0


def test_half_tube_plus_branch_miss():
    gt = make_phantom()
    orig = gt > 0
    pred = np.zeros_like(gt, dtype=bool)
    for z in range(4, 13):  # keep only the proximal half of the main tube
        pred[:, :, z] |= gt[:, :, z] == 1
    r = bi.compute_balt(orig, gt, pred, spacing=(1.0, 1.0, 1.0), pad_mm=2.0)
    n_keep = int(((gt == 1) & pred).sum())
    n_miss = r["n_orig_mj"] - n_keep
    assert abs(r["b_alt"] - n_miss / r["n_orig_mj"]) < 1e-12
    assert 0.3 < r["b_alt"] < 0.8


def test_no_major_gate_zero_support_is_nan():
    n = 16
    gt = np.zeros((n, n, n), dtype=np.uint8)
    gt[4:12, 4:12, 4:12] = 10  # only PDA (label 10), not in the major gate
    r = bi.compute_balt(gt > 0, gt, gt > 0, (1.0, 1.0, 1.0), 2.0)
    assert r["n_orig_mj"] == 0
    assert np.isnan(r["b_alt"])


def test_orig_superset_of_gate_counts_only_inside_ball():
    # gate = one voxel at the centre of a 9^3 grid, 1 mm isotropic.
    # orig = centre, 6 face neighbours, the edge-diagonal (1,1,0) (all inside the 2 mm ball) plus six
    # voxels at offset 3 (outside the ball). Only the 8 inside-ball voxels may count.
    gt = np.zeros((9, 9, 9), dtype=np.uint8)
    gt[4, 4, 4] = 1
    orig = np.zeros_like(gt, dtype=bool)
    inside = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (1, 1, 0)]
    outside = [(3, 0, 0), (-3, 0, 0), (0, 3, 0), (0, 0, 3), (0, -3, 0), (0, 0, -3)]
    for dx, dy, dz in inside + outside:
        orig[4 + dx, 4 + dy, 4 + dz] = True
    pred = np.zeros_like(orig)
    for dx, dy, dz in [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (3, 0, 0)]:  # 3 inside, 1 outside
        pred[4 + dx, 4 + dy, 4 + dz] = True
    r = bi.compute_balt(orig, gt, pred, (1.0, 1.0, 1.0), 2.0)
    assert r["n_orig_mj"] == 8
    assert r["b_alt"] == 0.625  # 1 - 3/8 exactly


def test_multiclass_prediction_values_count_as_covered():
    gt = make_phantom()
    pred = np.where(gt > 0, 2, 0).astype(np.uint8)  # every predicted voxel carries value 2
    r = bi.compute_balt(gt > 0, gt, pred, (1.0, 1.0, 1.0), 2.0)
    assert r["b_alt"] == 0.0


def test_dilate_mm_anisotropic_euclidean_ball():
    m = np.zeros((9, 9, 9), dtype=bool)
    m[4, 4, 4] = True
    out = bi.dilate_mm(m, spacing=(1.0, 2.0, 4.0), mm=2.0)
    # squared distance: dx^2*1 + dy^2*4 + dz^2*16 <= 4 -> dz = 0, (dx, dy) in {(0..2,0), (0,+-1)}
    assert out.sum() == 7
    assert out[2:7, 4, 4].all()
    assert out[4, 3, 4] and out[4, 5, 4]
    assert not out[5, 5, 4]  # sqrt(1 + 4) > 2
    assert not out[4, 4, 5]  # 4 mm away along z


def test_dilate_mm_coarse_spacing():
    m = np.zeros((9, 9, 9), dtype=bool)
    m[4, 4, 4] = True
    assert bi.dilate_mm(m, (2.0, 2.0, 2.0), 1.0).sum() == 1  # neighbours are 2 mm away
    assert bi.dilate_mm(m, (2.0, 2.0, 2.0), 2.0).sum() == 7  # centre + six face neighbours


def test_dilate_mm_empty_mask():
    assert not bi.dilate_mm(np.zeros((5, 5, 5), dtype=bool), (1.0, 1.0, 1.0), 2.0).any()


def test_nr_top_set_floor_and_fractional_ties():
    w, k = bi.nr_weights(np.arange(12, dtype=float))
    assert k == 2  # floor(0.2 * 12), not ceil
    assert w.sum() == pytest.approx(2.0)
    # four tied units at the cut-off share two slots: each gets weight 0.5
    frozen = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    b = np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    res = bi._flagged_diff(frozen, b, np.arange(10).astype(str), n_boot=5)
    assert res["estimate"] == pytest.approx(0.25 - 0.0625)


def test_c2_residual_exact_fit_null():
    rng = np.random.default_rng(0)
    n = 60
    burden = rng.random(n)
    crit = (rng.random(n) < 0.3).astype(float)
    b_alt = 0.5 * burden + 2.0 * crit
    assert np.abs(bi._c2_residual(b_alt, burden, crit)).max() < 1e-8


def test_c2_statistic_refits_inside_resample():
    rng = np.random.default_rng(1)
    n = 40
    frozen = rng.random(n)
    b = rng.random(n)
    b_v3 = rng.random(n)
    c = (rng.random(n) < 0.4).astype(float)
    rows = rng.integers(0, n, size=n)
    expected_resid = bi._c2_residual(b[rows], b_v3[rows], c[rows])
    from scipy.stats import spearmanr

    expected = spearmanr(frozen[rows], expected_resid)[0]
    assert bi._c2_stat(rows, frozen, b, b_v3, c) == pytest.approx(expected)


def test_cluster_bootstrap_resamples_whole_patients():
    clusters = np.repeat(np.array([f"p{i}" for i in range(10)]), 2)  # two segmenter rows per patient
    seen = []

    def stat(rows):
        seen.append(rows.copy())
        return 0.0

    cluster_bootstrap(clusters, stat, n_boot=20, seed=0)
    assert len(seen) == 21
    for rows in seen[1:]:
        _, counts = np.unique(clusters[rows], return_counts=True)
        assert np.all(counts % 2 == 0)


def test_decision_branches():
    assert bi.decision_branch(0.1, 0.2) == "(i)"
    assert bi.decision_branch(0.1, -0.2) == "(ii)"
    assert bi.decision_branch(0.1, 0.0) == "(ii)"
    assert bi.decision_branch(-0.1, 0.5) == "(iii)"
    assert bi.decision_branch(0.1, float("nan")) == "(iii)"
    assert bi.decision_branch(float("nan"), 0.5) == "(iii)"


def _synthetic_units(n=24):
    rng = np.random.default_rng(3)
    ids = [str(i + 1) for i in range(n)]
    units = pd.DataFrame({
        "method": "cas_net", "scan_id": ids, "status": "ok",
        "b_alt": rng.random(n), "burden_v3": rng.random(n),
        "critical_v3": (rng.random(n) < 0.4).astype(float),
    })
    scores = pd.DataFrame({"method": "cas_net", "scan_id": ids, "score": rng.random(n)})
    return units, scores


def test_stats_block_on_synthetic_units():
    units, scores = _synthetic_units()
    out = bi.stats_block(units, scores)
    assert set(out) == {"pooled", "cas_net"}
    e = out["cas_net"]
    assert e["n"] == 24 and e["n_patients"] == 24
    assert e["decision_branch"] in {"(i)", "(ii)", "(iii)"}
    assert e["verdict"] == bi.VERDICTS[e["decision_branch"]]
    for key in ("spearman_score_balt", "c2_spearman_score_resid", "flagged_top20_minus_rest_balt"):
        assert np.isfinite(e[key]["estimate"])


def test_stats_block_rejects_two_scans_for_one_patient():
    units, scores = _synthetic_units()
    units.loc[1, "scan_id"] = units.loc[0, "scan_id"]
    with pytest.raises(ValueError, match="one scan per patient"):
        bi.stats_block(units, scores)


def test_load_scores_keeps_leading_zeros(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("method,scan_id,score\ncas_net,007,0.5\n")
    df = bi.load_scores(p)
    assert df.loc[0, "scan_id"] == "007"
    assert df.loc[0, "method"] == "cas_net"


def _write_case(root, sid, method, gt, pred, pred_affine, orig, burden=0.1, critical=True):
    aff = np.diag([1.0, 1.0, 1.0, 1.0])
    seg = root / ICX_GT_DIR / "segmentations"
    seg.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(gt.astype(np.uint8), aff), seg / f"{sid}.coronary.nii.gz")
    pdir = root / "preds" / method / sid
    pdir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(pred.astype(np.uint8), pred_affine), pdir / "mask.nii.gz")
    qdir = root / "qa" / "v3" / method
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / f"{sid}.json").write_text(json.dumps({"burden": burden, "critical": critical}))
    orig_path = root / f"orig_{sid}.nii.gz"
    nib.save(nib.Nifti1Image(orig.astype(np.uint8), aff), orig_path)
    return orig_path


def test_score_unit_ok_and_affine_mismatch_is_error(tmp_path):
    gt = make_phantom()
    orig_path = _write_case(tmp_path, "1", "cas_net", gt, gt > 0, np.eye(4), gt > 0)
    ok = bi.score_unit(("cas_net", "1", str(tmp_path), str(orig_path)))
    assert ok["status"] == "ok" and ok["b_alt"] == 0.0 and ok["critical_v3"] == 1.0

    bad_aff = np.diag([1.5, 1.0, 1.0, 1.0])
    orig_path2 = _write_case(tmp_path, "2", "cas_net", gt, gt > 0, bad_aff, gt > 0)
    err = bi.score_unit(("cas_net", "2", str(tmp_path), str(orig_path2)))
    assert err["status"] == "error"
    assert "affine" in err["error"]


def test_score_unit_low_or_zero_support_is_excluded(tmp_path):
    gt = make_phantom()
    tiny = np.zeros_like(gt, dtype=bool)
    tiny[12, 12, 12] = True  # 1 voxel of support
    orig_path = _write_case(tmp_path, "3", "cas_net", gt, gt > 0, np.eye(4), tiny)
    r = bi.score_unit(("cas_net", "3", str(tmp_path), str(orig_path)))
    assert r["status"] == "excluded_low_support"
    assert np.isnan(r["b_alt"])

    empty = np.zeros_like(gt, dtype=bool)
    orig_path0 = _write_case(tmp_path, "4", "cas_net", gt, gt > 0, np.eye(4), empty)
    r0 = bi.score_unit(("cas_net", "4", str(tmp_path), str(orig_path0)))
    assert r0["status"] == "excluded_low_support" and r0["n_orig_mj"] == 0


def test_test_split_refused_without_canonical_read(tmp_path):
    scores = tmp_path / "s.csv"
    scores.write_text("method,scan_id,score\n")
    out = tmp_path / "out"
    base = ["--split", "test", "--methods", "cas_net", "--out", str(out)]
    with pytest.raises(SystemExit):
        bi.main(base + ["--scores", str(scores)])  # no --canonical-read
    with pytest.raises(SystemExit):
        bi.main(base + ["--scores", str(scores), "--canonical-read", str(tmp_path / "missing.json")])
    with pytest.raises(SystemExit):
        bi.main(base)  # no --scores
    out.mkdir()
    (out / "old.txt").write_text("x")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    with pytest.raises(SystemExit):
        bi.main(base + ["--scores", str(scores), "--canonical-read", str(manifest)])  # non-empty --out
