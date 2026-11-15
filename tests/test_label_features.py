"""Local test of the Modal per-case job logic (infra/modal/label_features.py) on a synthetic case
laid out like the Volume: GT NIfTI, LPS centerline VTKs, prediction mask/prob/meta."""

import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from phantoms import SPACING, TUBES, build_gt, lad_gap

REPO = Path(__file__).resolve().parents[1]
GT_DIR = "imagecas_x/ImageCAS-X_dataset_v3"


def _load_module():
    # The Modal entry point imports the modal package, which is not in the droplet inference venv.
    pytest.importorskip("modal", reason="modal is not installed; Modal entry point not exercised")
    path = REPO / "infra" / "modal" / "label_features.py"
    spec = importlib.util.spec_from_file_location("label_features", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _save(arr, path, dtype=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    aff = np.diag([-SPACING[0], SPACING[1], SPACING[2], 1.0])  # RAS, x flipped
    img = nib.Nifti1Image(arr if dtype is None else arr.astype(dtype), aff)
    img.header.set_zooms(SPACING)
    nib.save(img, str(path))


def _vtk_ascii(path, polylines_mm, flip_to_lps=True):
    """Legacy ASCII POLYDATA. Phantom mm (x, y, z) -> LPS (x, -y, z) like a real export."""
    pts, lines, k = [], [], 0
    for line in polylines_mm:
        idx = []
        for p in line:
            x, y, z = p
            pts.append((x, -y, z) if flip_to_lps else (-x, y, z))
            idx.append(k)
            k += 1
        lines.append(idx)
    header = ["# vtk DataFile Version 3.0", "cl", "ASCII", "DATASET POLYDATA"]
    text = header + [f"POINTS {len(pts)} float"] + [f"{a} {b} {c}" for a, b, c in pts]
    text.append(f"LINES {len(lines)} {sum(len(c) + 1 for c in lines)}")
    text += [f"{len(c)} " + " ".join(map(str, c)) for c in lines]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text) + "\n")


def _segment_samples(p0, p1, step=1.0):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = max(2, int(np.linalg.norm(p1 - p0) / step) + 1)
    return [tuple(p0 + (p1 - p0) * t) for t in np.linspace(0, 1, n)]


def _write_case(root, flip_to_lps=True, gap=(20.0, 35.0)):
    base = root / GT_DIR
    gt = build_gt()
    _save(gt, base / "segmentations" / "1.coronary.nii.gz", np.uint8)
    left = [_segment_samples(*s) for lab in (1, 2, 3, 4) for s in TUBES[lab]]
    right = [_segment_samples(*s) for lab in (9, 10) for s in TUBES[lab]]
    _vtk_ascii(base / "centerlines" / "1.coronary_left_centerline.vtk", left, flip_to_lps)
    _vtk_ascii(base / "centerlines" / "1.coronary_right_centerline.vtk", right, flip_to_lps)
    pred = lad_gap(gt > 0, *gap)
    pdir = root / "preds" / "cas_net" / "1"
    _save(pred, pdir / "mask.nii.gz", np.uint8)
    _save(np.where(pred, 0.9, 0.05).astype(np.float32), pdir / "prob.nii.gz", np.float32)
    (pdir / "meta.json").write_text(json.dumps({"prob_cropped": False, "warnings": []}))


def test_process_case_on_synthetic_volume_layout(tmp_path):
    lf = _load_module()
    _write_case(tmp_path)
    res = lf.process_case(str(tmp_path), "cas_net", "1")
    assert res["status"] == "ok", res
    out = json.loads((tmp_path / "qa" / lf.QA_VERSION / "cas_net" / "1.json").read_text())
    assert out["centerline_status"] == "vtk" and not out["centerline_fallback"]
    assert out["centerline_source"] == "provided"
    assert "F1" in out["flags"] and out["severe"]
    assert 0.0 < out["dice"] < 1.0
    assert out["prob_present"] and "entropy" not in out  # uncertainty keys live under 'uncertainty'
    assert "mean_entropy_mask" in out["uncertainty"]
    # idempotent: second call is skipped
    assert lf.process_case(str(tmp_path), "cas_net", "1")["status"] == "skipped"


def test_misaligned_centerlines_fall_back_and_are_flagged(tmp_path):
    lf = _load_module()
    _write_case(tmp_path, flip_to_lps=False)
    lf.process_case(str(tmp_path), "cas_net", "1", force=True)
    out = json.loads((tmp_path / "qa" / lf.QA_VERSION / "cas_net" / "1.json").read_text())
    assert out["status"] == "ok"
    assert out["centerline_fallback"] is True
    assert out["centerline_source"] == "gt_skeleton"


def test_missing_prediction_writes_error_json(tmp_path):
    lf = _load_module()
    _write_case(tmp_path)
    res = lf.process_case(str(tmp_path), "swin_unetr", "1")
    assert res["status"] == "error"
    out = json.loads((tmp_path / "qa" / lf.QA_VERSION / "swin_unetr" / "1.json").read_text())
    assert out["status"] == "error" and "no prediction" in out["error"]


def test_changed_prediction_invalidates_cache_and_shifted_grid_errors(tmp_path):
    lf = _load_module()
    _write_case(tmp_path)
    assert lf.process_case(str(tmp_path), "cas_net", "1")["status"] == "ok"
    pdir = tmp_path / "preds" / "cas_net" / "1"
    path = pdir / "mask.nii.gz"
    image = nib.load(path)
    shifted = image.affine.copy()
    shifted[0, 3] += 1.0
    nib.save(nib.Nifti1Image(np.asanyarray(image.dataobj), shifted), path)
    (pdir / "meta.json").write_text(json.dumps({"prob_cropped": False, "tta": True}))
    assert lf.process_case(str(tmp_path), "cas_net", "1")["status"] == "error"
    record = tmp_path / "qa" / lf.QA_VERSION / "cas_net" / "1.json"
    assert "affine differs" in json.loads(record.read_text())["error"]
