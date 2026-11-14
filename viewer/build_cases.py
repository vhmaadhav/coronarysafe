"""Create a private viewer bundle from world-aligned binary reconstruction masks."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from coronarysafe.geometry import bbox_slices
from coronarysafe.graph import branch_graph

ROOT = Path(__file__).resolve().parent


def centerlines(mask, spacing, affine, crop):
    lines = []
    origin = np.array([axis.start for axis in crop])
    for _, _, edge in branch_graph(mask[crop], spacing, prune_mm=2).edges(data=True):
        voxels = np.asarray(edge["voxels"]) + origin
        lines.append(nib.affines.apply_affine(affine, voxels).round(2).tolist())
    return lines


def case(name, reference, prediction, affine, events=None):
    """Use one crop for both layers to preserve alignment and physical coordinates."""
    reference = np.asarray(reference) > 0
    prediction = np.asarray(prediction) > 0
    if reference.ndim != 3 or prediction.shape != reference.shape or not reference.any():
        raise ValueError("expected a nonempty 3D reference and a prediction on the same grid")
    spacing = nib.affines.voxel_sizes(affine)
    crop = bbox_slices(reference | prediction, spacing, pad_mm=3)
    return {
        "name": name,
        "reference": centerlines(reference, spacing, affine, crop),
        "prediction": centerlines(prediction, spacing, affine, crop),
        "markers": events or [],
    }


def from_nifti(name, reference_path, prediction_path, markers_path):
    ref = nib.load(reference_path)
    pred = nib.load(prediction_path)
    if not np.allclose(ref.affine, pred.affine, rtol=0, atol=0.01):
        raise ValueError("reference and prediction have different physical transforms")
    markers = json.loads(markers_path.read_text()) if markers_path else []
    return case(name, np.asanyarray(ref.dataobj), np.asanyarray(pred.dataobj), ref.affine, markers)


def demo_cases():
    import sys

    sys.path.insert(0, str(ROOT.parent / "tests"))
    from phantoms import SPACING, build_gt, lad_gap, tube_mask

    gt = build_gt()
    transform = np.diag([*SPACING, 1])
    gap = lad_gap(gt > 0, 20, 35)
    leak = (gt > 0) | tube_mask((32, 8, 40), (39, 25, 46))
    return [
        case("Synthetic LAD gap", gt, gap, transform,
             [{"label": "LAD gap", "world_mm": [20, 4, 27.5]}]),
        case("Synthetic RCA spur", gt, leak, transform,
             [{"label": "RCA spur", "world_mm": [37, 21, 45]}]),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--name", default="Local reconstruction")
    parser.add_argument("--markers", type=Path, help="JSON list of label and world_mm objects")
    args = parser.parse_args()
    if bool(args.reference) != bool(args.prediction):
        parser.error("--reference and --prediction must be provided together")
    cases = (
        [from_nifti(args.name, args.reference, args.prediction, args.markers)]
        if args.reference else demo_cases()
    )
    encoded = json.dumps(cases, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    template = (ROOT / "template.html").read_text()
    args.output.write_text(template.replace("__CASES_JSON__", encoded))


if __name__ == "__main__":
    main()
