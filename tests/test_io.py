"""NIfTI and legacy VTK POLYDATA readers."""

import nibabel as nib
import numpy as np
import pytest

from coronarysafe.io import load_centerline_vtk, load_nifti, world_to_index


def test_nifti_spacing_and_affine(tmp_path):
    arr = np.zeros((4, 5, 6), dtype=np.uint8)
    arr[1, 2, 3] = 1
    aff = np.diag([0.4, 0.4, 0.5, 1.0])
    img = nib.Nifti1Image(arr, aff)
    img.header.set_zooms((0.4, 0.4, 0.5))
    path = tmp_path / "m.nii.gz"
    nib.save(img, str(path))
    got, sp, affine = load_nifti(path)
    assert got.shape == (4, 5, 6) and got[1, 2, 3] == 1
    assert sp == pytest.approx((0.4, 0.4, 0.5))
    assert np.allclose(affine, aff)
    idx = world_to_index(np.array([[0.8, 0.8, 1.0]]), affine)
    assert np.allclose(idx, [[2, 2, 2]])


ASCII_VTK = """# vtk DataFile Version 3.0
centerline
ASCII
DATASET POLYDATA
POINTS 4 float
0 0 0
1 0 0
2 0 0
5 5 5
LINES 2 6
3 0 1 2
1 3
POINT_DATA 4
SCALARS radius float 1
LOOKUP_TABLE default
1.0
1.5
2.0
3.0
"""


def test_ascii_legacy_polydata(tmp_path):
    p = tmp_path / "c.vtk"
    p.write_text(ASCII_VTK)
    pts, lines, arrays = load_centerline_vtk(p, prefer_vtk=False)
    assert pts.shape == (4, 3)
    assert [list(x) for x in lines] == [[0, 1, 2], [3]]
    assert np.allclose(arrays["radius"], [1.0, 1.5, 2.0, 3.0])


def test_binary_legacy_polydata(tmp_path):
    p = tmp_path / "b.vtk"
    pts = np.array([[0, 0, 0], [0, 0, 1]], dtype=">f4")
    lines = np.array([2, 0, 1], dtype=">i4")
    header = b"# vtk DataFile Version 3.0\ncl\nBINARY\nDATASET POLYDATA\n"
    body = (
        b"POINTS 2 float\n" + pts.tobytes() + b"\n"
        + b"LINES 1 3\n" + lines.tobytes() + b"\n"
    )
    p.write_bytes(header + body)
    got, polys, _ = load_centerline_vtk(p, prefer_vtk=False)
    assert np.allclose(got, [[0, 0, 0], [0, 0, 1]])
    assert [list(x) for x in polys] == [[0, 1]]


def test_binary_vtk51_offsets_connectivity(tmp_path):
    p = tmp_path / "v51.vtk"
    pts = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1]], dtype=">f4")
    offsets = np.array([0, 2, 3], dtype=">i8")
    conn = np.array([0, 1, 2], dtype=">i8")
    header = b"# vtk DataFile Version 5.1\ncl\nBINARY\nDATASET POLYDATA\n"
    body = (
        b"POINTS 3 float\n" + pts.tobytes() + b"\n"
        + b"LINES 2 3\nOFFSETS vtktypeint64\n" + offsets.tobytes() + b"\n"
        + b"CONNECTIVITY vtktypeint64\n" + conn.tobytes() + b"\n"
    )
    p.write_bytes(header + body)
    got, polys, _ = load_centerline_vtk(p, prefer_vtk=False)
    assert got.shape == (3, 3)
    assert [list(x) for x in polys] == [[0, 1], [2]]
