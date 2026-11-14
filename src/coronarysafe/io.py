"""I/O: NIfTI volumes (nibabel) and VTK POLYDATA centerlines.

``load_centerline_vtk`` uses the ``vtk`` package when it is installed (extra
``[vtk]``); otherwise a pure-numpy reader parses legacy ASCII/BINARY
POLYDATA files (LINES + POINTS + optional point scalars/vectors).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_VTK_DTYPES = {
    "bit": np.uint8,
    "unsigned_char": np.uint8,
    "char": np.int8,
    "signed_char": np.int8,
    "short": np.int16,
    "unsigned_short": np.uint16,
    "int": np.int32,
    "unsigned_int": np.uint32,
    "long": np.int64,
    "unsigned_long": np.uint64,
    "vtktypeint32": np.int32,
    "vtktypeuint32": np.uint32,
    "vtktypeint64": np.int64,
    "vtktypeuint64": np.uint64,
    "float": np.float32,
    "double": np.float64,
}


def load_nifti(path):
    """Load a NIfTI volume.

    Returns ``(array, spacing, affine)``: the array in nibabel order, voxel sizes
    in mm per array axis (header zooms), and the voxel-to-world affine.
    """
    import nibabel as nib

    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    return arr, spacing, np.asarray(img.affine, dtype=float)


def world_to_index(points_world, affine):
    """Map (N,3) world-space mm points to fractional voxel indices via the affine."""
    inv = np.linalg.inv(np.asarray(affine, dtype=float))
    pts = np.asarray(points_world, dtype=float).reshape(-1, 3)
    homo = np.column_stack([pts, np.ones(len(pts))])
    return (homo @ inv.T)[:, :3]


def _split_cells(flat):
    flat = np.asarray(flat).astype(np.int64)
    out, i = [], 0
    while i < len(flat):
        n = int(flat[i])
        out.append(flat[i + 1 : i + 1 + n])
        i += n + 1
    return out


def load_centerline_vtk(path, prefer_vtk: bool = True):
    """Read a legacy VTK POLYDATA centerline.

    Returns ``(points, polylines, point_arrays)``:
      * points: (N,3) float array in the file's coordinates (world mm),
      * polylines: list of int arrays indexing into points,
      * point_arrays: dict name -> (N,) or (N,k) array.
    """
    if prefer_vtk:
        try:
            import vtk  # noqa: F401
        except ImportError:
            pass
        else:
            return _read_with_vtk(path)
    return _read_legacy_polydata(path)


def _read_with_vtk(path):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    points = np.asarray(vtk_to_numpy(poly.GetPoints().GetData()), dtype=float)
    polylines = []
    if poly.GetNumberOfLines() > 0:
        polylines = _split_cells(vtk_to_numpy(poly.GetLines().GetData()))
    arrays = {}
    pdata = poly.GetPointData()
    for i in range(pdata.GetNumberOfArrays()):
        arr = pdata.GetArray(i)
        arrays[arr.GetName() or f"array{i}"] = np.asarray(vtk_to_numpy(arr)).copy()
    return points, polylines, arrays


def _read_legacy_polydata(path):
    buf = Path(path).read_bytes()
    state = {"pos": 0}

    def readline():
        p = state["pos"]
        if p >= len(buf):
            return ""
        end = buf.find(b"\n", p)
        end = len(buf) if end < 0 else end
        state["pos"] = end + 1
        return buf[p:end].decode("ascii", "replace").strip()

    def read_block(count, dtype_name, binary):
        key = dtype_name.lower()
        if key not in _VTK_DTYPES:
            raise ValueError(f"unsupported VTK data type: {dtype_name}")
        dt = np.dtype(_VTK_DTYPES[key])
        if binary:
            p = state["pos"]
            arr = np.frombuffer(buf, dtype=dt.newbyteorder(">"), count=count, offset=p).astype(dt)
            p += count * dt.itemsize
            if p < len(buf) and buf[p : p + 1] == b"\n":
                p += 1
            state["pos"] = p
            return arr
        vals = []
        while len(vals) < count:
            if state["pos"] >= len(buf):
                raise ValueError("unexpected end of VTK file")
            vals.extend(readline().split())
        return np.array(vals[:count], dtype=dt)

    readline()  # version header
    readline()  # title
    binary = readline().upper() == "BINARY"

    points = None
    n_points = 0
    polylines = []
    arrays = {}
    while state["pos"] < len(buf):
        line = readline()
        if not line:
            continue
        tok = line.split()
        kw = tok[0].upper()
        if kw == "DATASET":
            if tok[1].upper() != "POLYDATA":
                raise ValueError(f"only POLYDATA is supported, got {tok[1]}")
        elif kw == "POINTS":
            n_points = int(tok[1])
            points = read_block(3 * n_points, tok[2], binary).astype(float).reshape(-1, 3)
        elif kw in ("LINES", "VERTICES", "POLYGONS", "TRIANGLE_STRIPS"):
            ncells, size = int(tok[1]), int(tok[2])
            if buf.startswith(b"OFFSETS", state["pos"]):
                # VTK 5.1 layout: OFFSETS <type> then CONNECTIVITY <type> blocks
                otype = readline().split()[1]
                offsets = read_block(ncells + 1, otype, binary)
                ctype = readline().split()[1]
                conn = read_block(size, ctype, binary)
                cells = [conn[offsets[i] : offsets[i + 1]] for i in range(ncells)]
            else:
                cells = _split_cells(read_block(size, "int", binary))
            if kw == "LINES":
                polylines.extend(cells)
        elif kw == "POINT_DATA":
            n_points = int(tok[1])
        elif kw in ("CELL_DATA", "FIELD"):
            break  # trailing cell/field data (e.g. string segment names) is not read
        elif kw == "SCALARS":
            name, dtype_name = tok[1], tok[2]
            ncomp = int(tok[3]) if len(tok) > 3 else 1
            readline()  # LOOKUP_TABLE line
            vals = read_block(n_points * ncomp, dtype_name, binary).reshape(n_points, ncomp)
            arrays[name] = vals[:, 0] if ncomp == 1 else vals
        elif kw in ("VECTORS", "NORMALS"):
            name = tok[1] if len(tok) > 1 else kw.lower()
            dtype_name = tok[2] if len(tok) > 2 else "float"
            arrays[name] = read_block(n_points * 3, dtype_name, binary).reshape(n_points, 3)
        else:
            raise ValueError(f"unsupported VTK keyword: {kw}")

    if points is None:
        raise ValueError("no POINTS block in VTK file")
    return points, polylines, arrays
