"""CoronarySafe: QA layer for coronary CTA segmentations.

Conventions used throughout the package
---------------------------------------
* Volumes are numpy arrays in nibabel array order (axis 0, 1, 2 = i, j, k).
* ``spacing`` is a 3-tuple of voxel sizes in mm, one per array axis
  (nibabel ``header.get_zooms()[:3]``).
* Graph and point coordinates are *index-mm* (voxel index times spacing) or
  raw voxel indices (float) where stated. Use ``io.world_to_index`` to bring
  world-space centerlines (mm, RAS) into voxel-index space via the NIfTI affine.
"""

__version__ = "0.1.0"
