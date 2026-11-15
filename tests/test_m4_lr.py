"""Per-scan left/right orientation from the affine (no GT). Synthetic affines only."""

import numpy as np

from coronarysafe.m4.lr_convention import lr_from_affine


def _aff(diag, perm=(0, 1, 2)):
    A = np.zeros((4, 4))
    for i, (j, d) in enumerate(zip(perm, diag)):
        A[j, i] = d  # voxel axis i maps to world axis j with scale d
    A[3, 3] = 1.0
    return A


def test_identity_affine_left_is_plus_x():
    u = [1.0, 0.0, 0.0]
    assert lr_from_affine(_aff([1, 1, 1]), u) == (0, 1)


def test_flipped_axis_gives_negative_sign():
    u = [1.0, 0.0, 0.0]
    assert lr_from_affine(_aff([-0.5, 0.5, 0.5]), u) == (0, -1)


def test_permuted_axes_pick_the_right_voxel_axis():
    # voxel axis 2 is world x here; world +x is patient left
    u = [1.0, 0.0, 0.0]
    axis, sign = lr_from_affine(_aff([0.5, 0.5, 0.5], perm=(1, 2, 0)), u)
    assert axis == 2 and sign == 1
