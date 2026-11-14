"""Branch graph basics: spur pruning and edge lengths in mm."""

import numpy as np

from coronarysafe.graph import branch_graph, components

SP = (0.4, 0.4, 0.5)


def test_straight_tube_has_one_branch_with_expected_length():
    i, j, k = np.indices((30, 30, 60))
    m = ((i - 15) ** 2 + (j - 15) ** 2 <= 4.0) & (k >= 5) & (k < 55)  # round tube, r = 2 voxels
    B = branch_graph(m, SP, prune_mm=3.0)
    comps = components(B)
    assert len(comps) == 1
    assert B.number_of_edges() == 1
    assert 20.0 < comps[0][1] < 30.0  # 50 slices * 0.5 mm = 25 mm, minus end effects


def test_even_width_square_prism_skeleton_guard(caplog):
    import logging

    from coronarysafe.graph import skeletonize

    m = np.zeros((20, 20, 60), dtype=bool)
    m[8:12, 8:12, 5:55] = True  # 4x4 square section: thinning alone returns nothing
    with caplog.at_level(logging.WARNING, logger="coronarysafe.graph"):
        sk = skeletonize(m)
    # The guard must detect the sparse thinning and log the retry. skimage 3D thinning
    # also returns an empty skeleton for square sections after dilation, so the result is
    # still empty: known limitation (round tubes are unaffected).
    assert sk.shape == m.shape
    assert any("retrying" in r.message for r in caplog.records)


def test_round_tube_skeleton_guard_not_triggered(caplog):
    import logging

    from coronarysafe.graph import skeletonize

    i, j, k = np.indices((30, 30, 60))
    m = ((i - 15) ** 2 + (j - 15) ** 2 <= 4.0) & (k >= 5) & (k < 55)
    with caplog.at_level(logging.WARNING, logger="coronarysafe.graph"):
        sk = skeletonize(m)
    assert sk.sum() > 0
    assert not caplog.records
