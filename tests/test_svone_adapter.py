"""svOneDSolver adapter: deck structure and output parsing on a synthetic tree (droplet only)."""

import numpy as np

from coronarysafe.analysis import hemo1d as h
from coronarysafe.analysis.harm import nearest_gt_labels
from coronarysafe.analysis import svone_adapter as sv
from phantoms import SPACING, build_gt


def _tree():
    gt = build_gt()
    lab = nearest_gt_labels(gt, SPACING)
    from coronarysafe.analysis.hemo1d import pick_roots, build_tree, assign_outlets, solve
    from coronarysafe.graph import branch_graph
    B = branch_graph(gt > 0, SPACING, prune_mm=3.0)
    roots, _ = pick_roots(B, None)
    net, _ = build_tree(B, roots, SPACING, lab, h.H_MM, (gt > 0).astype(float))
    assign_outlets(net, 3e-6 * max(1, len(roots)))
    return net


def test_deck_has_one_segment_per_chain_and_resistance_outlets(tmp_path):
    net = _tree()
    deck, meta = sv.build_deck(net)
    n_chains = len(meta["chains"])
    segs = [ln for ln in deck.splitlines() if ln.startswith("SEGMENT ")]
    assert len(segs) == meta["n_segments"] == n_chains + (1 if meta["trunk"] else 0)
    assert "SOLVEROPTIONS" in deck and "INLETDATA PRESSURE" in deck
    assert "RESISTANCE" in deck
    total_len_cm = sum(float(ln.split()[3]) for ln in segs if ln.split()[1] != "trunk")
    assert abs(total_len_cm * 10.0 - float(net["L"].sum()) * 1000.0) < 1e-3 * float(net["L"].sum()) * 1000.0 + 1e-6


def test_parse_pressures_and_ffr_mapping(tmp_path):
    net = _tree()
    deck, meta = sv.build_deck(net)
    seg_ids = sorted(meta["seg_of"][c] for c in range(len(meta["chains"])))
    if meta["trunk"]:
        seg_ids = [0] + seg_ids
    for sid in seg_ids:
        n_nodes = len(meta["chains"][next(c for c in meta["seg_of"] if meta["seg_of"][c] == sid)]) + 1 if sid else 2
        row = " ".join(f"{sv.P_AORTA * sv.CGS_PER_PA * (0.99 - 0.001 * k):.6f}" for k in range(n_nodes))
        (tmp_path / f"pressure_seg{sid}.txt").write_text(f"{row}\n{row}\n")
    pressures = sv.read_pressures(str(tmp_path), seg_ids)
    ffr = sv.ffr_by_element(net, meta, pressures)
    assert ffr.shape == (len(net["parent"]),)
    assert np.all(np.isfinite(ffr))
    assert np.all((ffr > 0) & (ffr <= 1))
