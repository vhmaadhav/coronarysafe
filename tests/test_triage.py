"""Validation split and unseen-segmenter regression checks."""

import numpy as np
import pandas as pd
import pytest

from coronarysafe.eval.triage import GRAPH, UNCERTAINTY, evaluate


def _cohort(ids, methods):
    rows = []
    for method in methods:
        for scan in ids:
            value = float(scan % 2)
            rows.append(
                {"method": method, "scan_id": str(scan), "critical": bool(value)}
                | {key: value for key in GRAPH + UNCERTAINTY}
            )
    return pd.DataFrame(rows)


def test_independent_transfer_excludes_test_model_and_patients():
    development = _cohort(range(40), ["a", "b"])
    test = _cohort(range(40, 60), ["c"])
    test.loc[:, UNCERTAINTY] = np.nan  # not required for graph-only transfer
    result = evaluate(development, n_boot=20, test=test)["c"]
    assert result["model_for_paired_comparison"] == "unseen_segmenter_graph"
    assert result["scores"]["unseen_segmenter_graph"]["normalised_recall"]["estimate"] == 1
    with pytest.raises(ValueError, match="overlapping patients"):
        evaluate(development, n_boot=20, test=_cohort(range(30, 50), ["c"]))
