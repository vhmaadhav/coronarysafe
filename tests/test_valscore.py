"""Unit tests for the frozen-model val scorer (m4.valscore): frozen checks, feature assembly,
missing-feature flagging, provenance refusal, and frozen LR math."""

import json
import math

import numpy as np
import pytest

from coronarysafe.m4 import valscore as vs
from coronarysafe.m4.valscore import (
    apply_frozen_lr,
    assemble_features,
    m4_logit,
    refuse_unmatched_cache,
    verify_frozen_dir,
)

FROZEN = {
    "features": ["g_n_components", "g_max_radius_mm", "m3_n_ostia", "m4_logit"],
    "preprocessing": {
        "impute_median": {
            "g_n_components": 4.0,
            "g_max_radius_mm": 1.9,
            "m3_n_ostia": 2.0,
            "m4_logit": -0.83,
        },
        "scale_mean": {"g_n_components": 5.0, "g_max_radius_mm": 2.0, "m3_n_ostia": 2.0, "m4_logit": -1.0},
        "scale_std": {"g_n_components": 1.0, "g_max_radius_mm": 0.2, "m3_n_ostia": 0.5, "m4_logit": 1.0},
    },
    "logistic": {
        "coef": {"g_n_components": 0.5, "g_max_radius_mm": -0.3, "m3_n_ostia": 0.2, "m4_logit": -0.4},
        "intercept": -1.0,
        "C": 0.1,
    },
}


def test_m4_logit_frozen_clamp():
    assert m4_logit(0.5) == 0.0
    assert m4_logit(1e-9) == pytest.approx(math.log(1e-6), abs=1e-6)  # about -13.8155
    assert -13.9 < m4_logit(1e-9) < -13.8
    assert 13.8 < m4_logit(1.0 - 1e-9) < 13.9


def test_assemble_maps_names_and_reports_no_missing():
    qa = {"graph": {"n_components": 7, "max_radius_mm": 2.4, "n_gaps": 3}}
    unit = {"m3": {"n_ostia": 1.0, "unattached_len_mm": 40.0}, "m4_prob": 0.5}
    vals, missing = assemble_features(qa, unit, FROZEN["features"])
    assert missing == []
    assert vals == {"g_n_components": 7.0, "g_max_radius_mm": 2.4, "m3_n_ostia": 1.0, "m4_logit": 0.0}


def test_assemble_absent_source_key_is_flagged_not_imputed():
    vals, missing = assemble_features({}, {"m3": {}}, FROZEN["features"])
    assert missing == FROZEN["features"]  # every feature absent: flagged
    assert "g_n_components" not in vals


def test_assemble_present_nan_is_kept_for_frozen_imputation():
    qa = {"graph": {"n_components": 4, "max_radius_mm": 2.0}}
    unit = {"m3": {"n_ostia": float("nan")}, "m4_prob": 0.5}
    vals, missing = assemble_features(qa, unit, FROZEN["features"])
    assert missing == []
    assert math.isnan(vals["m3_n_ostia"])


def test_apply_frozen_lr_matches_hand_computation():
    vals = {"g_n_components": 7.0, "g_max_radius_mm": 2.4, "m3_n_ostia": 1.0, "m4_logit": 0.0}
    x = np.array([(7.0 - 5.0) / 1.0, (2.4 - 2.0) / 0.2, (1.0 - 2.0) / 0.5, (0.0 + 1.0) / 1.0])
    expected = 1.0 / (1.0 + math.exp(-(-1.0 + float(x @ np.array([0.5, -0.3, 0.2, -0.4])))))
    assert apply_frozen_lr(vals, FROZEN) == pytest.approx(expected, rel=1e-12)


def test_apply_frozen_lr_imputes_nan_with_frozen_median_only():
    vals = {"g_n_components": float("nan"), "g_max_radius_mm": 2.0, "m3_n_ostia": 2.0, "m4_logit": -1.0}
    x = np.array([(4.0 - 5.0) / 1.0, 0.0, 0.0, 0.0])
    expected = 1.0 / (1.0 + math.exp(-(-1.0 + float(x @ np.array([0.5, -0.3, 0.2, -0.4])))))
    assert apply_frozen_lr(vals, FROZEN) == pytest.approx(expected, rel=1e-12)


def test_apply_frozen_lr_refuses_absent_feature():
    vals = {"g_n_components": None, "g_max_radius_mm": 2.0, "m3_n_ostia": 2.0, "m4_logit": -1.0}
    with pytest.raises(ValueError):
        apply_frozen_lr(vals, FROZEN)


def test_verify_frozen_dir_hard_fails_on_mismatch(tmp_path):
    for name in vs.EXPECTED_SHA:
        (tmp_path / name).write_bytes(b"not the frozen file")
    with pytest.raises(SystemExit, match="sha256"):
        verify_frozen_dir(tmp_path)


def test_refuse_cached_units_without_matching_provenance(tmp_path):
    prov = {"split": "val", "frozen_sha256": {"x": "1"}, "code": "t"}
    (tmp_path / "ffr_unet__1.json").write_text(json.dumps({"status": "ok"}))
    with pytest.raises(SystemExit, match="provenance"):
        refuse_unmatched_cache(tmp_path, [("ffr_unet", "1")], prov)
    (tmp_path / "ffr_unet__1.json").write_text(json.dumps({"status": "ok", "provenance": prov}))
    refuse_unmatched_cache(tmp_path, [("ffr_unet", "1")], prov)  # matching: no error


def test_test_split_is_refused():
    with pytest.raises(SystemExit):
        vs.main(["--split", "test", "--root", "x", "--methods", "ffr_unet",
                 "--unit-dir", "x", "--aorta-dir", "x", "--frozen-dir", "x", "--out", "x"])
