"""The parity gate's metrics. The negative control matters most: cosine and
top-k are both blind to a uniformly scaled embedding, so the raw-error check
must catch it or the gate is not live."""
import numpy as np
import pytest

from scripts.preflight.miewid_dinov3 import metrics

LIMITS = {"max_abs": 5e-2, "rel_l2": 1e-2}


def test_identical_passes():
    a = np.random.default_rng(0).normal(size=(19, 1536)).astype(np.float32)
    assert metrics.check_stage("embedding", a, a, LIMITS)["pass"]
    assert metrics.topk_agreement(a, a, 5) == 1.0
    assert metrics.cosine_distance(a[0], a[0]) < 1e-6


def test_scaled_embedding_fails_raw_check_but_not_cosine():
    a = np.random.default_rng(0).normal(size=(19, 1536)).astype(np.float32)
    scaled = a * 1.05
    assert not metrics.check_stage("embedding", scaled, a, LIMITS)["pass"]
    assert metrics.cosine_distance(scaled[0], a[0]) < 1e-6
    assert metrics.topk_agreement(scaled, a, 5) == 1.0


def test_shape_mismatch_raises_instead_of_broadcasting():
    with pytest.raises(ValueError, match="shape"):
        metrics.check_stage("embedding", np.zeros((1, 1536)), np.zeros((19, 1536)), LIMITS)


def test_rel_l2_and_max_abs():
    ref = np.array([3.0, 4.0])
    got = np.array([3.0, 4.5])
    assert metrics.max_abs(got, ref) == 0.5
    assert abs(metrics.rel_l2(got, ref) - 0.1) < 1e-12


def test_topk_agreement_detects_reordering():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(10, 8))
    got = ref.copy()
    got[[0, 1]] = got[[1, 0]]
    assert metrics.topk_agreement(got, ref, 3) < 1.0


def test_check_chip():
    a = np.zeros((10, 12, 3), np.uint8)
    lim = {"max_abs": 8, "mean_abs": 0.5}
    assert metrics.check_chip(a, a, lim)["pass"]
    assert not metrics.check_chip(a, np.zeros((12, 10, 3), np.uint8), lim)["pass"]
    b = a.copy(); b[0, 0, 0] = 200
    assert not metrics.check_chip(b, a, lim)["pass"]


def test_router_bbox_truncates_toward_zero():
    assert metrics.router_bbox([612.9, -213.7, 1101.0, 2226.5]) == (612, -213, 1101, 2226)


# ---- F1/F2/F4: gallery completeness, known divergence, exit policy ----

import json
import pathlib

MANIFEST = json.load(open(
    pathlib.Path(__file__).resolve().parents[1] / "scripts/preflight/miewid_dinov3/manifest.json"))
EXPECTED = MANIFEST["expected_chips"]


def _rows(expected=EXPECTED):
    return [{"case": e["case"], "species": e["species"], "name": e["name"],
             "bbox": "[0,0,1,1]", "theta": "0"} for e in expected]


def test_manifest_lists_19_chips_and_3_known_divergence():
    assert len(EXPECTED) == 19
    assert all(isinstance(e["name"], str) for e in EXPECTED)
    assert [e["case"] for e in MANIFEST["known_divergence"]] == ["bbox_overhang"] * 3


def test_exact_19_accepted():
    assert metrics.check_expected_chips(_rows(), EXPECTED) == []


def test_one_row_reference_rejected():
    assert metrics.check_expected_chips(_rows()[:1], EXPECTED)


def test_missing_row_rejected():
    assert metrics.check_expected_chips(_rows()[:5] + _rows()[6:], EXPECTED)


def test_extra_row_rejected():
    assert metrics.check_expected_chips(_rows() + _rows()[:1], EXPECTED)


def test_duplicate_row_rejected():
    rows = _rows()
    rows[3] = dict(rows[2])
    assert metrics.check_expected_chips(rows, EXPECTED)


def test_reordered_rows_rejected():
    rows = _rows()
    rows[0], rows[1] = rows[1], rows[0]
    assert metrics.check_expected_chips(rows, EXPECTED)


def test_topk_raises_when_rows_not_more_than_k():
    a = np.random.default_rng(0).normal(size=(5, 8))
    with pytest.raises(ValueError, match="rows"):
        metrics.topk_agreement(a, a, 5)
    assert metrics.topk_agreement(np.vstack([a, a[:1] + 1]), np.vstack([a, a[:1] + 1]), 5) == 1.0


def test_row_selection_excludes_known_divergence_from_gated_e2e():
    sel = metrics.select_rows(EXPECTED, MANIFEST["known_divergence"])
    assert sel["arch"] == list(range(19))
    assert sel["reported"] == [4, 5, 6]
    assert sel["gated"] == [i for i in range(19) if i not in (4, 5, 6)]
    assert len(sel["gated"]) == 16


def test_known_divergence_must_match_expected_chip():
    bad = [{"case": "bbox_overhang", "species": "lion", "name": "1", "reason": "x"}]
    assert metrics.check_known_divergence(bad, EXPECTED)
    assert metrics.check_known_divergence(MANIFEST["known_divergence"], EXPECTED) == []


def test_nan_cosine_fails_the_stage():
    assert metrics.cosine_gate([1e-5, 2e-5], 1e-3)["pass"]
    assert not metrics.cosine_gate([1e-5, float("nan")], 1e-3)["pass"]
    assert not metrics.cosine_gate([float("inf")], 1e-3)["pass"]
    assert not metrics.cosine_gate([2e-3], 1e-3)["pass"]


def test_proposed_thresholds_exit_nonzero_unless_allowed():
    assert metrics.exit_code(True, "agreed", False) == 0
    assert metrics.exit_code(True, "proposed", False) != 0
    assert metrics.exit_code(True, "proposed", True) == 0
    assert metrics.exit_code(False, "agreed", True) != 0
