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
