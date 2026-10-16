"""MiewID inference: per-instance input size, DINOv3-only autocast, and no
non-finite embedding ever reaching Wildbook."""
import base64
import io
import pathlib
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from app.models import miewid as miewid_mod
from app.models.dinov3_vit import DINOv3ViTConfig
from app.models.miewid import (
    BACKBONE_DINOV3_VITB16, DinoV3MiewIdNet, MiewidModel, build_miewid_transform,
)

FIX = pathlib.Path(__file__).parent / "fixtures" / "dinov3"


def _png(arr):
    # Mode is inferred from the array: (H,W,4) uint8 -> RGBA, (H,W) uint8 -> L,
    # (H,W) uint16 -> I;16. (fromarray's `mode=` argument is deprecated.)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _rgb_png(h=80, w=60):
    rng = np.random.default_rng(0)
    return _png(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def _tiny_dinov3_model(device="cpu", precision="fp32"):
    """A MiewidModel wired to the tiny golden DINOv3 net (64x64 input)."""
    golden = torch.load(FIX / "golden_tiny.pt", weights_only=True)
    net = DinoV3MiewIdNet(DINOv3ViTConfig(**golden["config"]),
                          input_size=tuple(golden["input_size"])).eval()
    net.load_state_dict(golden["state_dict"], strict=True)
    m = MiewidModel()
    m.device = device
    m.model = net.to(device)
    m.backbone_name = BACKBONE_DINOV3_VITB16
    m.image_size = tuple(golden["input_size"])
    m.preprocess = build_miewid_transform(m.image_size)
    m.precision = precision
    return m


class _Const(torch.nn.Module):
    def __init__(self, value, dim=8):
        super().__init__()
        self.value, self.dim = value, dim

    def forward(self, x):
        return torch.full((x.shape[0], self.dim), self.value)


def test_new_without_init_still_extracts_fp32():
    """Existing tests build instances via MiewidModel.__new__ (skipping
    __init__); class-level defaults must keep extract_embeddings working."""
    m = MiewidModel.__new__(MiewidModel)
    m.inference_lock = threading.RLock()
    m.device = "cpu"
    m.model = _Const(1.0)
    m.preprocess = build_miewid_transform((440, 440))
    assert m.extract_embeddings(_rgb_png()).shape == (1, 8)


def test_bare_instance_defaults():
    m = MiewidModel()
    assert MiewidModel.IMAGE_SIZE == (440, 440)
    assert m.image_size == (440, 440)
    assert m.precision == "fp32"
    out = m.preprocess(image=np.zeros((30, 20, 3), np.uint8))["image"]
    assert tuple(out.shape) == (3, 440, 440)


def test_dinov3_instance_preprocesses_to_its_own_size():
    m = MiewidModel()
    m.image_size = (448, 448)
    m.preprocess = build_miewid_transform(m.image_size)
    assert tuple(m.preprocess(image=np.zeros((30, 20, 3), np.uint8))["image"].shape) == (3, 448, 448)


def test_tiny_dinov3_extract_returns_float32():
    m = _tiny_dinov3_model()
    emb = m.extract_embeddings(_rgb_png(), bbox=(5, 5, 40, 50), theta=0.3)
    assert emb.shape == (1, 128) and emb.dtype == np.float32
    assert np.isfinite(emb).all()


@pytest.mark.parametrize("make", [
    lambda: _png(np.zeros((40, 40, 4), np.uint8)),
    lambda: _png(np.zeros((40, 40), np.uint8)),
    lambda: _png(np.arange(1600, dtype=np.uint16).reshape(40, 40) * 40),
])
def test_dinov3_accepts_non_rgb_inputs(make):
    emb = _tiny_dinov3_model().extract_embeddings(make())
    assert emb.shape == (1, 128) and np.isfinite(emb).all()


def test_fp32_models_never_enter_autocast(monkeypatch):
    spy = MagicMock()
    monkeypatch.setattr(miewid_mod.torch, "autocast", spy)
    m = MiewidModel()
    m.device = "cpu"
    m.model = _Const(1.0)
    m.extract_embeddings(_rgb_png())
    spy.assert_not_called()


def test_dinov3_cuda_precision_selects_fp16_autocast(monkeypatch):
    spy = MagicMock()
    monkeypatch.setattr(miewid_mod.torch, "autocast", spy)
    m = MiewidModel()
    m.precision = "fp16_autocast"
    m._autocast()
    spy.assert_called_once_with(device_type="cuda", dtype=torch.float16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_dinov3_runs_fp16_internally_on_cuda_and_returns_float32():
    m = _tiny_dinov3_model(device="cuda", precision="fp16_autocast")
    seen = {}

    def record_dtype(mod, inp, out):
        # Must return None: a non-None return REPLACES the module output.
        seen["dtype"] = out.dtype

    m.model.backbone.model.layer[0].mlp.up_proj.register_forward_hook(record_dtype)
    emb = m.extract_embeddings(_rgb_png())
    assert seen["dtype"] == torch.float16
    assert next(m.model.parameters()).dtype == torch.float32
    assert emb.dtype == np.float32


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_embedding_is_a_500(bad):
    m = MiewidModel()
    m.device = "cpu"
    m.model = _Const(bad)
    with pytest.raises(HTTPException) as e:
        m.extract_embeddings(_rgb_png())
    assert e.value.status_code == 500


def _nan_model():
    m = MiewidModel()
    m.device = "cpu"
    m.model = _Const(float("nan"))
    return m


def _app(router, models, infos):
    app = FastAPI()
    app.include_router(router)
    handler = MagicMock()
    handler.get_model.side_effect = models.get
    handler.get_model_info.side_effect = infos.get
    handler.list_models.return_value = {k: {} for k in models}
    app.state.model_handler = handler
    return TestClient(app, raise_server_exceptions=False)


DATA_URI = "data:image/png;base64," + base64.b64encode(_rgb_png()).decode()


def test_extract_endpoint_returns_500_not_nan():
    from app.routers import extract_router
    client = _app(extract_router.router, {"e": _nan_model()}, {"e": {"config": {"version": 5}}})
    r = client.post("/extract/", json={"model_id": "e", "image_uri": DATA_URI})
    assert r.status_code == 500, r.text
    # FastAPI already 500s on NaN JSON, so "500" alone passes vacuously; the
    # guard's own message proves extraction ran and the guard fired.
    assert "non-finite embedding" in r.text, r.text


def test_pipeline_endpoint_returns_500_not_nan():
    from app.models.lightnet_model import LightNetModel
    from app.routers import pipeline_router
    detector = MagicMock(spec=LightNetModel)
    detector.predict.return_value = {"bboxes": [[1, 1, 20, 20]], "scores": [0.9],
                                     "class_names": ["x"], "num_detections": 1}
    client = _app(pipeline_router.router, {"p": detector, "e": _nan_model()},
                  {"p": {"config": {}}, "e": {"config": {"version": 5}}})
    r = client.post("/pipeline/", json={"predict_model_id": "p", "extract_model_id": "e",
                                        "image_uri": DATA_URI})
    assert r.status_code == 500, r.text
    # FastAPI already 500s on NaN JSON, so "500" alone passes vacuously; the
    # guard's own message proves extraction ran and the guard fired.
    assert "non-finite embedding" in r.text, r.text
