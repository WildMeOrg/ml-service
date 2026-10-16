"""PairX fails closed for DINOv3 MiewID models.

PairX hooks spatial CNN feature maps and has no ViT canonizer. A DINOv3
module name would pass the router's named_modules() check and then fail deep
inside PairX, so the capability check must run at model resolution, before
any image is fetched, whatever layer_key the caller sends.
"""
import inspect
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.models.miewid import MiewidModel, build_miewid_transform
from app.routers import explain_router


@pytest.fixture(autouse=True)
def _restore_default_model_id():
    original = explain_router._default_model_id
    yield
    explain_router._default_model_id = original


def _dinov3_stub():
    m = MagicMock(spec=MiewidModel)
    m.inference_lock = threading.RLock()
    m.supports_pairx = False
    m.backbone_name = "dinov3_vitb16"
    m.preprocess = build_miewid_transform((448, 448))
    return m


def _client(models):
    app = FastAPI()
    app.include_router(explain_router.router)
    handler = MagicMock()
    handler.get_model.side_effect = models.get
    handler.list_models.return_value = {k: {} for k in models}
    app.state.model_handler = handler
    app.state.device = "cpu"
    return TestClient(app, raise_server_exceptions=False)


def _body(**over):
    b = {"image1_uris": ["https://example.invalid/a.jpg"], "bb1": [[0, 0, 10, 10]],
         "image2_uris": ["https://example.invalid/b.jpg"], "bb2": [[0, 0, 10, 10]]}
    b.update(over)
    return b


@pytest.mark.parametrize("extra", [{}, {"layer_key": "backbone.model.layer.3"}])
def test_dinov3_model_is_400_before_any_fetch(extra):
    client = _client({"dino": _dinov3_stub()})
    with patch.object(explain_router, "process_image") as fetch:
        r = client.post("/explain/", json=_body(model_id="dino", **extra))
    assert r.status_code == 400, r.text
    assert "PairX is not supported" in r.text and "dinov3_vitb16" in r.text
    fetch.assert_not_called()


def test_default_model_id_pointing_at_dinov3_is_400():
    explain_router._default_model_id = "dino"
    client = _client({"dino": _dinov3_stub()})
    with patch.object(explain_router, "process_image") as fetch:
        r = client.post("/explain/", json=_body())
    assert r.status_code == 400, r.text
    fetch.assert_not_called()


def test_explain_preprocess_uses_the_instance_transform():
    out = explain_router.preprocess(np.zeros((50, 60, 3), np.uint8), _dinov3_stub())
    assert tuple(out.shape) == (3, 448, 448)


def test_explain_router_no_longer_reads_the_class_constant():
    assert "IMAGE_SIZE" not in inspect.getsource(explain_router)
