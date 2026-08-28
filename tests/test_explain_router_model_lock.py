"""PairX's forward hook must not see another request's forward.

Flukebook, 2026-08-28 20:06 -- `RuntimeError: element 0 of tensors does not
require grad and does not have a grad_fn` from
pairx/core.py:140 `intermediate_fm.backward(...)`.

pairx/core.py:23-43 captures the feature map with a forward hook on the
SHARED model submodule:

    handles.append(submodule.register_forward_hook(get_intermediate_hook(k)))
    embedding = model(img)      # captures a grad-bearing fm
                                # <-- any other forward here overwrites it
    for handle in handles: handle.remove()

/extract and /pipeline run `self.model(x)` under torch.no_grad() on that same
MiewID instance. A forward landing in that window replaces the captured tensor
with one that has no grad_fn, and PairX's backward then fails.

Before the PairX offload (#42) run_pairx ran inline on the event loop, which
blocked everything else and made the window unreachable by accident. The
offload removed that, so the exclusion now has to be explicit.
"""
import threading

import numpy as np
import pytest
import torch
import torch.nn as nn

from app.models.miewid import MiewidModel


class _SharedBackbone(nn.Module):
    """Stands in for MiewID: one instance, reachable by both endpoints."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())

    def forward(self, x):
        return self.blocks(x)


def test_concurrent_forward_corrupts_an_unguarded_hook():
    """Characterizes the bug itself, so the guard below is not vacuous."""
    model = _SharedBackbone()
    captured = {}
    handle = model.blocks[0].register_forward_hook(
        lambda m, i, o: captured.__setitem__("fm", o))

    img = torch.randn(1, 3, 8, 8, requires_grad=True)
    model(img)
    assert captured["fm"].grad_fn is not None, "PairX's own forward is grad-bearing"

    with torch.no_grad():                    # the interloping /extract forward
        model(torch.randn(1, 3, 8, 8))
    handle.remove()

    assert captured["fm"].grad_fn is None
    with pytest.raises(RuntimeError, match="does not require grad"):
        captured["fm"].backward(gradient=torch.zeros_like(captured["fm"]))


def test_miewid_exposes_a_lock_excluding_concurrent_forwards():
    """The model instance must offer the mutual exclusion both callers use."""
    m = MiewidModel()
    assert hasattr(m, "inference_lock"), \
        "MiewidModel needs a lock /explain and /extract can share"
    assert m.inference_lock.acquire(blocking=False)
    m.inference_lock.release()


def test_extract_embeddings_holds_the_lock_during_its_forward():
    """/extract must not run a forward while PairX holds the model."""
    m = MiewidModel()
    m.device = "cpu"
    m.model = _SharedBackbone()
    m.preprocess = lambda image: {
        "image": torch.zeros(3, 8, 8)}

    forward_ran = threading.Event()
    preprocessed = threading.Event()
    original = m.model.forward
    m.model.forward = lambda x: (forward_ran.set(), original(x))[1]
    # Signals that the worker has finished decode/preprocess and is at the
    # forward. Without this the assertion below could pass merely because a
    # slow PIL decode had not reached the lock yet.
    m.preprocess = lambda image: (preprocessed.set(),
                                  {"image": torch.zeros(3, 8, 8)})[1]

    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")

    m.inference_lock.acquire()               # stand in for PairX holding it
    try:
        t = threading.Thread(target=lambda: m.extract_embeddings(buf.getvalue()))
        t.start()
        assert preprocessed.wait(5), \
            "/extract never reached the forward; the test would be vacuous"
        assert not forward_ran.wait(0.5), \
            "/extract ran a forward while the model was held by another caller"
    finally:
        m.inference_lock.release()
        t.join(5)
    assert forward_ran.wait(5), "/extract must proceed once the lock is released"


def test_explain_holds_the_model_lock_across_pairx():
    """The /explain path must hold the model for the whole PairX call.

    And it must take the lock inside the worker thread, never on the event
    loop -- a threading lock acquired in the handler would reintroduce the
    starvation #42 fixed.
    """
    import httpx
    from unittest.mock import MagicMock, patch
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import cv2

    from app.routers import explain_router
    from app.utils import image_uri

    ok, buf = cv2.imencode(".png", np.full((8, 8, 3), 127, np.uint8))
    png = buf.tobytes()

    entry = MagicMock(spec=MiewidModel)
    entry.model = MagicMock()
    entry.inference_lock = threading.RLock()

    observed = {}
    loop_thread = {}

    real_process_image = explain_router.process_image

    async def recording_process_image(*a, **k):
        # process_image is awaited on the event loop, so this is that thread.
        loop_thread["t"] = threading.current_thread()
        return await real_process_image(*a, **k)

    def fake_run_pairx(*a, **k):
        # Held by us, on this thread -- so prove another thread cannot take it.
        taken = []
        t = threading.Thread(
            target=lambda: taken.append(
                entry.inference_lock.acquire(blocking=False)))
        t.start(); t.join(5)
        observed["excluded"] = taken == [False]
        observed["thread"] = threading.current_thread()
        return [np.zeros((4, 4, 3), np.uint8)]

    image_uri.init_image_fetch(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, content=png)))
    app = FastAPI()
    app.include_router(explain_router.router)
    handler = MagicMock()
    handler.get_model.side_effect = lambda mid: {"miewid-msv4.1": entry}.get(mid)
    handler.list_models.return_value = {"miewid-msv4.1": {}}
    app.state.model_handler = handler
    app.state.device = "cpu"

    with patch.object(explain_router, "run_pairx", fake_run_pairx), \
         patch.object(explain_router, "process_image", recording_process_image):
        r = TestClient(app, raise_server_exceptions=False).post("/explain/", json={
            "image1_uris": ["https://wb.example/a.jpg"], "bb1": [[0, 0, 8, 8]],
            "image2_uris": ["https://wb.example/b.jpg"], "bb2": [[0, 0, 8, 8]],
            "model_id": "miewid-msv4.1"})

    assert r.status_code == 200, r.text
    assert observed.get("excluded") is True, \
        "another thread could run a forward while PairX held the model"
    # Compared against the event-loop thread, NOT main_thread(): TestClient
    # runs handlers on an asyncio-portal thread, so a main_thread() check
    # would hold even for an inline call and prove nothing.
    assert observed["thread"] is not loop_thread["t"], \
        "PairX took the lock on the event loop; that reintroduces starvation"


def test_concurrent_extract_cannot_corrupt_pairx_capture():
    """The invariant, end to end, through the real extract_embeddings path.

    A PairX-shaped hook window is held open on a shared model while a genuine
    /extract call runs against the same instance. The captured feature map
    must still carry its grad_fn -- i.e. the tensor PairX backwards through is
    its own forward's, not /extract's.
    """
    import io
    import time
    from PIL import Image

    from unittest.mock import patch as _patch

    from app.routers import explain_router
    from app.routers.explain_router import run_pairx_locked

    m = MiewidModel()
    m.device = "cpu"
    m.model = _SharedBackbone()
    m.preprocess = lambda image: {"image": torch.zeros(3, 8, 8)}

    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    payload = buf.getvalue()

    captured = {}
    window_open = threading.Event()

    def pairx_shaped(*a, **k):
        """Mirrors pairx/core.py:23-43, with the window held open."""
        handle = m.model.blocks[0].register_forward_hook(
            lambda mod, i, o: captured.__setitem__("fm", o))
        try:
            m.model(torch.randn(1, 3, 8, 8, requires_grad=True))
            window_open.set()
            time.sleep(0.5)          # /extract gets its chance here
        finally:
            handle.remove()
        return captured["fm"]

    extracted = []
    t = threading.Thread(
        target=lambda: extracted.append(m.extract_embeddings(payload)))

    def start_extract_once_open():
        window_open.wait(5)
        t.start()

    starter = threading.Thread(target=start_extract_once_open)
    starter.start()

    with _patch.object(explain_router, "run_pairx", pairx_shaped):
        fm = run_pairx_locked(m.inference_lock)

    starter.join(5)
    t.join(5)

    assert extracted, "/extract must still complete, just not during the window"
    assert fm.grad_fn is not None, (
        "a concurrent /extract forward overwrote PairX's captured feature map; "
        "backward() would fail with 'does not require grad'")
    fm.backward(gradient=torch.zeros_like(fm))   # the call that used to raise
