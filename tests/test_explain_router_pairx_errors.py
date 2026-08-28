"""run_pairx must not swallow the exception behind its generic 500.

Flukebook, 2026-08-28 19:20: Wildbook logged
`postRaw() on .../explain/ failed with code=500 {"detail":"Internal Server
Error"}` and the retry succeeded. Two lines in this router produce that exact
body -- process_asyncio_result, which logs a traceback, and run_pairx's
generic `except`, which logged nothing at all. When the failure is the second
one there is no way to tell what actually broke.
"""
import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from fastapi import HTTPException

from app.routers import explain_router


def _run(raises):
    """Drive run_pairx with `explain` replaced by something that fails."""
    model = MagicMock()
    model.named_modules.return_value = [("backbone.blocks.3", MagicMock())]
    tensors = [torch.zeros(1, 3, 4, 4)]
    images = [np.zeros((4, 4, 3), np.uint8)]

    def boom(*a, **k):
        raise raises

    with patch.object(explain_router, "explain", boom):
        return explain_router.run_pairx(
            tensors, tensors, images, images, model,
            "backbone.blocks.3", 20, 5, "only_colors")


def test_pairx_failure_is_logged_with_its_cause(caplog):
    """The 500 stays generic on the wire; the cause must reach the log."""
    cause = RuntimeError(
        "CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total "
        "capacity of 23.68 GiB of which 1.12 GiB is free.")

    with caplog.at_level(logging.ERROR, logger=explain_router.logger.name):
        with pytest.raises(HTTPException) as excinfo:
            _run(cause)

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Internal Server Error", \
        "the wire response must stay generic -- no internals leak to the caller"
    assert "CUDA out of memory" in caplog.text, \
        "run_pairx swallowed the cause; the 500 is undiagnosable"
    assert "Traceback" in caplog.text, "the log entry needs the traceback"


def test_pairx_failure_does_not_leak_internals_to_the_caller():
    """Whatever we log, the response body stays opaque."""
    with pytest.raises(HTTPException) as excinfo:
        _run(RuntimeError("/opt/secret/path exploded"))
    assert "secret" not in str(excinfo.value.detail)
