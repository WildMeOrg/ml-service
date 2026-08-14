"""URL-branch tests for resolve_image_uri using httpx.MockTransport.

MockTransport cannot simulate timeouts by sleeping; handlers raise
httpx.ReadTimeout directly (per httpx docs)."""
import asyncio
import httpx
import pytest

from app.utils import image_uri


def _init_with(handler):
    image_uri.init_image_fetch(transport=httpx.MockTransport(handler))


def test_url_fetch_returns_bytes():
    _init_with(lambda req: httpx.Response(200, content=b"jpegbytes"))
    result = asyncio.run(image_uri.resolve_image_uri("https://wb.example/img.jpg"))
    assert result == b"jpegbytes"


def test_url_fetch_declared_oversize_rejected_before_body():
    _init_with(lambda req: httpx.Response(
        200, headers={"content-length": str(60 * 1024 * 1024)}, content=b""))
    with pytest.raises(image_uri.ImageTooLargeError):
        asyncio.run(image_uri.resolve_image_uri("https://wb.example/big.jpg"))


def test_url_fetch_undeclared_oversize_rejected_midstream(monkeypatch):
    monkeypatch.setenv("IMAGE_FETCH_MAX_BYTES", "10")
    async def no_length_body():
        # Generator body so httpx doesn't auto-set content-length
        yield b"x" * 20
    def handler(req):
        return httpx.Response(200, content=no_length_body())
    _init_with(handler)
    with pytest.raises(image_uri.ImageTooLargeError, match="mid-stream"):
        asyncio.run(image_uri.resolve_image_uri("https://wb.example/liar.jpg"))


def test_url_fetch_empty_body_rejected():
    _init_with(lambda req: httpx.Response(200, content=b""))
    with pytest.raises(image_uri.EmptyImageError):
        asyncio.run(image_uri.resolve_image_uri("https://wb.example/empty.jpg"))


def test_url_fetch_read_timeout_propagates():
    def handler(req):
        raise httpx.ReadTimeout("stalled", request=req)
    _init_with(handler)
    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(image_uri.resolve_image_uri("https://wb.example/slow.jpg"))


def test_url_fetch_total_deadline(monkeypatch):
    monkeypatch.setenv("IMAGE_FETCH_TOTAL_DEADLINE_S", "0.05")

    async def run():
        async def slow_drip():
            # slower than the deadline but each read under the read timeout
            yield b"a"
            await asyncio.sleep(0.2)
            yield b"b"
        def handler(req):
            return httpx.Response(200, content=slow_drip())
        image_uri.init_image_fetch(transport=httpx.MockTransport(handler))
        with pytest.raises(asyncio.TimeoutError):
            await image_uri.resolve_image_uri("https://wb.example/drip.jpg")
    asyncio.run(run())


def test_url_fetch_upstream_error_raises_status_error():
    _init_with(lambda req: httpx.Response(500))
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(image_uri.resolve_image_uri("https://wb.example/oops.jpg"))


def test_url_fetch_follows_redirect():
    def handler(req):
        if req.url.path == "/a":
            return httpx.Response(302, headers={"location": "https://cdn.example/b"})
        return httpx.Response(200, content=b"moved")
    _init_with(handler)
    result = asyncio.run(image_uri.resolve_image_uri("https://wb.example/a"))
    assert result == b"moved"
