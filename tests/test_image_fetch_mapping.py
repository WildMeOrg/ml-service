"""fetch_image_for_request must map every failure class to the spec's
status code, and admission_slot must 503 on wait timeout."""
import asyncio
import httpx
import pytest
from fastapi import HTTPException

from app.utils import image_uri
from tests.test_image_fetch_guards import ONE_PX_PNG


def _status_for(handler):
    image_uri.init_image_fetch(transport=httpx.MockTransport(handler))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(image_uri.fetch_image_for_request("https://wb.example/x.jpg"))
    return exc.value.status_code


def test_success_returns_bytes():
    image_uri.init_image_fetch(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=ONE_PX_PNG)))
    data = asyncio.run(image_uri.fetch_image_for_request("https://wb.example/ok.png"))
    assert data == ONE_PX_PNG


def test_read_timeout_maps_504():
    def h(req):
        raise httpx.ReadTimeout("stall", request=req)
    assert _status_for(h) == 504


def test_connect_error_maps_502():
    def h(req):
        raise httpx.ConnectError("refused", request=req)
    assert _status_for(h) == 502


def test_upstream_500_maps_502():
    assert _status_for(lambda r: httpx.Response(500)) == 502


def test_upstream_429_maps_502():
    assert _status_for(lambda r: httpx.Response(429)) == 502


def test_upstream_408_maps_502():
    assert _status_for(lambda r: httpx.Response(408)) == 502


@pytest.mark.parametrize("code", [401, 403, 404, 410, 415])
def test_upstream_permanent_4xx_maps_400(code):
    assert _status_for(lambda r, c=code: httpx.Response(c)) == 400


def test_too_many_redirects_maps_400():
    def h(req):
        return httpx.Response(302, headers={"location": str(req.url)})
    assert _status_for(h) == 400


def test_decoding_error_maps_502():
    def h(req):
        raise httpx.DecodingError("bad encoding", request=req)
    assert _status_for(h) == 502


def test_unparseable_image_maps_400():
    assert _status_for(lambda r: httpx.Response(200, content=b"mp4junk" * 4)) == 400


def test_invalid_data_uri_maps_400():
    image_uri.init_image_fetch()
    with pytest.raises(HTTPException) as exc:
        asyncio.run(image_uri.fetch_image_for_request("data:image/png;base64"))
    assert exc.value.status_code == 400


def test_total_deadline_maps_504(monkeypatch):
    monkeypatch.setenv("IMAGE_FETCH_TOTAL_DEADLINE_S", "0.05")

    async def run():
        async def drip():
            yield b"a"
            await asyncio.sleep(0.2)
            yield b"b"
        image_uri.init_image_fetch(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, content=drip())))
        with pytest.raises(HTTPException) as exc:
            await image_uri.fetch_image_for_request("https://wb.example/drip.jpg")
        assert exc.value.status_code == 504
    asyncio.run(run())


def test_admission_slot_503_on_wait_timeout(monkeypatch):
    monkeypatch.setenv("IMAGE_ADMISSION_LIMIT", "1")
    monkeypatch.setenv("IMAGE_ADMISSION_WAIT_S", "0.05")

    async def run():
        image_uri.init_image_fetch()
        async with image_uri.admission_slot():
            with pytest.raises(HTTPException) as exc:
                async with image_uri.admission_slot():
                    pass
            assert exc.value.status_code == 503
    asyncio.run(run())


def test_admission_slot_releases_on_exception():
    async def run():
        image_uri.init_image_fetch()
        with pytest.raises(RuntimeError):
            async with image_uri.admission_slot():
                raise RuntimeError("boom")
        # slot must be free again
        async with image_uri.admission_slot():
            pass
    asyncio.run(run())


def test_failure_log_contains_sanitized_uri(caplog):
    def h(req):
        raise httpx.ConnectError("refused", request=req)
    image_uri.init_image_fetch(transport=httpx.MockTransport(h))
    with pytest.raises(HTTPException):
        asyncio.run(image_uri.fetch_image_for_request("https://wb.example/y.jpg"))
    assert any("https://wb.example/y.jpg" in r.message for r in caplog.records)
