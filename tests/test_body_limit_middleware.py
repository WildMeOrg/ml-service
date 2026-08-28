from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware import BodyLimitMiddleware


def _make_app(max_bytes=64):
    app = FastAPI()
    app.add_middleware(BodyLimitMiddleware, max_bytes=max_bytes)

    @app.post("/echo")
    async def echo(payload: dict):
        return {"n": len(str(payload))}
    return TestClient(app)


def test_small_body_passes():
    client = _make_app()
    r = client.post("/echo", json={"a": 1})
    assert r.status_code == 200


def test_oversize_content_length_413():
    client = _make_app(max_bytes=8)
    r = client.post("/echo", json={"key": "value-far-over-eight-bytes"})
    assert r.status_code == 413


def test_oversize_chunked_body_413():
    client = _make_app(max_bytes=8)

    def gen():
        yield b'{"key": "'
        yield b"x" * 32
        yield b'"}'
    r = client.post("/echo", content=gen(),
                    headers={"content-type": "application/json"})
    assert r.status_code == 413


def test_non_http_scope_passthrough():
    # middleware must not touch lifespan scopes: app startup succeeding
    # under TestClient's context manager proves it
    with _make_app() as client:
        assert client.post("/echo", json={"a": 1}).status_code == 200
