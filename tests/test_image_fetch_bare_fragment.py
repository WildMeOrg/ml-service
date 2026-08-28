"""A bare '#' in an image filename must not truncate the request.

Flukebook, 2026-08-28 19:20:

    httpx GET .../3c8f4264-.../2000_174419#R8123A.jpg "HTTP/1.1 404"
    Image fetch failed (400, HTTPStatusError, 47 ms): .../2000_174419#R8123A.jpg
    POST /extract/ 400 Bad Request

'#' is the fragment delimiter, so httpx sent only
`/wildbook_data_dir/3/c/<uuid>/2000_174419` and Flukebook 404'd on the
truncated path. The file on disk is named `2000_174419#R8123A.jpg`; the '#'
has to reach the server percent-encoded.

Wildbook builds these URLs without encoding the filename, which is the real
upstream fix. This makes ml-service resolve them anyway.
"""
import asyncio

import httpx

from app.utils import image_uri

PNG = bytes.fromhex("89504e470d0a1a0a") + b"\x00" * 64


def _capture():
    """A transport that records the path each request actually asks for."""
    seen = {}

    def responder(request):
        seen["path"] = request.url.path
        seen["raw_path"] = request.url.raw_path.decode()
        return httpx.Response(200, content=PNG)

    return responder, seen


def test_bare_hash_in_filename_reaches_the_server_encoded():
    responder, seen = _capture()
    image_uri.init_image_fetch(transport=httpx.MockTransport(responder))

    asyncio.run(image_uri.resolve_image_uri(
        "https://wb.example/wildbook_data_dir/3/c/uuid/2000_174419#R8123A.jpg"))

    assert seen["path"].endswith("2000_174419#R8123A.jpg"), \
        f"request was truncated at the '#': {seen['path']}"
    assert "%23" in seen["raw_path"], \
        f"'#' must be percent-encoded on the wire: {seen['raw_path']}"


def test_already_encoded_hash_is_not_double_encoded():
    responder, seen = _capture()
    image_uri.init_image_fetch(transport=httpx.MockTransport(responder))

    asyncio.run(image_uri.resolve_image_uri("https://wb.example/a%23b.jpg"))

    assert seen["path"].endswith("a#b.jpg"), seen["path"]
    assert "%2523" not in seen["raw_path"], \
        f"already-encoded '#' was encoded again: {seen['raw_path']}"


def test_hash_in_filename_followed_by_a_signed_query():
    """`/a#b.jpg?sig=...` is all fragment under URI parsing.

    We deliberately reinterpret it as the producer means it -- a filename
    containing '#', then a query -- so assert the exact wire target.
    """
    responder, seen = _capture()
    image_uri.init_image_fetch(transport=httpx.MockTransport(responder))

    asyncio.run(image_uri.resolve_image_uri("https://wb.example/a#b.jpg?sig=xyz&t=1"))

    assert seen["raw_path"] == "/a%23b.jpg?sig=xyz&t=1", seen["raw_path"]


def test_genuine_fragment_after_a_real_query_is_left_as_a_fragment():
    """A '#' that follows a real '?' is a fragment and must stay dropped."""
    responder, seen = _capture()
    image_uri.init_image_fetch(transport=httpx.MockTransport(responder))

    asyncio.run(image_uri.resolve_image_uri(
        "https://wb.example/img.jpg?sig=xyz#fragment"))

    assert seen["raw_path"] == "/img.jpg?sig=xyz", seen["raw_path"]


def test_hash_in_the_authority_never_retargets_the_host():
    """`https://evil#@internal.example/a.jpg` names host `evil`.

    A blanket '#' -> '%23' turns it into userinfo `evil%23` on host
    `internal.example`, silently sending a caller-supplied fetch somewhere
    the URL never named. The path-only rule must leave it alone.
    """
    hostile = "https://evil#@internal.example/a.jpg"
    assert image_uri.encode_bare_fragment(hostile) == hostile
    assert httpx.URL(image_uri.encode_bare_fragment(hostile)).host == "evil"


def test_local_path_with_hash_still_resolves(tmp_path, monkeypatch):
    """A local filename containing '#' needs no encoding and must not get any."""
    calls = []
    monkeypatch.setattr(image_uri, "encode_bare_fragment",
                        lambda u: calls.append(u) or u)
    image_uri.init_image_fetch()
    f = tmp_path / "2000_174419#R8123A.jpg"
    f.write_bytes(PNG)

    assert asyncio.run(image_uri.resolve_image_uri(str(f))) == PNG
    assert calls == [], f"local path was passed through the URL normalizer: {calls}"


def test_data_uri_never_reaches_the_normalizer(monkeypatch):
    """Asserting the bytes come back proves nothing -- spy on the call."""
    import base64
    calls = []
    monkeypatch.setattr(image_uri, "encode_bare_fragment",
                        lambda u: calls.append(u) or u)
    image_uri.init_image_fetch()
    uri = "data:image/png;base64," + base64.b64encode(PNG).decode()

    assert asyncio.run(image_uri.resolve_image_uri(uri)) == PNG
    assert calls == [], f"data: URI was passed through the URL normalizer: {calls}"


def test_ordinary_url_with_query_is_untouched():
    """The common case must be byte-identical: no '#', nothing rewritten."""
    responder, seen = _capture()
    image_uri.init_image_fetch(transport=httpx.MockTransport(responder))

    asyncio.run(image_uri.resolve_image_uri(
        "https://wb.example/img.jpg?sig=xyz&expires=1"))

    assert seen["raw_path"] == "/img.jpg?sig=xyz&expires=1", seen["raw_path"]


def test_wbia_compat_loader_also_encodes_the_bare_hash():
    """/wbia-compat has its own httpx.get and the identical exposure.

    Fixing only the shared path would leave the same filenames unfetchable
    through this router.
    """
    from unittest.mock import patch

    from app.routers import wbia_compat_router

    asked = {}

    class _Resp:
        content = PNG

        def raise_for_status(self):
            pass

    def fake_get(url, **kw):
        asked["url"] = str(url)
        return _Resp()

    with patch.object(wbia_compat_router.httpx, "get", fake_get):
        data = wbia_compat_router._load_image(
            "https://wb.example/wildbook_data_dir/3/c/uuid/2000_174419#R8123A.jpg")

    assert data == PNG
    assert "%23R8123A.jpg" in asked["url"], \
        f"request would truncate at the '#': {asked['url']}"


def test_scheme_relative_uri_is_returned_unchanged():
    """Pins the contract: absolute http(s) only, no guessing.

    Both call sites gate on an 'http://' / 'https://' prefix, so a
    scheme-relative URI cannot reach the normalizer. It is left alone rather
    than parsed heuristically.
    """
    assert image_uri.encode_bare_fragment("//h/a#b.jpg") == "//h/a#b.jpg"
