"""Pure-ASGI request-body cap.

Bounds per-request parse-time memory: FastAPI/Pydantic materializes the
whole body before route handlers run, so the cap must sit below them.
Aggregate concurrency is bounded separately by uvicorn --limit-concurrency."""

import logging

logger = logging.getLogger(__name__)


class BodyLimitMiddleware:
    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await self._reject(send)
            return

        received = 0
        rejected = False          # we sent the 413; swallow the app's output
        response_started = False  # app began responding; too late to reject

        async def counting_receive():
            nonlocal received, rejected
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    if not response_started and not rejected:
                        rejected = True
                        await self._reject(send)
                    # abort the app's body read; connection is done
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message):
            nonlocal response_started
            if rejected:
                return  # 413 already sent; drop the app's response
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, guarded_send)
        except Exception as exc:
            # the disconnect we injected may surface as ClientDisconnect (or
            # similar) from the app; if we already answered 413 that is
            # expected — anything else is a real error
            if not rejected:
                raise
            logger.warning(
                "Suppressed exception after body-cap 413 was delivered: %r", exc)

    @staticmethod
    async def _reject(send):
        body = b'{"detail":"Request body too large"}'
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})
