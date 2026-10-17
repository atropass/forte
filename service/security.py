"""HTTP hardening: request IDs, body-size limits, security headers, CSRF origin checks, client IP."""
from __future__ import annotations

import json
import re
import secrets
from urllib.parse import urlsplit

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .logs import request_id

CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
       "img-src 'self' data: blob:; media-src 'self' blob:; font-src 'self' data:; connect-src 'self'; "
       "object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'")
_RID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class _TooLarge(Exception):
    pass


async def _send_json(send: Send, status: int, body: dict, headers: list | None = None):
    data = json.dumps(body).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(data)).encode())]
                + (headers or [])})
    await send({"type": "http.response.body", "body": data})


class HardeningMiddleware:
    """Pure ASGI (does not buffer bodies): request id, body limits, security headers on every response."""

    def __init__(self, app: ASGIApp, upload_limit: int, upload_paths: tuple[str, ...], default_limit: int = 1 << 20,
                 hsts: bool = True):
        self.app, self.upload_limit, self.upload_paths = app, upload_limit, upload_paths
        self.default_limit, self.hsts = default_limit, hsts

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.decode().lower(): v.decode("latin-1") for k, v in scope["headers"]}
        rid = headers.get("x-request-id", "")
        rid = rid if _RID_RE.match(rid) else secrets.token_hex(8)
        token = request_id.set(rid)
        path = scope["path"]
        limit = self.upload_limit if path in self.upload_paths else self.default_limit
        extra = [(b"x-request-id", rid.encode())]
        try:
            cl = headers.get("content-length")
            if cl is not None and (not cl.isdigit() or int(cl) > limit):
                return await _send_json(send, 413, {"detail": "request body too large"}, extra)
            received = 0
            started = False

            async def limited_receive() -> Message:
                nonlocal received
                msg = await receive()
                if msg["type"] == "http.request":
                    received += len(msg.get("body", b""))
                    if received > limit:
                        raise _TooLarge()
                return msg

            async def send_wrapper(msg: Message):
                nonlocal started
                if msg["type"] == "http.response.start":
                    started = True
                    h = list(msg.get("headers", []))
                    names = {k.lower() for k, _ in h}
                    add = [(b"x-content-type-options", b"nosniff"), (b"x-frame-options", b"DENY"),
                           (b"referrer-policy", b"same-origin"), (b"content-security-policy", CSP.encode()),
                           (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
                           (b"cross-origin-opener-policy", b"same-origin")] + extra
                    if self.hsts:
                        add.append((b"strict-transport-security", b"max-age=31536000; includeSubDomains"))
                    if path.startswith("/api/") or path in ("/healthz", "/readyz", "/metrics"):
                        add.append((b"cache-control", b"no-store"))
                    h += [(k, v) for k, v in add if k not in names]
                    msg = {**msg, "headers": h}
                await send(msg)

            try:
                await self.app(scope, limited_receive, send_wrapper)
            except _TooLarge:
                if not started:
                    await _send_json(send, 413, {"detail": "request body too large"}, extra)
        finally:
            request_id.reset(token)


def client_ip(request, trust_proxy: bool) -> str:
    if trust_proxy:
        for h in ("fly-client-ip", "x-real-ip"):
            v = request.headers.get(h)
            if v:
                return v.strip()
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def origin_allowed(request, public_origins: tuple[str, ...]) -> bool:
    """CSRF defence for cookie-authenticated, state-changing requests: Origin (or Referer) must be this site."""
    origin = request.headers.get("origin") or request.headers.get("referer")
    if not origin:
        return False
    o = urlsplit(origin)
    origin_base = f"{o.scheme}://{o.netloc}".rstrip("/")
    if public_origins:
        return origin_base in public_origins
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
    return o.netloc == host
