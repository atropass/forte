"""Health of the GPU services the workers depend on (vLLM pod, CV pod).

Used by workers as a circuit breaker before a job starts, and by /readyz. Results are cached in Redis for a
few seconds so many API replicas do not hammer the pods. URLs and tokens never leave this module.
"""
from __future__ import annotations

import json
import os
import time

from .redis_conn import PREFIX, get_redis

CACHE_S = 10


def targets() -> dict[str, tuple[str, str]]:
    """name -> (health URL, bearer token); only the backends that are remote in this deployment."""
    out = {}
    if os.getenv("VLM_BACKEND", "openai").lower() in ("openai", "remote") and os.getenv("VLM_API_BASE"):
        base = os.getenv("VLM_API_BASE", "").rstrip("/")
        base = base[:-3] if base.endswith("/v1") else base
        out["vlm"] = (base + "/health", os.getenv("VLM_API_KEY", ""))
    det = os.getenv("DET_BACKEND", "auto").lower()
    if os.getenv("DET_API_BASE") and det in ("auto", "remote"):
        out["detector"] = (os.getenv("DET_API_BASE", "").rstrip("/") + "/health",
                           os.getenv("DET_API_KEY") or os.getenv("VLM_API_KEY", ""))
    return out


def _probe(url: str, token: str, timeout: float = 5.0) -> bool:
    import httpx
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.get(url, headers={"Authorization": f"Bearer {token}"} if token else {})
        return r.status_code < 500
    except Exception:  # noqa: BLE001
        return False


def status(use_cache: bool = True) -> dict[str, bool]:
    r = get_redis()
    key = PREFIX + "upstream"
    if use_cache:
        try:
            v = r.get(key)
            if v:
                return json.loads(v)
        except Exception:  # noqa: BLE001
            pass
    res = {name: _probe(url, tok) for name, (url, tok) in targets().items()}
    try:
        r.set(key, json.dumps(res), ex=CACHE_S)
    except Exception:  # noqa: BLE001
        pass
    return res


def wait_ready(max_wait_s: float) -> bool:
    """Circuit breaker: True as soon as every upstream answers; False after max_wait_s of failures."""
    deadline = time.time() + max_wait_s
    delay = 2.0
    while True:
        st = status(use_cache=False)
        if all(st.values()):
            return True
        if time.time() + delay > deadline:
            return False
        time.sleep(delay)
        delay = min(delay * 2, 15)
