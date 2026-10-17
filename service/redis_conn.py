"""Redis connections. Two clients on the same server: text (app state) and binary (RQ stores pickled jobs).

redis-py connection pools reset themselves after fork, so the clients are safe in RQ work-horses.
"""
from __future__ import annotations

import redis

from .config import get_settings

_text: redis.Redis | None = None
_raw: redis.Redis | None = None

PREFIX = "fv:"


def get_redis() -> redis.Redis:
    global _text
    if _text is None:
        _text = redis.Redis.from_url(get_settings().redis_url, decode_responses=True, health_check_interval=30,
                                     socket_connect_timeout=5, socket_timeout=10, retry_on_timeout=True)
    return _text


def get_rq_redis() -> redis.Redis:
    global _raw
    if _raw is None:
        _raw = redis.Redis.from_url(get_settings().redis_url, health_check_interval=30,
                                    socket_connect_timeout=5, retry_on_timeout=True)
    return _raw


def set_clients(text: redis.Redis | None, raw: redis.Redis | None):
    """Inject clients (tests use fakeredis)."""
    global _text, _raw
    _text, _raw = text, raw
