"""Accounts, sessions, API keys and login rate limits, stored in Redis.

* Users are created by an administrator (`python -m service.admin create-user`); there is no self-registration.
  Passwords: scrypt (n=2^14, r=8, p=1), random 16-byte salt.
* Sessions: random token in an HttpOnly cookie; only its SHA-256 is stored, with a TTL.
* API keys (header X-API-Key): only the SHA-256 is stored, with a name (e.g. "jury"); revocable at runtime.
  Keys from env API_KEYS / API_KEY are accepted as bootstrap keys.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
from dataclasses import dataclass

from .config import get_settings
from .redis_conn import PREFIX, get_redis

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
MIN_PASSWORD = 12
MAX_PASSWORD = 256
_SCRYPT = {"n": 2**14, "r": 8, "p": 1, "dklen": 32}
_DUMMY_SALT = b"\x00" * 16


class AuthError(ValueError):
    pass


@dataclass(frozen=True)
class Caller:
    kind: str  # "api" (X-API-Key) | "user" (session)
    name: str  # key name or username

    @property
    def is_api(self) -> bool:
        return self.kind == "api"

    @property
    def id(self) -> str:
        return f"{self.kind}:{self.name}"


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)


def _ukey(username: str) -> str:
    return f"{PREFIX}user:{username.lower()}"


# ------------------------------------------------------------------ users
def create_user(username: str, password: str) -> str:
    username = username.strip()
    if not USERNAME_RE.match(username):
        raise AuthError("username: 3-32 chars, latin letters, digits, _ . -")
    validate_password(password)
    r = get_redis()
    salt = secrets.token_bytes(16)
    created = r.hsetnx(_ukey(username), "username", username)
    if not created:
        raise AuthError("user already exists")
    r.hset(_ukey(username), mapping={"salt": salt.hex(), "pw_hash": _hash_password(password, salt).hex(),
                                     "created": str(time.time())})
    r.sadd(PREFIX + "users", username.lower())
    return username


def import_user(username: str, salt: bytes, pw_hash: bytes, created: float) -> bool:
    """Migration helper: copy an existing scrypt hash (same parameters) without knowing the password."""
    r = get_redis()
    if not r.hsetnx(_ukey(username), "username", username):
        return False
    r.hset(_ukey(username), mapping={"salt": salt.hex(), "pw_hash": pw_hash.hex(), "created": str(created)})
    r.sadd(PREFIX + "users", username.lower())
    return True


def validate_password(password: str):
    if not (MIN_PASSWORD <= len(password) <= MAX_PASSWORD):
        raise AuthError(f"password: {MIN_PASSWORD}-{MAX_PASSWORD} characters")


def set_password(username: str, password: str):
    validate_password(password)
    r = get_redis()
    if not r.exists(_ukey(username)):
        raise AuthError("no such user")
    salt = secrets.token_bytes(16)
    r.hset(_ukey(username), mapping={"salt": salt.hex(), "pw_hash": _hash_password(password, salt).hex()})
    revoke_sessions(username)


def delete_user(username: str):
    r = get_redis()
    revoke_sessions(username)
    r.delete(_ukey(username))
    r.srem(PREFIX + "users", username.lower())


def list_users() -> list[str]:
    r = get_redis()
    return sorted(r.hget(_ukey(u), "username") or u for u in r.smembers(PREFIX + "users"))


def check_password(username: str, password: str) -> str:
    if len(password) > MAX_PASSWORD or len(username) > 64:
        raise AuthError("invalid credentials")
    row = get_redis().hgetall(_ukey(username.strip()))
    if not row or "pw_hash" not in row:
        _hash_password(password, _DUMMY_SALT)  # same cost as a real check: no username probing by timing
        raise AuthError("invalid credentials")
    if not hmac.compare_digest(_hash_password(password, bytes.fromhex(row["salt"])), bytes.fromhex(row["pw_hash"])):
        raise AuthError("invalid credentials")
    return row["username"]


# ------------------------------------------------------------------ sessions
def create_session(username: str) -> str:
    token = secrets.token_urlsafe(32)
    r = get_redis()
    ttl = get_settings().session_ttl_s
    h = _sha(token)
    p = r.pipeline()
    p.set(f"{PREFIX}session:{h}", username, ex=ttl)
    p.sadd(f"{_ukey(username)}:sessions", h)
    p.expire(f"{_ukey(username)}:sessions", ttl)
    p.execute()
    return token


def session_user(token: str | None) -> str | None:
    if not token or len(token) > 128:
        return None
    return get_redis().get(f"{PREFIX}session:{_sha(token)}")


def delete_session(token: str):
    r = get_redis()
    h = _sha(token)
    user = r.get(f"{PREFIX}session:{h}")
    r.delete(f"{PREFIX}session:{h}")
    if user:
        r.srem(f"{_ukey(user)}:sessions", h)


def revoke_sessions(username: str):
    r = get_redis()
    sk = f"{_ukey(username)}:sessions"
    hs = r.smembers(sk)
    if hs:
        r.delete(*[f"{PREFIX}session:{h}" for h in hs])
    r.delete(sk)


# ------------------------------------------------------------------ API keys
def create_api_key(name: str) -> str:
    if not USERNAME_RE.match(name):
        raise AuthError("key name: 3-32 chars, latin letters, digits, _ . -")
    key = "fv_" + secrets.token_urlsafe(32)
    r = get_redis()
    h = _sha(key)
    r.hset(f"{PREFIX}apikey:{h}", mapping={"name": name, "created": str(time.time()), "prefix": key[:7]})
    r.sadd(PREFIX + "apikeys", h)
    return key


def list_api_keys() -> list[dict]:
    r = get_redis()
    out = []
    for h in r.smembers(PREFIX + "apikeys"):
        d = r.hgetall(f"{PREFIX}apikey:{h}")
        if d:
            out.append({"id": h[:12], **d})
    return sorted(out, key=lambda d: float(d.get("created", 0)))


def revoke_api_key(name_or_id: str) -> int:
    r = get_redis()
    n = 0
    for h in list(r.smembers(PREFIX + "apikeys")):
        d = r.hgetall(f"{PREFIX}apikey:{h}")
        if d.get("name") == name_or_id or h.startswith(name_or_id) and len(name_or_id) >= 8:
            r.delete(f"{PREFIX}apikey:{h}")
            r.srem(PREFIX + "apikeys", h)
            n += 1
    return n


def api_key_name(key: str | None) -> str | None:
    if not key or len(key) > 256:
        return None
    ok = False
    for i, k in enumerate(get_settings().api_keys):  # constant-time compare against every bootstrap key
        if hmac.compare_digest(key.encode(), k.encode()):
            ok = True
            name = "env" if i == 0 else f"env{i}"
    if ok:
        return name
    return get_redis().hget(f"{PREFIX}apikey:{_sha(key)}", "name")


# ------------------------------------------------------------------ rate limits
def hit(bucket: str, limit: int, window_s: int) -> bool:
    """Fixed-window counter; True while under the limit."""
    r = get_redis()
    k = f"{PREFIX}rl:{bucket}"
    n = r.incr(k)
    if n == 1 or r.ttl(k) < 0:
        r.expire(k, window_s)
    return int(n) <= limit


def failures(bucket: str) -> int:
    v = get_redis().get(f"{PREFIX}rl:{bucket}")
    return int(v) if v else 0


def reset(bucket: str):
    get_redis().delete(f"{PREFIX}rl:{bucket}")
