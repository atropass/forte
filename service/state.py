"""Job records in Redis.

fv:job:<id>             hash, every value JSON-encoded (owner, source, status, stage, progress, timestamps, ...)
fv:jobs:active          set of job ids that are queued or processing (scanned by the reaper)
fv:user:<name>:jobs     sorted set of a user's job ids by creation time (history)
fv:dedup:<caller>:<sha> job id of an identical upload by the same caller (idempotent re-POST)
fv:hb:<id>              heartbeat of the worker running the job (expires when the worker dies)
fv:metrics:*            counters for /metrics
"""
from __future__ import annotations

import json
import re
import secrets
import time

from .config import get_settings
from .redis_conn import PREFIX, get_redis

JOB_ID_RE = re.compile(r"^[0-9a-f]{8,32}$")
ACTIVE = PREFIX + "jobs:active"
TERMINAL = ("done", "failed")


def _k(jid: str) -> str:
    return f"{PREFIX}job:{jid}"


def _enc(d: dict) -> dict:
    return {k: json.dumps(v, ensure_ascii=False) for k, v in d.items()}


def _dec(d: dict) -> dict:
    return {k: json.loads(v) for k, v in d.items()}


def new_id() -> str:
    r = get_redis()
    while True:
        jid = secrets.token_hex(8)
        if not r.exists(_k(jid)):
            return jid


def valid_id(jid: str) -> bool:
    return bool(JOB_ID_RE.match(jid))


def create(jid: str, **fields) -> dict:
    r = get_redis()
    now = time.time()
    job = {"job_id": jid, "status": "processing", "stage": "queued", "progress": 0.0, "created": now,
           "attempts": 0, **fields}
    ttl = get_settings().job_ttl_s
    p = r.pipeline()
    p.hset(_k(jid), mapping=_enc(job))
    p.expire(_k(jid), ttl)
    p.sadd(ACTIVE, jid)
    if job.get("owner"):
        uk = f"{PREFIX}user:{job['owner'].lower()}:jobs"
        p.zadd(uk, {jid: now})
        p.expire(uk, ttl)
    p.execute()
    return job


def get(jid: str) -> dict | None:
    if not valid_id(jid):
        return None
    d = get_redis().hgetall(_k(jid))
    return _dec(d) if d else None


def update(jid: str, **fields):
    r = get_redis()
    if fields and r.exists(_k(jid)):
        r.hset(_k(jid), mapping=_enc(fields))


def incr(jid: str, field: str) -> int:
    return int(get_redis().hincrby(_k(jid), field, 1))


def finish(jid: str, status: str, **fields) -> bool:
    """Terminal transition; returns False if the job already ended (e.g. the watchdog failed it first)."""
    job = get(jid)
    if job is None or job.get("status") in TERMINAL:
        return False
    now = time.time()
    started = job.get("started") or now
    update(jid, status=status, stage=status, finished=now, elapsed_s=round(now - started, 1),
           progress=1.0 if status == "done" else job.get("progress", 0.0), **fields)
    r = get_redis()
    p = r.pipeline()
    p.srem(ACTIVE, jid)
    p.incr(f"{PREFIX}metrics:jobs:{status}:{job.get('source', 'api')}")
    if status == "failed":
        p.incr(f"{PREFIX}metrics:errors:{fields.get('error', 'internal_error')}")
    else:
        p.incrbyfloat(f"{PREFIX}metrics:duration_sum", now - started)
        p.incr(f"{PREFIX}metrics:duration_count")
        if job.get("queued_at") and job.get("started"):
            p.incrbyfloat(f"{PREFIX}metrics:queue_wait_sum", job["started"] - job["queued_at"])
            p.incr(f"{PREFIX}metrics:queue_wait_count")
    p.execute()
    return True


def active_ids() -> list[str]:
    return sorted(get_redis().smembers(ACTIVE))


def remove_active(jid: str):
    get_redis().srem(ACTIVE, jid)


def user_jobs(owner: str, limit: int = 100) -> list[dict]:
    r = get_redis()
    uk = f"{PREFIX}user:{owner.lower()}:jobs"
    out, gone = [], []
    for jid in r.zrevrange(uk, 0, limit - 1):
        j = get(jid)
        (out.append(j) if j else gone.append(jid))
    if gone:
        r.zrem(uk, *gone)
    return out


def count_active_for(owner: str) -> int:
    return sum(1 for j in user_jobs(owner, 50) if j.get("status") not in TERMINAL)


def dedup_get(caller: str, sha: str) -> str | None:
    return get_redis().get(f"{PREFIX}dedup:{caller}:{sha}")


def dedup_set(caller: str, sha: str, jid: str):
    get_redis().set(f"{PREFIX}dedup:{caller}:{sha}", jid, ex=get_settings().job_ttl_s)


def heartbeat(jid: str, ttl: int):
    get_redis().set(f"{PREFIX}hb:{jid}", str(time.time()), ex=ttl)


def heartbeat_alive(jid: str) -> bool:
    return bool(get_redis().exists(f"{PREFIX}hb:{jid}"))


def clear_heartbeat(jid: str):
    get_redis().delete(f"{PREFIX}hb:{jid}")


def set_info(name: str, value: dict, ttl: int = 3600):
    get_redis().set(f"{PREFIX}info:{name}", json.dumps(value, ensure_ascii=False), ex=ttl)


def get_info(name: str) -> dict | None:
    v = get_redis().get(f"{PREFIX}info:{name}")
    return json.loads(v) if v else None
