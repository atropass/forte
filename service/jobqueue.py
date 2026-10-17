"""RQ queues. Jury jobs (X-API-Key) go to `api`, web-UI jobs to `ui`; workers always drain `api` first."""
from __future__ import annotations

import time

from rq import Queue

from . import state
from .config import get_settings
from .redis_conn import get_rq_redis

QUEUES = ("api", "ui")


def queue(name: str) -> Queue:
    return Queue(name, connection=get_rq_redis())


def enqueue(jid: str, source: str, at_front: bool = False) -> str:
    """Queue (or re-queue) a job; returns the RQ job id (job id + attempt, unique per attempt)."""
    s = get_settings()
    job = state.get(jid) or {}
    rq_id = f"{jid}-{int(job.get('attempts', 0)) + 1}"
    qname = "api" if source == "api" else "ui"
    state.update(jid, rq_id=rq_id, queue=qname, queued_at=time.time(), status="processing", stage="queued")
    queue(qname).enqueue("service.worker.run_job", jid, job_id=rq_id,
                         job_timeout=s.job_timeout_s + 30,  # the job's own watchdog fires first
                         result_ttl=3600, failure_ttl=86400, ttl=s.job_ttl_s, at_front=at_front)
    return rq_id


def position(jid: str) -> int:
    """1-based position among jobs waiting for a worker (jury queue counts first); 0 when not waiting."""
    job = state.get(jid) or {}
    rq_id, qname = job.get("rq_id"), job.get("queue")
    if not rq_id or job.get("stage") != "queued":
        return 0
    try:
        pos = queue(qname).get_job_position(rq_id)
    except Exception:  # noqa: BLE001
        return 0
    if pos is None:
        return 0
    ahead = len(queue("api")) if qname == "ui" else 0
    return ahead + pos + 1


def depth() -> dict[str, int]:
    return {q: len(queue(q)) for q in QUEUES}
