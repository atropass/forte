"""Reconciles job records with reality, so no job stays "processing" forever.

* a processing job whose worker stopped heart-beating (process killed, OOM, host restart) is re-queued once
  (front of its queue) or failed with "worker_lost";
* a processing job past its deadline is failed with "timeout";
* a queued job whose queue entry vanished is re-queued.
Runs every REAPER_INTERVAL_S in the API and in the worker supervisor; a Redis lock keeps it single-flight.
"""
from __future__ import annotations

import logging
import threading
import time

from . import jobqueue, state
from .config import get_settings
from .redis_conn import PREFIX, get_redis, get_rq_redis
from .worker import _cleanup_video

log = logging.getLogger("reaper")
INTERVAL_S = 15


def _fail(jid: str, job: dict, code: str):
    if state.finish(jid, "failed", error=code):
        log.warning("job %s failed by reaper: %s", jid, code)
    _cleanup_video(job)


def reconcile() -> int:
    s = get_settings()
    r = get_redis()
    if not r.set(PREFIX + "lock:reaper", "1", nx=True, ex=INTERVAL_S):
        return 0
    from rq.job import Job
    from rq.exceptions import NoSuchJobError
    now, n = time.time(), 0
    for jid in state.active_ids():
        job = state.get(jid)
        if job is None or job.get("status") in state.TERMINAL:
            state.remove_active(jid)
            continue
        if job.get("stage") == "queued":
            rq_id = job.get("rq_id")
            try:
                st = Job.fetch(rq_id, connection=get_rq_redis()).get_status(refresh=True) if rq_id else None
            except NoSuchJobError:
                st = None
            if st is None or str(getattr(st, "value", st)) in ("failed", "stopped", "canceled"):
                if now - (job.get("queued_at") or job.get("created") or now) > 30:
                    if int(job.get("attempts", 0)) < s.job_max_attempts:
                        log.warning("job %s lost its queue entry; re-queueing", jid)
                        jobqueue.enqueue(jid, job.get("source", "api"), at_front=True)
                    else:
                        _fail(jid, job, "worker_lost")
                    n += 1
            continue
        started = job.get("started") or job.get("created") or now
        if state.heartbeat_alive(jid):
            if now - started > s.job_timeout_s + 90:
                _fail(jid, job, "timeout")
                n += 1
            continue
        if now - started < s.heartbeat_s * 3:
            continue  # just started, first heartbeat pending
        if int(job.get("attempts", 0)) < s.job_max_attempts and now - started < s.job_timeout_s:
            log.warning("job %s: worker lost; re-queueing", jid)
            jobqueue.enqueue(jid, job.get("source", "api"), at_front=True)
        else:
            _fail(jid, job, "timeout" if now - started >= s.job_timeout_s else "worker_lost")
        n += 1
    return n


def start_background(stop: threading.Event) -> threading.Thread:
    def loop():
        while not stop.wait(INTERVAL_S):
            try:
                reconcile()
            except Exception:  # noqa: BLE001
                log.exception("reaper pass failed")
    t = threading.Thread(target=loop, daemon=True, name="reaper")
    t.start()
    return t
