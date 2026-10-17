"""Worker supervisor: `python -m service.worker_main` runs WORKER_COUNT RQ workers and keeps them alive.

* each worker drains queue `api` (jury) before `ui`;
* a worker that dies is restarted (with backoff);
* SIGTERM/SIGINT -> warm shutdown: running jobs finish (up to JOB_TIMEOUT_S), then exit;
* WORKER_MAX_JOBS > 0 recycles a worker after that many jobs.
In Kubernetes/compose you can also run WORKER_COUNT=1 per container and scale replicas instead.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import socket
import threading
import time

from . import logs
from .config import get_settings

log = logging.getLogger("worker-main")


def run_worker(index: int):
    s = get_settings()
    logs.setup(s.log_format, s.log_level)
    from rq import SimpleWorker, Worker

    from . import jobqueue, worker
    from .redis_conn import get_rq_redis
    if s.worker_mode == "simple":
        log.info("worker %d: loading models in-process", index)
        worker.set_analyzer(worker.build_analyzer(warmup=True))
        cls = SimpleWorker
    else:
        import pipeline.analyze  # noqa: F401  pre-import so each forked work-horse starts fast
        cls = Worker
    w = cls([jobqueue.queue(q) for q in jobqueue.QUEUES], connection=get_rq_redis(),
            name=f"{socket.gethostname()}.{index}.{os.getpid()}")
    w.work(max_jobs=s.worker_max_jobs or None, logging_level=s.log_level, with_scheduler=False)


def main():
    s = get_settings()
    logs.setup(s.log_format, s.log_level)
    from . import reaper, worker
    from .redis_conn import get_redis
    for i in range(30):
        try:
            get_redis().ping()
            break
        except Exception as e:  # noqa: BLE001
            log.warning("waiting for Redis (%s)", type(e).__name__)
            time.sleep(2)
    worker.publish_info()

    count = s.worker_count
    if s.worker_mode == "simple" and count > 1:
        log.warning("WORKER_MODE=simple (in-process model): forcing WORKER_COUNT=1")
        count = 1
    stop = threading.Event()
    reaper.start_background(stop)
    if count <= 1:
        run_worker(0)
        stop.set()
        return

    ctx = mp.get_context("spawn")
    procs: dict[int, mp.Process] = {}
    started_at: dict[int, float] = {}
    backoff: dict[int, float] = {}

    def spawn(i: int):
        p = ctx.Process(target=run_worker, args=(i,), name=f"rq-worker-{i}", daemon=False)
        p.start()
        procs[i], started_at[i] = p, time.time()
        log.info("worker %d started (pid %s)", i, p.pid)

    def shutdown(signum, _frame):
        if stop.is_set():
            return
        log.info("signal %s: warm shutdown of %d workers", signum, len(procs))
        stop.set()
        for p in procs.values():
            if p.is_alive():
                os.kill(p.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    for i in range(count):
        spawn(i)
    while not stop.is_set():
        time.sleep(2)
        for i, p in list(procs.items()):
            if p.is_alive() or stop.is_set():
                continue
            quick = time.time() - started_at[i] < 30
            backoff[i] = min(backoff.get(i, 1) * 2, 60) if quick else 1
            log.warning("worker %d exited (code %s); restarting in %ss", i, p.exitcode, backoff[i])
            time.sleep(backoff[i])
            if not stop.is_set():
                spawn(i)
    deadline = time.time() + s.job_timeout_s + 30
    for p in procs.values():
        p.join(max(0.0, deadline - time.time()))
        if p.is_alive():
            p.kill()
    log.info("all workers stopped")


def check() -> int:
    """Container health check: Redis reachable and at least one worker of this host registered."""
    from rq import Worker

    from .redis_conn import get_rq_redis
    try:
        host = socket.gethostname() + "."
        return 0 if any(w.name.startswith(host) for w in Worker.all(connection=get_rq_redis())) else 1
    except Exception:  # noqa: BLE001
        return 1


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["check"]:
        sys.exit(check())
    main()
