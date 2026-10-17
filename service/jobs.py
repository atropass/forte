"""Persistent single-worker job queue.

Each job lives in DATA_DIR/jobs/<job_id>/: input video, job.json (status/progress), result.json.
One GPU worker thread processes jobs in FIFO order. On restart, jobs still marked "processing"
are re-queued (their video is on disk), so a restart never loses accepted work.
"""
from __future__ import annotations

import json
import logging
import queue
import secrets
import threading
import time
import traceback
from pathlib import Path
from typing import Callable

log = logging.getLogger("jobs")


class JobStore:
    def __init__(self, root: Path):
        self.root = root / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def new_id(self) -> str:
        while True:
            jid = secrets.token_hex(4)
            if not (self.root / jid).exists():
                return jid

    def dir(self, jid: str) -> Path:
        return self.root / jid

    def valid(self, jid: str) -> bool:
        return len(jid) == 8 and all(c in "0123456789abcdef" for c in jid) and (self.root / jid / "job.json").exists()

    def read(self, jid: str) -> dict:
        return json.loads((self.root / jid / "job.json").read_text())

    def write(self, jid: str, **fields) -> dict:
        with self._lock:
            p = self.root / jid / "job.json"
            job = json.loads(p.read_text()) if p.exists() else {}
            job.update(fields)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(job, ensure_ascii=False))
            tmp.replace(p)
            return job

    def result(self, jid: str) -> dict | None:
        p = self.root / jid / "result.json"
        return json.loads(p.read_text()) if p.exists() else None

    def save_result(self, jid: str, res: dict):
        p = self.root / jid / "result.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, ensure_ascii=False))
        tmp.replace(p)

    def unfinished(self) -> list[str]:
        out = []
        for d in self.root.iterdir():
            f = d / "job.json"
            if f.exists():
                j = json.loads(f.read_text())
                if j.get("status") in ("queued", "processing"):
                    out.append((j.get("created", 0), d.name))
        return [jid for _, jid in sorted(out)]


class Worker:
    """Runs `fn(video_path, job_id, progress_cb) -> result dict` for queued jobs, one at a time."""

    def __init__(self, store: JobStore, fn: Callable, timeout_s: float):
        self.store = store
        self.fn = fn
        self.timeout_s = timeout_s
        self.q: queue.Queue[str] = queue.Queue()
        self.ready = threading.Event()
        self.current: str | None = None
        self.thread = threading.Thread(target=self._loop, daemon=True, name="gpu-worker")

    def start(self):
        for jid in self.store.unfinished():
            log.info("re-queue unfinished job %s", jid)
            self.store.write(jid, status="processing", stage="queued", progress=0.0)
            self.q.put(jid)
        self.thread.start()

    def submit(self, jid: str):
        self.q.put(jid)

    def queue_position(self, jid: str) -> int:
        items = list(self.q.queue)
        return items.index(jid) + 1 if jid in items else 0

    def _loop(self):
        self.ready.wait()
        while True:
            jid = self.q.get()
            self.current = jid
            job = self.store.read(jid)
            t0 = time.time()
            self.store.write(jid, status="processing", stage="started", progress=0.02, started=t0)

            def progress(stage: str, pct: float, _jid=jid):
                self.store.write(_jid, stage=stage, progress=round(float(pct), 3))

            try:
                res = self.fn(self.store.dir(jid) / job["video"], jid, progress)
                elapsed = time.time() - t0
                res["job_id"] = jid
                self.store.save_result(jid, res)
                self.store.write(jid, status="done", stage="done", progress=1.0, finished=time.time(),
                                 elapsed_s=round(elapsed, 1), over_budget=elapsed > self.timeout_s)
                log.info("job %s done in %.1fs", jid, elapsed)
            except Exception as e:  # noqa: BLE001 - every failure must surface as status=failed
                log.error("job %s failed: %s\n%s", jid, e, traceback.format_exc())
                self.store.write(jid, status="failed", stage="failed", error=f"{type(e).__name__}: {e}",
                                 finished=time.time())
            finally:
                self.current = None
                self.q.task_done()
