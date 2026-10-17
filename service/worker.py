"""Job execution: `run_job(job_id)` is what an RQ worker runs for every queued video.

  video from storage -> tmp -> Analyzer.analyze() (pipeline unchanged) -> result + log to storage
  -> video deleted (always, also on failure).

WORKER_MODE=fork (remote VLM/detector, default): RQ forks one work-horse process per job. The analyzer is built
inside the horse, so a crash or memory leak dies with the job. A watchdog thread enforces JOB_TIMEOUT_S: it marks
the job failed ("timeout"), cleans up and terminates the horse.
WORKER_MODE=simple (in-process vLLM / MLX): one long-lived process loads the model once; the watchdog marks the job
failed but cannot kill the process (the model would have to reload).
"""
from __future__ import annotations

import gc
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from . import blob, jobqueue, state, upstream
from .config import get_settings
from .logs import TEXT_FMT, context_filter, job_id as job_ctx

log = logging.getLogger("worker")

RETRYABLE = {"inference_unavailable", "worker_lost"}
_analyzer = None


class UpstreamUnavailable(RuntimeError):
    pass


# ------------------------------------------------------------------ analyzer runtime
def build_analyzer(warmup: bool = False):
    import numpy as np

    from pipeline.analyze import Analyzer
    from pipeline.perception import make_detector
    from pipeline.vlm import VLM
    from pipeline.vlm_backends import make_backend

    aux_id = os.getenv("VLM_AUX_MODEL", "").strip()
    backend = make_backend(gpu_util=float(os.getenv("VLLM_GPU_UTIL_MAIN", "0.40")) if aux_id else None)
    aux = VLM(make_backend(model_id=aux_id, gpu_util=float(os.getenv("VLLM_GPU_UTIL_AUX", "0.35"))),
              max_side=848) if aux_id else None
    det = make_detector()
    an = Analyzer(VLM(backend, max_side=848), det, aux_vlm=aux)
    if warmup:
        dummy = np.full((848, 464, 3), 128, np.uint8)
        det(dummy[None])
        an.vlm.ask([("cashier_at_desk", [dummy])])
        an.vlm.prob([("cashier_at_desk", [dummy])])
        if aux is not None:
            aux.choose([("Answer with one letter: A or B.", [dummy], ["A", "B"])])
    return an


def analyzer():
    global _analyzer
    if _analyzer is None:
        factory = os.getenv("ANALYZER_FACTORY", "").strip()  # "module:callable", for integration tests / plugins
        if factory:
            import importlib
            mod, fn = factory.split(":", 1)
            _analyzer = getattr(importlib.import_module(mod), fn)()
        else:
            _analyzer = build_analyzer()
    return _analyzer


def set_analyzer(an):
    """Tests / simple mode: use a pre-built analyzer."""
    global _analyzer
    _analyzer = an


def release_memory():
    """Between jobs in a long-lived process: drop per-video caches held by the analyzer."""
    an = _analyzer
    if an is not None:
        for v in (getattr(an, "vlm", None), getattr(an, "aux_vlm", None)):
            if v is not None and isinstance(getattr(v, "cache", None), dict):
                v.cache.clear()
        if hasattr(an, "_store"):
            an._store = None
    try:
        from pipeline import frames
        getattr(frames, "_probe_cache", {}).clear()
    except Exception:  # noqa: BLE001
        pass
    gc.collect()


def models_info() -> dict:
    """What the workers run, for the UI and /readyz (no URLs or tokens)."""
    backend = os.getenv("VLM_BACKEND", "openai").lower()
    det = "RT-DETRv2 (remote CV service)" if os.getenv("DET_API_BASE") else os.getenv("DET_MODEL", "PekingU/rtdetr_v2_r18vd")
    return {"vlm_backend": backend, "vlm": os.getenv("VLM_MODEL", "Qwen/Qwen3-VL-4B-Instruct"),
            "vlm_aux": os.getenv("VLM_AUX_MODEL") or None, "detector": det}


def publish_info():
    """Called by the worker supervisor at start: models + calibration status for the API (read-only use of pipeline)."""
    info = {"models": models_info(), "calibration": None, "at": time.time()}
    try:
        from pipeline.vlm import calibration_info
        served = None
        if info["models"]["vlm_backend"] in ("openai", "remote") and os.getenv("VLM_API_BASE"):
            try:
                from pipeline.vlm_backends import make_backend
                b = make_backend()
                served = b.served_models()
                b.client.close()
            except Exception as e:  # noqa: BLE001
                served = [f"unavailable: {type(e).__name__}"]
        info["calibration"] = calibration_info(served)
        for w in info["calibration"]["warnings"]:
            log.warning("CALIBRATION: %s", w)
    except Exception:  # noqa: BLE001
        log.exception("could not read calibration info")
    state.set_info("workers", info, ttl=7 * 86400)
    return info


# ------------------------------------------------------------------ errors
def classify(e: BaseException) -> str:
    from rq.timeouts import JobTimeoutException

    from .video import InvalidVideo
    if isinstance(e, JobTimeoutException):
        return "timeout"
    if isinstance(e, UpstreamUnavailable):
        return "inference_unavailable"
    if isinstance(e, (InvalidVideo, subprocess.CalledProcessError)):
        return "invalid_video"
    try:
        import httpx
        if isinstance(e, httpx.TransportError):
            return "inference_unavailable"
    except ImportError:
        pass
    msg = str(e)
    if isinstance(e, RuntimeError) and (msg.startswith("HTTP 5") or msg.startswith("HTTP 429")):
        return "inference_unavailable"
    return "internal_error"


# ------------------------------------------------------------------ job
def _cleanup_video(job: dict):
    key = job.get("video_key")
    if key:
        try:
            blob.get_blob().delete(key)
        except Exception:  # noqa: BLE001
            log.exception("could not delete video %s", key)


def _upload_log(jid: str, path: Path):
    try:
        if path.exists():
            blob.get_blob().put_bytes(blob.job_key(jid, "log.txt"), path.read_bytes(), "text/plain; charset=utf-8")
    except Exception:  # noqa: BLE001
        log.exception("could not store job log")


class _Watchdog(threading.Thread):
    """Heartbeat for the reaper + hard deadline for the job."""

    def __init__(self, jid: str, started: float, on_timeout):
        super().__init__(daemon=True, name=f"watchdog-{jid}")
        s = get_settings()
        self.jid, self.deadline, self.on_timeout = jid, started + s.job_timeout_s, on_timeout
        self.hb, self.stop = s.heartbeat_s, threading.Event()

    def run(self):
        last = 0.0
        while not self.stop.wait(1.0):
            now = time.time()
            if now - last >= self.hb:
                try:
                    state.heartbeat(self.jid, self.hb * 3)
                except Exception:  # noqa: BLE001
                    pass
                last = now
            if now > self.deadline:
                self.on_timeout()
                return


def run_job(jid: str):
    s = get_settings()
    job = state.get(jid)
    if job is None or job.get("status") in state.TERMINAL:
        log.warning("job %s is gone or finished; skipping", jid)
        return
    try:
        from rq import get_current_job
        cur = get_current_job()
        if cur is not None and job.get("rq_id") and cur.id != job["rq_id"]:
            log.warning("stale queue entry %s for job %s; skipping", cur.id, jid)
            return
    except Exception:  # noqa: BLE001
        pass

    token = job_ctx.set(jid)
    attempts = state.incr(jid, "attempts")
    started = time.time()
    state.update(jid, status="processing", stage="started", progress=0.02, started=started, worker=os.getpid())
    state.heartbeat(jid, s.heartbeat_s * 3)

    tmp = Path(tempfile.mkdtemp(prefix=f"fv-{jid}-"))
    log_path = tmp / "log.txt"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(TEXT_FMT))
    handler.addFilter(context_filter())
    logging.getLogger().addHandler(handler)
    ended = threading.Event()

    def on_timeout():
        if ended.is_set():
            return
        log.error("job %s exceeded %ss: failing", jid, s.job_timeout_s)
        if state.finish(jid, "failed", error="timeout"):
            _cleanup_video(job)
            handler.flush()
            _upload_log(jid, log_path)
        state.clear_heartbeat(jid)
        if s.worker_mode == "fork":
            shutil.rmtree(tmp, ignore_errors=True)
            os._exit(70)  # work-horse only: the RQ parent survives and takes the next job

    dog = _Watchdog(jid, started, on_timeout)
    dog.start()
    retried = False

    def progress(stage: str, pct: float):
        state.update(jid, stage=stage, progress=round(float(pct), 3))

    log.info("job %s started (attempt %d, %s bytes)", jid, attempts, job.get("size_bytes"))
    try:
        if upstream.targets() and not upstream.wait_ready(s.infra_wait_s):
            raise UpstreamUnavailable("GPU services unreachable")
        video = tmp / f"input{Path(job['video_key']).suffix}"
        blob.get_blob().get_file(job["video_key"], video)
        res = analyzer().analyze(video, job_id=jid, progress=progress)
        res["job_id"] = jid
        blob.get_blob().put_json(blob.job_key(jid, "result.json"), res)
        ended.set()
        if state.finish(jid, "done", verdict=res.get("verdict"), n_violations=len(res.get("violations") or []),
                        result_key=blob.job_key(jid, "result.json")):
            log.info("job %s done in %.1fs: %s", jid, time.time() - started, res.get("verdict"))
    except BaseException as e:  # noqa: BLE001 - every failure must end as status=failed (or one retry)
        ended.set()
        code = classify(e)
        log.error("job %s failed (%s): %s: %s", jid, code, type(e).__name__, e, exc_info=code == "internal_error")
        cur = state.get(jid) or {}
        if code in RETRYABLE and attempts < s.job_max_attempts and cur.get("status") not in state.TERMINAL:
            retried = True
            jobqueue.enqueue(jid, job.get("source", "api"), at_front=True)
            log.warning("job %s re-queued after %s", jid, code)
        else:
            state.finish(jid, "failed", error=code)
        if not isinstance(e, Exception):
            raise
    finally:
        dog.stop.set()
        state.clear_heartbeat(jid)
        logging.getLogger().removeHandler(handler)
        handler.close()
        _upload_log(jid, log_path)
        shutil.rmtree(tmp, ignore_errors=True)
        if not retried:
            _cleanup_video(job)
        if s.worker_mode != "fork":
            release_memory()
        job_ctx.reset(token)
