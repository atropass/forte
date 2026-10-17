"""FastAPI service: public API for the jury + the web UI. Stateless: state lives in Redis, files in storage.

Jury API (header X-API-Key; keys are issued with `python -m service.admin create-api-key --name jury`):
  POST /api/v1/analyze              multipart field `file` (.mov) -> {"job_id", "status": "processing"}
  GET  /api/v1/jobs/{job_id}        processing | done (exact spec schema) | failed (+ "error" code)
Web UI (login session cookie; accounts are created with `python -m service.admin create-user`):
  POST /api/v1/auth/login | logout, GET /api/v1/auth/me
  GET  /api/v1/jobs                 history of the logged-in user
  GET  /api/v1/jobs/{job_id}/progress | details, GET /api/v1/catalog   (also with X-API-Key)
Operations (no auth): GET /healthz (liveness), GET /readyz (readiness, ?strict=1 also requires GPU services
and workers); GET /metrics (Prometheus, Bearer METRICS_TOKEN).

API-key callers see jobs created with an API key; a user sees only their own jobs. Videos are deleted after
analysis; results and logs are kept for JOB_TTL_DAYS. Workers run separately: `python -m service.worker_main`.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # locality: never reach the hub at inference time
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import redis  # noqa: E402
import yaml  # noqa: E402
from fastapi import Cookie, Depends, FastAPI, File, Header, HTTPException, Request, Response, UploadFile  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse, PlainTextResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from . import auth, blob, jobqueue, logs, state, upstream  # noqa: E402
from .auth import AuthError, Caller  # noqa: E402
from .config import ROOT, get_settings  # noqa: E402
from .redis_conn import get_redis  # noqa: E402
from .security import HardeningMiddleware, client_ip, origin_allowed  # noqa: E402
from .video import ALLOWED_EXT, InvalidVideo, check_limits, looks_like_mp4, probe  # noqa: E402

log = logging.getLogger("service")
SESSION_COOKIE = "fv_session"
SPEC_KEYS = ("job_id", "status", "verdict", "violations")
SAFE_METHODS = ("GET", "HEAD", "OPTIONS")
PUBLIC_JOB_KEYS = ("job_id", "status", "stage", "progress", "created", "started", "finished", "elapsed_s",
                   "filename", "size_bytes", "error", "verdict", "n_violations", "duration_s", "source")
HISTORY_KEYS = ("job_id", "filename", "size_bytes", "created", "finished", "elapsed_s", "status", "stage", "error",
                "verdict", "n_violations")
_FILENAME_BAD = re.compile(r"[^\w.\- ()\[\]]+", re.UNICODE)


class Credentials(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


# ------------------------------------------------------------------ auth dependencies
def get_caller(request: Request, x_api_key: str | None = Header(default=None),
               fv_session: str | None = Cookie(default=None)) -> Caller:
    if x_api_key:
        name = auth.api_key_name(x_api_key)
        if name is None:
            raise HTTPException(status_code=401, detail="invalid API key")
        return Caller("api", name)
    user = auth.session_user(fv_session)
    if user is None:
        raise HTTPException(status_code=401, detail="missing X-API-Key or login session")
    if request.method not in SAFE_METHODS and not origin_allowed(request, get_settings().public_origins):
        raise HTTPException(status_code=403, detail="cross-site request rejected")
    return Caller("user", user)


def require_user(fv_session: str | None = Cookie(default=None)) -> str:
    user = auth.session_user(fv_session)
    if user is None:
        raise HTTPException(status_code=401, detail="not logged in")
    return user


def _visible(job: dict | None, who: Caller) -> bool:
    if job is None:
        return False
    if who.is_api:
        return job.get("source") == "api"
    return job.get("owner") == who.name


def _get_job(jid: str, who: Caller) -> dict:
    job = state.get(jid)
    if not _visible(job, who):  # other callers' jobs look exactly like missing ones
        raise HTTPException(status_code=404, detail="job not found")
    return job


def _result(job: dict) -> dict:
    key = job.get("result_key")
    res = blob.get_blob().get_json(key) if key else None
    return res or {}


def _models() -> dict:
    return (state.get_info("workers") or {}).get("models", {})


def _workers_alive() -> int:
    from rq import Worker

    from .redis_conn import get_rq_redis
    return Worker.count(connection=get_rq_redis())


# ------------------------------------------------------------------ upload ingest (thread pool)
def _ingest(src, ext: str, jid: str, caller_id: str) -> dict:
    s = get_settings()
    fd, tmp_name = tempfile.mkstemp(suffix=ext, prefix="fv-upload-")
    tmp = Path(tmp_name)
    try:
        h, size = hashlib.sha256(), 0
        with os.fdopen(fd, "wb") as out:
            while chunk := src.read(1 << 20):
                size += len(chunk)
                if size > s.max_upload_bytes:
                    raise HTTPException(status_code=413, detail=f"file larger than {s.max_upload_bytes >> 20} MB")
                h.update(chunk)
                out.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="empty file")
        sha = h.hexdigest()
        info = None
        if s.validate_video:
            try:
                info = probe(tmp)
                check_limits(info, s.max_video_s, s.max_video_px)
            except InvalidVideo as e:
                raise HTTPException(status_code=422, detail=str(e)) from None
        existing = state.dedup_get(caller_id, sha)
        if existing:
            job = state.get(existing)
            if job and job.get("status") != "failed":
                return {"duplicate_of": existing}
        key = blob.job_key(jid, f"input{ext}")
        blob.get_blob().put_file(tmp, key, "video/quicktime" if ext == ".mov" else "video/mp4")
        return {"size": size, "sha": sha, "key": key, "duration": round(info.duration, 2) if info else None}
    finally:
        tmp.unlink(missing_ok=True)


def _safe_filename(name: str | None) -> str:
    base = Path(name or "video.mov").name
    base = _FILENAME_BAD.sub("_", base).strip() or "video.mov"
    return base[-120:]


# ------------------------------------------------------------------ app
def create_app() -> FastAPI:
    s = get_settings()
    logs.setup(s.log_format, s.log_level)

    @asynccontextmanager
    async def lifespan(_app):
        stop = threading.Event()
        if os.getenv("REAPER_IN_API", "1") != "0":
            from . import reaper
            reaper.start_background(stop)
        try:
            if not s.api_keys and not auth.list_api_keys():
                log.warning("no API key configured: create one with `python -m service.admin create-api-key --name jury`")
        except redis.RedisError:
            log.error("Redis is not reachable at startup")
        yield
        stop.set()

    docs = s.enable_docs
    app = FastAPI(title="ForteVision AI — Cash Compliance API", version="2.0.0", lifespan=lifespan,
                  docs_url="/api/docs" if docs else None, redoc_url=None,
                  openapi_url="/api/openapi.json" if docs else None)
    origins = [o for o in s.cors_origins if o != "*"]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True,
                           allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"])
    app.add_middleware(HardeningMiddleware, upload_limit=s.max_upload_bytes + (1 << 20),
                       upload_paths=("/api/v1/analyze",), hsts=s.cookie_secure)

    @app.exception_handler(redis.RedisError)
    async def redis_down(request: Request, exc: redis.RedisError):
        log.error("redis error on %s: %s", request.url.path, type(exc).__name__)
        return JSONResponse({"detail": "service temporarily unavailable"}, status_code=503)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse({"detail": "internal error", "request_id": logs.request_id.get()}, status_code=500)

    # -------------------------------------------------------------- auth
    def _start_session(username: str, response: Response) -> dict:
        response.set_cookie(SESSION_COOKIE, auth.create_session(username), max_age=s.session_ttl_s, httponly=True,
                            samesite="strict", secure=s.cookie_secure, path="/")
        return {"username": username}

    @app.post("/api/v1/auth/login")
    def login(cred: Credentials, request: Request, response: Response):
        if not origin_allowed(request, s.public_origins):
            raise HTTPException(status_code=403, detail="cross-site request rejected")
        ip = client_ip(request, s.trust_proxy)
        fail_bucket = f"login:fail:{cred.username.strip().lower()}"
        if not auth.hit(f"login:ip:{ip}", s.login_ip_limit, s.login_window_s) or \
                auth.failures(fail_bucket) >= s.login_user_fail_limit:
            log.warning("login throttled (ip=%s)", ip)
            raise HTTPException(status_code=429, detail="too many login attempts, try again later")
        try:
            name = auth.check_password(cred.username, cred.password)
        except AuthError:
            auth.hit(fail_bucket, 10**6, s.login_window_s)
            raise HTTPException(status_code=401, detail="invalid username or password") from None
        auth.reset(fail_bucket)
        log.info("login ok: %s", name)
        return _start_session(name, response)

    @app.post("/api/v1/auth/logout")
    def logout(response: Response, fv_session: str | None = Cookie(default=None)):
        if fv_session:
            auth.delete_session(fv_session)
        response.delete_cookie(SESSION_COOKIE, path="/", secure=s.cookie_secure, httponly=True, samesite="strict")
        return {"ok": True}

    @app.get("/api/v1/auth/me")
    def me(user: str = Depends(require_user)):
        return {"username": user}

    # -------------------------------------------------------------- ops
    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/readyz")
    def readyz(strict: bool = False):
        checks: dict[str, bool] = {}
        try:
            checks["redis"] = bool(get_redis().ping())
        except Exception:  # noqa: BLE001
            checks["redis"] = False
        try:
            blob.get_blob().check()
            checks["storage"] = True
        except Exception:  # noqa: BLE001
            checks["storage"] = False
        body: dict = {"ok": all(checks.values()), "checks": checks}
        if checks["redis"]:
            try:
                ups = upstream.status()
                workers = _workers_alive()
                info = state.get_info("workers") or {}
                cal = info.get("calibration") or {}
                body.update(workers=workers, upstream=ups, queue=jobqueue.depth(),
                            calibration={k: cal.get(k) for k in ("calibrated", "model_match", "warnings")})
                body["degraded"] = workers == 0 or not all(ups.values())
            except Exception:  # noqa: BLE001
                body["degraded"] = True
        else:
            body["degraded"] = True
        ok = body["ok"] and not (strict and body["degraded"])
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.get("/metrics")
    def metrics(authorization: str | None = Header(default=None)):
        import hmac
        if not s.metrics_token:
            raise HTTPException(status_code=404, detail="not found")
        if not authorization or not hmac.compare_digest(authorization.encode(), f"Bearer {s.metrics_token}".encode()):
            raise HTTPException(status_code=401, detail="unauthorized")
        from .metrics import render
        return PlainTextResponse(render(), media_type="text/plain; version=0.0.4")

    # -------------------------------------------------------------- jobs
    @app.post("/api/v1/analyze")
    async def analyze(file: UploadFile = File(...), who: Caller = Depends(get_caller)):
        name = _safe_filename(file.filename)
        ext = Path(name).suffix.lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(status_code=415, detail="unsupported file type; expected .mov")
        if not who.is_api and state.count_active_for(who.name) >= s.ui_max_active_jobs:
            raise HTTPException(status_code=429, detail="too many videos in progress, wait for them to finish")
        head = await file.read(16)
        await file.seek(0)
        if not head:
            raise HTTPException(status_code=400, detail="empty file")
        if not looks_like_mp4(head):
            raise HTTPException(status_code=415, detail="file is not a MOV/MP4 video")
        jid = state.new_id()
        res = await run_in_threadpool(_ingest, file.file, ext, jid, who.id)
        if "duplicate_of" in res:
            dup = res["duplicate_of"]
            log.info("duplicate upload by %s -> job %s", who.id, dup)
            return {"job_id": dup, "status": (state.get(dup) or {}).get("status", "processing")}
        source = "api" if who.is_api else "ui"
        state.create(jid, owner=None if who.is_api else who.name, source=source, caller=who.id, filename=name,
                     size_bytes=res["size"], video_key=res["key"], sha256=res["sha"], duration_s=res["duration"])
        state.dedup_set(who.id, res["sha"], jid)
        jobqueue.enqueue(jid, source)
        log.info("job %s queued by %s (%d bytes)", jid, who.id, res["size"])
        return {"job_id": jid, "status": "processing"}

    @app.get("/api/v1/jobs")
    def list_jobs(user: str = Depends(require_user)):
        return {"items": [{k: j.get(k) for k in HISTORY_KEYS} for j in state.user_jobs(user)]}

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str, who: Caller = Depends(get_caller)):
        job = _get_job(job_id, who)
        if job["status"] == "done":
            res = _result(job)
            if not res:
                return JSONResponse({"job_id": job_id, "status": "failed", "error": "result_unavailable"})
            return JSONResponse({k: res.get(k) for k in SPEC_KEYS} | {"job_id": job_id, "status": "done"})
        if job["status"] == "failed":
            return {"job_id": job_id, "status": "failed", "error": job.get("error") or "internal_error"}
        return {"job_id": job_id, "status": "processing"}

    @app.get("/api/v1/jobs/{job_id}/progress")
    def get_progress(job_id: str, who: Caller = Depends(get_caller)):
        job = _get_job(job_id, who)
        ups = upstream.status()
        return {k: job.get(k) for k in PUBLIC_JOB_KEYS} | {
            "queue_position": jobqueue.position(job_id),
            "models_ready": _workers_alive() > 0 and all(ups.values())}

    @app.get("/api/v1/jobs/{job_id}/details")
    def get_details(job_id: str, who: Caller = Depends(get_caller)):
        job = _get_job(job_id, who)
        res = _result(job) if job.get("status") == "done" else {}
        return {"job": {k: job.get(k) for k in PUBLIC_JOB_KEYS}, "debug": res.get("_debug", {}), "models": _models()}

    @app.get("/api/v1/catalog", dependencies=[Depends(get_caller)])
    def catalog():
        import json
        cat = json.loads((ROOT / "legal" / "violations_catalog.source.json").read_text())
        cov = yaml.safe_load((ROOT / "legal" / "coverage.yaml").read_text())
        norms = yaml.safe_load((ROOT / "legal" / "norms.yaml").read_text())
        items = []
        for v in cat["violations"]:
            c = cov["codes"].get(v["code"], {})
            ref = norms["refs"].get(v["code"])
            items.append({"code": v["code"], "category": v["category"], "category_ru": cov["categories"].get(v["category"]),
                          "type_ru": v["type_ru"], "status": c.get("status", "not_detected"), "note": c.get("note"),
                          "clause": ref["clause"] if ref else None, "verified": bool(ref and ref.get("verified"))})
        for code, c in cov["codes"].items():  # codes beyond the team catalog (X01, X02, X03, ...)
            if not code.startswith("X"):
                continue
            ref = norms["refs"].get(code) or {}
            cat_key = c.get("category", "extra")
            items.append({"code": code, "category": cat_key, "category_ru": cov["categories"].get(cat_key),
                          "type_ru": c["type_ru"], "status": c["status"], "note": c["note"],
                          "clause": ref.get("clause"), "verified": bool(ref.get("verified"))})
        return {"items": items, "categories": cov["categories"], "models": _models()}

    if s.frontend_dir.exists():
        app.mount("/", StaticFiles(directory=s.frontend_dir, html=True), name="ui")
    return app


app = create_app()
