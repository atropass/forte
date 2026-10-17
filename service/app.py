"""FastAPI service: API for the jury + the web UI.

API (routes under /api/v1 require header X-API-Key = env API_KEY, or a login session cookie, else 401):
  POST /api/v1/analyze              multipart field `file` (.mov) -> {"job_id", "status": "processing"}
  GET  /api/v1/jobs/{job_id}        processing | done (exact spec schema) | failed (+ "error")
  GET  /api/v1/jobs/{job_id}/progress   stage / progress / queue position / timings (for the UI)
  GET  /api/v1/jobs/{job_id}/details    events, scene-state timeline, timings (for the UI)
  GET  /api/v1/catalog              violation catalog with what the service checks (for the UI)
  GET  /api/v1/jobs                 history of the logged-in user (session only)
  POST /api/v1/auth/register | login | logout, GET /api/v1/auth/me   accounts for the web UI (users.py)
A logged-in user sees only their own jobs; the X-API-Key caller sees all. Videos are deleted after analysis;
reports and logs are kept on the volume and archived to object storage when BUCKET_NAME is set (archive.py).
Unauthenticated: GET /healthz, and the static UI at / (FRONTEND_DIR).

Env: API_KEY (required), DATA_DIR (./data_service), VLM_BACKEND (mlx|vllm), VLM_MODEL, DET_MODEL,
MAX_UPLOAD_MB (500), JOB_TIMEOUT_S (280), FRONTEND_DIR, CORS_ORIGINS (comma list, dev only).
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # locality: never reach the hub at inference time
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import yaml  # noqa: E402
from fastapi import Cookie, Depends, FastAPI, File, Header, HTTPException, Request, Response, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from .archive import Archive  # noqa: E402
from .jobs import JobStore, Worker  # noqa: E402
from .users import SESSION_TTL_S, AuthError, UserStore  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("service")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger().setLevel(os.getenv("LOG_LEVEL", "INFO"))  # also when a host (uvicorn, pytest) configured logging first: per-job log.txt needs INFO

API_KEY = os.getenv("API_KEY", "")
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data_service"))
MAX_UPLOAD = int(os.getenv("MAX_UPLOAD_MB", "500")) * 1024 * 1024
JOB_TIMEOUT_S = float(os.getenv("JOB_TIMEOUT_S", "280"))
FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", ROOT / "forte-vision-ai-development" / "out"))
ALLOWED_EXT = {".mov", ".mp4", ".m4v"}
SPEC_KEYS = ("job_id", "status", "verdict", "violations")

if not API_KEY:
    # No key configured: generate one once, persist it next to the jobs and print it, so the service still
    # starts with zero configuration but is never open without a key.
    _kf = DATA_DIR / "api_key.txt"
    if _kf.exists():
        API_KEY = _kf.read_text().strip()
    else:
        import secrets
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        API_KEY = secrets.token_urlsafe(24)
        _kf.write_text(API_KEY)
        _kf.chmod(0o600)
    log.warning("API_KEY not set - using key from %s:  %s", _kf, API_KEY)

store = JobStore(DATA_DIR)
users = UserStore(DATA_DIR / "users.db")
archive = Archive()
SESSION_COOKIE = "fv_session"
state = {"analyzer": None, "loaded_at": None, "load_error": None, "models": {}}


def _run(video: Path, jid: str, progress):
    if state["analyzer"] is None:  # VLM was unreachable at startup (e.g. GPU pod stopped): retry now
        progress("loading_models", 0.01)
        load_models()
    if state["analyzer"] is None:
        raise RuntimeError(f"VLM unavailable: {state['load_error']}")
    return state["analyzer"].analyze(video, job_id=jid, progress=progress)


def _finished(jid: str):
    """Summary for the history list, then archive report + log (never the video) to object storage."""
    res = store.result(jid) or {}
    job = store.write(jid, verdict=res.get("verdict"), n_violations=len(res.get("violations") or []))
    archive.upload_async(store.dir(jid), job, on_done=lambda key: store.write(jid, archive_key=key))


worker = Worker(store, _run, JOB_TIMEOUT_S, on_finish=_finished)


def load_models():
    """Load detector + VLM once, warm up, then release the worker."""
    try:
        t = time.time()
        import numpy as np

        from pipeline.analyze import Analyzer
        from pipeline.perception import Detector
        from pipeline.vlm import VLM
        from pipeline.vlm_backends import make_backend

        aux_id = os.getenv("VLM_AUX_MODEL", "").strip()
        backend = make_backend(gpu_util=float(os.getenv("VLLM_GPU_UTIL_MAIN", "0.40")) if aux_id else None)
        aux = None
        if aux_id:  # second model for hand-over frames only (V05/V13 ensemble); shares the GPU with the main one
            aux = VLM(make_backend(model_id=aux_id, gpu_util=float(os.getenv("VLLM_GPU_UTIL_AUX", "0.35"))), max_side=848)
        det = Detector()
        an = Analyzer(VLM(backend, max_side=848), det, aux_vlm=aux)
        # warm-up: detector batch + one generation + one decision-mode call per model
        dummy = np.full((848, 464, 3), 128, np.uint8)
        det(dummy[None])
        an.vlm.ask([("cashier_at_desk", [dummy])])
        an.vlm.prob([("cashier_at_desk", [dummy])])
        if aux is not None:
            aux.choose([("Answer with one letter: A or B.", [dummy], ["A", "B"])])
        state.update(analyzer=an, loaded_at=time.time(), load_error=None,
                     models={"vlm_backend": backend.name, "vlm": backend.model_id, "vlm_aux": aux_id or None,
                             "detector": det.model_id, "device": det.device})
        log.info("models loaded in %.1fs: %s", time.time() - t, state["models"])
    except Exception as e:  # noqa: BLE001
        state["load_error"] = f"{type(e).__name__}: {e}"
        log.exception("model loading failed")
    finally:
        worker.ready.set()


@asynccontextmanager
async def lifespan(_app):
    if os.getenv("VLM_BACKEND", "mlx").lower() == "vllm":
        load_models()  # vLLM must be created in the main thread (signal handlers); blocks startup ~1-3 min
    else:
        threading.Thread(target=lambda: load_models(), daemon=True, name="model-loader").start()
    worker.start()
    yield


app = FastAPI(title="ForteVision AI — Cash Compliance", version="1.0.0", lifespan=lifespan,
              docs_url="/api/docs", openapi_url="/api/openapi.json")
origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
if origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


def caller(x_api_key: str | None = Header(default=None), fv_session: str | None = Cookie(default=None)) -> str | None:
    """Username of the logged-in user, or None for the X-API-Key caller (full access)."""
    if x_api_key and hmac.compare_digest(x_api_key.encode(), API_KEY.encode()):
        return None
    user = users.session_user(fv_session) if fv_session else None
    if user is None:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
    return user


def require_user(fv_session: str | None = Cookie(default=None)) -> str:
    user = users.session_user(fv_session) if fv_session else None
    if user is None:
        raise HTTPException(status_code=401, detail="not logged in")
    return user


class Credentials(BaseModel):
    username: str
    password: str


def _start_session(username: str, request: Request, response: Response) -> dict:
    https = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
    response.set_cookie(SESSION_COOKIE, users.create_session(username), max_age=SESSION_TTL_S, httponly=True,
                        samesite="lax", secure=https, path="/")
    return {"username": username}


@app.post("/api/v1/auth/register")
def register(cred: Credentials, request: Request, response: Response):
    try:
        name = users.register(cred.username, cred.password)
    except AuthError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    return _start_session(name, request, response)


@app.post("/api/v1/auth/login")
def login(cred: Credentials, request: Request, response: Response):
    try:
        name = users.check_password(cred.username, cred.password)
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e)) from None
    return _start_session(name, request, response)


@app.post("/api/v1/auth/logout")
def logout(response: Response, fv_session: str | None = Cookie(default=None)):
    if fv_session:
        users.delete_session(fv_session)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


@app.get("/api/v1/auth/me")
def me(user: str = Depends(require_user)):
    return {"username": user}


@app.get("/healthz")
def healthz():
    return {"ok": state["analyzer"] is not None, "models": state["models"], "load_error": state["load_error"],
            "queue": worker.q.qsize(), "current_job": worker.current}


@app.post("/api/v1/analyze")
async def analyze(file: UploadFile = File(...), who: str | None = Depends(caller)):
    name = Path(file.filename or "video.mov").name
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=415, detail=f"unsupported file type {ext!r}; expected .mov")
    jid = store.new_id()
    d = store.dir(jid)
    d.mkdir(parents=True)
    dest = d / f"input{ext}"
    size = 0
    with dest.open("wb") as fh:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD:
                fh.close()
                dest.unlink(missing_ok=True)
                d.rmdir()
                raise HTTPException(status_code=413, detail=f"file larger than {MAX_UPLOAD >> 20} MB")
            fh.write(chunk)
    if size == 0:
        dest.unlink(missing_ok=True)
        d.rmdir()
        raise HTTPException(status_code=400, detail="empty file")
    store.write(jid, job_id=jid, status="processing", stage="queued", progress=0.0, created=time.time(),
                filename=name, video=dest.name, size_bytes=size, owner=who)
    worker.submit(jid)
    return {"job_id": jid, "status": "processing"}


def _get_job(jid: str, who: str | None) -> dict:
    job = store.read(jid) if store.valid(jid) else None
    if job is None or (who is not None and job.get("owner") != who):  # other users' jobs look like missing ones
        raise HTTPException(status_code=404, detail="job not found")
    return job


HISTORY_KEYS = ("job_id", "filename", "size_bytes", "created", "finished", "elapsed_s", "status", "stage", "error",
                "verdict", "n_violations")


@app.get("/api/v1/jobs")
def list_jobs(user: str = Depends(require_user)):
    return {"items": [{k: j.get(k) for k in HISTORY_KEYS} for j in store.list(user)]}


@app.get("/api/v1/jobs/{job_id}")
def get_job(job_id: str, who: str | None = Depends(caller)):
    job = _get_job(job_id, who)
    if job["status"] == "done":
        res = store.result(job_id) or {}
        return JSONResponse({k: res.get(k) for k in SPEC_KEYS} | {"job_id": job_id, "status": "done"})
    if job["status"] == "failed":
        return {"job_id": job_id, "status": "failed", "error": job.get("error", "unknown error")}
    return {"job_id": job_id, "status": "processing"}


@app.get("/api/v1/jobs/{job_id}/progress")
def get_progress(job_id: str, who: str | None = Depends(caller)):
    job = _get_job(job_id, who)
    return {k: job.get(k) for k in ("job_id", "status", "stage", "progress", "created", "started", "finished",
                                    "elapsed_s", "filename", "size_bytes", "error")} | {
        "queue_position": worker.queue_position(job_id), "models_ready": state["analyzer"] is not None}


@app.get("/api/v1/jobs/{job_id}/details")
def get_details(job_id: str, who: str | None = Depends(caller)):
    job = _get_job(job_id, who)
    res = store.result(job_id) or {}
    return {"job": job, "debug": res.get("_debug", {}), "models": state["models"]}


@app.get("/api/v1/catalog", dependencies=[Depends(caller)])
def catalog():
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
        items.append({"code": code, "category": cat_key, "category_ru": cov["categories"].get(cat_key), "type_ru": c["type_ru"],
                      "status": c["status"], "note": c["note"], "clause": ref.get("clause"), "verified": bool(ref.get("verified"))})
    return {"items": items, "categories": cov["categories"], "models": state["models"]}


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")
