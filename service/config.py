"""All service settings, read once from the environment (12-factor). See .env.example for the full list."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None or v == "" else v.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    v = os.getenv(name, "").strip()
    return int(v) if v else default


def _float(name: str, default: float) -> float:
    v = os.getenv(name, "").strip()
    return float(v) if v else default


def _list(name: str) -> list[str]:
    return [x.strip() for x in os.getenv(name, "").split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    env: str
    redis_url: str
    # storage: fs (shared volume) | s3 (any S3 API: MinIO, Ceph, AWS, R2, Tigris, ...)
    storage_backend: str
    data_dir: Path
    s3_endpoint: str | None
    s3_bucket: str
    s3_region: str
    s3_access_key: str | None
    s3_secret_key: str | None
    s3_sse: str | None
    # auth
    api_keys: tuple[str, ...]
    session_ttl_s: int
    cookie_secure: bool
    public_origins: tuple[str, ...]
    trust_proxy: bool
    login_ip_limit: int
    login_user_fail_limit: int
    login_window_s: int
    # uploads
    max_upload_bytes: int
    validate_video: bool
    max_video_s: float
    max_video_px: int
    ui_max_active_jobs: int
    # jobs
    job_timeout_s: int
    job_max_attempts: int
    job_ttl_s: int
    infra_wait_s: float
    heartbeat_s: int
    # workers
    worker_count: int
    worker_max_jobs: int
    worker_mode: str
    # http
    enable_docs: bool
    cors_origins: tuple[str, ...]
    metrics_token: str | None
    frontend_dir: Path
    log_format: str
    log_level: str
    extra: dict = field(default_factory=dict)

    @property
    def production(self) -> bool:
        return self.env == "production"


def _worker_mode() -> str:
    mode = os.getenv("WORKER_MODE", "auto").lower()
    if mode == "auto":
        # in-process GPU backends load the model once and must stay in one process; remote backends fork per job
        return "simple" if os.getenv("VLM_BACKEND", "openai").lower() in ("vllm", "mlx") else "fork"
    return mode


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_dir = Path(os.getenv("DATA_DIR", str(ROOT / "data_service")))
    keys = _list("API_KEYS") + _list("API_KEY")
    env = os.getenv("APP_ENV", "production").lower()
    return Settings(
        env=env,
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        storage_backend=os.getenv("STORAGE_BACKEND", "fs").lower(),
        data_dir=data_dir,
        s3_endpoint=os.getenv("S3_ENDPOINT") or os.getenv("AWS_ENDPOINT_URL_S3") or None,
        s3_bucket=os.getenv("S3_BUCKET") or os.getenv("BUCKET_NAME", ""),
        s3_region=os.getenv("S3_REGION") or os.getenv("AWS_REGION", "us-east-1"),
        s3_access_key=os.getenv("S3_ACCESS_KEY") or os.getenv("AWS_ACCESS_KEY_ID") or None,
        s3_secret_key=os.getenv("S3_SECRET_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY") or None,
        s3_sse=os.getenv("S3_SSE") or None,
        api_keys=tuple(keys),
        session_ttl_s=int(_float("SESSION_TTL_H", 12) * 3600),
        cookie_secure=_bool("COOKIE_SECURE", True),
        public_origins=tuple(o.rstrip("/") for o in _list("PUBLIC_ORIGIN")),
        trust_proxy=_bool("TRUST_PROXY", True),
        login_ip_limit=_int("LOGIN_IP_LIMIT", 20),
        login_user_fail_limit=_int("LOGIN_USER_FAIL_LIMIT", 5),
        login_window_s=_int("LOGIN_WINDOW_S", 900),
        max_upload_bytes=_int("MAX_UPLOAD_MB", 100) * 1024 * 1024,
        validate_video=_bool("VALIDATE_VIDEO", True),
        max_video_s=_float("MAX_VIDEO_S", 900),
        max_video_px=_int("MAX_VIDEO_PX", 3840),
        ui_max_active_jobs=_int("UI_MAX_ACTIVE_JOBS", 3),
        job_timeout_s=_int("JOB_TIMEOUT_S", 290),
        job_max_attempts=_int("JOB_MAX_ATTEMPTS", 2),
        job_ttl_s=_int("JOB_TTL_DAYS", 30) * 86400,
        infra_wait_s=_float("INFRA_WAIT_S", 60),
        heartbeat_s=_int("HEARTBEAT_S", 10),
        worker_count=_int("WORKER_COUNT", 2),
        worker_max_jobs=_int("WORKER_MAX_JOBS", 0),
        worker_mode=_worker_mode(),
        enable_docs=_bool("ENABLE_DOCS", False),
        cors_origins=tuple(_list("CORS_ORIGINS")),
        metrics_token=os.getenv("METRICS_TOKEN") or None,
        frontend_dir=Path(os.getenv("FRONTEND_DIR", str(ROOT / "forte-vision-ai-development" / "out"))),
        log_format=os.getenv("LOG_FORMAT", "text").lower(),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )


def reset_settings():
    get_settings.cache_clear()
