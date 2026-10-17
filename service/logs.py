"""Logging: plain text for development, one JSON object per line for production (LOG_FORMAT=json).

Every record carries request_id / job_id from context variables. Secrets, API keys, passwords and video
contents are never logged by the service.
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import time

request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
job_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("job_id", default=None)


class _Context(logging.Filter):
    def filter(self, record):
        record.request_id = request_id.get()
        record.job_id = getattr(record, "job_id", None) or job_id.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record):
        d = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
             "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        for k in ("request_id", "job_id"):
            v = getattr(record, k, None)
            if v:
                d[k] = v
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return json.dumps(d, ensure_ascii=False)


TEXT_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup(fmt: str = "text", level: str = "INFO"):
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter() if fmt == "json" else logging.Formatter(TEXT_FMT))
    h.addFilter(_Context())
    root.addHandler(h)
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "botocore", "boto3", "urllib3", "s3transfer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def context_filter() -> logging.Filter:
    return _Context()
