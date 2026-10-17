"""Archive of finished reports in S3-compatible object storage (Tigris on Fly.io: `fly storage create`).

Videos are never uploaded. Per job we store the report (result.json: verdict, violations with explanations and
evidence frames), the job record (job.json: file name, owner, timings) and the analysis log (log.txt) under
    reports/<owner>/<YYYY-MM-DD>_<job_id>_<video name>/
Enabled when BUCKET_NAME is set; credentials and endpoint come from the standard AWS_* env vars that
`fly storage create` sets as app secrets. Without it the archive is a no-op and reports stay on the volume only.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from pathlib import Path

log = logging.getLogger("archive")
FILES = ("job.json", "result.json", "log.txt")


def prefix(job: dict) -> str:
    day = time.strftime("%Y-%m-%d", time.gmtime(job.get("created", time.time())))
    name = re.sub(r"[^\w.\-]+", "_", Path(job.get("filename") or "video").stem, flags=re.UNICODE)[:60]
    return f"reports/{job.get('owner') or 'api'}/{day}_{job['job_id']}_{name}/"


class Archive:
    def __init__(self):
        self.bucket = os.getenv("BUCKET_NAME", "").strip()
        self._client = None

    @property
    def enabled(self) -> bool:
        return bool(self.bucket)

    def client(self):
        if self._client is None:
            import boto3  # lazy: only needed when the archive is configured
            self._client = boto3.client("s3", endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3"),
                                        region_name=os.getenv("AWS_REGION", "auto"))
        return self._client

    def upload(self, job_dir: Path, job: dict) -> str | None:
        """Upload the job's report files; returns the key prefix, or None when disabled / on error."""
        if not self.enabled:
            return None
        pre = prefix(job)
        try:
            for f in FILES:
                p = job_dir / f
                if p.exists():
                    ctype = "text/plain; charset=utf-8" if f.endswith(".txt") else "application/json"
                    self.client().upload_file(str(p), self.bucket, pre + f, ExtraArgs={"ContentType": ctype})
            log.info("archived job %s to s3://%s/%s", job["job_id"], self.bucket, pre)
            return pre
        except Exception:  # noqa: BLE001 - archiving must never break the service
            log.exception("archiving job %s failed", job["job_id"])
            return None

    def upload_async(self, job_dir: Path, job: dict, on_done=None):
        def run():
            pre = self.upload(job_dir, job)
            if pre and on_done:
                on_done(pre)
        threading.Thread(target=run, daemon=True, name=f"archive-{job['job_id']}").start()
