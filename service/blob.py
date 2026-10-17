"""Object storage for uploaded videos (transient), results and job logs.

STORAGE_BACKEND=fs  directory under DATA_DIR/blobs (single host or a shared volume)
STORAGE_BACKEND=s3  any S3 API: S3_ENDPOINT, S3_BUCKET, S3_REGION, S3_ACCESS_KEY, S3_SECRET_KEY, S3_SSE (AES256)

Keys are built by the service only (`jobs/<job_id>/<name>`) and validated, never taken from user input.
Clients are created lazily per process (boto3 / urllib3 pools must not cross a fork).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

from .config import get_settings

_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9/_.-]{0,200}$")


def _check(key: str) -> str:
    if not _KEY_RE.match(key) or ".." in key:
        raise ValueError(f"invalid storage key {key!r}")
    return key


def job_key(job_id: str, name: str) -> str:
    return _check(f"jobs/{job_id}/{name}")


class Blob:
    def put_file(self, src: Path, key: str, content_type: str = "application/octet-stream"): ...
    def get_file(self, key: str, dest: Path) -> Path: ...
    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream"): ...
    def get_bytes(self, key: str) -> bytes | None: ...
    def delete(self, key: str): ...
    def delete_older_than(self, prefix: str, max_age_s: float) -> int: ...
    def check(self): ...

    def put_json(self, key: str, obj) -> None:
        self.put_bytes(key, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

    def get_json(self, key: str):
        b = self.get_bytes(key)
        return None if b is None else json.loads(b)


class FSBlob(Blob):
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, key: str) -> Path:
        return self.root / _check(key)

    def put_file(self, src, key, content_type="application/octet-stream"):
        p = self._p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".part")
        shutil.copyfile(src, tmp)
        os.replace(tmp, p)

    def get_file(self, key, dest):
        shutil.copyfile(self._p(key), dest)
        return Path(dest)

    def put_bytes(self, key, data, content_type="application/octet-stream"):
        p = self._p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, p)

    def get_bytes(self, key):
        p = self._p(key)
        return p.read_bytes() if p.is_file() else None

    def delete(self, key):
        p = self._p(key)
        p.unlink(missing_ok=True)
        try:
            p.parent.rmdir()
        except OSError:
            pass

    def delete_older_than(self, prefix, max_age_s):
        base = self._p(prefix.rstrip("/"))
        if not base.exists():
            return 0
        cutoff, n = time.time() - max_age_s, 0
        for f in base.rglob("*"):
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
                n += 1
        for d in sorted((d for d in base.rglob("*") if d.is_dir()), reverse=True):
            try:
                d.rmdir()
            except OSError:
                pass
        return n

    def check(self):
        if not os.access(self.root, os.W_OK):
            raise RuntimeError("storage directory is not writable")


class S3Blob(Blob):
    def __init__(self, s):
        if not s.s3_bucket:
            raise RuntimeError("S3_BUCKET is not set")
        import boto3
        from botocore.config import Config
        self.bucket = s.s3_bucket
        self.extra = {"ServerSideEncryption": s.s3_sse} if s.s3_sse else {}
        self.client = boto3.session.Session().client(
            "s3", endpoint_url=s.s3_endpoint, region_name=s.s3_region,
            aws_access_key_id=s.s3_access_key, aws_secret_access_key=s.s3_secret_key,
            config=Config(retries={"max_attempts": 5, "mode": "standard"}, connect_timeout=5, read_timeout=60,
                          s3={"addressing_style": "path"}))

    def put_file(self, src, key, content_type="application/octet-stream"):
        self.client.upload_file(str(src), self.bucket, _check(key), ExtraArgs={"ContentType": content_type, **self.extra})

    def get_file(self, key, dest):
        self.client.download_file(self.bucket, _check(key), str(dest))
        return Path(dest)

    def put_bytes(self, key, data, content_type="application/octet-stream"):
        self.client.put_object(Bucket=self.bucket, Key=_check(key), Body=data, ContentType=content_type, **self.extra)

    def get_bytes(self, key):
        try:
            return self.client.get_object(Bucket=self.bucket, Key=_check(key))["Body"].read()
        except self.client.exceptions.NoSuchKey:
            return None

    def delete(self, key):
        self.client.delete_object(Bucket=self.bucket, Key=_check(key))

    def delete_older_than(self, prefix, max_age_s):
        cutoff, n = time.time() - max_age_s, 0
        for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix):
            old = [{"Key": o["Key"]} for o in page.get("Contents", []) if o["LastModified"].timestamp() < cutoff]
            for i in range(0, len(old), 1000):
                self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": old[i:i + 1000], "Quiet": True})
                n += len(old[i:i + 1000])
        return n

    def check(self):
        self.client.head_bucket(Bucket=self.bucket)

    def ensure_bucket(self):
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except Exception:  # noqa: BLE001
            self.client.create_bucket(Bucket=self.bucket)


_cache: dict[int, Blob] = {}


def get_blob() -> Blob:
    pid = os.getpid()
    if pid not in _cache:
        s = get_settings()
        _cache.clear()
        _cache[pid] = S3Blob(s) if s.storage_backend == "s3" else FSBlob(s.data_dir / "blobs")
    return _cache[pid]


def reset_blob():
    _cache.clear()
