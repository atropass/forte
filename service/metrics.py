"""Prometheus metrics computed from Redis at scrape time (shared by every API replica and worker)."""
from __future__ import annotations

from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily

from . import jobqueue, upstream
from .redis_conn import PREFIX, get_redis, get_rq_redis


class _Collector:
    def collect(self):
        r = get_redis()
        jobs = CounterMetricFamily("fv_jobs", "Finished jobs", labels=["status", "source"])
        errors = CounterMetricFamily("fv_job_errors", "Failed jobs by error code", labels=["code"])
        for k in r.scan_iter(PREFIX + "metrics:jobs:*"):
            _, _, _, status, source = k.split(":", 4)
            jobs.add_metric([status, source], float(r.get(k) or 0))
        for k in r.scan_iter(PREFIX + "metrics:errors:*"):
            errors.add_metric([k.rsplit(":", 1)[1]], float(r.get(k) or 0))
        yield jobs
        yield errors
        for name, help_ in (("duration", "Job processing time"), ("queue_wait", "Time jobs waited for a worker")):
            c = CounterMetricFamily(f"fv_job_{name}_seconds", help_, labels=["stat"])
            c.add_metric(["sum"], float(r.get(f"{PREFIX}metrics:{name}_sum") or 0))
            c.add_metric(["count"], float(r.get(f"{PREFIX}metrics:{name}_count") or 0))
            yield c
        q = GaugeMetricFamily("fv_queue_depth", "Jobs waiting for a worker", labels=["queue"])
        for name, n in jobqueue.depth().items():
            q.add_metric([name], n)
        yield q
        yield GaugeMetricFamily("fv_jobs_active", "Jobs queued or processing", value=r.scard(PREFIX + "jobs:active"))
        from rq import Worker
        yield GaugeMetricFamily("fv_workers", "Registered RQ workers", value=Worker.count(connection=get_rq_redis()))
        up = GaugeMetricFamily("fv_upstream_up", "GPU service reachable", labels=["service"])
        for name, ok in upstream.status().items():
            up.add_metric([name], 1.0 if ok else 0.0)
        yield up


_registry = CollectorRegistry()
_registry.register(_Collector())


def render() -> bytes:
    return generate_latest(_registry)
