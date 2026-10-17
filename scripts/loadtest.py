"""Load test of the public API, the way the jury uses it: N videos submitted at once, then polled.

    python scripts/loadtest.py --url https://forte.example.kz --key "$FV_API_KEY" \
        --videos open/ --burst 10 [--unique] [--out reports/loadtest.md]

Per job it reports the time waiting for a worker, the processing time and the end-to-end time seen by the client,
and whether the job finished within --deadline (300 s, the jury limit). --unique appends a small ISO-BMFF `free`
atom with random bytes to each upload, so repeated files are not de-duplicated by the service (the video itself is
unchanged). The videos never leave the machine running this script except to the service under test.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import statistics
import struct
import time
from pathlib import Path

import httpx


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def run_one(url: str, key: str, path: Path, unique: bool, deadline: float, poll: float) -> dict:
    data = path.read_bytes()
    if unique:
        data += struct.pack(">I", 24) + b"free" + os.urandom(16)
    h = {"X-API-Key": key}
    t0 = time.time()
    with httpx.Client(timeout=120) as c:
        r = c.post(f"{url}/api/v1/analyze", headers=h, files={"file": (path.name, data, "video/quicktime")})
        if r.status_code != 200:
            return {"video": path.name, "status": f"http_{r.status_code}", "e2e": time.time() - t0}
        jid = r.json()["job_id"]
        res = {}
        while time.time() - t0 < deadline + 120:
            res = c.get(f"{url}/api/v1/jobs/{jid}", headers=h).json()
            if res.get("status") != "processing":
                break
            time.sleep(poll)
        e2e = time.time() - t0
        p = c.get(f"{url}/api/v1/jobs/{jid}/progress", headers=h).json()
    wait = p["started"] - p["created"] if p.get("started") and p.get("created") else None
    proc = p["finished"] - p["started"] if p.get("finished") and p.get("started") else None
    return {"video": path.name, "job_id": jid, "status": res.get("status", "timeout"), "error": res.get("error"),
            "verdict": res.get("verdict"), "e2e": e2e, "wait": wait, "proc": proc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--key", default=os.getenv("FV_API_KEY"), required=os.getenv("FV_API_KEY") is None)
    ap.add_argument("--videos", required=True, help="a .mov file or a directory of them")
    ap.add_argument("--burst", type=int, default=5, help="videos submitted at the same time")
    ap.add_argument("--unique", action="store_true")
    ap.add_argument("--deadline", type=float, default=300)
    ap.add_argument("--poll", type=float, default=3)
    ap.add_argument("--out")
    a = ap.parse_args()

    src = Path(a.videos)
    files = sorted(src.glob("*.mov")) + sorted(src.glob("*.MOV")) if src.is_dir() else [src]
    if not files:
        raise SystemExit("no .mov files found")
    batch = [files[i % len(files)] for i in range(a.burst)]
    url = a.url.rstrip("/")
    print(f"submitting {len(batch)} videos at once to {url}")
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=len(batch)) as ex:
        rows = list(ex.map(lambda p: run_one(url, a.key, p, a.unique, a.deadline, a.poll), batch))
    wall = time.time() - t0

    ok = [r for r in rows if r["status"] == "done"]
    in_time = [r for r in ok if r["e2e"] <= a.deadline]
    e2e = [r["e2e"] for r in ok]
    proc = [r["proc"] for r in ok if r.get("proc") is not None]
    wait = [r["wait"] for r in ok if r.get("wait") is not None]

    def f(v):
        return f"{v:.0f}" if isinstance(v, (int, float)) else "—"

    lines = [
        f"# Load test: {len(batch)} videos submitted at once",
        "",
        f"Service: `{url}` · {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} · wall time {wall:.0f} s",
        "",
        "| metric | value |", "|---|---|",
        f"| done | {len(ok)}/{len(rows)} |",
        f"| done within {a.deadline:.0f} s (end-to-end) | {len(in_time)}/{len(rows)} |",
        f"| processing p50 / p95 / max (s) | {f(pct(proc, 50))} / {f(pct(proc, 95))} / {f(max(proc, default=None))} |",
        f"| queue wait p50 / p95 / max (s) | {f(pct(wait, 50))} / {f(pct(wait, 95))} / {f(max(wait, default=None))} |",
        f"| end-to-end p50 / p95 / max (s) | {f(pct(e2e, 50))} / {f(pct(e2e, 95))} / {f(max(e2e, default=None))} |",
        f"| mean processing (s) | {f(statistics.mean(proc) if proc else None)} |",
        "", "| video | job | status | verdict | wait s | processing s | end-to-end s |", "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        err = f" ({r['error']})" if r.get("error") else ""
        lines.append(f"| {r['video']} | {r.get('job_id', '—')} | {r['status']}{err} | {r.get('verdict') or '—'} "
                     f"| {f(r.get('wait'))} | {f(r.get('proc'))} | {f(r['e2e'])} |")
    report = "\n".join(lines) + "\n"
    print(report)
    if a.out:
        Path(a.out).write_text(report)


if __name__ == "__main__":
    main()
