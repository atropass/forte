"""Evaluate the DEPLOYED service through its public API, exactly as the jury will use it.

python scripts/eval_remote.py --url https://<host> --key <API_KEY> [--parallel 5] [--only X-26,X-27] [--force]

Uploads every unique training video (POST /api/v1/analyze), polls GET /api/v1/jobs/{id}, validates the
response against the spec schema, stores results in reports/eval/remote/, and writes reports/eval_remote.md
with the same metrics as `make eval` plus the wall-clock time per video (upload -> done, including queueing).
--parallel N submits N videos at once to simulate the jury and check the 5-minute limit under load.
Needs only httpx + jsonschema locally; the videos stay between your machine and your server.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import jsonschema

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from pipeline.eval import load_labels, metrics  # noqa: E402
from test_api import SPEC_SCHEMA  # noqa: E402

OUT = ROOT / "reports" / "eval" / "remote"
LIMIT_S = 300


def run_one(client: httpx.Client, url: str, key: str, vid: str, path: Path) -> dict:
    t0 = time.time()
    with path.open("rb") as fh:
        r = client.post(f"{url}/api/v1/analyze", headers={"X-API-Key": key},
                        files={"file": (path.name, fh, "video/quicktime")}, timeout=300)
    r.raise_for_status()
    jid = r.json()["job_id"]
    while True:
        j = client.get(f"{url}/api/v1/jobs/{jid}", headers={"X-API-Key": key}, timeout=30).json()
        if j["status"] != "processing":
            break
        if time.time() - t0 > LIMIT_S + 120:
            j = {"status": "failed", "error": "client timeout"}
            break
        time.sleep(2)
    wall = round(time.time() - t0, 1)
    schema_ok = True
    if j["status"] == "done":
        try:
            jsonschema.validate(j, SPEC_SCHEMA)
        except jsonschema.ValidationError as e:
            schema_ok = False
            print(f"{vid}: SCHEMA ERROR {e.message}", flush=True)
    for v in j.get("violations", []):
        for f in v.get("evidence_frames", []):
            f["image_base64"] = f"<{len(f['image_base64'])} chars>"
    j.update(_wall_s=wall, _schema_ok=schema_ok, _job_id=jid)
    if j["status"] != "done":
        j.update(verdict="no_violation", violations=[])
    (OUT / f"{vid}.json").write_text(json.dumps(j, ensure_ascii=False, indent=1))
    flag = "" if wall <= LIMIT_S else "  !! OVER 5 MIN"
    print(f"{vid}: {j['status']} {j.get('verdict')} {[(v.get('type_code'), v['timecode_sec']) for v in j.get('violations', [])]} "
          f"wall={wall}s{flag}", flush=True)
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    url = a.url.rstrip("/")
    OUT.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    vids = [v for v in sorted(labels, key=lambda x: int(x.split("-")[1])) if not a.only or v in a.only.split(",")]
    todo = [v for v in vids if a.force or not (OUT / f"{v}.json").exists()]
    with httpx.Client() as client:
        health = client.get(f"{url}/healthz", timeout=30).json()
        print("service:", health)
        with ThreadPoolExecutor(max_workers=max(1, a.parallel)) as ex:
            list(ex.map(lambda v: run_one(client, url, a.key, v, ROOT / labels[v]["file"]), todo))
    walls = [json.loads((OUT / f"{v}.json").read_text())["_wall_s"] for v in vids if (OUT / f"{v}.json").exists()]
    md = metrics(labels, res_dir=OUT, out_md=ROOT / "reports" / "eval_remote.md",
                 title=f"Evaluation through the deployed API ({url}, parallel={a.parallel})")
    extra = (f"\n\nWall time per video (upload → done, incl. queue): max {max(walls):.0f} s, "
             f"mean {sum(walls) / len(walls):.0f} s; over 5 min: {sum(w > LIMIT_S for w in walls)}/{len(walls)}.\n")
    (ROOT / "reports" / "eval_remote.md").write_text(md + extra)
    print(md.split("## Per video")[0] + extra)


if __name__ == "__main__":
    main()
