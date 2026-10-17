"""CLI demo: analyse one video and print the API-style JSON (base64 frames shortened).

python scripts/demo.py open/videos/Х-26.mov [--out result.json]
"""
import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.analyze import Analyzer  # noqa: E402
from pipeline.perception import make_detector  # noqa: E402
from pipeline.vlm import VLM  # noqa: E402
from pipeline.vlm_backends import make_backend  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--out", default=None)
a = ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(message)s")
an = Analyzer(VLM(make_backend(), max_side=848, cache_dir=ROOT / "cache" / "vlm"), make_detector())
res = an.analyze(a.video, job_id="demo")
if a.out:
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
short = json.loads(json.dumps(res))
for v in short["violations"]:
    for f in v["evidence_frames"]:
        f["image_base64"] = f["image_base64"][:24] + "..."
short.pop("_debug", None)
print(json.dumps(short, ensure_ascii=False, indent=1))
