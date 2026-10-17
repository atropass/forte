"""Ground scene objects (safe/desk/door/laptop/counting machine) for all videos -> cache/grounding.json.

Uses the median of 3 groundings (frames at 10%, 50%, 90% of the video) per object to resist occlusion.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.frames import frame_at, probe  # noqa: E402
from pipeline.vlm import VLM  # noqa: E402
from pipeline.vlm_backends import make_backend  # noqa: E402

model = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3-VL-4B-Instruct-4bit"
vlm = VLM(make_backend("mlx", model), max_side=848, cache_dir=ROOT / "cache" / "vlm")
out_f = ROOT / "cache" / "grounding.json"
out = json.loads(out_f.read_text()) if out_f.exists() else {}
for p in sorted((ROOT / "open" / "videos").glob("*.mov")):
    vid = p.stem.replace("Х", "X")
    if vid in out:
        continue
    dur = probe(p).duration
    qs = (0.1, 0.5, 0.9) if os.getenv("GROUND_FRAMES", "3") == "3" else (0.5,)
    gs = [vlm.ground(frame_at(p, dur * q)) for q in qs]
    merged = {}
    for k in ("safe", "desk", "door", "laptop", "counting_machine"):
        boxes = [g[k] for g in gs if k in g]
        if boxes:
            merged[k] = np.median(np.array(boxes), axis=0).round(1).tolist()
    out[vid] = merged
    out_f.write_text(json.dumps(out, indent=1))
    print(vid, merged, flush=True)
