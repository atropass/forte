"""Print perception timeline per video next to the label (uses cache/det and cached grounding)."""
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.perception import Scene, in_room  # noqa: E402
from pipeline.timeline import build_timeline  # noqa: E402

gfile = ROOT / "cache" / "grounding.json"
ground = json.loads(gfile.read_text()) if gfile.exists() else {}
labels = {r["video"]: r for r in csv.DictReader((ROOT / "data" / "labels.csv").open())}
only = sys.argv[1:]
for f in sorted((ROOT / "cache" / "det").glob("*.pkl"), key=lambda p: int(p.stem.split("-")[1])):
    vid = f.stem
    if only and vid not in only:
        continue
    d = pickle.load(f.open("rb"))
    w, h = d["wh"]
    scene = Scene.from_grounding(ground.get(vid, {}))
    inroom = [[x.box for x in dets if x.label == "person" and in_room(x.box, w, h, scene, x.score)]
              for dets in d["dets"]]
    ev, dbg = build_timeline(d["ts"], inroom, (w, h), d["fps"], scene.desk)
    lab = labels[vid]
    print(f"{vid:5s} label={lab['timecode_sec'] or '-':>5} {lab['mapped_type'][:32]:32s} occ={''.join(map(str, dbg['occ']))}")
    for e in ev:
        print(f"        {e.kind:16s} t={e.t:6.1f} {('-> %.1f' % e.t_end) if e.t_end else ''} {e.info}")
