"""Run the detector once on all training videos at PIPE_FPS and cache results (cache/det/<vid>.pkl)."""
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.frames import extract_frames  # noqa: E402
from pipeline.perception import Detector  # noqa: E402

FPS, SIDE = 2.0, 640
out = ROOT / "cache" / "det"
out.mkdir(parents=True, exist_ok=True)
det = Detector()
for p in sorted((ROOT / "open" / "videos").glob("*.mov")):
    vid = p.stem.replace("Х", "X")
    f = out / f"{vid}.pkl"
    if f.exists():
        continue
    t = time.time()
    ts, frames = extract_frames(p, fps=FPS, max_side=SIDE)
    dets = det(frames)
    pickle.dump({"ts": ts, "wh": (frames.shape[2], frames.shape[1]), "fps": FPS, "dets": dets}, f.open("wb"))
    print(vid, len(ts), f"{time.time() - t:.1f}s", flush=True)
