"""Train the key-in-lock classifier (pipeline/keyclf.py) with leave-one-video-out validation.

Positives: frames of key-in-lock videos after the key was left (X-22 >= 15 s, X-24 >= 10 s, X-46 all).
Negatives: every 2 s of all clean videos. Frames where a person box overlaps the safe are skipped.
Writes models/keyclf.joblib and reports/keyclf.md.
"""
import json
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from feasibility import video_path  # noqa: E402
from pipeline.frames import frame_at, probe  # noqa: E402
from pipeline.keyclf import MODEL_PATH, features, lock_patch, locate_lock, safe_crop  # noqa: E402
from pipeline.vlm import VLM  # noqa: E402
from pipeline.vlm_backends import make_backend  # noqa: E402

POS = {"X-22": (15.0, 21.0), "X-24": (10.0, 14.5), "X-46": (0.5, 19.0)}
CLEAN = ["X-3", "X-5", "X-8", "X-10", "X-12", "X-14", "X-16", "X-19", "X-21", "X-23", "X-25", "X-27", "X-29",
         "X-31", "X-35", "X-37", "X-39", "X-43", "X-45"]

g = json.loads((ROOT / "cache" / "grounding.json").read_text())
vlm = VLM(make_backend("mlx", "mlx-community/Qwen3-VL-4B-Instruct-4bit"), max_side=896, cache_dir=ROOT / "cache" / "vlm")


def occluded(vid, t, safe):
    d = pickle.load(open(ROOT / "cache" / "det" / f"{vid}.pkl", "rb"))
    w, h = d["wh"]
    i = int(np.argmin(np.abs(d["ts"] - t)))
    sx1, sy1, sx2, sy2 = [v / 1000 for v in safe]
    for x in d["dets"][i]:
        if x.label != "person" or x.score < 0.5:
            continue
        b = x.box
        bx1, by1, bx2, by2 = b[0] / w, b[1] / h, b[2] / w, b[3] / h
        if min(bx2, sx2) - max(bx1, sx1) > 0.01 and min(by2, sy2) - max(by1, sy1) > 0.01:
            return True
    return False


X, y, grp, meta = [], [], [], []
for vid in list(POS) + CLEAN:
    safe = g[vid].get("safe")
    if not safe:
        print("no safe box", vid)
        continue
    p = video_path(vid)
    dur = probe(p).duration
    a, b = POS.get(vid, (0.5, dur - 0.3))
    step = 0.5 if vid in POS else 2.0
    lock = None
    for t in np.arange(a, min(b, dur - 0.2), step):
        if occluded(vid, float(t), safe):
            continue
        s = safe_crop(frame_at(p, float(t)), safe)
        if lock is None:
            lock = locate_lock(vlm, s) or []
        X.append(features(lock_patch(s, lock or None)))
        y.append(int(vid in POS))
        grp.append(vid)
        meta.append((vid, float(t)))
X, y, grp = np.array(X), np.array(y), np.array(grp)
print("samples", len(y), "pos", y.sum(), "videos", len(set(grp)))


def make():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))


# leave-one-video-out: per-frame probabilities for each held-out video
probs = np.zeros(len(y))
for vid in sorted(set(grp)):
    m = grp == vid
    clf = make().fit(X[~m], y[~m])
    probs[m] = clf.predict_proba(X[m])[:, 1]

lines = ["# Key-in-lock classifier — leave-one-video-out", "",
         f"Samples: {len(y)} patches ({y.sum()} positive) from {len(set(grp))} videos.", "",
         "| video | label | frames | mean p | frac ≥ 0.5 |", "|---|---|---|---|---|"]
for vid in sorted(set(grp), key=lambda v: int(v.split("-")[1])):
    m = grp == vid
    lines.append(f"| {vid} | {'key' if vid in POS else 'clean'} | {m.sum()} | {probs[m].mean():.2f} | {(probs[m] >= 0.5).mean():.2f} |")
# choose threshold: highest value that still catches >= 2/3 of each positive video's frames on average,
# reported with the clean-video frame false-positive rate
best = None
for thr in np.arange(0.3, 0.96, 0.05):
    rec = np.mean([(probs[grp == v] >= thr).mean() for v in POS if (grp == v).any()])
    fpr = (probs[y == 0] >= thr).mean()
    lines.append(f"\nthr {thr:.2f}: mean per-video positive-frame recall {rec:.2f}, clean-frame FPR {fpr:.3f}")
    if rec >= 0.66 and (best is None or fpr < best[2] or (fpr == best[2] and thr > best[0])):
        best = (float(thr), rec, fpr)
thr = best[0] if best else 0.5
lines.append(f"\n**Chosen threshold {thr:.2f}** (rule needs 2 of 3 frames ≥ threshold).")
(ROOT / "reports" / "keyclf.md").write_text("\n".join(lines))
print("\n".join(lines))
MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump({"model": make().fit(X, y), "threshold": thr}, MODEL_PATH)
