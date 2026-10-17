"""Phase 5: robustness to camera geometry. Re-runs a subset of videos after geometric augmentations and
compares video-level verdicts and timecodes with the original run.

Augmentations (ffmpeg): hflip (mirrored room), rot8 (8° rotation, cropped to avoid black corners),
zoom85 (85 % centre crop, i.e. a tighter/closer camera), persp (perspective warp, simulates a camera
mounted further left and lower).
python scripts/robustness.py [--videos X-26,X-27,...] [--augs hflip,rot8,zoom85,persp]
Writes reports/robustness.md.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.eval import RES_DIR, load_labels, strip_b64  # noqa: E402

AUGS = {
    "hflip": "hflip",
    "rot8": "rotate=8*PI/180:ow=iw:oh=ih,crop=iw*0.8:ih*0.8,scale=464:848",
    "zoom85": "crop=iw*0.85:ih*0.85,scale=464:848",
    "persp": "perspective=x0=0:y0=0:x1=W:y1=H*0.06:x2=0:y2=H:x3=W:y3=H*0.94:sense=destination,crop=iw*0.9:ih*0.88,scale=464:848",
}
DEFAULT_VIDEOS = ["X-15", "X-16", "X-18", "X-19", "X-26", "X-27", "X-28", "X-29", "X-32", "X-31", "X-40", "X-43"]


def augment(src: Path, aug: str, out: Path):
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", AUGS[aug], "-c:v", "libx264", "-preset",
                    "veryfast", "-crf", "20", "-an", str(out)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=",".join(DEFAULT_VIDEOS))
    ap.add_argument("--augs", default=",".join(AUGS))
    a = ap.parse_args()
    labels = load_labels()
    from pipeline.analyze import Analyzer
    from pipeline.perception import make_detector
    from pipeline.vlm import VLM
    from pipeline.vlm_backends import make_backend
    an = Analyzer(VLM(make_backend(), max_side=848, cache_dir=ROOT / "cache" / "vlm"), make_detector())
    out_dir = ROOT / "reports" / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for vid in a.videos.split(","):
        lab = labels[vid]
        orig = json.loads((RES_DIR / f"{vid}.json").read_text()) if (RES_DIR / f"{vid}.json").exists() else None
        for aug in a.augs.split(","):
            res_f = out_dir / f"{vid}_{aug}.json"
            if res_f.exists():
                res = json.loads(res_f.read_text())
            else:
                src = ROOT / lab["file"]
                dst = ROOT / "cache" / "aug" / f"{vid}_{aug}.mp4"
                augment(src, aug, dst)
                res = strip_b64(an.analyze(dst, job_id=f"{vid}_{aug}"))
                res_f.write_text(json.dumps(res, ensure_ascii=False, indent=1))
            pred = res["verdict"] == "violation"
            codes = [v["type_code"] for v in res["violations"]]
            tc = min((abs(v["timecode_sec"] - t) for v in res["violations"] if v["type_code"] in lab["codes"]
                      for t in lab["tc"]), default=None)
            rows.append(dict(video=vid, aug=aug, label=lab["viol"], pred=pred, ok=pred == lab["viol"],
                             orig_ok=None if orig is None else (orig["verdict"] == "violation") == lab["viol"],
                             codes=codes, tc_err=tc))
            print(rows[-1], flush=True)
    lines = ["# Phase 5 — Robustness to camera geometry", "",
             "Same videos, geometrically transformed. Verdict accuracy per augmentation vs the original run.", "",
             "| augmentation | verdict accuracy | original accuracy on same videos | recall | FP |", "|---|---|---|---|---|"]
    for aug in a.augs.split(","):
        r = [x for x in rows if x["aug"] == aug]
        acc = sum(x["ok"] for x in r) / len(r)
        o = [x for x in r if x["orig_ok"] is not None]
        oacc = sum(x["orig_ok"] for x in o) / len(o) if o else float("nan")
        pos = [x for x in r if x["label"]]
        neg = [x for x in r if not x["label"]]
        lines.append(f"| {aug} | {acc:.2f} | {oacc:.2f} | {sum(x['pred'] for x in pos)}/{len(pos)} | "
                     f"{sum(x['pred'] for x in neg)}/{len(neg)} |")
    lines += ["", "| video | aug | label | predicted codes | ok | tc err |", "|---|---|---|---|---|---|"]
    for x in rows:
        lines.append(f"| {x['video']} | {x['aug']} | {int(x['label'])} | {', '.join(x['codes']) or '—'} | "
                     f"{'✅' if x['ok'] else '❌'} | {'' if x['tc_err'] is None else round(x['tc_err'], 1)} |")
    (ROOT / "reports" / "robustness.md").write_text("\n".join(lines))
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
