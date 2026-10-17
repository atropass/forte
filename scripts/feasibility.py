"""Phase 2 feasibility: per-question accuracy of the VLM on hand-picked moments + latency.

python scripts/feasibility.py --model mlx-community/Qwen3-VL-4B-Instruct-4bit [--backend mlx]
Writes reports/feasibility/<model>.json and failing frames to reports/feasibility/fail/<model>/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.frames import frame_at  # noqa: E402
from pipeline.vlm import QUESTIONS, VLM, crop_box, to_pil  # noqa: E402
from pipeline.vlm_backends import make_backend  # noqa: E402

VIDEO_DIR = ROOT / "open" / "videos"


def video_path(vid: str) -> Path:
    for p in VIDEO_DIR.glob("*.mov"):
        if p.stem.replace("Х", "X") == vid:
            return p
    raise FileNotFoundError(vid)


def upscale_min(img: np.ndarray, min_side: int = 448) -> np.ndarray:
    h, w = img.shape[:2]
    if min(h, w) >= min_side:
        return img
    s = min_side / min(h, w)
    return np.asarray(Image.fromarray(img).resize((round(w * s), round(h * s)), Image.BICUBIC))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", default="mlx")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    items = yaml.safe_load((ROOT / "scripts" / "feasibility_items.yaml").read_text())
    tag = args.model.split("/")[-1]
    out_dir = ROOT / "reports" / "feasibility"
    fail_dir = out_dir / "fail" / tag
    fail_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    backend = make_backend(args.backend, args.model)
    load_s = time.time() - t0
    vlm = VLM(backend, max_side=848, cache_dir=ROOT / "cache" / "vlm")

    # one grounding call per video (frame at 0.5 s) -> boxes for crops
    ground, ground_lat = {}, []
    vids = sorted({it[0] for q in items.values() for lab in q.values() for it in lab})
    for vid in vids:
        f = frame_at(video_path(vid), 0.5)
        t = time.time()
        ground[vid] = vlm.ground(f)
        ground_lat.append(time.time() - t)
        im = to_pil(f)
        d = ImageDraw.Draw(im)
        for k, b in ground[vid].items():
            bb = [b[0] * im.width / 1000, b[1] * im.height / 1000, b[2] * im.width / 1000, b[3] * im.height / 1000]
            d.rectangle(bb, outline=(255, 0, 0), width=2)
            d.text((bb[0] + 2, bb[1] + 2), k, fill=(255, 255, 0))
        (out_dir / "grounding" / tag).mkdir(parents=True, exist_ok=True)
        im.save(out_dir / "grounding" / tag / f"{vid}.jpg", quality=85)

    results = []
    for qid, labs in items.items():
        if args.only and qid not in args.only.split(","):
            continue
        scope = QUESTIONS["questions"][qid]["scope"]
        modes = ["full"] + (["crop"] if scope.startswith("crop:") else [])
        for expected, lst in labs.items():
            expected = {True: "yes", False: "no"}.get(expected, expected)  # YAML 1.1 parses yes/no as bool
            for vid, t in lst:
                path = video_path(vid)
                if isinstance(t, list):
                    frames = [np.asarray(to_pil(frame_at(path, x), 448)) for x in t]
                    mode_imgs = {"frames": frames}
                else:
                    f = frame_at(path, t)
                    mode_imgs = {"full": [f]}
                    if "crop" in modes:
                        obj = scope.split(":")[1]
                        box = ground[vid].get(obj)
                        mode_imgs["crop"] = [upscale_min(crop_box(f, box))] if box else [f]
                for mode, imgs in mode_imgs.items():
                    tq = time.time()
                    a = vlm.ask([(qid, imgs)])[0]
                    lat = time.time() - tq
                    ok = a.answer == expected
                    results.append(dict(q=qid, mode=mode, vid=vid, t=t, expected=expected, answer=a.answer,
                                        evidence=a.evidence, ok=ok, latency=round(lat, 2), n_img=len(imgs)))
                    print(f"{qid:22s} {mode:6s} {vid:5s} {str(t):16s} exp={expected:3s} got={a.answer:7s} {'OK ' if ok else 'BAD'} {lat:4.1f}s | {a.evidence}", flush=True)
                    if not ok:
                        strip = np.concatenate([np.asarray(to_pil(i, 400).resize((round(400 * i.shape[1] / i.shape[0]), 400))) for i in imgs], axis=1)
                        Image.fromarray(strip).save(fail_dir / f"{qid}_{mode}_{vid}_{str(t).replace(' ', '')}.jpg", quality=80)

    summary = {}
    for r in results:
        k = f"{r['q']}|{r['mode']}"
        s = summary.setdefault(k, {"n": 0, "ok": 0, "unclear": 0, "lat": []})
        s["n"] += 1
        s["ok"] += r["ok"]
        s["unclear"] += r["answer"] == "unclear"
        s["lat"].append(r["latency"])
    for s in summary.values():
        s["acc"] = round(s["ok"] / s["n"], 3)
        s["lat_mean"] = round(float(np.mean(s["lat"])), 2)
        del s["lat"]
    out = dict(model=args.model, backend=args.backend, load_s=round(load_s, 1),
               ground_latency_mean=round(float(np.mean(ground_lat)), 2) if ground_lat else None,
               summary=summary, results=results, grounding=ground)
    (out_dir / f"{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
