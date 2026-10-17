"""Phase 4 evaluation: run the analyzer on every unique training video and compare with labels.

python -m pipeline.eval                 # run all (results cached per video in reports/eval/results/)
python -m pipeline.eval --reuse         # recompute metrics from saved results only
python -m pipeline.eval --only X-26,X-22 --force
Writes reports/eval.md and reports/eval/per_video.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES_DIR = ROOT / "reports" / "eval" / "results"
# accepted alternative timecodes (DECISIONS #3, #10)
EXTRA_TC = {"X-4": [68.0], "X-34": [7.0]}
DUPLICATES = {"X-47", "X-48"}


def load_labels() -> dict:
    out = {}
    for r in csv.DictReader((ROOT / "data" / "labels.csv").open()):
        if r["video"] in DUPLICATES:
            continue
        codes = [c for c in r.get("catalog_code", "").split("|") if c]
        out[r["video"]] = {
            "file": r["file"], "viol": r["has_violation"] == "1",
            "tc": [float(r["timecode_sec"])] + EXTRA_TC.get(r["video"], []) if r["timecode_sec"] else [],
            "codes": [c.rstrip("?") for c in codes], "confirmed": not any(c.endswith("?") for c in codes),
            "types": r["mapped_type"],
        }
    return out


def strip_b64(res: dict) -> dict:
    res = json.loads(json.dumps(res))
    for v in res.get("violations", []):
        for f in v.get("evidence_frames", []):
            f["image_base64"] = f"<{len(f['image_base64'])} chars>"
    return res


def run(videos: list[str], labels: dict, force: bool):
    from .analyze import Analyzer
    from .perception import make_detector
    from .vlm import VLM
    from .vlm_backends import make_backend
    an = Analyzer(VLM(make_backend(), max_side=848, cache_dir=ROOT / "cache" / "vlm"), make_detector())
    RES_DIR.mkdir(parents=True, exist_ok=True)
    for vid in videos:
        out = RES_DIR / f"{vid}.json"
        if out.exists() and not force:
            continue
        t = time.time()
        res = an.analyze(ROOT / labels[vid]["file"], job_id=vid)
        out.write_text(json.dumps(strip_b64(res), ensure_ascii=False, indent=1))
        print(f"{vid}: {res['verdict']} {[(v['type_code'], v['timecode_sec']) for v in res['violations']]} "
              f"label={labels[vid]['codes']}@{labels[vid]['tc']} ({time.time() - t:.0f}s)", flush=True)


def metrics(labels: dict, res_dir: Path = RES_DIR, out_md: Path | None = None, title: str | None = None) -> str:
    rows = []
    for vid, lab in sorted(labels.items(), key=lambda x: int(x[0].split("-")[1])):
        f = res_dir / f"{vid}.json"
        if not f.exists():
            continue
        res = json.loads(f.read_text())
        preds = [(v["type_code"], v["timecode_sec"], v["type_id"]) for v in res["violations"]]
        pred_v = res["verdict"] == "violation"
        type_ok = tc_err = tc_err_any = None
        if lab["viol"] and preds:
            tc_err_any = min(abs(p[1] - t) for p in preds for t in lab["tc"])
            matching = [p for p in preds if p[0] in lab["codes"]]
            if lab["confirmed"]:
                type_ok = bool(matching)
            if matching:
                tc_err = min(abs(p[1] - t) for p in matching for t in lab["tc"])
        rows.append(dict(video=vid, label=int(lab["viol"]), label_codes="|".join(lab["codes"]),
                         label_tc="|".join(f"{t:g}" for t in lab["tc"]), pred=int(pred_v),
                         pred_codes="|".join(f"{c}@{t:g}" for c, t, _ in preds), verdict_ok=int(pred_v == lab["viol"]),
                         type_ok="" if type_ok is None else int(type_ok),
                         tc_err="" if tc_err is None else round(tc_err, 1),
                         tc_err_any="" if tc_err_any is None else round(tc_err_any, 1),
                         timing=json.dumps(res.get("_debug", {}).get("timing_sec", res.get("_wall_s", {})))))
    if not rows:
        return "no results"
    out_md = out_md or ROOT / "reports" / "eval.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.with_name(out_md.stem + "_per_video.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    pos = [r for r in rows if r["label"]]
    neg = [r for r in rows if not r["label"]]
    acc = sum(r["verdict_ok"] for r in rows) / len(rows)
    rec = sum(r["pred"] for r in pos) / max(1, len(pos))
    fpr = sum(r["pred"] for r in neg) / max(1, len(neg))
    typed = [r for r in pos if r["type_ok"] != ""]
    tacc = sum(r["type_ok"] for r in typed) / max(1, len(typed))
    errs = sorted(r["tc_err"] for r in pos if r["tc_err"] != "")
    import statistics as st
    lines = [
        f"# {title or 'Phase 4 — Evaluation on the training videos'}", "",
        f"Videos: {len(rows)} unique ({len(pos)} with violation, {len(neg)} clean). Duplicates X-47/X-48 excluded.", "",
        "| metric | value |", "|---|---|",
        f"| video-level verdict accuracy | **{acc:.3f}** ({sum(r['verdict_ok'] for r in rows)}/{len(rows)}) |",
        f"| recall on violation videos | **{rec:.3f}** ({sum(r['pred'] for r in pos)}/{len(pos)}) |",
        f"| false-positive rate on clean videos | **{fpr:.3f}** ({sum(r['pred'] for r in neg)}/{len(neg)}) |",
        f"| type accuracy (label code among predicted codes; confirmed labels only) | {tacc:.3f} ({sum(r['type_ok'] for r in typed)}/{len(typed)}) |",
        f"| timecode error, matching type (s): mean / median | "
        f"{(st.mean(errs) if errs else float('nan')):.1f} / {(st.median(errs) if errs else float('nan')):.1f} (n={len(errs)}) |",
        "", "## Per video", "",
        "| video | label | label codes @ t | predicted codes @ t | verdict | type | tc err |", "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['video']} | {r['label']} | {r['label_codes']} @ {r['label_tc']} | {r['pred_codes'] or '—'} | "
                     f"{'✅' if r['verdict_ok'] else '❌'} | {'' if r['type_ok'] == '' else ('✅' if r['type_ok'] else '❌')} | {r['tc_err']} |")
    md = "\n".join(lines)
    out_md.write_text(md)
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    labels = load_labels()
    vids = [v for v in sorted(labels, key=lambda x: int(x.split("-")[1])) if not a.only or v in a.only.split(",")]
    if not a.reuse:
        run(vids, labels, a.force)
    print(metrics(labels))


if __name__ == "__main__":
    main()
