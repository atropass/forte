"""Calibrate decision thresholds on the training videos and write pipeline/thresholds.yaml.

Run where the models are deployed (same model + backend as production), e.g. on the GPU server:
    docker compose exec backend python3 scripts/calibrate.py
Needs the training videos in open/videos/. Writes $THRESHOLDS_FILE (default pipeline/thresholds.yaml; in Docker
/data/thresholds.yaml, persisted) and reports/calibration.md.
Restart the backend afterwards so the new thresholds are loaded.

Yes/no questions: positives/negatives from scripts/feasibility_items.yaml (violation moments vs the same
moments in the clean twin videos); threshold = the value with the best balanced accuracy, ties broken
towards the LOWER threshold (a missed violation is worse than a false alarm, FAQ #16).
Hand-over episodes (V05/V13): scripts/calibration_episodes.yaml; grid over (handover_id, handover_act).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline import handover  # noqa: E402
from pipeline.analyze import handover_crop  # noqa: E402
from pipeline.frames import frame_at, probe  # noqa: E402
from pipeline.vlm import VLM, image_for  # noqa: E402
from pipeline.vlm_backends import make_backend  # noqa: E402

VIDEOS = ROOT / "open" / "videos"
GRID = [round(x, 2) for x in np.arange(0.15, 0.91, 0.05)]


def video_path(vid: str) -> Path:
    for p in VIDEOS.glob("*.mov"):
        if p.stem.replace("Х", "X") == vid:
            return p
    raise FileNotFoundError(vid)


def best_threshold(pos: list[float], neg: list[float]) -> tuple[float, float, float, float]:
    """Best balanced accuracy; among equally good thresholds take the MIDDLE of that range (a peaked model
    gives the same accuracy for 0.15..0.85, and an edge value would be fragile on unseen footage)."""
    rows = []
    for thr in GRID:
        tpr = float(np.mean([p >= thr for p in pos])) if pos else 1.0
        tnr = float(np.mean([p < thr for p in neg])) if neg else 1.0
        rows.append((thr, (tpr + tnr) / 2, tpr, tnr))
    top = max(r[1] for r in rows)
    best = [r for r in rows if r[1] >= top - 1e-9]
    return best[len(best) // 2]


def main():
    vlm = VLM(make_backend(), max_side=848, cache_dir=ROOT / "cache" / "vlm" if os.getenv("CALIB_CACHE") else None)
    ground: dict[str, dict] = {}

    def boxes(vid):
        if vid not in ground:
            p = video_path(vid)
            ground[vid] = vlm.ground(frame_at(p, probe(p).duration * 0.5))  # same as analyze.py
        return ground[vid]

    items = yaml.safe_load((ROOT / "scripts" / "feasibility_items.yaml").read_text())
    enabled, stats = [], {}
    out, lines = {}, ["# Threshold calibration", "", f"Model: `{vlm.backend.model_id}` ({vlm.backend.name})", "",
                      "| question | threshold | balanced acc | recall | specificity | n+ / n- |", "|---|---|---|---|---|---|"]
    for qid, labs in items.items():
        if qid in ("id_shown", "bag_handover", "banknote_manual_check", "key_in_safe"):
            continue  # replaced by the hand-over sequence analysis / key: invisible (3 px), keep the strict default
        batch, ys = [], []
        for expected, lst in labs.items():
            y = {True: 1, False: 0, "yes": 1, "no": 0}[expected]
            for vid, t in lst:
                batch.append((qid, [image_for(qid, frame_at(video_path(vid), float(t)), boxes(vid))]))
                ys.append(y)
        ps = [a.p for a in vlm.prob(batch)]
        pos = [p for p, y in zip(ps, ys) if y]
        neg = [p for p, y in zip(ps, ys) if not y]
        thr, bal, tpr, tnr = best_threshold(pos, neg)
        out[qid] = float(thr)
        stats[qid] = (bal, tpr)
        lines.append(f"| {qid} | {thr:.2f} | {bal:.2f} | {tpr:.2f} | {tnr:.2f} | {len(pos)} / {len(neg)} |")
        print(qid, thr, round(bal, 2), "pos", [round(p, 2) for p in pos], "neg", [round(p, 2) for p in neg], flush=True)

    # hand-over episodes
    eps = yaml.safe_load((ROOT / "scripts" / "calibration_episodes.yaml").read_text())
    ep_probs = []
    for e in eps:
        p = video_path(e["video"])
        times = handover.episode_times(e["t0"], e["t1"], probe(p).duration)
        imgs = [handover_crop(frame_at(p, t), boxes(e["video"])) for t in times]
        ep_probs.append((times, vlm.choose([(handover.PROMPT, [im], handover.OPTIONS) for im in imgs])))
    best = None
    for id_thr in GRID:
        for act_thr in (0.35, 0.4, 0.45, 0.5, 0.55, 0.6):
            res = [handover.decide_episode(t, pr, id_thr, act_thr) for t, pr in ep_probs]
            ok = [r.kind == e["expected"] for r, e in zip(res, eps)]
            caught = sum(r.kind == e["expected"] for r, e in zip(res, eps) if e["expected"])
            fp = sum(r.kind is not None for r, e in zip(res, eps) if not e["expected"])
            score = (sum(ok), caught, -fp, -abs(id_thr - 0.5))
            if best is None or score > best[0]:
                best = (score, id_thr, act_thr, res)
    (score, id_thr, act_thr, res) = best
    out["handover_id"], out["handover_act"] = float(id_thr), float(act_thr)
    # calibration-gated rules: switched on only when their signal separates violation from clean
    bal, tpr = stats.get("cash_in_machine", (0, 0))
    if bal >= 0.8 and tpr >= 0.67:
        enabled.append("cash_in_machine_no_client")
    judged = [(r, e) for r, e in zip(res, eps) if "partial" in e]
    partial_ok = judged and all((r.partial_at is not None) == e["partial"] for r, e in judged)
    if partial_ok:
        enabled.append("partial_disbursement")
    out["enabled_rules"] = enabled
    lines += ["", f"## Hand-over episodes (V05 / V13): handover_id={id_thr}, handover_act={act_thr}", "",
              f"{score[0]}/{len(eps)} episodes correct.", "",
              "| video | expected | got | handover t | label t | ID seen at |", "|---|---|---|---|---|---|"]
    for r, e in zip(res, eps):
        lines.append(f"| {e['video']} | {e['expected'] or 'clean'} | {r.kind or 'clean'} | {r.t} | {e.get('handover_t', '')} | {r.id_seen_at} |")
    lines += ["", "| video | X03 expected | 2nd hand-over at | label t |", "|---|---|---|---|"]
    lines += [f"| {e['video']} | {'partial' if e['partial'] else 'whole'} | {r.partial_at} | {e.get('partial_t', '')} |"
              for r, e in judged]
    lines += ["", f"**Calibration-gated rules enabled:** {', '.join(enabled) or 'none'} "
              "(cash_in_machine_no_client needs balanced acc >= 0.8 and recall >= 0.67; partial_disbursement needs "
              "every judged episode right).", "",
              "Small sample (10 episodes, 4 violations): treat as a starting point, re-check on new footage."]
    pins = yaml.safe_load((ROOT / "scripts" / "threshold_overrides.yaml").read_text()) or {}
    out.update({k: float(v) for k, v in pins.items()})  # evidence-based pins from the full evaluation
    lines += ["", f"Pinned from the full evaluation (scripts/threshold_overrides.yaml): {pins}"]
    thr_file = Path(os.getenv("THRESHOLDS_FILE", ROOT / "pipeline" / "thresholds.yaml"))
    thr_file.parent.mkdir(parents=True, exist_ok=True)
    thr_file.write_text(
        "# written by scripts/calibrate.py — model " + vlm.backend.model_id + "\n" + yaml.safe_dump(out, sort_keys=True))
    (ROOT / "reports" / "calibration.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps(out, indent=1))
    print(f"written {thr_file} — restart the backend to load it: docker compose restart backend")


if __name__ == "__main__":
    main()
