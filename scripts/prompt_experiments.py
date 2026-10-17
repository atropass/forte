"""Phase 2 prompt/crop iteration for the weak questions. Re-uses feasibility items + grounding.

python scripts/prompt_experiments.py --model <id>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from feasibility import upscale_min, video_path  # noqa: E402
from pipeline.frames import frame_at  # noqa: E402
from pipeline.vlm import QUESTIONS, VLM, crop_box, parse_answer, parse_json, to_pil  # noqa: E402
from pipeline.vlm_backends import make_backend  # noqa: E402

PRE = QUESTIONS["preamble"].strip()

VARIANTS = {
    "money_on_desk": [
        ("crop_tight", "desk", 0.05, 896, None),
        ("crop_tight_p2", "desk", 0.05, 896,
         "Look carefully at the desk surface. Is any banknote, stack or bundle of money (small rectangular paper "
         "items, usually beige/green/blue) lying on it? Ignore the counting machine, laptop and phone."),
        ("full_p2", None, 0, 848,
         "Look carefully at the desk surface. Is any banknote, stack or bundle of money (small rectangular paper "
         "items, usually beige/green/blue) lying on it? Ignore the counting machine, laptop and phone."),
    ],
    "door_open": [
        ("crop", "door", 0.15, 640, None),
        ("crop_p2", "door", 0.15, 640,
         "The room entrance is a glass door set in a glass wall. Is the door leaf swung open, i.e. turned at an "
         "angle to the wall leaving a gap/opening to the corridor? Answer no if it is shut flush with the wall."),
        ("full_p2", None, 0, 848,
         "The room entrance is a glass door set in a glass wall. Is the door leaf swung open, i.e. turned at an "
         "angle to the wall leaving a gap/opening to the corridor? Answer no if it is shut flush with the wall."),
    ],
    "key_in_safe": [
        ("crop_tight", "safe", 0.05, 896, None),
    ],
}

HANDOVER_MC = (
    "These frames are in time order and show a visitor at a bank cashier's desk. What does the visitor hand "
    "to the cashier FIRST? Reply JSON only: {\"first_item\": \"id_document\" | \"banknotes\" | \"bag\" | "
    "\"nothing\" | \"unclear\", \"evidence\": \"<max 15 words>\"}. An id_document is a small card or passport "
    "that the cashier looks at and usually gives back."
)
HANDOVER_ITEMS = {
    # (vid, times, expected first item)
    "id_document": [("X-3", [5, 6, 7, 8]), ("X-1", [4, 5, 6, 7]), ("X-35", [6, 8, 10, 12])],
    "banknotes": [("X-2", [7, 8, 9, 10]), ("X-34", [4, 6, 7, 8])],
    "bag": [("X-17", [5, 6, 7, 8]), ("X-16", [8, 10, 12, 14])],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    tag = args.model.split("/")[-1]
    items = yaml.safe_load((ROOT / "scripts" / "feasibility_items.yaml").read_text())
    ground = json.loads((ROOT / "reports" / "feasibility" / f"{tag}.json").read_text())["grounding"]
    vlm = VLM(make_backend("mlx", args.model), max_side=896, cache_dir=ROOT / "cache" / "vlm")
    res = {}
    for qid, variants in VARIANTS.items():
        for name, obj, pad, side, text in variants:
            ok = n = 0
            for expected, lst in items[qid].items():
                expected = {True: "yes", False: "no"}.get(expected, expected)
                for vid, t in lst:
                    f = frame_at(video_path(vid), t)
                    img = f
                    if obj:
                        box = ground.get(vid, {}).get(obj)
                        img = upscale_min(crop_box(f, box, pad=pad), min_side=min(side, 448)) if box else f
                    prompt = PRE + "\nQuestion: " + (text or QUESTIONS["questions"][qid]["text"]).strip()
                    a = parse_answer(vlm.raw([(prompt, [to_pil(img, side)])])[0])
                    n += 1
                    ok += a.answer == expected
                    print(f"{qid} {name:14s} {vid} {t} exp={expected} got={a.answer} | {a.evidence}")
            res[f"{qid}|{name}"] = round(ok / n, 3)
            print(f"==> {qid}|{name}: {ok}/{n}")
    # handover multiple choice: crop around desk (with margin to include the visitor), 4 frames
    for crop_mode in ("full", "desk"):
        ok = n = 0
        for expected, lst in HANDOVER_ITEMS.items():
            for vid, ts in lst:
                frames = []
                for t in ts:
                    f = frame_at(video_path(vid), t)
                    if crop_mode == "desk":
                        f = crop_box(f, ground.get(vid, {}).get("desk"), pad=0.25)
                    frames.append(to_pil(f, 512))
                txt = vlm.raw([(HANDOVER_MC, frames)], max_tokens=60)[0]
                j = parse_json(txt) or {}
                got = j.get("first_item", "unclear")
                n += 1
                ok += got == expected
                print(f"handover {crop_mode} {vid} exp={expected} got={got} | {j.get('evidence', txt[:80])}")
        res[f"handover_mc|{crop_mode}"] = round(ok / n, 3)
        print(f"==> handover_mc|{crop_mode}: {ok}/{n}")
    (ROOT / "reports" / "feasibility" / f"experiments_{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
