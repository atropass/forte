"""Phase 1 data audit: parse Excel labels, match to videos, probe metadata, write data/labels.csv.

Usage: python scripts/audit.py  (paths relative to repo root)
"""
import csv
import hashlib
import json
import re
import subprocess
import unicodedata
from pathlib import Path

import openpyxl
import yaml

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "open" / "videos"
XLSX = ROOT / "Yerkes.xlsx"
OUT_CSV = ROOT / "data" / "labels.csv"
OUT_META = ROOT / "data" / "video_meta.json"

# Cyrillic "Х" and Latin "X" are mixed in both filenames and the sheet -> normalise to Latin.
def norm_id(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s)).strip()
    return s.replace("Х", "X").replace("х", "x").upper()


def parse_seconds(s):
    if s is None or str(s).strip() == "":
        return None
    s = str(s).lower()
    m = re.search(r"(\d+)\s*мин\w*\s*(\d+)?", s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2) or 0)
    m = re.search(r"(\d+(?:[.,]\d+)?)", s)
    return float(m.group(1).replace(",", ".")) if m else None


# Mapping of the teammate's free-text (col 4) to type ids. Derived by reading every description and
# looking at the frames around each timecode (reports/label_strips/). See reports/data_audit.md.
TYPE_BY_VIDEO = {
    "X-1": ["unchecked_rejected_banknotes?"],  # no description; hypothesis, see audit
    "X-2": ["client_no_id"],
    "X-4": ["prev_client_money_on_desk"],
    "X-11": ["laptop_unlocked_unattended"],
    "X-13": ["unknown?"],  # no description; hypotheses in audit
    "X-15": ["bag_left_on_desk"],
    "X-17": ["collector_no_id"],
    "X-18": ["second_client_during_service"],
    "X-20": ["safe_left_open"],
    "X-22": ["key_left_in_safe"],
    "X-24": ["key_left_in_safe"],
    "X-26": ["money_left_unattended"],
    "X-28": ["laptop_unlocked_unattended"],
    "X-30": ["client_admitted_workplace_not_ready"],
    "X-32": ["door_left_open"],
    "X-33": ["door_left_open"],
    "X-34": ["client_no_id"],
    "X-38": ["prev_client_money_on_desk"],
    "X-40": ["personal_items_on_desk"],
    "X-44": ["unchecked_rejected_banknotes?"],  # no description; hypothesis
    "X-46": ["personal_items_on_desk", "money_left_on_desk", "key_left_in_safe"],
    "X-47": ["prev_client_money_on_desk"],
}


RULE_CODES = {k: v.get("code", "") for k, v in
              yaml.safe_load((ROOT / "pipeline" / "rules.yaml").read_text())["types"].items()}
RULE_CODES["money_left_on_desk"] = "V03"
RULE_CODES["unknown"] = "?"


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames:format=duration,size",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    st, fm = j["streams"][0], j["format"]
    num, den = map(int, st["avg_frame_rate"].split("/"))
    return {
        "codec": st["codec_name"], "width": st["width"], "height": st["height"],
        "r_fps": st["r_frame_rate"], "avg_fps": round(num / den, 3), "nb_frames": int(st.get("nb_frames", 0)),
        "duration": round(float(fm["duration"]), 3), "size_mb": round(int(fm["size"]) / 1e6, 2),
        "md5": hashlib.md5(path.read_bytes()).hexdigest(),
    }


def main():
    videos = {norm_id(p.stem): p for p in sorted(VIDEO_DIR.glob("*.mov"))}
    meta = {vid: {"file": str(p.relative_to(ROOT)), **probe(p)} for vid, p in videos.items()}

    ws = openpyxl.load_workbook(XLSX).active
    rows, unmatched = [], []
    for r in list(ws.iter_rows(values_only=True))[1:]:
        if not r[0]:
            continue
        vid = norm_id(r[0])
        verdict = str(r[2] or "").strip().lower()
        has_v = verdict == "нарушение"
        t = parse_seconds(r[1])
        desc = (r[3] or "").strip()
        if vid not in videos:
            unmatched.append(vid)
        types = TYPE_BY_VIDEO.get(vid, []) if has_v else []
        codes = [RULE_CODES.get(x.rstrip("?"), "?") + ("?" if x.endswith("?") else "") for x in types]
        rows.append({
            "video": vid, "file": meta.get(vid, {}).get("file", ""), "has_violation": int(has_v),
            "timecode_sec": t if has_v else "", "description_ru": desc,
            "mapped_type": "|".join(types), "catalog_code": "|".join(codes), "type_confident": int(has_v and not any(x.endswith("?") for x in types)),
        })

    labelled = {r["video"] for r in rows}
    no_label = sorted(set(videos) - labelled)

    # duplicate detection
    by_md5 = {}
    for vid, m in meta.items():
        by_md5.setdefault(m["md5"], []).append(vid)
    dups = [v for v in by_md5.values() if len(v) > 1]
    for r in rows:
        grp = next((g for g in dups if r["video"] in g), None)
        r["duplicate_of"] = sorted(grp, key=lambda x: int(x.split("-")[1]))[0] if grp and sorted(grp, key=lambda x: int(x.split("-")[1]))[0] != r["video"] else ""

    OUT_CSV.parent.mkdir(exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=1))

    print(f"videos={len(videos)} label_rows={len(rows)} violations={sum(r['has_violation'] for r in rows)}")
    print("unmatched label rows:", unmatched)
    print("videos without label:", no_label)
    print("duplicates (md5):", dups)
    for r in rows:
        if r["has_violation"] and r["timecode_sec"] != "" and r["video"] in meta and r["timecode_sec"] > meta[r["video"]]["duration"]:
            print("timecode beyond duration:", r["video"], r["timecode_sec"])


if __name__ == "__main__":
    main()
