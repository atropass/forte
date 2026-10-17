# Progress

| Phase | Status | Notes |
|---|---|---|
| 1 Data audit | ✅ done | `reports/data_audit.md`, `data/labels.csv`. 40 unique videos (21 violation / 19 clean), 2 duplicate pairs, 3 rows without description. |
| 2 Feasibility | ✅ done | `reports/feasibility.md`. Qwen3-VL-4B chosen. Crops: safe/bag/laptop/personal items 1.00, door 0.86, money 0.75; VLM presence count 21/21. Infeasible as-is: key in lock (3 px), client ID (card ≈ cash). Mac bench 137 s without the presence pass; GPU bench pending (`make bench`). |
| 3 Pipeline | ⏳ next (frames/perception/timeline/vlm/rules/evidence/legal drafted) | |
| 4 Eval | — | |
| 5 Robustness | — | |
| 6 Service | — | |
| 7 Legal | — | |
| 8 Docs | — | |


## Evaluation on all training videos (2026-10-17, Qwen3-VL-4B on GPU pod + RT-DETRv2 CV pod)
`reports/eval.md` (per video), history in `reports/eval_run{1..4}.md`.

| metric | value |
|---|---|
| video-level verdict accuracy | **80% (32/40)** |
| recall on violation videos | 62% (13/21) |
| false positives on clean videos | **0% (0/19)** |
| type accuracy (confirmed labels) | 92% (12/13) |
| timecode error (matching type) | mean 1.1 s, median 0.9 s |

Misses (8): X-2, X-34 (client ID, ~10 px), X-17 (collector documents), X-24 (key in lock, ~3 px),
X-1 (partial disbursement, rule failed calibration → off), X-44 (cash in machine, failed calibration → off),
X-13 (type unconfirmed by the team), X-46 (drinking: 0.62 vs threshold 0.9, raising recall would make clean X-16 fire).
Caveat: thresholds were tuned on these same 40 videos, so numbers on new footage will be lower.
