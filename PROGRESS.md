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
