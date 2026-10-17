# Phase 4 — Evaluation on the training videos

Videos: 8 unique (4 with violation, 4 clean). Duplicates X-47/X-48 excluded.

| metric | value |
|---|---|
| video-level verdict accuracy | **0.625** (5/8) |
| recall on violation videos | **0.250** (1/4) |
| false-positive rate on clean videos | **0.000** (0/4) |
| type accuracy (label code among predicted codes; confirmed labels only) | 1.000 (1/1) |
| timecode error, matching type (s): mean / median | 2.5 / 2.5 (n=1) |

## Per video

| video | label | label codes @ t | predicted codes @ t | verdict | type | tc err |
|---|---|---|---|---|---|---|
| X-1 | 1 | V06 @ 22 | — | ❌ |  |  |
| X-2 | 1 | V05 @ 10 | — | ❌ |  |  |
| X-3 | 0 |  @  | — | ✅ |  |  |
| X-4 | 1 | V03 @ 57|68 | V03@59.5 | ✅ | ✅ | 2.5 |
| X-5 | 0 |  @  | — | ✅ |  |  |
| X-8 | 0 |  @  | — | ✅ |  |  |
| X-10 | 0 |  @  | — | ✅ |  |  |
| X-11 | 1 | X01 @ 18 | — | ❌ |  |  |