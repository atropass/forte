# Phase 4 — Evaluation on the training videos

Videos: 8 unique (4 with violation, 4 clean). Duplicates X-47/X-48 excluded.

| metric | value |
|---|---|
| video-level verdict accuracy | **0.750** (6/8) |
| recall on violation videos | **0.500** (2/4) |
| false-positive rate on clean videos | **0.000** (0/4) |
| type accuracy (label code among predicted codes; confirmed labels only) | 1.000 (2/2) |
| timecode error, matching type (s): mean / median | 4.3 / 4.3 (n=2) |

## Per video

| video | label | label codes @ t | predicted codes @ t | verdict | type | tc err |
|---|---|---|---|---|---|---|
| X-1 | 1 | X03 @ 22 | — | ❌ |  |  |
| X-2 | 1 | V05 @ 10 | — | ❌ |  |  |
| X-3 | 0 |  @  | — | ✅ |  |  |
| X-4 | 1 | V03 @ 57|68 | V03@58.8 | ✅ | ✅ | 1.8 |
| X-5 | 0 |  @  | — | ✅ |  |  |
| X-8 | 0 |  @  | — | ✅ |  |  |
| X-10 | 0 |  @  | — | ✅ |  |  |
| X-11 | 1 | X01 @ 18 | X01@11.2|V03@11.2 | ✅ | ✅ | 6.8 |