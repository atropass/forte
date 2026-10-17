# Decisions log

| # | Date | Decision | Why |
|---|---|---|---|
| 1 | 2026-10-17 | Use Latin `X-n` as the canonical video id. | Sheet and filenames mix Cyrillic Х / Latin X. |
| 2 | 2026-10-17 | The Excel file has 4 columns: col 3 is the verdict, col 4 the description. Adapted the parser instead of the assumed 3-column layout. | Real file structure. |
| 3 | 2026-10-17 | Merge duplicates X-47→X-4 and X-48→X-5 in eval. Accept either 57 or 68 s for X-4. | Identical MD5. |
| 4 | 2026-10-17 | Timestamps come from PTS (`ffprobe`/decoder), never frame_idx/30. | All files are VFR (avg 29.76–30.0 fps). |
| 5 | 2026-10-17 | Added two types not in the seed list: `client_admitted_workplace_not_ready` (X-30) and `personal_items_on_desk` (X-40, X-46). | Teammate descriptions. |
| 6 | 2026-10-17 | X-1/X-44 tentatively `unchecked_rejected_banknotes`, X-13 `unknown`. Flagged for teammate confirmation; excluded from per-type accuracy until confirmed (still count for verdict recall). | No description in the sheet. |
| 7 | 2026-10-17 | **Hardware differs from the brief:** this machine is an Apple M1 Pro with 16 GB unified memory and no CUDA. Phase 2 will benchmark the VLM via MLX/MPS locally. The production target (Linux + NVIDIA, vLLM) stays the deployment path in Docker; code keeps a backend switch (`mlx` / `transformers` / `vllm`). | The environment section in the brief was left as a template (`<FILL IN>`). |
| 8 | 2026-10-17 | Timecode = start of the triggering behaviour (see data_audit §4). | Matches the labeller's convention on 19/21 videos. |
