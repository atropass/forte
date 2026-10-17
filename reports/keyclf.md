# Key-in-lock classifier — leave-one-video-out

Samples: 332 patches (47 positive) from 22 videos.

| video | label | frames | mean p | frac ≥ 0.5 |
|---|---|---|---|---|
| X-3 | clean | 10 | 0.01 | 0.00 |
| X-5 | clean | 33 | 0.02 | 0.00 |
| X-8 | clean | 17 | 0.17 | 0.06 |
| X-10 | clean | 8 | 0.02 | 0.00 |
| X-12 | clean | 16 | 0.03 | 0.00 |
| X-14 | clean | 44 | 0.07 | 0.02 |
| X-16 | clean | 11 | 0.00 | 0.00 |
| X-19 | clean | 28 | 0.01 | 0.00 |
| X-21 | clean | 7 | 0.49 | 0.71 |
| X-22 | key | 11 | 0.00 | 0.00 |
| X-23 | clean | 2 | 0.55 | 0.50 |
| X-24 | key | 9 | 0.01 | 0.00 |
| X-25 | clean | 7 | 0.00 | 0.00 |
| X-27 | clean | 3 | 0.44 | 0.67 |
| X-29 | clean | 4 | 0.94 | 1.00 |
| X-31 | clean | 8 | 0.00 | 0.00 |
| X-35 | clean | 26 | 0.00 | 0.00 |
| X-37 | clean | 8 | 0.00 | 0.00 |
| X-39 | clean | 35 | 0.02 | 0.03 |
| X-43 | clean | 8 | 0.00 | 0.00 |
| X-45 | clean | 10 | 0.11 | 0.00 |
| X-46 | key | 27 | 0.01 | 0.00 |

thr 0.30: mean per-video positive-frame recall 0.00, clean-frame FPR 0.056

thr 0.35: mean per-video positive-frame recall 0.00, clean-frame FPR 0.056

thr 0.40: mean per-video positive-frame recall 0.00, clean-frame FPR 0.056

thr 0.45: mean per-video positive-frame recall 0.00, clean-frame FPR 0.056

thr 0.50: mean per-video positive-frame recall 0.00, clean-frame FPR 0.053

thr 0.55: mean per-video positive-frame recall 0.00, clean-frame FPR 0.046

thr 0.60: mean per-video positive-frame recall 0.00, clean-frame FPR 0.035

thr 0.65: mean per-video positive-frame recall 0.00, clean-frame FPR 0.032

thr 0.70: mean per-video positive-frame recall 0.00, clean-frame FPR 0.025

thr 0.75: mean per-video positive-frame recall 0.00, clean-frame FPR 0.021

thr 0.80: mean per-video positive-frame recall 0.00, clean-frame FPR 0.021

thr 0.85: mean per-video positive-frame recall 0.00, clean-frame FPR 0.021

thr 0.90: mean per-video positive-frame recall 0.00, clean-frame FPR 0.011

thr 0.95: mean per-video positive-frame recall 0.00, clean-frame FPR 0.011

**Chosen threshold 0.50** (rule needs 2 of 3 frames ≥ threshold).
## Conclusion
Not shipped. Held-out key videos score ≈ 0 while clean twins of the same setup (X-21, X-23, X-27, X-29)
score high: with 3 positive videos the model learns the camera setup, not the key. key_left_in_safe
therefore stays on the VLM check, which answers "unclear" on these frames, so the rule effectively does not
fire. Known miss: X-22, X-24 (X-46 is still caught via V01). Fix requires a higher-resolution camera or
labelled lock close-ups.
