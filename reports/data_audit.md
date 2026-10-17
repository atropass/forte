# Phase 1 — Data audit

Produced by `make audit` (`scripts/audit.py`). Generated files: `data/labels.csv`, `data/video_meta.json`,
`reports/contact_sheets/*.jpg` (1 fps), `reports/label_strips/*.jpg` (7 s around each labelled second),
`reports/first_frames_mosaic.jpg`.

## 1. Inputs found
| Input | Path | Notes |
|---|---|---|
| Spec | `Кейс_ForteBank.docx` | read fully; API/JSON schema in §10, scoring in §9, FAQ |
| Videos | `open/videos/*.mov` | 42 files |
| Labels | `Yerkes.xlsx`, sheet `labels` | 42 rows, **4 columns** (not 3): file id · timecode text · `нарушение`/`без нарушения` · free-text description |

Label sheet quirks:
* IDs mix Cyrillic `Х` (40 files) and Latin `X` (`X-1`, `X-2`) — normalised to Latin `X-n` everywhere.
* Timecodes are Russian text (`"22 секунда"`, `"1минута 8 секунда"`) — parsed to seconds, integer resolution.
* One timecode per video; no video has several rows.
* 3 violation rows have **no description** (X-1, X-13, X-44).
* Every row matches a file and every file has a row. Missing numbers in the series (6, 7, 9, 36, 41, 42) simply don't exist.

## 2. Video properties
* All 42: H.264, **464×848 portrait**, no audio track, nominal 30 fps but **variable frame rate** (avg 29.76–30.00) → timestamps must come from PTS, not `frame_idx / 30`.
* Duration 11.6–142.5 s (mean 34.4, median 22.7, total 24 min). Size 2.4–28.7 MB.
* **Duplicates (identical MD5):** `X-4 ≡ X-47` and `X-5 ≡ X-48`.
  * X-5/X-48: both `без нарушения` — consistent.
  * X-4/X-47: both "некст клиент деньги на столе" but labelled **57 s** vs **68 s**. At 57 s the next client walks in while the cashier still has the previous client's money on the desk; at 68 s that client is seated and banknotes are on the desk. Treated as one video with two acceptable timecodes; eval counts it once (X-47 marked `duplicate_of=X-4`), timecode error = min over {57, 68}.
* Effective dataset: **40 unique videos, 21 unique violation videos, 19 clean.**

### Scene setups (same room, ≥4 camera/furniture arrangements)
| Setup | Look | Videos |
|---|---|---|
| A | cashier with back to camera, client faces camera, safe on the right | 1,2,3,4,5,16,17,18,19,47,48 |
| B | cashier faces camera, desk diagonal, safe behind cashier on the right, door at top-left | 8,10–15,20–29,30,31,32 |
| C | like B but with a blue client chair, cashier on right side | 33,40,43,44,45,46 |
| D | cashier (white blouse) with back to camera, laptop on left | 34,35,37,38,39 |

Training already contains several angles. This is good for robustness: we can hold out whole setups during eval. Fixed pixel zones would break even inside the training set.

## 3. Type mapping (Excel column 4 → type id)
Mapped by reading each description and checking the frames around each timecode (`reports/label_strips/`).

| Type id | Seed # | Videos (t, s) | What is visible | How the labelled second relates to the event |
|---|---|---|---|---|
| `client_no_id` | 1 | X-2 (10), X-34 (13) | Client sits down and puts money on the desk; no ID card is shown at any point. In clean setup-A videos (X-3, X-1) the client hands a card first. | X-2: the moment the money is put down. X-34: money at 7 s, client leaves 11–13 s → label ≈ **end of the client episode**. Inconsistent, so we report the handover moment. |
| `collector_no_id` | 2 | X-17 (7) | Collector comes in with a green CIT bag and hands it straight over; no ID. | the **handover moment** |
| `unchecked_rejected_banknotes` (hypothesis) | 3 | X-1 (22)?, X-44 (16)? | X-1: client showed ID (6–7 s); money goes through the counting machine (14 s); at 20–23 s the cashier takes notes from the machine and hands them on without inspecting them. X-44: cashier alone, takes notes out of the machine (11–13 s) and handles them at 14–19 s. | around the moment notes leave the machine. **No description in the sheet → unconfirmed; needs teammate check.** |
| `prev_client_money_on_desk` | 4 | X-4 (57 / dup 68), X-38 (33) | Banknotes from the previous operation still lie on the desk when the next client walks in. | **next client crosses the door / approaches the desk** |
| `second_client_during_service` | 5 | X-18 (20) | A second person enters at 17 s while client 1 is seated; reaches the desk and hands something over at 20 s. | second client **reaches the counter** (3 s after entering) |
| `client_admitted_workplace_not_ready` | new | X-30 (13) | Cashier is still standing, setting up the counting machine, when a client enters (12–13 s). | client **appears in the room** |
| `door_left_open` | 6 | X-32 (5), X-33 (6) | Cashier enters through the glass door at 4–5 s and doesn't close it behind her. | ~1 s after the cashier passes the door (door visibly stays open) |
| `safe_left_open` | 7 | X-20 (5) | Cashier opens the safe (0–4 s) and sits down at the desk with the safe door open. | moment she **turns away from the open safe / sits down** |
| `key_left_in_safe` | 8 | X-22 (14), X-24 (8), X-46 (11, secondary) | Cashier locks the safe, the key stays in the lock, and she walks out. | cashier **turns away from the safe and walks to the door** (start of leaving) |
| `money_left_unattended` | 12 | X-26 (9), X-46 (secondary) | Money on the desk; cashier gets up (9 s) and leaves (10–11 s). | cashier **stands up** |
| `laptop_unlocked_unattended` | 10 | X-11 (18), X-28 (7) | Laptop screen bright/on; cashier gets up and leaves. | cashier **starts walking away** |
| `bag_left_on_desk` | 11 | X-15 (15) | Green CIT bag stays on the desk after she put the money in the safe; she leaves. | cashier **walks to the door** |
| `personal_items_on_desk` | new | X-40 (17), X-46 (11) | X-40: cashier puts a personal handbag/coat on the work desk. X-46: cashier drinks from a cup at the desk with cash on the desk; sheet also says "key in safe". | moment the item **is placed / used** |
| `unknown` | — | X-13 (47) | Long solo clip: cashier empties a CIT bag, counts money, puts it in the safe (45–53 s), locks it and leaves (58 s). Candidates at 47 s: money put in the safe **without** going through the counting machine (manual count only); the safe opened while banknotes are spread on the desk. **Needs teammate check.** | — |

Seed types with **no examples** in training: #9 "left without safe key" (hard to separate visually from #8) and the generic case.
New types found: `client_admitted_workplace_not_ready` and `personal_items_on_desk`.

### Counts (unique videos; X-47 merged into X-4)
| Type | # videos |
|---|---|
| key_left_in_safe | 2 (+1 secondary) |
| laptop_unlocked_unattended | 2 |
| door_left_open | 2 |
| client_no_id | 2 |
| prev_client_money_on_desk | 2 |
| unchecked_rejected_banknotes (hyp.) | 2 |
| personal_items_on_desk | 2 |
| collector_no_id, second_client_during_service, client_admitted_workplace_not_ready, safe_left_open, money_left_unattended, bag_left_on_desk, unknown | 1 each |

19 clean videos: X-3, 5, 8, 10, 12, 14, 16, 19, 21, 23, 25, 27, 29, 31, 35, 37, 39, 43, 45 (+ X-48 = X-5).

## 4. Timecode convention (what we will emit)
The labeller marks **the moment the triggering behaviour starts**, at integer seconds:
* "left unattended" types (key, money, laptop, bag, safe): the second the cashier **turns away / stands up / heads for the door** — not when she leaves the frame (that's 1–3 s later).
* "arrival" types (next client, second client, client while not ready): when the arriving person **enters the room or reaches the desk**.
* "handover without ID" types: the **handover moment** (X-34 is an outlier at the end of the episode).
* door open: ~1 s after the cashier passes through the door.

Our rule engine will emit `timecode_sec` = start of the trigger event. Evidence frames come from trigger to trigger+2 s.

## 5. Anomalies / open questions for the team
1. X-4 vs X-47 duplicate with conflicting timecodes (57 vs 68).
2. X-1, X-13, X-44 have no description. Types are my hypotheses. **Please confirm.**
3. X-34 timecode convention differs from X-2 (end of episode vs handover).
4. X-46 has three types in one description; the cashier never leaves. "Key in safe" here means the key stays in the lock while she works.
5. X-11: camera auto-exposure drops sharply at 16 s (scene goes dark). The perception layer must tolerate global brightness jumps.
6. People are visible **through the glass wall** (outside the room) in most videos. Person detection must not count them as clients, so we need a room/door model, not just a person count.
