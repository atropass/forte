# Threshold calibration

Model: `Qwen/Qwen3-VL-4B-Instruct` (openai)

| question | threshold | balanced acc | recall | specificity | n+ / n- |
|---|---|---|---|---|---|
| safe_open | 0.15 | 1.00 | 1.00 | 1.00 | 4 / 5 |
| money_on_desk | 0.50 | 0.88 | 0.75 | 1.00 | 4 / 4 |
| bag_on_desk | 0.55 | 0.83 | 0.67 | 1.00 | 3 / 4 |
| laptop_on | 0.55 | 1.00 | 1.00 | 1.00 | 3 / 4 |
| door_open | 0.65 | 1.00 | 1.00 | 1.00 | 3 / 4 |
| personal_items | 0.90 | 0.88 | 0.75 | 1.00 | 4 / 12 |
| cashier_at_desk | 0.55 | 1.00 | 1.00 | 1.00 | 4 / 4 |
| cash_in_machine | 0.20 | 0.58 | 0.33 | 0.83 | 3 / 6 |

## Hand-over episodes (V05 / V13): handover_id=0.5, handover_act=0.35

7/10 episodes correct.

| video | expected | got | handover t | label t | ID seen at |
|---|---|---|---|---|---|
| X-1 | clean | clean | 21.0 |  | 5.0 |
| X-3 | clean | clean | 14.5 |  | 4.0 |
| X-19 | clean | clean | None |  | None |
| X-35 | clean | clean | None |  | None |
| X-37 | clean | clean | None |  | None |
| X-39 | clean | clean | None |  | None |
| X-16 | clean | clean | 4.0 |  | 3.5 |
| X-2 | client_no_id | clean | None | 9.5 | None |
| X-34 | client_no_id | clean | None | 7.0 | None |
| X-17 | collector_no_id | clean | 4.0 | 7.0 | 6.0 |

| video | X03 expected | 2nd hand-over at | label t |
|---|---|---|---|
| X-1 | partial | None | 22.0 |
| X-3 | whole | None |  |
| X-19 | whole | None |  |
| X-35 | whole | None |  |

**Calibration-gated rules enabled:** none (cash_in_machine_no_client needs balanced acc >= 0.8 and recall >= 0.67; partial_disbursement needs every judged episode right).

Small sample (10 episodes, 4 violations): treat as a starting point, re-check on new footage.
