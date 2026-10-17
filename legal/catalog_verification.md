# Team violation catalog: verification and coverage

Source files (team, 2026-10-17): `legal/VIOLATIONS_GUIDE_EN.source.md`, `legal/violations_catalog.source.json`.
Each legal claim was checked against the act text (zakon.uchet.kz mirror of adilet, which includes the
2025–2026 amendments; footnotes from No. 71 of 10.11.2025 and No. 25 of 16.03.2026 are visible in it).
`legal/norms.yaml` is the corrected version the service uses.

## 1. Clause checks
| Catalog claim | Result |
|---|---|
| No. 231 п.12 (personal items / photo-video devices) | ✅ word for word, edition of 10.11.2025 No. 71 |
| No. 231 п.22 (cashier may not leave before service ends) | ✅ |
| No. 231 п.20 абз.1 (count in client's presence), абз.2 (only client's cash on the desk; earlier cash locked away) | ✅ |
| No. 231 п.21 (separate acceptance per document) | ✅ |
| No. 231 п.19 пп.1, п.31 пп.1 (visual ID check) | ✅ |
| No. 231 п.19 пп.2, п.57, п.9 (sheet count + authenticity) | ✅ |
| No. 231 п.24 (accepting from collection: document, bag integrity, seal) | ✅ (no explicit "collector shows ID") |
| No. 231 п.42 (counting/sorting with a supervising employee) | ✅ |
| No. 231 п.80 (storage of keys to safes / cash drawers) | ✅ used for the "key left in safe" basis |
| No. 231 пп.59–61 (doubtful / counterfeit notes) | ✅ |
| No. 231 **п.7** cited for ID checks | ⚠️ п.7 is about AML due diligence ("меры по надлежащей проверке"), not visual ID. **Not cited** |
| No. 231 **п.75** cited for unauthorized persons in the cash desk | ⚠️ п.75 regulates entry to the **vault / safe room**. **Not cited** for the cash room |
| No. 231 "подпункт 7) пункта 3" (definition of cash storage) | ➖ definition exists in the current п.3, but it adds nothing. Not cited |
| No. 14 п.8 (cash unit isolated by one lockable entrance), п.16 пп.1–6 (lockable door, transfer device, cameras) | ✅ |
| No. 49 п.41, п.11 (exchange offices) | ⚠️ applies to **exchange offices**, not bank cash desks; clause text not verified. **Not cited** |
| "Глава 2, Параграф N" labels | not relied on; we cite only пункт numbers, which are unambiguous |
| No. 159 (CIT transport), ARDFM No. 188 (risk management) | not verified; not needed for detected types |

## 2. Coverage: which codes the service detects
| Code | Catalog type | Status | Our rule(s) | Training videos |
|---|---|---|---|---|
| V01 | Personal belongings / phone in the cash desk | **detected** (personal items; phone use not yet) | personal_items_on_desk | X-40, X-46 |
| V02 | Cashier leaves during client service | **detected, untested** | cashier_left_during_service | — |
| V03 | Unsecured cash on desk / open safe | **detected**: money left on leaving, previous client's money, safe open, key in safe lock, CIT bag left, generic open check | money_left_unattended, prev_client_money_on_desk, safe_left_open, key_left_in_safe, bag_left_on_desk, unsecured_valuables_generic | X-4, X-15, X-20, X-22, X-24, X-26, X-38, X-46 |
| V04 | Mixing cash across several documents | not detected — documents not distinguishable at 464 px | — | — |
| V05 | No ID check of the client | **detected, low confidence** (card ≈ cash at this resolution, see feasibility) | client_no_id | X-2, X-34 |
| V06 | No sheet count / authenticity check | rule present, **disabled** until the team confirms X-1/X-44 | unchecked_rejected_banknotes | X-1?, X-44? |
| V07 | Counting in client's absence | not detected (not in data; would need client-presence × counting) | — | — |
| V08 | Documents not signed / no receipt | not detected — signatures/stamps not visible | — | — |
| V09 | Disbursing without counting | not detected (not in data) | — | — |
| V10 | Counting without a supervisor | **deliberately off**: a cashier counting alone appears in clean training videos (X-43, X-45), so the dataset treats it as normal and enabling it would cause false positives | — | — |
| V11 | Cash handed between employees without a record | not detected — the record itself is invisible | — | — |
| V12 | Improper strapping / packaging | not detected — straps not visible at 464 px | — | — |
| V13 | CIT bag accepted without checks/documents | **detected, low confidence** | collector_no_id | X-17 |
| V14 | Damaged CIT bag not opened in collector's presence | not detected (not in data) | — | — |
| V15 | Collector without body armour / cash outside a bag | not detected (not in data); possible future VLM check | — | — |
| V16 | Unauthorized person in the cash room | **detected** as a second visitor during service | second_client_during_service | X-18 |
| V17 | Cash room door open | **detected** | door_left_open | X-32, X-33 |
| V18 | Cash passed bypassing the tray / theft / camera blocked | not detected (this setup has no tray/booth) | — | — |
| V19 | Counterfeit returned to client | not detected (not in data) | — | — |
| X01 | *(not in catalog)* Workstation computer left unlocked | **detected** | laptop_unlocked_unattended | X-11, X-28 |
| X02 | *(not in catalog)* Client served before the workplace is ready | **detected** | client_admitted_workplace_not_ready | X-30 |

The API returns the catalog `type_ru` as `type`; the specific situation is in `subtype` and in
`description`; the code is in `type_code`.
