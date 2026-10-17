# Bank Cash Discipline Compliance Violation Guide (Datathon 26 — ForteBank)

This document provides the complete classification framework for visually observable cash desk, cash-in-transit (CIT / inkassatsiya), and physical security compliance violations under the laws and regulations of the **Republic of Kazakhstan** (National Bank of the Republic of Kazakhstan — NBRK, Agency for Regulation and Development of the Financial Market — ARDFM, and the Official Legal Information System `adilet.zan.kz`).

This guide is designed for:
1. **AI Service Event Classification** — populating `verdict`, `type`, `timecode_sec`, `description`, and `evidence_frames` in the evaluation API response.
2. **Local Vision-Language Model (VLM) & CV Pipeline Prompting** — detecting objects, roles, and procedural anomalies from overhead (top-down) CCTV camera recordings.
3. **Regulatory Justification for Jury Evaluation** — maximizing Criterion #4 ("Rule Justification & Regulatory Accuracy", 15 points).

---

## 1. Regulatory Sources (Republic of Kazakhstan)

All violation categories in this guide are strictly grounded in publicly enacted regulations of the Republic of Kazakhstan (as required by Section 6 and FAQ items 30–32 of the Case Specification):

1. **Resolution of the Board of the National Bank of the Republic of Kazakhstan (NBRK) No. 231 dated November 29, 2019**
   * **Official Title:** *"On Approval of the Rules for Conducting Cash Operations and Operations on Collection (Inkassatsiya) of Banknotes, Coins, and Valuables in Second-Tier Banks, Branches of Non-Resident Banks of the Republic of Kazakhstan, the National Postal Operator, and Legal Entities Whose Exclusive Activity is Collection of Banknotes, Coins, and Valuables"* (Registered with the Ministry of Justice of the RK under No. 19680).
   * **Latest Amendments:** Includes amendments introduced by NBRK Board Resolutions No. 19 (28.02.2022), No. 125 (19.12.2022), No. 71 (10.11.2025, `V2500037429`), No. 25 (16.03.2026, `V2600038217`), and No. 78 (30.06.2026, `V2600039191`).
   * **Official Link:** `https://adilet.zan.kz/rus/docs/V1900019680`

2. **Resolution of the Board of the National Bank of the Republic of Kazakhstan (NBRK) No. 14 dated February 24, 2020**
   * **Official Title:** *"On Approval of the Rules for Organizing Security and Arrangement of Premises of Second-Tier Banks, Branches of Non-Resident Banks of the Republic of Kazakhstan, the National Postal Operator, Legal Entities Whose Exclusive Activity is Collection of Banknotes, Coins, and Valuables, and Legal Entities Operating Exclusively Through Exchange Offices Under a License of the National Bank of the Republic of Kazakhstan for Exchange Operations with Foreign Cash"* (Registered with the Ministry of Justice of the RK under No. 20075; amended by NBRK Resolution No. 88 dated 24.12.2024, `V2400035552`).
   * **Official Link:** `https://adilet.zan.kz/rus/docs/V2000020075`

3. **Resolution of the Board of the National Bank of the Republic of Kazakhstan (NBRK) No. 49 dated April 4, 2019**
   * **Official Title:** *"On Approval of the Rules for Conducting Exchange Operations with Foreign Cash in the Republic of Kazakhstan"* (Registered with the Ministry of Justice of the RK under No. 18545).
   * **Official Link:** `https://adilet.zan.kz/rus/docs/V1900018545`

4. **Resolution of the Board of the National Bank of the Republic of Kazakhstan (NBRK) No. 230 dated November 29, 2017**
   * **Official Title:** *"On Approval of the Rules for Determining the Solvency (Fitness/Authenticity) of Banknotes and Coins of the National Currency of the Republic of Kazakhstan"* (Registered with the Ministry of Justice of the RK under No. 16120).
   * **Official Link:** `https://adilet.zan.kz/rus/docs/V1700016120`

5. **Resolution of the Board of the National Bank of the Republic of Kazakhstan (NBRK) No. 159 dated September 12, 2019**
   * **Official Title:** *"On Approval of the Instruction on Organizing Motor Cash-in-Transit (Inkassatsiya) Transportation in the Republic of Kazakhstan"* (Registered with the Ministry of Justice of the RK under No. 19391; incorporated into Chapter 3, Paragraph 2-1 of Rules No. 231 in 2026).
   * **Official Link:** `https://adilet.zan.kz/rus/docs/V1900019391`

6. **Resolution of the Board of the ARDFM / NBRK Resolution No. 188 dated November 12, 2019**
   * **Official Title:** *"On Approval of the Rules for Forming the Risk Management and Internal Control System for Second-Tier Banks and Branches of Non-Resident Banks of the Republic of Kazakhstan"* (`V1900019632` and the updated ARDFM Resolution provided in `ob-utverzhdenii-pravil-formirovaniya-sistemy-upravleniya-riskami-i-vnutrennego-kontrolya-dlya-bankov.docx`).
   * **Official Link:** `https://adilet.zan.kz/rus/docs/V1900019632`

---

## 2. Core Constraints for Video-Based Compliance Analysis (Section 7 of Case Spec)

1. **Strictly Visual Evidence Only:** The AI system must never claim a violation if proving it requires invisible backend data (such as core banking system / ABIS entries or ledger reconciliations).
2. **No Facial Recognition / Biometrics:** Identifying individuals via facial recognition is strictly prohibited (FAQ #29). Detection must rely on spatial zones, roles (cashier inside the booth, client at the window, supervising officer, CIT officer with a bag/vest), and objects/actions.
3. **No Cloud APIs:** All inference must run locally on the server without calling external cloud LLM/VLM APIs.
4. **Structured Justification (`description`):** Every reported violation must state:
   * *What is visually observed on the video*;
   * *Why this indicates a compliance violation / operational risk*;
   * *Exact regulatory citation* (Document title, Resolution number, Chapter, Paragraph, Clause/Subclause).

---

## 3. Complete Taxonomy of Visually Observable Violations (19 Types across 6 Categories)

### Category 1: Cashier Workstation Discipline & Cash Storage Inside the Booth

#### `V01` — Use of a Smartphone (Photo/Video Capable Device) or Presence of Personal Belongings in the Cash Booth
* **Violation Type (`type`):** Наличие личных вещей или использование мобильного телефона в кассе *(Personal belongings or use of a smartphone in the cash desk)*
* **Visual Cues on Video:**
  * The cashier holds, looks at, types on, takes photos with, or talks on a smartphone inside the cash booth.
  * Personal items (handbag, backpack, personal wallet/purse, personal cash, food/drinks) are placed on the cashier's desk or inside the cash booth.
* **Regulatory Citation:**
  * **NBRK Resolution No. 231 (29.11.2019)**, Chapter 2, Paragraph 1, **Clause 12** (as amended by NBRK Resolution No. 71 dated 10.11.2025):
    > *"12. Storage of personal belongings, cash, and valuables of bank employees is carried out outside the cash unit in the manner established by internal documents of the bank. Bank employees are permitted to bring medications, medical devices, and communication devices without photo and video recording functions into the cash unit."*
* **Explanation Template (`description`):**
  > "The video shows the cashier using a smartphone (a communication device with photo/video recording capability) or keeping personal belongings on the workstation inside the cash booth. This violates Clause 12 of the Rules for Conducting Cash Operations (NBRK Board Resolution No. 231 dated 29.11.2019), which prohibits storing personal belongings and personal cash inside the cash unit and forbids bringing communication devices equipped with photo/video recording functions."

---

#### `V02` — Cashier Leaving the Workstation Before Completing Client Service
* **Violation Type (`type`):** Оставление рабочего места кассиром во время обслуживания клиента *(Cashier leaving the workstation during client service)*
* **Visual Cues on Video:**
  * A client is standing at the cash window, cash or transaction documents are on the desk (the transaction is still in progress), and the cashier stands up and walks away from the workstation or exits the cash booth.
* **Regulatory Citation:**
  * **NBRK Resolution No. 231 (29.11.2019)**, Chapter 2, Paragraph 2, **Clause 22**:
    > *"22. The cash employee is not permitted to leave the workstation until the completion of client service."*
* **Explanation Template (`description`):**
  > "The video shows the cashier leaving the workstation before completing the transaction while the client is still being served at the window. This violates Clause 22 of the Rules for Conducting Cash Operations (NBRK Board Resolution No. 231 dated 29.11.2019), which strictly prohibits a cashier from leaving the workstation until client service is finished."

---

#### `V03` — Previously Accepted Cash Left Unsecured on the Desk / Unclosed Safe, Cabinet, or Drawer
* **Violation Type (`type`):** Нарушение порядка хранения наличных денег на рабочем месте кассира *(Unsecured cash on the cashier's desk or unlocked safe/drawer)*
* **Visual Cues on Video:**
  * While serving a client, stacks of banknotes or loose cash from previous transactions are lying openly on the cashier's desk alongside the current client's money.
  * After completing a transaction, the cashier leaves accepted cash on the desk instead of placing it into a lockable storage device (safe, metal cabinet, pedestal drawer, trolley) and locking it with a key.
  * The safe door or cash storage drawer remains wide open and unlocked during subsequent operations or when the cashier steps away.
* **Regulatory Citation:**
  * **NBRK Resolution No. 231 (29.11.2019)**, Chapter 2, Paragraph 2, **Clause 20 (Paragraph 2)** and Paragraph 8, **Clause 80**:
    > *"20. ... Only the client's cash shall be present on the cash employee's workstation. All previously accepted cash must be stored in special devices located at the cash employee's workstation that lock with a key and are designed for storing cash (metal cabinet, safe, trolley, desk drawers, pedestal)."*
* **Explanation Template (`description`):**
  > "The video shows previously accepted cash lying openly on the cashier's desk during client service (or the safe/cash drawer left open and unlocked). This violates Clause 20 (paragraph 2) of the Rules for Conducting Cash Operations (NBRK Board Resolution No. 231 dated 29.11.2019), which mandates that only the current client's cash may be on the desk, while all previously accepted cash must be kept in locked storage devices (safe, metal cabinet, locked drawer)."

---

#### `V04` — Mixing Cash Across Multiple Incoming Cash Documents
* **Violation Type (`type`):** Смешивание наличных денег при приеме по нескольким приходным документам *(Mixing cash when accepting funds under multiple incoming cash documents)*
* **Visual Cues on Video:**
  * A client submits multiple incoming cash vouchers/orders along with multiple bundles of cash, and the cashier combines all the cash into a single pile and counts it together instead of accepting and counting the cash separately for each document.
* **Regulatory Citation:**
  * **NBRK Resolution No. 231 (29.11.2019)**, Chapter 2, Paragraph 2, **Clause 21**:
    > *"21. If a client deposits cash into the bank's cash desk under several incoming cash documents, the cash employee accepts cash under each incoming cash document separately."*
* **Explanation Template (`description`):**
  > "The video shows the cashier combining and counting cash from multiple incoming cash documents simultaneously rather than processing each document separately. This violates Clause 21 of the Rules for Conducting Cash Operations (NBRK Board Resolution No. 231 dated 29.11.2019)."

---

### Category 2: Violations in Accepting Cash from Clients

#### `V05` — Accepting or Disbursing Cash Without Visual Client Identification (No ID Verification)
* **Violation Type (`type`):** Проведение кассовой операции без идентификации клиента по документу, удостоверяющему личность *(Conducting a cash transaction without verifying the client's identity document)*
* **Visual Cues on Video:**
  * The client hands over cash for deposit (or receives cash during withdrawal/exchange), but never presents an identity document (ID card, passport, or digital ID on a smartphone screen), and the cashier processes the transaction without inspecting any ID or visually identifying the client.
* **Regulatory Citation:**
  * **NBRK Resolution No. 231 (29.11.2019)**, Chapter 2, Paragraph 1, **Clause 7**; Paragraph 2, **Subclause 1) of Clause 19** (cash acceptance); Paragraph 3, **Subclause 1) of Clause 31** (cash disbursement):
    > *"19. The bank's cash employee performs the following actions: 1) conducts visual identification of the client depositing cash against their identity document, or data confirming (identifying) the client's identity obtained via the digital documents service..."*
    > *"31. The bank's cash employee performs the following actions: 1) conducts visual identification of the client based on the identity document data, or data confirming (identifying) the client's identity obtained via the digital documents service..."*
  * **NBRK Resolution No. 49 (04.04.2019)**, Chapter 1, **Clause 4** and Chapter 6, **Clause 51** (for currency exchange operations).
* **Explanation Template (`description`):**
  > "The video shows the cashier accepting (or disbursing) cash without requesting or inspecting the client's identity document (or digital ID) and without performing visual identification of the client. This violates Clause 7 and Subclause 1) of Clause 19 (for acceptance) / Subclause 1) of Clause 31 (for disbursement) of the Rules for Conducting Cash Operations (NBRK Board Resolution No. 231 dated 29.11.2019)."

---

#### `V06` — Accepting Cash Without Sheet-by-Sheet Counting or Without Authenticity/Fitness Verification on Detection Equipment
* **Violation Type (`type`):** Прием наличных денег без полистного пересчета и проверки подлинности *(Accepting cash without sheet-by-sheet counting and authenticity verification)*
* **Visual Cues on Video:**
  * The cashier takes banknotes or coins from the client via the transfer tray and immediately places them into the drawer/safe (or signs the receipt) **without counting the banknotes sheet-by-sheet** (by hand or in a counting machine) and/or **without checking their authenticity and fitness** using a UV/magnetic detector or currency sorting machine.
* **Regulatory Citation:**
  * **NBRK Resolution No. 231 (29.11.2019)**, Chapter 2, Paragraph 1, **Clause 9**; Paragraph 2, **Subclause 2) of Clause 19**; Paragraph 3, **Clause 36**; Paragraph 4, **Clause 41**; Paragraph 5, **Clauses 57, 58**:
    > *"9. The bank ensures the availability of special equipment for determining the fitness and authenticity of banknotes when conducting cash operations."*
    > *"19. The bank's cash employee performs the following actions: ... 2) accepts banknotes with sheet-by-sheet counting, and coins by counting piece-by-piece, checking the authenticity and fitness of the cash in accordance with Clauses 57 and 58 of the Rules..."*
  * **NBRK Resolution No. 230 (29.11.2017)** *"Rules for Determining the Fitness of Banknotes and Coins of the National Currency of the RK"*.
  * **NBRK Resolution No. 49 (04.04.2019)**, Chapter 2, **Subclause 3) of Clause 11** (mandatory UV and magnetic mark verification).
* **Explanation Template (`description`):**
  > "The video shows the cashier accepting cash from the client without performing a sheet-by-sheet count (or piece-by-piece coin count) and without verifying banknote authenticity and fitness on specialized detection equipment. This violates Clause 9, Subclause 2) of Clause 19, and Clause 57 of the Rules for Conducting Cash Operations (NBRK Board Resolution No. 231 dated 29.11.2019)."

---

#### `V07` — Accepting or Counting Client Cash in the Client's Absence (Out of Client's View)
* **Violation Type (`type`):** Пересчет принятых от клиента наличных денег в отсутствие клиента *(Counting client cash in the absence of the client)*
* **Visual Cues on Video:**
  * The client leaves the cash window before the cash is counted, or the cashier moves the client's cash out of the client's line of sight to count it without the client present.
* **Regulatory Citation:**
  * **NBRK Resolution No. 231 (29.11.2019)**, Chapter 2, Paragraph 2, **Clause 20 (Paragraph 1)**:
    > *"20. Cash is accepted and counted by the cash employee in the presence of the client."*
* **Explanation Template (`description`):**
  > "The video shows the cashier accepting and counting the client's cash in the client's absence (or outside the client's field of view). This violates Clause 20 (paragraph 1) of the Rules for Conducting Cash Operations (NBRK Board Resolution No. 231 dated 29.11.2019), which requires cash to be accepted and counted strictly in the presence of the client."

---

#### `V08` — Failure to Sign/Stamp Cash Documents or Failure to Issue a Confirming Receipt to the Client
* **Violation Type (`type`):** Неоформление кассового документа (отсутствие подписи, штампа или невыдача подтверждающего документа клиенту) *(Failure to sign/stamp cash documents or failure to issue a confirming document to the client)*
* **Visual Cues on Video:**
  * The cashier accepts or disburses cash but fails to check the client's signature, fails to sign and stamp all copies of the incoming/outgoing cash order, and/or fails to hand over a confirming receipt/order/check to the client through the tray before the client leaves.
* **Regulatory Citation:**
  * **NBRK Resolution No. 231 (29.11.2019)**, Chapter 2, Paragraph 2, **Clause 17**, **Subclauses 3), 5), 6) of Clause 19** (acceptance) and Paragraph 3, **Clause 29**, **Subclauses 2), 5), 6) of Clause 31** (disbursement):
    > *"19. The bank's cash employee performs the following actions: ... 3) verifies the presence of the client's signature... 5) signs all copies of the incoming cash document and affixes the bank cash employee's stamp on all copies... 6) issues to the client a document confirming the execution of the cash acceptance operation."*
  * **NBRK Resolution No. 49 (04.04.2019)**, Chapter 6, **Clause 54** (issuing a fiscal control receipt for exchange transactions).
* **Explanation Template (`description`):**
  > "The video shows the cashier completing a cash transaction without properly executing the cash document (omitting the cashier's signature and stamp) and/or failing to issue a confirming receipt/voucher to the client. This violates Subclauses 3), 5), and 6) of Clause 19 (for acceptance) / Subclauses 2), 5), and 6) of Clause 31 (for disbursement) of the Rules for Conducting Cash Operations (NBRK Board Resolution No. 231 dated 29.11.2019)."

---

### Category 3: Violations in Disbursing Cash to Clients

#### `V09` — Disbursing Cash to a Client Without Prior Counting in the Client's Presence
* **Violation Type (`type`):** Выдача наличных денег клиенту без пересчета в его присутствии *(Disbursing cash to a client without counting it in their presence)*
* **Visual Cues on Video:**
  * When disbursing cash, the cashier takes loose banknotes or banknotes from an incomplete/opened strap (or coins from an opened bag) and places them directly into the client tray without counting the cash sheet-by-sheet in front of the client.
* **Regulatory Citation:**
  * **NBRK Resolution No. 231 (29.11.2019)**, Chapter 2, Paragraph 3, **Subclause 4) of Clause 31** and **Clause 32**:
    > *"31. The bank's cash employee performs the following actions: ... 4) counts the cash prepared for disbursement in the presence of the client, and hands the cash to the client..."*
    > *"32. The cash employee disburses banknotes to the client in full or partial bundles and/or individual straps according to the amounts indicated on the packaging without sheet-by-sheet counting provided the packaging is undamaged. Banknotes from incomplete or damaged straps are disbursed by the cash employee with sheet-by-sheet counting."*
* **Explanation Template (`description`):**
  > "The video shows the cashier handing loose banknotes (or banknotes from an opened/incomplete strap) to the client without counting the prepared amount in the client's presence. This violates Subclause 4) of Clause 31 and Clause 32 of the Rules for Conducting Cash Operations (NBRK Board Resolution No. 231 dated 29.11.2019)."

---

### Category 4: Violations in Internal Counting, Sorting, Packaging, and Employee-to-Employee Cash Handover

#### `V10` — Counting, Sorting, and Packaging Cash (Outside Client Service) Without a Supervising Employee Present
* **Violation Type (`type`):** Пересчет, сортировка и упаковка наличных денег без присутствия контролирующего работника *(Counting, sorting, and packaging cash without a supervising employee present)*
* **Visual Cues on Video:**
  * Outside of client service (no client at the window), a single cashier performs bulk cash counting, sorting by denomination/fitness, strapping, and packaging bundles/bags alone, without a second supervising bank employee (`контролирующий работник`) observing the process.
* **Regulatory Citation:**
  * **NBRK Resolution No. 231 (29.11.2019)**, Chapter 1, **Subclause 2) of Clause 2** and Chapter 2, Paragraph 4, **Clause 42**:
    > *"42. Cash operations for counting, sorting, and packaging cash are carried out by the cash employee in the presence of a supervising employee, except in cases of performing cash operations for accepting and disbursing cash to a client."*
* **Explanation Template (`description`):**
  > "The video shows the cashier performing internal cash counting, sorting, and packaging operations alone without the presence of a supervising bank employee. This violates Clause 42 of the Rules for Conducting Cash Operations (NBRK Board Resolution No. 231 dated 29.11.2019), which requires counting, sorting, and packaging of cash (outside of direct client acceptance/disbursement) to be conducted in the presence of a supervising employee."

---

#### `V11` — Handover of Cash Between Bank Employees Without Documentary Recording and Signatures
* **Violation Type (`type`):** Передача наличных денег между работниками банка без оформления документа учета и подписи *(Transferring cash between bank employees without signing the cash transfer log/document)*
* **Visual Cues on Video:**
  * Another bank employee (e.g., head cashier or second cashier) enters or approaches the cash desk, and cash bundles/straps are handed over from hand to hand without recording the transfer in the cash acceptance/issuance log and without both employees signing the transfer document.
* **Regulatory Citation:**
  * **NBRK Resolution No. 231 (29.11.2019)**, Chapter 2, Paragraph 1, **Clause 15**; Paragraph 2, **Clause 23**; Paragraph 3, **Clauses 28, 37**:
    > *"15. The transfer of cash between cash employees of the bank is recorded in the document for accounting accepted and issued cash and valuables, which contains the following information: 1) date of cash transfer; 2) full name, position, and signature of the cash employee transferring the cash; 3) amount of cash in figures and words, quantity, and type of cash; 4) full name, position, and signature of the cash employee receiving the cash."*
* **Explanation Template (`description`):**
  > "The video shows cash being transferred directly from hand to hand between bank employees without recording the operation and obtaining signatures from both the transferring and receiving employees in the cash accounting document. This violates Clauses 15 and 23 of the Rules for Conducting Cash Operations (NBRK Board Resolution No. 231 dated 29.11.2019)."

---

#### `V12` — Improper Sorting, Strapping, or Packaging of Banknotes and Coins
* **Violation Type (`type`):** Нарушение требований к сортировке, обандероливанию и упаковке наличных денег *(Improper sorting, strapping, or packaging of banknotes and coins)*
* **Visual Cues on Video:**
  * Banknotes are bound with rubber bands or clips instead of paper straps of 100 sheets.
  * Straps or bundle top covers (`накладки`) are not stamped or signed by the cashier.
  * Full bundles of 1,000 banknotes are not fitted with top/bottom covers, not tied with twine and lead-sealed (`пломбиратор`), or not vacuum-sealed in polyethylene packs.
  * Unfit/worn (`ветхие`) banknotes are packaged together with fit banknotes.
* **Regulatory Citation:**
  * **NBRK Resolution No. 231 (29.11.2019)**, Chapter 2, Paragraph 4, **Clauses 43, 45, 46, 47, 48, 51, 52**.
* **Explanation Template (`description`):**
  > "The video shows a violation of mandatory cash sorting and packaging procedures (failure to properly strap 100-sheet bundles, omit stamps/signatures on straps or bundle covers, failure to seal or vacuum-pack bundles, or mixing fit and unfit banknotes). This violates Clauses 45–48 of the Rules for Conducting Cash Operations (NBRK Board Resolution No. 231 dated 29.11.2019)."

---

### Category 5: Violations in Cash-in-Transit (CIT / Inkassatsiya) Operations

#### `V13` — Accepting or Handing Over a CIT Bag Without Verifying Bag Integrity, Seals, and Accompanying Documents
* **Violation Type (`type`):** Нарушение процедуры приема или сдачи инкассаторской сумки (отсутствие проверки пломбы, целостности сумки и документов) *(Accepting or handing over a CIT bag without checking bag integrity, seal, and accompanying documents)*
* **Visual Cues on Video:**
  * A CIT officer (`инкассатор`) delivers or picks up a collection bag/container/cassette, and the cashier (or CIT collector) takes the bag without inspecting the fabric/body integrity, without checking the lead seal (`пломба`) against the sample seal card, and without verifying/signing the accompanying waybill (`препроводительный документ`) and attendance card (`явочная карточка`).
* **Regulatory Citation:**
  * **NBRK Resolution No. 231 (29.11.2019)**, Chapter 2, Paragraph 2, **Clause 24** (acceptance from CIT); Paragraph 3, **Clauses 38, 39**; Chapter 3, Paragraph 1, **Clauses 94, 101** and Paragraph 2, **Clauses 106, 107, 108**:
    > *"24. Cash operations for accepting client cash from the bank's collection unit or a collection organization are carried out by the bank's cash employee based on an accompanying document meeting the requirements of Clause 106 of the Rules. The cash employee: 1) verifies the required information on the accompanying document; 2) registers the amount of accepted cash...; 3) checks the integrity of the bag; 4) checks the correctness of the bag's sealing."*
* **Explanation Template (`description`):**
  > "The video shows the cashier (or CIT collector) accepting/handing over a cash-in-transit bag without inspecting the physical integrity of the bag, verifying the seal against the certified sample, and checking/signing the accompanying transport documents. This violates Clauses 24, 38–39, and 107 of the Rules for Conducting Cash Operations and Inkassatsiya (NBRK Board Resolution No. 231 dated 29.11.2019)."

---

#### `V14` — Accepting a Damaged/Unsealed CIT Bag Without Opening and Counting Contents in the CIT Officer's Presence
* **Violation Type (`type`):** Нарушение порядка действий при обнаружении повреждения инкассаторской сумки или пломбы *(Failure to open and count a damaged/unsealed CIT bag in the presence of the CIT officer)*
* **Visual Cues on Video:**
  * A CIT bag has visible damage, a broken seal, or is unzipped/open, yet the cashier accepts it without immediately opening and counting the cash in the presence of the CIT officer (or opens it only after the CIT officer has left).
* **Regulatory Citation:**
  * **NBRK Resolution No. 231 (29.11.2019)**, Chapter 2, Paragraph 2, **Clause 25**:
    > *"25. If a breach of the bag's integrity is discovered, the cash employee, in the presence of the employee of the bank's collection unit or collection organization, opens and counts the cash and valuables contained therein. If a shortage or surplus is found, a recount act is drawn up..."*
* **Explanation Template (`description`):**
  > "The video shows a CIT bag with compromised integrity (open or unsealed) being accepted without immediately opening and counting the cash and valuables in the presence of the CIT officer and drawing up a recount act. This violates Clause 25 of the Rules for Conducting Cash Operations and Inkassatsiya (NBRK Board Resolution No. 231 dated 29.11.2019)."

---

#### `V15` — CIT Officer Lacking Mandatory Protective Equipment (Body Armor) or Transporting Cash Outside a Sealed CIT Bag
* **Violation Type (`type`):** Нарушение требований к экипировке инкассаторов или транспортировке наличных денег без спецсумки *(CIT officer lacking protective gear/body armor or carrying cash outside a sealed CIT bag)*
* **Visual Cues on Video:**
  * A CIT officer performs collection/delivery without personal protective equipment (body armor / tactical vest) or carries loose bundles of cash in their hands or an ordinary plastic bag instead of a numbered, sealed CIT bag or specialized ink-dye container.
* **Regulatory Citation:**
  * **NBRK Resolution No. 231 (29.11.2019)**, Chapter 3, Paragraph 1, **Clauses 93, 103, 105** and Paragraph 2-1, **Subclause 1) of Clause 109-6** (and Clause 8 of Instruction No. 159):
    > *"93. The bank performing collection... uses bags that ensure the safety of banknotes, coins, and valuables and do not allow opening without visible signs of tampering..."*
    > *"103. The team of employees of the bank's collection unit... is provided with gear, personal protective equipment, including bulletproof vests, helmets, and service weapons."*
    > *"105. Banknotes, coins, and valuables subject to collection are packed into bags with the preparation of an accompanying document."*
* **Explanation Template (`description`):**
  > "The video shows the cash-in-transit (CIT) officer performing collection operations without mandatory personal protective equipment (bulletproof vest) and/or carrying cash outside of a sealed specialized collection bag/container. This violates Clauses 93, 103, 105, and Subclause 1) of Clause 109-6 of the Rules for Conducting Cash Operations and Inkassatsiya (NBRK Board Resolution No. 231 dated 29.11.2019)."

---

### Category 6: Physical Security, Access Control, and Cash Booth Premises Violations

#### `V16` — Unauthorized Person Entering or Present Inside the Cash Booth (Cash Unit)
* **Violation Type (`type`):** Присутствие посторонних лиц в помещении кассы *(Unauthorized person present inside the cash booth)*
* **Visual Cues on Video:**
  * A client, visitor, cleaner, or other unauthorized individual enters the enclosed cash booth (behind the partition, into the cashier's working zone).
* **Regulatory Citation:**
  * **NBRK Resolution No. 49 (04.04.2019)**, Chapter 5, **Clause 41**:
    > *"41. The presence of unauthorized persons inside the premises of operational cash desks of exchange offices is not permitted."*
  * **NBRK Resolution No. 14 (24.02.2020)**, Chapter 3, **Clause 8** and **Subclause 1) of Clause 16**:
    > *"8. The cash unit must be isolated from other production, utility, basement, and service premises by a single lockable service entrance."*
    > *"16. The arrangement of the cash desk premises meets the following parameters: 1) each cash desk has a separate entrance with a door that locks..."*
  * **NBRK Resolution No. 231 (29.11.2019)**, Chapter 2, Paragraph 1, **Clause 14** and Paragraph 8, **Clause 75**.
* **Explanation Template (`description`):**
  > "The video shows an unauthorized person (client or third party) entering and remaining inside the isolated cash booth (cashier's working area). This violates Clause 41 of the Rules for Conducting Exchange Operations (NBRK Board Resolution No. 49 dated 04.04.2019), Clause 8 and Subclause 1) of Clause 16 of the Rules for Organizing Security and Arrangement of Bank Premises (NBRK Board Resolution No. 14 dated 24.02.2020), and Clause 14 of the Rules for Conducting Cash Operations (NBRK Board Resolution No. 231 dated 29.11.2019)."

---

#### `V17` — Cash Booth Entrance Door Left Open (Unlocked) During Cash Operations
* **Violation Type (`type`):** Открытая входная дверь в помещение кассы во время работы с наличными деньгами *(Cash booth entrance door left open during cash operations)*
* **Visual Cues on Video:**
  * The entrance door to the cash booth is left ajar or wide open (unlocked) while the cashier is handling cash, counting banknotes, or serving a client, or when the cashier leaves the booth unattended.
* **Regulatory Citation:**
  * **NBRK Resolution No. 14 (24.02.2020)**, Chapter 3, **Clause 8** and **Subclauses 1), 2) of Clause 16**:
    > *"16. The arrangement of the cash desk premises meets the following parameters: 1) each cash desk has a separate entrance with a door that locks; 2) the walls, ceiling, and floor of the cash desk are equipped with protective means ensuring the safety of the cash employee..."*
  * **NBRK Resolution No. 231 (29.11.2019)**, Chapter 2, Paragraph 1, **Subclause 7) of Clause 3** and **Clause 5**.
* **Explanation Template (`description`):**
  > "The video shows that the entrance door to the cash booth is left open (unlocked) while cash operations are being conducted. This violates the cash booth isolation and security requirements established by Clause 8 and Subclauses 1)–2) of Clause 16 of the Rules for Organizing Security and Arrangement of Bank Premises (NBRK Board Resolution No. 14 dated 24.02.2020) and Clause 5 of the Rules for Conducting Cash Operations (NBRK Board Resolution No. 231 dated 29.11.2019)."

---

#### `V18` — Passing Cash Bypassing the Transfer Tray/Bunker, Camera Obstruction, or Misappropriation (Theft) of Cash
* **Violation Type (`type`):** Передача наличных денег в обход специального транспортирующего устройства либо противоправные действия с наличностью (хищение) *(Passing cash bypassing the transfer tray, obstructing CCTV, or misappropriating cash)*
* **Visual Cues on Video:**
  * The cashier hands cash to (or receives cash from) a client through the open cash booth door or around the partition instead of using the secure transfer tray (`передаточный лоток`) or CIT bunker (`инкассаторский бункер`).
  * The cashier secretly pockets banknotes into their clothing, hides cash under the desk/in personal items, or discards cash into a trash bin.
  * The cashier intentionally blocks or obstructs the overhead CCTV camera's view of the workstation or cash.
* **Regulatory Citation:**
  * **NBRK Resolution No. 14 (24.02.2020)**, Chapter 3, **Subclauses 3), 4), 6) of Clause 16**:
    > *"16. ... 3) the front part of the cash desk has a window opening equipped with protective means ensuring the safety of the cash employee and preventing theft of cash and valuables; 4) the front part of the cash desk is equipped with special transport devices for transferring cash and valuables... 6) the cash desk is equipped with video cameras of a video surveillance system..."*
  * **NBRK Resolution No. 231 (29.11.2019)**, Chapter 2, Paragraph 1, **Clauses 12, 14, 16, 20**.
  * **NBRK Resolution No. 49 (04.04.2019)**, Chapter 2, **Subclause 2) of Clause 11** (unobstructed CCTV view of the cashier's working zone and client).
* **Explanation Template (`description`):**
  > "The video shows the cashier transferring cash bypassing the dedicated secure transfer tray/bunker (or concealing banknotes in personal clothing/items, or obstructing the surveillance camera), creating a direct risk of cash theft. This violates Subclauses 3), 4), and 6) of Clause 16 of the Rules for Organizing Security and Arrangement of Bank Premises (NBRK Board Resolution No. 14 dated 24.02.2020) and Clauses 12, 14, and 20 of the Rules for Conducting Cash Operations (NBRK Board Resolution No. 231 dated 29.11.2019)."

---

#### `V19` — Returning Suspected Counterfeit or Doubtful Banknotes to the Client Without Drawing Up an Act
* **Violation Type (`type`):** Нарушение порядка работы с сомнительными или имеющими признаки подделки банкнотами *(Improper handling of doubtful or suspected counterfeit banknotes)*
* **Visual Cues on Video:**
  * Upon inspecting a banknote under the detector, the cashier identifies a doubtful, dye-stained, or suspected counterfeit note, but instead of retaining it, calling the head cashier, and drawing up an examination/seizure act, the cashier hands the suspect banknote back to the client.
* **Regulatory Citation:**
  * **NBRK Resolution No. 231 (29.11.2019)**, Chapter 2, Paragraph 5, **Clauses 59, 60, 61**:
    > *"59. Upon discovering doubtful banknotes and coins, the cash employee, in the presence of the client and the head of the cash desk..., draws up an act of acceptance for examination of doubtful banknotes and coins in two copies..."*
    > *"60. Law enforcement agencies are notified upon discovery of banknotes and coins showing signs of counterfeiting..."*
    > *"61. Banknotes and coins showing signs of counterfeiting are not returned to the client, and the amount of accepted banknotes and coins is not credited to the client's bank account."*
* **Explanation Template (`description`):**
  > "The video shows that after detecting a doubtful or suspected counterfeit banknote on the verification device, the cashier returns the banknote to the client instead of retaining it and drawing up an official examination act in the presence of the head cashier. This violates Clauses 59–61 of the Rules for Conducting Cash Operations (NBRK Board Resolution No. 231 dated 29.11.2019)."

---

## 4. Timecode (`timecode_sec`) & Evidence Frame (`evidence_frames`) Selection Strategy

1. **Active Violations (Commission):**
   * Examples: Using a smartphone (`V01`), walking out of the booth (`V02`), unauthorized person entering (`V16`), open booth door (`V17`), pocketing cash (`V18`).
   * `timecode_sec`: The exact timestamp (in seconds from the start of the video) when the violation action begins or reaches its clearest visual peak.
   * `evidence_frames`: 1 to 3 JPEG frames (base64-encoded) showing the start, peak (clearest unobstructed frame), and continuation of the violation.
2. **Omission Violations (Failure to Perform a Mandatory Step):**
   * Examples: Accepting cash without ID verification (`V05`), storing accepted cash without counting/checking (`V06`), failing to issue a receipt (`V08`), accepting a CIT bag without checking the seal (`V13`).
   * `timecode_sec`: The exact timestamp when the mandatory step was bypassed (e.g., the moment the cashier puts uncounted cash into the drawer, or the moment the client walks away without receiving a signed receipt).
   * `evidence_frames`: 2 to 3 sequential frames demonstrating the transition from receiving the cash/bag directly to completing the step without the required verification.
