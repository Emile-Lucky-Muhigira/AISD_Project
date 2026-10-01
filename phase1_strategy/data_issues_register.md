# ElimuSentry — Data Issues Register (Phase 1 evidence base)

All numbers are from the training table (Wave 1 → Wave 2, N = 1,307) and the test table (Wave 2 → Wave 3, N = 1,395) built in `notebooks/01_build_dataset.ipynb`, unless stated otherwise. Each issue lists the evidence, the assignment item it feeds, and the decision you need to make. Where you must decide, the options are listed with a recommendation; the choice and wording are yours.

---

## 0. Corrections already applied while building the table

| # | Issue | Evidence | How it was handled |
|---|---|---|---|
| 0.1 | Legacy text encoding | 5 files fail as UTF-8 (byte 0x96 = Windows en dash) | Read with `encoding="latin1"` |
| 0.2 | Numbers stored as text | Most Wave 1 numeric columns have `object` dtype | `pd.to_numeric(errors="coerce")` |
| 0.3 | Duplicate person rows | 2 duplicated (household, member) rows in W1 education | Kept the first |
| 0.4 | Codes differ between waves | Sex 1/2 (W1) vs 1/5 (W2); feeding 1/2 vs 1/5; textbooks 1/2/3 vs 1/3/0; parent status 1/2/3 vs 1/3/5; parent education starts at 1 (W1) vs 0 (W2) | Harmonised to Wave 1 codes |
| 0.5 | Parent education "missing" by skip pattern | Question skipped when parent lives in the household; looked ~54% missing | Filled from the parent's own education record → 7.7% (father) / 3.8% (mother) missing |
| 0.6 | Code 99 = "on vacation" (W1 hours) | 183 baseline children; interviews in **January and December** (school holidays) — confirms the meaning | Set to missing + flag `on_vacation` |
| 0.7 | Hours and minutes in two boxes | Minutes often filled, hours blank (and vice versa) | Blank box = 0 if the other box is filled |
| 0.8 | Test answers stored as letters (W1) / codes 1–5 (W2) | No right/wrong in W1–W2 | Scored with key recovered from W3 (matches modal answer on every item) |
| 0.9 | District codes repeat across regions | 112 raw codes vs 136 real districts | `district_id = region × 1000 + district` |
| 0.10 | Wave 2 has no location file | — | Region and EA decoded from household ID; district, urban/rural, weight from W1 lookup |
| 0.11 | Consumption aggregate only for W1 | Built by separate Stata scripts, no W2 equivalent | Not used; wealth from assets, durables value, electricity, crowding (**deviation from proposal**) |

---

## 1. Issues that need a fix in the build notebook (recommended before the audit)

| # | Issue | Evidence | Recommended fix |
|---|---|---|---|
| 1.1 | **`on_vacation` is constant (0) in the test set** | W2 had no 99 code. Instead 183 test children have `attend_hours = 0`, and 151 of them also `missed_hours = 0`; these interviews cluster in **April–May (Easter break) and August (long vacation)** | In W2, set `on_vacation = 1` and attendance to missing when `attend_hours == 0 and missed_hours == 0`. Otherwise the feature means different things in train and test |
| 1.2 | Impossible values (entry errors, not outliers) | `travel_minutes` > 300 (5 h/day): 62 train / 12 test, max 2,735 and 2,400 (whole numbers typed in the hours box); `attend_hours` > 45 h/week: 10 train (max 75); `teacher_absent_days` > 22 school days/month: 3 test (= 30); `rooms` = 0: 2 test | Plausibility rule → set to missing (or convert hours→minutes where the pattern is clear). Decide thresholds and justify them physically |
| 1.3 | Invalid category codes | `relationship_to_head = 2` (spouse) for 3 test children aged 8–13; grade codes 21, 30 for 2 train children (already → missing) | Set spouse code to missing or "other"; mention in report |

---

## 2. A — Dataset & Problem Audit

**A1 Shape.** Train 1,307 × 39 (33 features, 5 ID/grouping columns, 1 target); test 1,395 × 39. Report also the full flow (Figure `fig0_sample_flow.png`): 2,532 enrolled W1 children → 1,307 labelled; 2,049 enrolled W2 children → 1,395 labelled.

**A2 Variable types (suggested classification — check each):**
- Continuous: `attend_hours`, `missed_hours`, `travel_minutes`, `school_spend`, `durables_value`, `persons_per_room`
- Discrete (counts/scores): `age`, `math_score` (0–8), `english_score` (0–7), `teacher_absent_days`, `hh_size`, `asset_count` (0–10), `rooms`
- Ordinal: `grade_level` (0–9), `over_age`, `father_edu`, `mother_edu` (0–4), `has_textbooks` (1 all, 2 some, 3 none — reversed order)
- Nominal: `regioncode` (10), `school_type` (4), `relationship_to_head` (5–6 levels), `math_status`, `english_status` (3 levels)
- Binary: `sex`, `urbrur`, `on_vacation`, `feeding_program`, `father_alive`, `mother_alive`, `father_in_hh`, `mother_in_hh`, `orphan`, `electricity`
- Identifiers / not features: `FPrimary`, `hhmid`, `district_id`, `eacode`, `hhweight3`
- No datetime/text features (interview dates exist in raw data but are not features)

**A3 Missingness (train, after recoding sentinels):** overall 8.78% of feature cells; **only 54 of 1,307 rows (4%) are complete** → listwise deletion is not an option. Highest: `missed_hours` 77.7%, `teacher_absent_days` 56.8%, `english_score` 54.8%, `math_score` 33.4%, `attend_hours` 21.6%. Test set: `missed_hours` 0.9%, `teacher_absent_days` 31.6%, `english_score` 39.9%, `math_score` 24.4%, `attend_hours` 0.9%.

**A4 Target.** Binary. Train 188/1,307 = 14.4% (≈ 1 : 6.0); test 145/1,395 = 10.4% (≈ 1 : 8.6). Mitigation per proposal: class weighting (keeps probabilities calibratable); evaluate with PR-AUC.

---

## 3. B1 — Missingness mechanisms (evidence)

| Feature | Evidence | Suggested mechanism |
|---|---|---|
| `missed_hours` (W1) | Missingness unrelated to dropout (14.2% vs 15.1%, χ² p = 0.76) and to age/urban/wealth; in W2, where the tablet forced an answer, median = 0 | Administrative blank, most likely "0 hours missed" → **MCAR-like / recording convention**. Decision: fill 0 (+ indicator) vs median |
| `teacher_absent_days` | Asked only when the child has one main teacher; missing rises from ~50% (primary) to 89–100% (JSS); missingness **predicts dropout** (17.2% vs 10.6%, p = 0.001) | **Structural (not applicable)**; missingness informative → keep indicator |
| `math_score` / `english_score` | Age 8: not tested by design (ages 9–26 only) → MAR on age. Ages 9–13 untested ~164 in train but ~6 in test (administration differs by wave). English missing children are more rural, poorer, more northern → MAR on region/wealth. `blank_sheet` math: dropout **36%** vs 14% (n = 22); English 19% vs 13% | Mixed: MAR (age, region) + likely **MNAR** for blank sheets (cannot read). Keep `*_status` categories |
| `father_edu` / `mother_edu` | Remaining missing = "don't know" about absent parents; missing group is more urban and wealthier | MAR on parent presence / household type |
| `attend_hours` (W1) | 79 = vacation; 203 blank; no relation to dropout (p = 0.18) | Vacation: structural; rest MCAR-like |
| `electricity` | 17 missing; dropout 35% vs 14% (p = 0.03) | Small n; flag only |
| **Attrition (label missing)** | Lost children are older (p < 0.001), far less often living with their parents (father in household 52% vs 66%; mother 65% vs 83%; p < 10⁻¹¹), and grandchildren/foster/other relatives are lost 54–59% vs 33% of own children | **MNAR-risk for the label**: lost children share known dropout risk factors → labelled sample likely under-represents at-risk children; dropout rate probably understated. Report as a limitation; consider weighting sensitivity |
| `id_mismatch` (164 excluded) | Poorer, more rural/northern, lower test scores than labelled children. 145 have matching sex but age gap outside 2–7 (40 have gap +1, 19 have +8); widening the window to 1–8 would add 59 children (13 leavers) | Exclusion removes a higher-risk group. Option: sensitivity check with wider age window or date of birth |

---

## 4. B2–B3 — Distributions and outliers

Skewness (train): `school_spend` 10.8, `durables_value` 10.1, `travel_minutes` 6.7, `rooms` 4.8, `teacher_absent_days` 3.0, `missed_hours` 2.1, `hh_size` 1.3, `persons_per_room` 1.1, `attend_hours` −0.8 → almost everything is right-skewed → IQR rule (not z-score) and median (not mean) imputation are the defensible defaults; `attend_hours` is the only left-skewed one.

Two-stage rule recommended: (1) **plausibility** (physically impossible → missing, Section 1.2); (2) **IQR capping** on the rest, bounds computed on X_train only. Note: IQR on `travel_minutes` gives an upper fence of 85 min and would cap 153 plausible long commutes — discuss whether a log transform before capping is better.

`rooms` > 15: 13 train households (max 52) — likely compound houses; `persons_per_room` is more meaningful than raw `rooms`.

---

## 5. B4 — Correlation and multicollinearity

- **Exact linear dependency:** `over_age = age − grade_level − 5` for all 1,226 rows where defined → VIF = ∞. You must drop one of the three (recommendation: keep `over_age` + `age`, drop `grade_level`, because over-age-for-grade is the known dropout signal).
- `orphan` is derived from `father_alive`/`mother_alive` (r = −0.85 with `father_alive`; VIF 20) → keep either the flag or the two components.
- Spearman ≥ 0.6: `rooms`–`persons_per_room` (−0.86), `relationship_to_head`–`mother_in_hh` (−0.77), `asset_count`–`durables_value` (0.71), `age`–`grade_level` (0.64), `math`–`english` (0.61).
- After removing `over_age`/`grade_level` redundancy and `orphan`, all VIF < 2.
- Most features are skewed/ordinal → Spearman ρ is more appropriate than Pearson r; use point-biserial / Mann-Whitney for numeric vs binary target, χ² / Cramér's V for categorical, mutual information for all.

---

## 6. C1 — Encoding and scaling notes

- `has_textbooks` order is reversed (1 = all … 3 = none) — re-map so higher = more books before ordinal use.
- `relationship_to_head`: 5–6 levels, dominated by "child" (84%) → collapse to {own child, grandchild, other relative/foster/non-relative}.
- `regioncode` 10 levels → one-hot (cardinality 10). But see shift in Section 7.
- XGBoost needs no scaling; logistic regression baseline and PCA do (StandardScaler after log transform of skewed money variables).

---

## 7. C2 — Splits, grouping and leakage

- **Siblings:** 552 of 1,307 training children share a household with another sample child (227 two-child, 30 three-child, 2 four-child households) → group by household at minimum.
- **Districts:** 136 districts, median 0 dropouts per district, 78 districts with no dropout → `StratifiedGroupKFold` (stratify on dropout, group on district) is needed to keep dropouts in every fold.
- **Temporal test:** W2→W3 is out-of-time. **220 children appear in both train and test** (enrolled at both waves) — not future leakage (their training label is known by test time), but reduces test independence. Decide: keep (realistic re-scoring) or report results with and without them.
- **Leakage check:** all features are measured at wave *t*; nothing from *t+1* enters the features; IDs and survey weight are excluded from features. ✓
- **Distribution shift between periods (important for the report):**
  - Dropout falls from 14.4% to 10.4%; in Northern region 28% → 12% and Upper West 46% → 12% (regional effects may not transfer).
  - `durables_value` median ×2.6 (largely inflation, nominal cedis) — use within-wave relative measures, log + robust scaling, or rely on `asset_count`.
  - `asset_count` 2.9 → 3.7, electricity 50% → 65% (real development).
  - Test administration: English blank sheets 105 (train) vs 251 (test); untested ages 9–13 ~164 vs ~6; W2 tests include a 5th option "none of the above" (harder to guess) → scores not perfectly comparable.
  - `missed_hours` missingness 77.7% vs 0.9% — a missingness indicator learned in training would mean something different in test → strong reason to resolve it by imputation rule rather than an indicator.

---

## 8. Open decisions (yours) — checklist

1. Plausibility thresholds for travel time, class hours, teacher absence, rooms (Section 1.2).
2. `missed_hours` blanks in W1: 0 vs median vs indicator.
3. Test-score missingness: status categories vs fill + indicator; how to treat blank sheets.
4. Which of `age` / `grade_level` / `over_age` to drop; `orphan` vs alive flags; `rooms` vs `persons_per_room`.
5. Wealth across waves: how to handle inflation in `durables_value`.
6. Keep or drop the 220 overlapping children in the test evaluation.
7. Region as a feature given the regional shift.
8. Whether to run an attrition / identity-window sensitivity check.
