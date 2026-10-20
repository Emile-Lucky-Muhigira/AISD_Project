# ElimuSentry: An Interpretable Early-Warning System for School Dropout

**Author:** Emile Lucky Muhigira (emuhigir@andrew.cmu.edu), Carnegie Mellon University Africa
**Course:** 04-652 Artificial Intelligence System Design

## 1. Problem Statement

School dropout in Sub-Saharan Africa is a gradual process: absences accumulate and learning falls behind before a child finally leaves school. Most regional dropout models are trained on one-visit household surveys (e.g., DHS, MICS) that record only whether a child is *already* out of school, so they can describe risk but cannot warn in time to act.

**Objective:** Given a child aged 8–13 who is enrolled in school at survey wave *t*, use information measured at wave *t* (attendance, math and English test performance, household wealth, parents' education, orphanhood and school context) to predict whether the child will be **out of school at wave *t+1* without having completed basic education** (Junior High School in Ghana).

**Domain context:** The project supports **SDG 4 (Quality Education)**, Targets 4.1 and 4.5, by helping schools identify at-risk children early. The central hypothesis is that dynamic signals (attendance and performance) improve prediction beyond static background factors (wealth, parents' education).

## 2. Dataset

### 2.1 Source and Survey Structure

This project uses the **Ghana Socioeconomic Panel Survey (GSPS)**, produced through a collaboration between the University of Ghana’s Institute of Statistical, Social and Economic Research (ISSER), Yale University and Northwestern University.

The survey follows households and individuals across repeated interviews. This project uses **2009/10, 2013/14 and 2017/18** to connect children’s circumstances at one interview with their schooling outcomes several years later.

The original dataset contains **317 Stata (`.dta`) files** covering household membership, education, learning assessments, possessions, housing and other topics. Depending on the file, a row describes a person, a household or an asset record. The project joins selected files using household and member identifiers and harmonises differences in question names and answer codes across survey waves.

**Data source:** [GSPS on Harvard Dataverse](https://doi.org/10.7910/DVN/E5QP0F).

### 2.2 Project Dataset at a Glance

| Item | Description |
|---|---|
| **Task** | Binary classification: predict school dropout at the following survey interview |
| **One analysis row** | One child aged 8–13, enrolled at the starting interview, with a classified schooling outcome at follow-up |
| **Training and cross-validation sample (N)** | 1,307 children: Wave 1 circumstances → Wave 2 outcomes |
| **Later-period sample (N)** | 1,395 children: Wave 2 circumstances → Wave 3 outcomes |
| **Potential predictors (P)** | 33 features describing the child, schooling, learning, family and living conditions |
| **Other columns** | Five identification/grouping/weight columns and one outcome, giving 39 columns per table |
| **Target (Y)** | `dropout`: 1 = classified as leaving school without completing basic education; 0 = still enrolled |
| **Class distribution** | Training: 188 dropouts (14.4%); later period: 145 dropouts (10.4%) |
| **Project tables** | `train_w1_w2_raw.csv` and `test_w2_w3_raw.csv` in `data/processed/` |

The predictors come from the **starting interview**; the following interview supplies the outcome. The outcome describes schooling status at follow-up, rather than the exact date a child left school.

### 2.3 Understanding the Columns

The **33 potential predictors** fall into five groups:

- **Child and location:** age, sex, region, urban/rural residence and relationship to the household head.
- **Schooling:** school type, grade, age relative to grade, weekly attendance and missed hours, holiday/no-class-week indicator, travel time, teacher absence, annual school spending, textbook access and school feeding.
- **Learning:** mathematics and English scores, with separate assessment-status fields distinguishing scored responses, blank sheets and absent assessment records.
- **Family:** parents’ education, whether each parent is alive or lives in the household, loss of a parent and household size.
- **Living conditions:** ownership of ten selected asset types, total durable-goods value, main lighting source, number of rooms and people per room.

The remaining columns serve different purposes:

- **`FPrimary` and `hhmid`:** identify the household and the member within it.
- **`district_id` and `eacode`:** identify geographic groups used to understand the sample and organise evaluation.
- **`hhweight3`:** retains the household sampling weight; it is not an ordinary predictor.
- **`dropout`:** records the outcome the model aims to predict.

See the [variable inventory](phase1_strategy/A2_variable_inventory.csv) for every column’s name, definition, statistical type and observed training values or range.

### 2.4 Important Interpretation Notes

**Sample inclusion.** The labelled tables retain children classified as still enrolled or as dropouts. Children who completed basic education or whose follow-up outcome cannot be established reliably are excluded. A missing follow-up record is **not automatically counted as dropout**.

**Missing information.** A blank can mean an unanswered question, a questionnaire skip, an unavailable assessment, an unsuccessful record match or a value rejected by a documented correction rule. It does not automatically mean zero or “no.” Likewise, a missing assessment score differs from a score of zero.

**Shared households and people.** Siblings share household information, and some children appear in both time transitions. The primary temporal evaluation therefore uses **731 children with no shared people or households** with training. A secondary evaluation uses **1,175 new people**, allowing shared households. All 1,395 later-period rows remain available for descriptive analysis.

**Derived fields and preprocessing.** The Wave 2 holiday indicator is an inferred no-class-week proxy, not confirmed holiday status. Files named `*_raw.csv` are already linked and harmonised project tables, before fitted preprocessing; the original survey files remain unchanged.

## 3. Task Type

**Binary Classification** (dropout = 1 vs. still enrolled = 0), with an imbalanced minority positive class (imbalance ratio 5.95 : 1 in training).

## 4. Repository Structure

```
AISD_Project/
├── .gitignore                 # excludes raw/processed data, virtual env, caches
├── requirements.txt           # exact package versions (pip freeze)
├── README.md                  # this file
├── data/
│   ├── raw/                   # original GSPS files (not tracked; see Setup)
│   └── processed/             # cleaned, encoded, scaled feature dataset (not tracked)
├── notebooks/                 # 01_build_dataset (joins survey files), 02_phase1_audit (Phase 1 numbers and figures)
├── phase1_strategy/
│   ├── Phase1_Strategy_Report.pdf   # Phase 1 deliverable
│   ├── Phase1_Strategy_Report.tex   # LaTeX source of the report
│   └── *.csv                        # result tables behind every table in the report
├── phase2_implementation/
│   ├── notebooks/
│   │   └── pipeline_execution.ipynb # Phase 2: runs the pipeline, produces the 8 figures
│   ├── decision_log.csv             # Phase 1 plan vs Phase 2 implementation
│   ├── Phase2_Final_Report.pdf      # Phase 2 deliverable
│   └── Phase2_Final_Report.tex      # LaTeX source of the report
├── src/
│   ├── __init__.py
│   ├── transformers.py        # custom scikit-learn transformers
│   └── pipeline_builder.py    # ColumnTransformer and Pipeline definitions
└── figures/                   # diagnostic plots (PNG, 300 DPI): fig1_1 ... fig5_1
```

`notebooks/` is an addition to the required structure: the GSPS is distributed as separate files per wave, so a dedicated linkage step (`notebooks/01_build_dataset.ipynb`) is needed before any preprocessing.

## 5. Setup & Reproduction

**Requirements:** Python 3.11, Windows/macOS/Linux.

1. Clone the repository:
```bash
git clone https://github.com/Emile-Lucky-Muhigira/AISD_Project.git
cd AISD_Project
```
2. Create and activate a virtual environment, then install dependencies:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
3. Download the dataset from https://doi.org/10.7910/DVN/E5QP0F and unzip it into `data/raw/`. You should then have three folders, `data/raw/Ghana_Panel_Survey/Data/Wave 1/`, `Wave 2/` and `Wave 3/` (all three are used). As a check, this file should exist: `data/raw/Ghana_Panel_Survey/Data/Wave 1/s1d.dta`.
4. Run the notebooks in order:
   - `notebooks/01_build_dataset.ipynb` builds the training and test tables in `data/processed/`, including separate `*_pre_correction.csv` snapshots before fixed value corrections. Unknown parent status remains unknown in the derived orphan indicator; 0 requires both parents known alive.
   - `notebooks/02_phase1_audit.ipynb` reproduces the Phase 1 tables and figures, reports missingness before and after fixed value corrections (`A3_baseline_missingness.csv`, `A3_missingness.csv`, `A3_missingness_summary.csv`), and saves the fold selection diagnostic (`C2_feature_selection_stability.csv`).
   - `phase2_implementation/notebooks/pipeline_execution.ipynb` fits the preprocessing pipeline, saves the 8 diagnostic figures to `figures/`, the processed tables to `data/processed/` and the decision log to `phase2_implementation/decision_log.csv`.
5. Note: some files use a legacy Windows text encoding and must be read with `encoding="latin1"`.

## 6. Pipeline Summary

The preprocessing is one scikit-learn `Pipeline` built around a `ColumnTransformer` (`src/pipeline_builder.py`). It is fitted on the training children only (X_train, 2009/10) and then applied unchanged to the test children (X_test, 2013/14).

1. **Fixed rules (nothing learned):** recording-error corrections; 2013/14 money converted to 2009/10 prices (factor 1.62, World Bank consumer prices for Ghana).
2. **Row-wise steps:** blank missed hours are assumed zero when attendance was recorded (with a training-only median-fill sensitivity); 4 fixed design/family-context missing flags; mixed-distance grade KNN with k selected from 3, 5, 9, 15, 25 inside three district-grouped inner folds of each fit; 3 structural features removed (33 → 30). Numeric age/assets and categorical sex/urban/region use different distance components; region codes have no ordering.
3. **Column-wise steps:** gaps filled with mean / median / most common answer according to skew; extreme values capped at the log-IQR limit (no child deleted); money ranked 0–1 against training values; one-hot encoding of labels; `StandardScaler` for logistic regression and PCA only.
4. **Output:** 53 columns for XGBoost, 48 for logistic regression; 0 missing values; the 3 removed features can be rebuilt exactly from the kept ones.
5. **Validation:** 5-fold `StratifiedGroupKFold` inside training. Primary temporal evaluation uses 731 new-household children (76 dropouts, zero shared households/people); secondary uses 1,175 new people but permits 370 shared households. All 1,395 temporal rows are retained for descriptive analysis. Prior inspection of later-period covariates and outcomes is disclosed; this is retrospective evaluation, not a pristine confirmatory holdout. Masks use IDs only; no temporal score chooses the method.

## 7. Project Status

**Phase 1 (strategy): implemented, locally verified and revised against the supplied assignment rubric.** See [Phase 1 report](phase1_strategy/Phase1_Strategy_Report.pdf).

**Phase 2 (execution and diagnostics): the required deliverables were already implemented in this repository before the recent review; they have now been revised and rerun with mixed-distance KNN.** The notebook, diagnostic figures, decision log and final report are existing project work, not deferred future work.

### Phase 2: questionnaire versus repository (checked 20 October 2026)

This comparison uses pages 9–10 and the repository layout on page 2 of **Assignment 2: Data Preprocessing** (`Assignment 2 Data Preprocessing-1.pdf`). The supplied brief says “More details will follow”; this check does not assume that any later Phase 2 instructions have been supplied.

- **Executable notebook using `Pipeline` and `ColumnTransformer` — implemented and verified.** [pipeline_execution.ipynb](phase2_implementation/notebooks/pipeline_execution.ipynb) uses the modular code in `src/transformers.py` and `src/pipeline_builder.py`. All 18 code cells have saved execution counts, with no error outputs; the notebook was rerun end to end during review.
- **Strict training/test separation — implemented, with retrospective limits disclosed.** Training uses Wave 1→2 and temporal evaluation uses Wave 2→3. Learned transformations fit on training only and are refitted within district-grouped validation folds. The primary cohort contains 731 children with no shared people or households; the secondary contains 1,175 new people and allows shared households. Earlier inspection of later-period features/outcomes means this is not an untouched confirmatory holdout. See `phase1_strategy/C2_evaluation_protocol.json`.
- **All eight publication-quality diagnostics — implemented.** Each is saved in `figures/` at approximately 300 DPI, displayed in the execution notebook, and included in the final report:
  - **1.1:** missingness matrix across observations and features — `fig1_1_missingness_heatmap.png`.
  - **1.2:** class-balance bars with counts and percentages — `fig1_2_target_distribution.png`.
  - **2.1:** before/after boxplots for capped features, with learned limits — `fig2_1_outlier_boxplots.png`.
  - **3.1:** masked upper-triangle correlation heatmap, highlighting pairs at the declared absolute-correlation threshold of 0.85 — `fig3_1_correlation_matrix.png`.
  - **3.2:** redundant-feature scatterplots with fitted linear trendlines; the report explains rank correlation and the exact derived-feature identity — `fig3_2_collinear_scatterplots.png`.
  - **3.3:** horizontal feature-to-target mutual-information ranking — `fig3_3_target_importance.png`.
  - **4.1:** 2×2 Raw → Log → Min-Max → Standardized distribution grid — `fig4_1_transformation_grid.png`.
  - **5.1:** PCA bars plus cumulative-variance line and 90%/95% cutoffs — `fig5_1_pca_scree_plot.png`.
- **Strategy-versus-reality reflection report — implemented.** [Phase2_Final_Report.pdf](phase2_implementation/Phase2_Final_Report.pdf), with editable `.tex` source alongside it, contains the decision log, deviations and technical rationale. The machine-readable decision log is `phase2_implementation/decision_log.csv`.
- **Final dimensions and feature-reduction verification — implemented and checked.** There are 1,307 training rows and 1,395 temporal rows; 33 original features become 30 retained features before encoding, then 53 XGBoost or 48 logistic-regression inputs, with no missing output values. `phase2_implementation/feature_reconstruction.csv` records 3,921 checks with zero differences. Exact reconstruction applies to the three structural removals; capping, ranking and imputation intentionally transform values, so the entire pipeline is not claimed to be lossless.

**Remaining documentation cleanup:** a few notebook markdown cells still use older wording about excluding only repeated children, normality as the reason for Spearman correlation, and fold-by-fold selection of the same removal list. The executed code and updated reports use the stricter primary/secondary cohorts, the robustness/monotonicity rationale, and fixed structural removals. These explanations should be aligned before treating the submission as fully polished.

**Outside the supplied Phase 2 deliverables:** final model tuning, calibration, decision-threshold selection and final temporal prediction scoring are not requested in the supplied Phase 2 section. Their absence must not be used to describe the existing Phase 2 implementation as missing. No instructor grade or Canvas submission confirmation is claimed.

**Review practice for subsequent changes:** check both phases against the supplied questionnaire and existing repository before changing status; trace each requirement to its code, output or report evidence; distinguish prior work from revisions; and record remaining gaps explicitly.

Audit notes: `on_vacation` is survey-coded in Wave 1 but only an inferred no-class-week proxy in Wave 2 (zero attended and zero missed hours); it is not confirmed holiday status. Baseline missingness is measured after linkage, harmonisation and special-code recognition. The primary temporal cohort excludes shared households and people; shared households are allowed only in the secondary cohort.

### Mixed-distance KNN, statistical checks and temporal cohorts

The active `GradeKNNImputer` uses mixed numeric/category distances and training-only inner grouped selection of k. Full training selects k=15. Nested grade-validation MAE is 1.128, versus 1.110 for legacy numeric-code k=5 and 1.444 for median filling; the small legacy advantage is retained in the report, not hidden. Run section B1.3 of the Phase 1 audit to reproduce the four `B1_knn_*.csv` tables. These masked observed-grade scores do not reveal the truth of genuinely missing grades.

Feature-target and missingness screens check categorical expected cells. Sparse 2x2 tables use Fisher exact; larger sparse tables use a seeded fixed-margin Fisher Monte Carlo test (99,999 samples). The 33 feature-target tests form one BH-adjusted family, with BY as a conservative dependence sensitivity. The 166 missingness screens form a separate family. These are exploratory row-level associations, not causal, survey-weighted or cluster-robust population inference.

`C2_temporal_cohorts.csv` and `C2_evaluation_protocol.json` record identity-only cohorts, overlap checks, effective allocations and raw-input hashes. The primary 731-child cohort shares neither households nor people with training; it still contains familiar districts. A genuinely new future/external cohort is needed for confirmatory transport claims.

Processed training files contain 53 XGBoost or 48 logistic-regression inputs, plus five ID/weight columns and the target. Processed test files additionally contain three cohort metadata flags: `in_both_tables`, `in_primary_test`, `in_secondary_test`. These flags must never be used as predictors. Raw GSPS and linked pre-pipeline datasets are preserved.

Run `python -m unittest discover -s tests -v` for KNN invariance, fitting-boundary, sparse-test, multiplicity and cohort-mask checks. Both full notebooks also run end to end; Phase 2 saves refreshed `cv_results.csv`, `pca_variance.csv`, `feature_reconstruction.csv` and the decision log. Current untuned training CV PR-AUC is 0.266 (LR) and 0.252 (XGBoost); this is not final temporal-test performance or an instructor grade.
