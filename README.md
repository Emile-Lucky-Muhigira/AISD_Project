# ElimuSentry: An Interpretable Early-Warning System for School Dropout

**Author:** Emile Lucky Muhigira (emuhigir@andrew.cmu.edu), Carnegie Mellon University Africa
**Course:** 04-652 Artificial Intelligence System Design

## 1. Problem Statement

School dropout in Sub-Saharan Africa is a gradual process: absences accumulate and learning falls behind before a child finally leaves school. Most regional dropout models are trained on one-visit household surveys (e.g., DHS, MICS) that record only whether a child is *already* out of school, so they can describe risk but cannot warn in time to act.

**Objective:** Given a child aged 8–13 who is enrolled in school at survey wave *t*, use information measured at wave *t* (attendance, math and English test performance, household wealth, parents' education, orphanhood and school context) to predict whether the child will be **out of school at wave *t+1* without having completed basic education** (Junior High School in Ghana).

**Domain context:** The project supports **SDG 4 (Quality Education)**, Targets 4.1 and 4.5, by helping schools identify at-risk children early. The central hypothesis is that dynamic signals (attendance and performance) improve prediction beyond static background factors (wealth, parents' education).

## 2. Dataset Overview

| Item | Description |
|---|---|
| **Source** | Ghana Socioeconomic Panel Survey (GSPS), Waves 1–3, by ISSER (University of Ghana), Yale University and Northwestern University |
| **Link** | Harvard Dataverse: https://doi.org/10.7910/DVN/E5QP0F |
| **Waves** | 2009/10, 2013/14, 2017/18 (same households and individuals re-interviewed) |
| **Raw format** | 317 Stata (`.dta`) files, one per questionnaire module per wave, linked by household ID (`FPrimary`) and member ID (`hhmid`) |
| **Analysis unit** | One row per child aged 8–13 enrolled in school at wave *t* |
| **Sample size (N)** | Training (Wave 1 → 2): 1,307 children (188 dropouts, 14.4%) · Test (Wave 2 → 3): 1,395 children (145 dropouts, 10.4%) |
| **Feature count (P)** | 33 features (+ 5 identifier/weight columns and the target; 39 columns in total) |
| **Target (Y)** | `dropout`: 1 = out of school at wave *t+1* without completing basic education; 0 = still enrolled |

**Training / test transitions:** Wave 1 → Wave 2 (training and cross-validation); Wave 2 → Wave 3 (out-of-time test set).

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

**Phase 1 (strategy): locally verified and revised against the assignment rubric.** See `phase1_strategy/Phase1_Strategy_Report.pdf`.
**Phase 2 (implementation and diagnostics): refreshed with the active mixed-distance KNN.** Both execution notebooks, processed inputs, diagnostic figures, decision log and reports have been regenerated and checked locally. Final model tuning, calibration and temporal prediction scoring belong to the modelling phase.
Audit notes: `on_vacation` is survey-coded in Wave 1 but only an inferred no-class-week proxy in Wave 2 (zero attended and zero missed hours); the proxy is not confirmed holiday status. Missingness before value correction is measured after linkage, harmonisation and survey special-code recognition, not on untouched survey modules. MI-only feature selection is unstable across folds; the pipeline uses the fixed structural list. The temporal test excludes repeated children, but some households still appear across time.

### Mixed-distance KNN, statistical checks and temporal cohorts

The active `GradeKNNImputer` uses mixed numeric/category distances and training-only inner grouped selection of k. Full training selects k=15. Nested grade-validation MAE is 1.128, versus 1.110 for legacy numeric-code k=5 and 1.444 for median filling; the small legacy advantage is retained in the report, not hidden. Run section B1.3 of the Phase 1 audit to reproduce the four `B1_knn_*.csv` tables. These masked observed-grade scores do not reveal the truth of genuinely missing grades.

Feature-target and missingness screens check categorical expected cells. Sparse 2x2 tables use Fisher exact; larger sparse tables use a seeded fixed-margin Fisher Monte Carlo test (99,999 samples). The 33 feature-target tests form one BH-adjusted family, with BY as a conservative dependence sensitivity. The 166 missingness screens form a separate family. These are exploratory row-level associations, not causal, survey-weighted or cluster-robust population inference.

`C2_temporal_cohorts.csv` and `C2_evaluation_protocol.json` record identity-only cohorts, overlap checks, effective allocations and raw-input hashes. The primary 731-child cohort shares neither households nor people with training; it still contains familiar districts. A genuinely new future/external cohort is needed for confirmatory transport claims.

Processed training files contain 53 XGBoost or 48 logistic-regression inputs, plus five ID/weight columns and the target. Processed test files additionally contain three cohort metadata flags: `in_both_tables`, `in_primary_test`, `in_secondary_test`. These flags must never be used as predictors. Raw GSPS and linked pre-pipeline datasets are preserved.

Run `python -m unittest discover -s tests -v` for KNN invariance, fitting-boundary, sparse-test, multiplicity and cohort-mask checks. Both full notebooks also run end to end; Phase 2 saves refreshed `cv_results.csv`, `pca_variance.csv`, `feature_reconstruction.csv` and the decision log. Current untuned training CV PR-AUC is 0.266 (LR) and 0.252 (XGBoost); this is not final temporal-test performance or an instructor grade.
