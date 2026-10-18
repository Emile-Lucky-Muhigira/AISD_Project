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
| **Raw format** | ~330 Stata (`.dta`) files, one per questionnaire module per wave, linked by household ID (`FPrimary`) and member ID (`hhmid`) |
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
   - `notebooks/01_build_dataset.ipynb` builds the training and test tables in `data/processed/`.
   - `notebooks/02_phase1_audit.ipynb` reproduces every number and table in the Phase 1 report.
   - `phase2_implementation/notebooks/pipeline_execution.ipynb` fits the preprocessing pipeline, saves the 8 diagnostic figures to `figures/`, the processed tables to `data/processed/` and the decision log to `phase2_implementation/decision_log.csv`.
5. Note: some files use a legacy Windows text encoding and must be read with `encoding="latin1"`.

## 6. Pipeline Summary

The preprocessing is one scikit-learn `Pipeline` built around a `ColumnTransformer` (`src/pipeline_builder.py`). It is fitted on the training children only (X_train, 2009/10) and then applied unchanged to the test children (X_test, 2013/14).

1. **Fixed rules (nothing learned):** recording-error corrections; 2013/14 money converted to 2009/10 prices (factor 1.62, World Bank consumer prices for Ghana).
2. **Row-wise steps:** blank "hours missed" = 0 when attendance was recorded; 4 "was missing" flags; missing school grade filled from the 5 most similar children (KNN); 3 repeating features removed (33 → 30).
3. **Column-wise steps:** gaps filled with mean / median / most common answer according to skew; extreme values capped at the log-IQR limit (no child deleted); money ranked 0–1 against training values; one-hot encoding of labels; `StandardScaler` for logistic regression and PCA only.
4. **Output:** 53 columns for XGBoost, 48 for logistic regression; 0 missing values; the 3 removed features can be rebuilt exactly from the kept ones.
5. **Validation:** 5-fold `StratifiedGroupKFold` grouped by district; time-based test set; the 220 children who appear in both tables are excluded from the main test result.

## 7. Project Status

**Phase 1 (strategy): complete.** See `phase1_strategy/Phase1_Strategy_Report.pdf`.
**Phase 2 (implementation and diagnostics): complete.** See `phase2_implementation/Phase2_Final_Report.pdf`.