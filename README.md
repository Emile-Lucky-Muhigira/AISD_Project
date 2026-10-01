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
| **Source** | Ghana Socioeconomic Panel Survey (GSPS), Waves 1–3 — ISSER (University of Ghana), Yale University, Northwestern University |
| **Link** | Harvard Dataverse: https://doi.org/10.7910/DVN/E5QP0F |
| **Waves** | 2009/10, 2013/14, 2017/18 (same households and individuals re-interviewed) |
| **Raw format** | ~330 Stata (`.dta`) files, one per questionnaire module per wave, linked by household ID (`FPrimary`) and member ID (`hhmid`) |
| **Analysis unit** | One row per child aged 8–13 enrolled in school at wave *t* |
| **Sample size (N)** | Training (Wave 1 → 2): 1,307 children (188 dropouts, 14.4%) · Test (Wave 2 → 3): 1,395 children (145 dropouts, 10.4%) |
| **Feature count (P)** | 33 features (+ 5 identifier/grouping columns and the target; 39 columns in total) |
| **Target (Y)** | `dropout`: 1 = out of school at wave *t+1* without completing basic education; 0 = still enrolled |

**Training / test transitions:** Wave 1 → Wave 2 (training and cross-validation); Wave 2 → Wave 3 (out-of-time test set).

## 3. Task Type

**Binary Classification** (dropout = 1 vs. still enrolled = 0), with an imbalanced minority positive class.

## 4. Repository Structure

```
AISD_Project/
├── .gitignore                 # excludes raw/processed data, virtual env, caches
├── requirements.txt           # exact package versions (pip freeze)
├── README.md                  # this file
├── data/
│   ├── raw/                   # original GSPS files (not tracked; see Setup)
│   └── processed/             # cleaned, encoded, scaled feature dataset (not tracked)
├── notebooks/                 # exploratory / audit notebooks supporting Phase 1
├── phase1_strategy/
│   └── Phase1_Strategy_Report.pdf
├── phase2_implementation/
│   ├── notebooks/
│   │   └── pipeline_execution.ipynb
│   └── Phase2_Final_Report.pdf
├── src/
│   ├── __init__.py
│   ├── build_dataset.py       # links the three waves into the child-level analysis table
│   ├── transformers.py        # custom scikit-learn transformers
│   └── pipeline_builder.py    # ColumnTransformer and Pipeline definitions
└── figures/                   # diagnostic plots (PNG, 300 DPI)
```

`notebooks/` and `src/build_dataset.py` are additions to the required structure: the GSPS is distributed as separate modules per wave, so a dedicated linkage step is needed before any preprocessing.

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
3. Download the dataset from https://doi.org/10.7910/DVN/E5QP0F and unzip it into `data/raw/`, so that this path exists:
   `data/raw/Ghana_Panel_Survey/Data/Wave 1/s1d.dta`
4. Note: some files use a legacy Windows text encoding and must be read with `encoding="latin1"`.

## 6. Pipeline Summary

*Phase 1 (strategy) in progress. The preprocessing pipeline will be summarised here after Phase 2.*
