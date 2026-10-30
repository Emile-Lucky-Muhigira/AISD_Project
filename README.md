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
| **Class distribution** | Training: 188 dropouts (14.4%); later period: 145 dropouts (10.4%) |
| **Project tables** | `train_w1_w2_raw.csv` and `test_w2_w3_raw.csv` in `data/processed/` |
| **Target (Y)** | `dropout`: 1 = classified as leaving school without completing basic education; 0 = still enrolled |

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
├── requirements.txt           # pinned package versions for reproduction
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
│   │   └── pipeline_execution.ipynb # preprocessing, model evaluation and diagnostic figures
│   ├── decision_log.csv             # Phase 1 plan vs Phase 2 implementation
│   ├── Phase2_Final_Report.pdf      # Phase 2 deliverable
│   ├── Phase2_Final_Report.tex      # LaTeX source of the report
│   ├── final_validation.json        # integrated execution and reproducibility evidence
│   └── *.csv / *.json               # metrics, predictions, protocols and diagnostic evidence
├── src/
│   ├── __init__.py
│   ├── transformers.py        # custom scikit-learn transformers
│   ├── pipeline_builder.py    # ColumnTransformer and Pipeline definitions
│   └── *.py                   # tuning, calibration, interpretation and sensitivity analyses
├── tests/                     # preprocessing and evaluation regression tests
└── figures/                   # eight required preprocessing figures and added model diagnostics
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
   - `phase2_implementation/notebooks/pipeline_execution.ipynb` runs preprocessing and the professor-requested model comparisons, nested tuning, calibration, subgroup and temporal evaluation, explanations and exploratory sensitivity analyses. It saves the eight required preprocessing figures plus added diagnostics to `figures/`, processed tables to `data/processed/`, and results and the decision log to `phase2_implementation/`. Select the project `.venv` as the notebook kernel after installing the pinned dependencies. The original 41-cell verified run took about 61 minutes on the development machine; runtime varies. The three appended robustness cells display saved evidence by default. Set `RECOMPUTE_ROBUSTNESS = True` in the appended extension cell to rerun the robustness, weighting and regional-selection experiments; these add substantial computation and were executed separately for the recorded extension.
5. Run `python -m unittest discover -s tests -v` to check preprocessing and evaluation behavior. The current suite passes 100 tests; the earlier integrated notebook run passed the then-current 83 tests.
6. Note: some files use a legacy Windows text encoding and must be read with `encoding="latin1"`.

## 6. Pipeline Summary

The preprocessing is one scikit-learn `Pipeline` built around a `ColumnTransformer` (`src/pipeline_builder.py`). It is fitted on the training children only (X_train, 2009/10) and then applied unchanged to the test children (X_test, 2013/14).

1. **Fixed rules (nothing learned):** recording-error corrections; 2013/14 money converted to 2009/10 prices (factor 1.62, World Bank consumer prices for Ghana).
2. **Row-wise steps:** blank missed hours are assumed zero when attendance was recorded (with a training-only median-fill sensitivity); 4 fixed design/family-context missing flags; mixed-distance grade KNN with k selected from 3, 5, 9, 15, 25 inside three district-grouped inner folds of each fit; 3 structural features removed (33 → 30). Numeric age/assets and categorical sex/urban/region use different distance components; region codes have no ordering.
3. **Column-wise steps:** gaps filled with mean / median / most common answer according to skew; extreme values capped at the log-IQR limit (no child deleted); money ranked 0–1 against training values; one-hot encoding of labels; `StandardScaler` for logistic regression and PCA only.
4. **Output:** 53 columns for XGBoost, 48 for logistic regression; 0 missing values; the 3 removed features can be rebuilt exactly from the kept ones.
5. **Validation:** 5-fold `StratifiedGroupKFold` inside training. Primary temporal evaluation uses 731 new-household children (76 dropouts, zero shared households/people); secondary uses 1,175 new people but permits 370 shared households. All 1,395 temporal rows are retained for descriptive analysis. Prior inspection of later-period covariates and outcomes is disclosed; this is retrospective evaluation, not a pristine confirmatory holdout. Masks use IDs only; no temporal score chooses the method.

## 7. Project Status

**Phase 1 (strategy): implemented, locally verified and revised against the supplied assignment rubric.** See [Phase 1 report](phase1_strategy/Phase1_Strategy_Report.pdf).

**Phase 2 (execution and diagnostics): the supplied preprocessing deliverables are implemented and verified; the model-evaluation extensions requested in the professor's feedback are substantially implemented, including the separately documented robustness and follow-up-weighting sensitivities below.** The approved mixed-distance KNN, structural feature removals, log-IQR capping and grouped validation remain in the default pipeline. Separate cleaning and feature experiments do not replace that pipeline or establish improved temporal performance.

### Phase 2: questionnaire versus repository (checked 28 October 2026)

This comparison uses pages 9–10 and the repository layout on page 2 of **Assignment 2: Data Preprocessing** (`Assignment 2 Data Preprocessing-1.pdf`). The supplied brief says “More details will follow”; this check does not assume that any later Phase 2 instructions have been supplied.

- **Executable notebook using `Pipeline` and `ColumnTransformer` — implemented and verified.** [pipeline_execution.ipynb](phase2_implementation/notebooks/pipeline_execution.ipynb) uses the modular code in `src/transformers.py` and `src/pipeline_builder.py`. The original 41 code cells executed in a fresh kernel, passing 83 tests and reproducing 88 result CSVs byte for byte. New experiments were executed separately, three appended evidence-display cells were checked in a fresh kernel, and the current suite passes 100 tests. The combined notebook has not been fully rerun in this extension; see [original validation](phase2_implementation/final_validation.json) and [extension validation](phase2_implementation/robustness_validation.json).
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
- **Strategy-versus-reality reflection report — implemented.** [Phase2_Final_Report.pdf](phase2_implementation/Phase2_Final_Report.pdf), with editable `.tex` source alongside it, contains the decision log, deviations and technical rationale. The [machine-readable decision log](phase2_implementation/decision_log.csv) has 32 entries. The 15-page report includes temporal comparison, calibration and final-model SHAP figures, with the appendix and references each starting on a fresh page.
- **Final dimensions and feature-reduction verification — implemented and checked.** There are 1,307 training rows and 1,395 temporal rows; 33 original features become 30 retained features before encoding, then 53 XGBoost or 48 logistic-regression inputs, with no missing output values. `phase2_implementation/feature_reconstruction.csv` records 3,921 checks with zero differences. Exact reconstruction applies to the three structural removals; capping, ranking and imputation intentionally transform values, so the entire pipeline is not claimed to be lossless.

### Professor feedback: implemented work and remaining gaps

- **Preserved methodology:** mixed-distance grade KNN with training-only grouped selection of k, three justified structural removals, log-IQR capping, and district-grouped validation that keeps households together remain in place.
- **Attrition and family context:** compared observable and unobservable children and reported performance by parent co-residence. Unknown outcomes are not classified as non-dropout. Results describe children with observable follow-up; generalisation to children lost to follow-up remains uncertain. **A separate follow-up-weighted sensitivity is implemented**, with known completers treated as observed and grouped cross-fitted observation weights. Its missing-at-random, positivity and observation-model assumptions remain unverified.
- **Model comparison and tuning:** evaluated XGBoost as the primary benchmark alongside LightGBM, CatBoost, ElasticNet logistic regression, Random Forest and Extra Trees. Hyperparameter selection uses nested district-grouped training validation; later outcomes do not select the model or settings.
- **Calibration and intervention policy:** fitted sigmoid/isotonic calibration using training validation predictions and reported average precision (AP, used here as the PR-AUC summary), recall and precision at capacity, Brier score, calibration curves and confusion counts. The frozen temporal comparison uses training-fitted sigmoid mappings and a top-20% allocation, alongside random-allocation and demographic comparators.
- **Missingness and geographic sensitivity:** internal ablations compare models with and without missingness indicators, geography, or the region predictor. Their AP, recall, precision and uncalibrated Brier scores provide sensitivity evidence; they do not prove a survey-mode effect. **The robustness extension now includes calibrated comparisons and temporal no-geography evaluation**, using six families, identical folds and a separate protocol. Removing explicit flags does not remove every missingness signal.
- **Temporal and regional drift:** evaluated both later-period cohorts, regional and high-/low-risk-region performance, and changes in prevalence, missingness and calibration. Sparse groups are identified. Earlier inspection of later outcomes is disclosed, so this is retrospective evidence rather than untouched external confirmation.
- **Interpretability:** added global and local SHAP analyses, final Extra Trees explanations and penalised logistic model-score odds ratios. These describe model associations, not causal effects. Extra Trees SHAP contributions reconstruct raw probability, not calibrated risk.

### Verified prediction results and practical limits

Extra Trees was selected using **training validation** (mean fold AP approximately **0.303**). On the **primary retrospective cohort of 731 children, including 76 dropouts**, its frozen calibrated model achieved **AP 0.134**. At 20% intervention capacity, it flagged **146 children**: **20 true positives, 126 false positives and 56 missed dropouts**. Recall was **26.3%**, precision **13.7%**, and Brier score **0.0938**. Mean-fold training AP and pooled temporal AP are different summaries; the comparison nevertheless reveals limited later-period ranking performance. For all six models, the same primary and secondary cohorts, and the fixed 20% capacity, see the [precision, recall and F1 table](phase2_implementation/temporal_precision_recall_f1.csv); the report gives these results in Table IX. The values are proportions. See the [temporal metrics](phase2_implementation/temporal_metrics.csv) and [report](phase2_implementation/Phase2_Final_Report.pdf) for all six models, uncertainty intervals and both cohorts.

Geographic concentration is substantial: regions 8 and 10 receive 128 of the 146 primary places. Family-group recall also differs: 17 of 48 dropouts are caught in both-parent households, versus 3 of 20 in mother-only households. These are descriptive disparities, not causal findings. Average calibration does not remove these allocation problems.

**The model is not ready for operational deployment.** Increasing capacity catches more dropouts but also increases false alarms. Separate source-audited cleaning alternatives preserve identities, outcomes and target-last ordering and give mixed training-validation gains; these cleaning comparisons alone do not establish a reliable temporal improvement. The exploratory model extensions below show modest point gains with uncertainty that includes no improvement. The original data and frozen evaluation remain available. See [cleaning sensitivity results](phase2_implementation/recleaning_summary.csv).

**Remaining work:** independent future/external validation and stronger evidence about unobserved outcomes. The three previously listed analysis gaps are addressed through calibrated ablations, temporal geography comparisons and a follow-up-weighting sensitivity. Their implementation does not prove a deployable model or guarantee a grade. Further analysis of the inspected later cohort remains exploratory.

**Evidence:** [executed notebook](phase2_implementation/notebooks/pipeline_execution.ipynb), [final report](phase2_implementation/Phase2_Final_Report.pdf), [report source](phase2_implementation/Phase2_Final_Report.tex), [decision log](phase2_implementation/decision_log.csv), and [validation manifest](phase2_implementation/final_validation.json). The eight original questionnaire diagnostics remain included alongside the added professor-requested analyses.

### Controlled robustness extension (28 October 2026)

A bounded 72-candidate study compares six families, four missingness/geography configurations and three class-weight strengths on source-confirmed corrected tables. Capacity-focused and AP-focused selection are repeated inside training-only calibration folds. The frozen model remains available; later outcomes do not select a replacement.

At the same **146 primary places**, capacity selection catches **20 of 76 dropouts with 126 false alarms** (AP 0.136); the AP control catches **21 with 125 false alarms**. A prespecified equal-weight ensemble catches **22 with 124 false alarms**, and follow-up-weighted Extra Trees catches **21 with 125 false alarms**. The frozen baseline caught 20 with 126 false alarms. Capacity selection's paired recall difference is **+0.0 percentage points**, with a 95% district-bootstrap interval of **[-3.5, +3.6]**, conditional on fitted models and fixed selections.

The additional **region-held-out selection** test catches **23 dropouts with 123 false alarms** for capacity selection and **23 with 123 false alarms** for AP selection. It changes inner validation groups, retains the candidate grid and outer district folds, and repeats selection within calibration. Both regional procedures select Random Forest without geography, with balanced class weights. Primary AP increases from 0.134 for the frozen baseline to 0.158, and recall from 26.3% to 30.3%. Capacity selection's primary recall-difference interval is [-11.4, +19.2] percentage points and includes no improvement. Its training-validation recall is lower than the district-based procedure. This test was motivated by prior retrospective findings and is not independent confirmation.

Weighting treats known completion separately from attrition. Its assumptions remain unverified, and performance is measured among observed children. No unknown dropout labels were invented; target-last ordering is preserved.

The detailed methods, assumptions, Table X and Figures 12–13 are in the **Controlled Robustness Extension** section of the [report](phase2_implementation/Phase2_Final_Report.pdf). Supporting evidence includes the [regional protocol](phase2_implementation/region_stress_protocol.json), [regional results](phase2_implementation/region_stress_summary.csv), [protocol](phase2_implementation/robustness_protocol.json), [results](phase2_implementation/robustness_summary.csv), [paired comparisons](phase2_implementation/robustness_paired_comparison.csv), [weighting assumptions](phase2_implementation/followup_weighted_protocol.json), and [extension validation](phase2_implementation/robustness_validation.json).

**Review practice for subsequent changes:** check both phases against the supplied questionnaire and existing repository before changing status; trace each requirement to its code, output or report evidence; distinguish prior work from revisions; and record remaining gaps explicitly.

Audit notes: `on_vacation` is survey-coded in Wave 1 but only an inferred no-class-week proxy in Wave 2 (zero attended and zero missed hours); it is not confirmed holiday status. Baseline missingness is measured after linkage, harmonisation and special-code recognition. The primary temporal cohort excludes shared households and people; shared households are allowed only in the secondary cohort.

### Mixed-distance KNN, statistical checks and temporal cohorts

The active `GradeKNNImputer` uses mixed numeric/category distances and training-only inner grouped selection of k. Full training selects k=15. Nested grade-validation MAE is 1.128, versus 1.110 for legacy numeric-code k=5 and 1.444 for median filling; the small legacy advantage is retained in the report, not hidden. Run section B1.3 of the Phase 1 audit to reproduce the four `B1_knn_*.csv` tables. These masked observed-grade scores do not reveal the truth of genuinely missing grades.

Feature-target and missingness screens check categorical expected cells. Sparse 2x2 tables use Fisher exact; larger sparse tables use a seeded fixed-margin Fisher Monte Carlo test (99,999 samples). The 33 feature-target tests form one BH-adjusted family, with BY as a conservative dependence sensitivity. The 166 missingness screens form a separate family. These are exploratory row-level associations, not causal, survey-weighted or cluster-robust population inference.

`C2_temporal_cohorts.csv` and `C2_evaluation_protocol.json` record identity-only cohorts, overlap checks, effective allocations and raw-input hashes. The primary 731-child cohort shares neither households nor people with training; it still contains familiar districts. A genuinely new future/external cohort is needed for confirmatory transport claims.

Processed training files contain 53 XGBoost or 48 logistic-regression inputs, plus five ID/weight columns and the target. Processed test files additionally contain three cohort metadata flags: `in_both_tables`, `in_primary_test`, `in_secondary_test`. These flags must never be used as predictors. Raw GSPS and linked pre-pipeline datasets are preserved.

Run `python -m unittest discover -s tests -v` for KNN invariance, fitting-boundary, sparse-test, multiplicity and cohort-mask checks. The dataset-building and Phase 1 audit notebooks retain their separate roles; the integrated Phase 2 run is recorded in `final_validation.json`. Phase 2 saves refreshed `cv_results.csv`, `pca_variance.csv`, `feature_reconstruction.csv` and the decision log. Current untuned training CV PR-AUC is 0.266 (LR) and 0.252 (XGBoost); this is not final temporal-test performance or an instructor grade.
