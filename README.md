# ElimuSentry: An Interpretable Early-Warning System for School Dropout

**Author:** Emile Lucky Muhigira (emuhigir@andrew.cmu.edu), Carnegie Mellon University Africa
**Course:** 04-652 Artificial Intelligence System Design

**Selected model:** 14-input regularized logistic regression without region identifiers or explicit missingness flags. On the retrospective household-disjoint primary cohort: precision **20.55%**, recall **39.47%**, F1 **27.03%**, AP **0.214** at 146 alerts. Final specification choice uses temporal comparisons; independent confirmation is still needed.

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
   - `phase2_implementation/notebooks/pipeline_execution.ipynb` runs preprocessing, model comparisons, nested tuning, calibration, subgroup and temporal evaluation, explanations and exploratory sensitivity analyses. It saves the eight required preprocessing figures plus added diagnostics to `figures/`, processed tables to `data/processed/`, and results and the decision log to `phase2_implementation/`. Select the project `.venv` as the notebook kernel after installing the pinned dependencies. The original 41-cell verified run took about 61 minutes on the development machine; runtime varies. The three appended robustness cells display saved evidence by default. Set `RECOMPUTE_ROBUSTNESS = True` in the appended extension cell to rerun the robustness, weighting and regional-selection experiments; these add substantial computation and were executed separately for the recorded extension.
5. Run `python -m unittest discover -s tests -v` to check preprocessing and evaluation behavior. Dated validation records identify the suite and execution scope for each run.
6. Note: some files use a legacy Windows text encoding and must be read with `encoding="latin1"`.

## 6. Comprehensive Benchmark Pipeline

The comprehensive benchmark preprocessing is one scikit-learn `Pipeline` built around a `ColumnTransformer` (`src/pipeline_builder.py`). It is fitted on the training children only (X_train, 2009/10) and then applied unchanged to the test children (X_test, 2013/14).

1. **Fixed rules (nothing learned):** recording-error corrections; 2013/14 money converted to 2009/10 prices (factor 1.62, World Bank consumer prices for Ghana).
2. **Row-wise steps:** blank missed hours are assumed zero when attendance was recorded (with a training-only median-fill sensitivity); 4 fixed design/family-context missing flags; mixed-distance grade KNN with k selected from 3, 5, 9, 15, 25 inside three district-grouped inner folds of each fit; 3 structural features removed (33 → 30). Numeric age/assets and categorical sex/urban/region use different distance components; region codes have no ordering.
3. **Column-wise steps:** gaps filled with mean / median / most common answer according to skew; extreme values capped at the log-IQR limit (no child deleted); money ranked 0–1 against training values; one-hot encoding of labels; `StandardScaler` for logistic regression and PCA only.
4. **Output:** 53 columns for XGBoost, 48 for logistic regression; 0 missing values; the 3 removed features can be rebuilt exactly from the kept ones.
5. **Validation:** 5-fold `StratifiedGroupKFold` inside training. Primary temporal evaluation uses 731 new-household children (76 dropouts, zero shared households/people); secondary uses 1,175 new people but permits 370 shared households. All 1,395 temporal rows are retained for descriptive analysis. Prior inspection of later-period covariates and outcomes is disclosed; this is retrospective evaluation, not a pristine confirmatory holdout. Masks use IDs only. Benchmark tuning uses grouped training folds; final compact-model selection uses the retrospective temporal comparison described below.

## 7. Project Status

**Phase 1 (strategy): implemented and verified against the assignment requirements.** See [Phase 1 report](phase1_strategy/Phase1_Strategy_Report.pdf).

**Phase 2 (execution and diagnostics): the supplied preprocessing deliverables are implemented and verified; the additional model-evaluation requirements are substantially implemented, including the separately documented robustness and follow-up-weighting sensitivities below.** The comprehensive benchmark pipeline uses mixed-distance KNN, structural feature removals, log-IQR capping and grouped validation. Separate cleaning and feature experiments do not replace that pipeline or establish improved temporal performance.

### Phase 2: assignment requirements and evidence

This comparison uses pages 9–10 and the repository layout on page 2 of **Assignment 2: Data Preprocessing** (`Assignment 2 Data Preprocessing-1.pdf`). The supplied brief says “More details will follow”; this check does not assume that any later Phase 2 instructions have been supplied.

- **Executable notebook using `Pipeline` and `ColumnTransformer` — implemented and verified.** [pipeline_execution.ipynb](phase2_implementation/notebooks/pipeline_execution.ipynb) uses the modular code in `src/transformers.py` and `src/pipeline_builder.py`. The original 41 code cells executed in a fresh kernel, passing 83 tests and reproducing 88 result CSVs byte for byte. New experiments were executed separately, three appended evidence-display cells were checked in a fresh kernel, and the current suite passes 100 tests. All 45 code cells passed a fresh-kernel run on 2 November 2026; final display cells load saved experimental evidence. See [current verification](phase2_implementation/full_notebook_validation.json), [experiment inventory](phase2_implementation/EXPERIMENT_EVIDENCE.md), [original validation](phase2_implementation/final_validation.json) and [extension validation](phase2_implementation/robustness_validation.json).
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
- **Strategy-versus-reality reflection report — implemented.** [Phase2_Final_Report.pdf](phase2_implementation/Phase2_Final_Report.pdf), with editable `.tex` source alongside it, contains the decision log, deviations and technical rationale. The [machine-readable decision log](phase2_implementation/decision_log.csv) has 33 entries. The report includes compact-model selection, temporal comparison, calibration, selected-model coefficients and benchmark SHAP figures, with the appendix and references each starting on a fresh page.
- **Final dimensions and feature-reduction verification — implemented and checked.** There are 1,307 training rows and 1,395 temporal rows; 33 original features become 30 retained features before encoding, then 53 XGBoost or 48 logistic-regression inputs, with no missing output values. `phase2_implementation/feature_reconstruction.csv` records 3,921 checks with zero differences. Exact reconstruction applies to the three structural removals; capping, ranking and imputation intentionally transform values, so the entire pipeline is not claimed to be lossless.

### Model evaluation: implemented work and remaining gaps

- **Preserved methodology:** mixed-distance grade KNN with training-only grouped selection of k, three justified structural removals, log-IQR capping, and district-grouped validation that keeps households together remain in place.
- **Attrition and family context:** compared observable and unobservable children and reported performance by parent co-residence. Unknown outcomes are not classified as non-dropout. Results describe children with observable follow-up; generalisation to children lost to follow-up remains uncertain. **A separate follow-up-weighted sensitivity is implemented**, with known completers treated as observed and grouped cross-fitted observation weights. Its missing-at-random, positivity and observation-model assumptions remain unverified.
- **Model comparison and tuning:** evaluated XGBoost as the primary benchmark alongside LightGBM, CatBoost, ElasticNet logistic regression, Random Forest and Extra Trees. Hyperparameter selection uses nested district-grouped training validation; benchmark tuning excludes later outcomes; final compact-specification choice uses retrospective temporal comparisons.
- **Calibration and intervention policy:** fitted sigmoid/isotonic calibration using training validation predictions and reported average precision (AP, used here as the PR-AUC summary), recall and precision at capacity, Brier score, calibration curves and confusion counts. The frozen temporal comparison uses training-fitted sigmoid mappings and a top-20% allocation, alongside random-allocation and demographic comparators.
- **Missingness and geographic sensitivity:** internal ablations compare models with and without missingness indicators, geography, or the region predictor. Their AP, recall, precision and uncalibrated Brier scores provide sensitivity evidence; they do not prove a survey-mode effect. **The robustness extension now includes calibrated comparisons and temporal no-geography evaluation**, using six families, identical folds and a separate protocol. Removing explicit flags does not remove every missingness signal.
- **Temporal and regional drift:** evaluated both later-period cohorts, regional and high-/low-risk-region performance, and changes in prevalence, missingness and calibration. Sparse groups are identified. Earlier inspection of later outcomes is disclosed, so this is retrospective evidence rather than untouched external confirmation.
- **Interpretability:** added global and local SHAP analyses, Extra Trees benchmark explanations and penalised logistic model-score odds ratios. These describe model associations, not causal effects. Extra Trees SHAP contributions reconstruct raw probability, not calibrated risk.

### Selected model and results

The selected specification is compact regularized logistic regression (`compact_no_region`), implemented in `src/phase3_readiness.py` and scored through `src/selected_model.py`. It uses 14 substantive inputs, median filling, standardization, `C=0.1`, no class weighting and a development-only sigmoid calibrator. Raw region IDs, explicit missingness flags and assessment-status categories are excluded; urban/rural residence remains. The fitted artifact is saved locally as `data/processed/selected_model.joblib` and is reproduced from repository code and local survey data.

A fixed split reserves 28 districts (258 children, 42 dropouts), leaving 108 development districts (1,049 children, 146 dropouts). Model coefficients and calibration use development children only. The saved artifact is the same fit used to obtain the reported scores; it is not refitted on reserved or temporal labels.

On the **primary household-disjoint cohort** of 731 children and 76 dropouts, the selected model flags 146 children: **30 TP, 116 FP, 46 FN, 539 TN**. Precision is **20.55%**, recall **39.47%**, F1 **27.03%**, AP **0.214**, and Brier score **0.0933**. The secondary person-disjoint cohort contains 1,175 children and 115 dropouts; at 235 alerts the selected model catches 37 (precision 15.74%, recall 32.17%, F1 21.14%).

The 17-input regional-context comparison catches 29 primary and 39 secondary dropouts. Development-fold mean AP favors that context specification (0.195 versus 0.177). Final project choice prioritizes the no-region specification's primary temporal results, simpler representation and less concentrated allocation. **This is retrospective model selection, not an untouched test or a claim of superiority on every cohort.** The selected model allocates 57 primary alerts to Regions 8 and 10, catching 21 dropouts elsewhere. Its primary district-bootstrap recall interval is approximately 25.0%-54.1%, conditional on fitted predictions and allocation; selection uncertainty is not included.

### Train and score

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.phase3_readiness
.\.venv\Scripts\python.exe -m src.selected_model --input data/processed/predictors.csv --output data/processed/risk_scores.csv --money-basis 2013_14
```

The scoring input must be a harmonized predictor table containing the 14 columns listed in `CORE`, plus unique `FPrimary`/`hhmid` identifiers, without `dropout`. Declare `2009_10` for already price-aligned spending or `2013_14` for nominal Wave 2 spending; the latter applies the fixed 174/107.5 (approximately 1.62) conversion to school spending. Other periods require a justified harmonization rule. The output contains child identifiers, calibrated risk and a cohort-level top-20% selection flag. The 20% capacity is a workload assumption, not a 50% probability threshold.

### Evidence and interpretation

- `selected_model.json`: selection basis, input hashes, fitted artifact hash and reproduction check.
- `selected_model_metrics.csv`: precision, recall, F1, confusion counts, AP, Brier and district-bootstrap intervals for the chosen model.
- `temporal_precision_recall_f1.csv`: six comprehensive benchmarks and both compact specifications.
- `selected_model_coefficients.csv`, `selected_model_calibration.csv`, `selected_model_capacity.csv`: selected-model interpretation and workload evidence.
- `phase3_readiness/`: aggregate results for both compact candidates, attrition scenarios and protocol. Child-level predictions and the partition manifest are generated locally and excluded from Git.
- `experiments/README.md`, `EXPERIMENT_EVIDENCE.md`: supporting explorations, including unsuccessful configurations.
- `REQUIREMENTS_EVIDENCE.md`: mapping to the supplied assignment requirements.

The notebook trains the selected model and displays its evidence. The comprehensive 45-code-cell execution and selected-model execution have separate validation records; running the compact experiment does not rerun every benchmark search.

### Limits and next validation

All three waves are already used: Wave 1 predictors → Wave 2 outcomes for training, and Wave 2 predictors → Wave 3 outcomes for temporal evaluation. Adding labelled Wave 2→3 rows to training does not create an independent Wave 3 temporal test. Independent later or external outcomes are needed for expanding-window confirmation.

Reserved districts and temporal data were previously explored. Final compact-model choice uses temporal comparisons, so scores may be optimistic. There are only 76 primary temporal events. The 14-input model has 10.4 development events per input (intercept excluded); this is a descriptive complexity check, not a guarantee of stable estimation.

Attrition leaves 1,158 of 2,532 baseline outcomes unresolved. Worst-case baseline dropout bounds span 7.4%-53.2%; these are identification bounds, not estimates. Inverse-probability weighting does not resolve unmeasured selection. Median imputation may still encode missingness, and geographic information remains in substantive predictors. The project is a research prototype requiring independent validation and human assessment before operational use.
