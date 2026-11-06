# Phase 2 requirement-to-evidence map
Scope: supplied Assignment 2 pages 9-10 and review recommendations. No numeric mark allocation or minimum model score was supplied; this map does not guarantee a grade. Historical temporal outcomes have been inspected, so the evaluation is retrospective exploratory evidence.

- Executable Pipeline / ColumnTransformer, fold-local fitting, target separation: `notebooks/pipeline_execution.ipynb`, `../src/pipeline_builder.py`, `../src/transformers.py`.
- Group/temporal separation: notebook split assertions, `../src/evaluation_split.py`, `temporal_cohorts.csv`.
- Eight required figures in notebook and PDF: `../figures/fig1_1_missingness_heatmap.png`, `fig1_2_target_distribution.png`, `fig2_1_outlier_boxplots.png`, `fig3_1_correlation_matrix.png`, `fig3_2_collinear_scatterplots.png`, `fig3_3_target_importance.png`, `fig4_1_transformation_grid.png`, `fig5_1_pca_scree_plot.png` (all in figures).
- Plan versus implementation and deviations: `decision_log.csv`, report Decision Log and Deviations sections.
- Output dimensions and structural redundancy: report Pipeline Output Verification, notebook assertions, `feature_reconstruction.csv`. Exact reconstruction concerns structural duplicates, not a general claim that imputation loses no information.
- Six families / training-only tuning: `model_comparison_metrics.csv`, `tuned_metrics.csv`, `tuning_results.csv` where present; notebook and `../src/model_tuning.py` are authoritative.
- OOF sigmoid/isotonic calibration: `calibration_metrics.csv`, `calibration_curves.csv`, `../src/model_calibration.py`.
- Temporal metrics / all confusion counts / capacity: `temporal_metrics.csv`, `temporal_evaluation_protocol.json`, `capacity_policy_analysis.csv`.
- Geography and missingness ablations: `sensitivity_metrics.csv`, `region_only_metrics.csv`, robustness outputs.
- Temporal / regional drift and subgroup performance: temporal diagnostics and subgroup outputs, report calibration/subgroup sections. Small groups remain descriptive.
- Attrition and inverse follow-up weights: `attrition_comparison.csv`, `followup_weighted_metrics.csv`, follow-up diagnostics. Missing outcomes remain unknown; weights require MAR and positivity assumptions.
- Global/local explanations: report SHAP figures and interpretation outputs; associations are not causal effects.
- Operational comparator: random expected capture and age/sex baseline in temporal results; 20% is a workload assumption, not training-set size.
- Recent exploratory work and negative findings: `EXPERIMENT_EVIDENCE.md`, `experiments/README.md`, scripts and saved aggregate metrics; no automatic model replacement or deletion.
- Reproduction status: consult dated execution evidence. A successful prior or partial run is not a fresh full-notebook run. Check dependencies with `python -m pip check`; run notebook from a fresh kernel with repository data installed.

Model roles: Extra Trees is the frozen training-selected candidate; XGBoost primary benchmark; ElasticNet interpretable baseline. Random Forest has the largest primary temporal top-20% capture under sigmoid calibration; CatBoost has the largest primary temporal AP. Reporting these distinctions avoids choosing a new winner from evaluated outcomes.

## Current verification (2026-11-02)
All 45 code cells executed from a fresh project-environment kernel without errors; 100 regression tests and the dependency check passed. Final display cells load saved robustness and exploratory results rather than retraining those searches. The identity-sensitivity experiment was separately rerun (45 fits), reproducing saved results. All 24 experiment-manifest entries now match, following refresh of the stale script fingerprint. See `full_notebook_validation.json` and `EXPERIMENT_EVIDENCE.md`. Historical validation records retain their original scope. This verification does not guarantee a grade or establish deployment readiness.

- Geographic grouping challenger: `experiments/geographic_grouping_results/README.md`, 360 training fits and matched temporal evaluation; appended notebook evidence cell checked separately. Training gains did not transfer to primary temporal capture; frozen model retained.
