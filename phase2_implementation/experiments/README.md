# Exploratory training experiments
These experiments were developed after inspecting historical outcomes. They are exploratory, not independent confirmation and do not replace the frozen Extra Trees candidate. Original source tables are never overwritten. Run scripts from any working directory with the project Python environment; the scripts locate the repository automatically. Each uses district-separated folds and refits preprocessing on its training partition. All validation children remain included. Target values never determine missingness or anomaly exclusion rules.

Run each listed script with `python phase2_implementation/experiments/<script>.py`:
- selective_cleaning_search.py: 18 rules, three models, three repetitions, 810 fits; resumes completed rule/repetition blocks from its CSV.
- anomaly_cleaning.py: predictor-only Isolation Forest, 0/1/3/5 percent exclusion; 180 fits.
- nested_cleaning.py: six families, two retention choices, three inner district folds, five outer folds; 180 search fits plus 10 refits. Selection maximizes inner top-20% capture with AP as tie-breaker; no hyperparameter search or calibration in this experiment.
- homework_experiment.py: baseline versus homework, 90 fits. Missing homework uses training median; holiday codes remain missing. Combining a recorded hours or minutes component assumes the absent component is zero, an unverified measurement assumption. No later-wave transfer claim.
- combined_homework.py: homework, retention and geography combinations, 180 fits.
- label_audit.py: source outcome-link duplicates and completion-code checks; aggregate outputs only.
- identity_sensitivity.py: 50 birth-year discrepancies exceeding two years; training-only exclusion, 45 fits. Unknown birth years remain eligible. Discrepancies do not establish wrong identities.

Counts include baseline fits repeated across experiments. Three repetitions reuse the same children; their averages are not counts of independent children. AP is mean fold average precision. All model settings are fixed comparison defaults except the explicitly nested model/retention choice. Probabilities here are uncalibrated. Screening capacity is 20 percent, not the training fraction. No candidate is promoted using later outcomes. Missingness and outlier exclusions may introduce selection bias.
