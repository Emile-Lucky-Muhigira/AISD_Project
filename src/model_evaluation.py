"""Untuned, uncalibrated training-only baselines on supplied grouped folds."""
import json
import numpy as np
import pandas as pd
from .pipeline_builder import TARGET, build_pipeline, default_estimator
from .evaluation_metrics import evaluate_binary_risk, select_top_risk


def evaluate_baselines(train, splits, capacity=0.20, progress=True, models=("lr", "xgb"), *, include_missing_flags=True, include_geography=True, include_region_predictor=True):
    """Fit fresh model pipelines per fold; return metrics and OOF records.

    The input is the labelled pre-pipeline training table, not the globally
    transformed matrix. No temporal data are accepted. Splits are positional.
    These fixed settings are benchmarks, not nested-tuned or calibrated models.
    """
    if train[["FPrimary", "hhmid", "district_id", TARGET]].isna().any().any():
        raise ValueError("Identity, group and outcome values must be present.")
    if train.duplicated(["FPrimary", "hhmid"]).any():
        raise ValueError("Duplicate child identities.")
    if not train[TARGET].isin([0, 1]).all():
        raise ValueError("Binary outcomes required.")
    splits = [(np.asarray(a), np.asarray(b)) for a, b in splits]
    if len(splits) != 5:
        raise ValueError("Exactly five outer folds are required.")
    seen = np.zeros(len(train), dtype=int)
    for a, b in splits:
        if (a.ndim != 1 or b.ndim != 1 or a.dtype.kind not in 'iu' or b.dtype.kind not in 'iu'
                or not len(a) or not len(b) or min(a.min(), b.min()) < 0
                or max(a.max(), b.max()) >= len(train)):
            raise ValueError("Invalid positional fold indices.")
        if len(np.unique(a)) != len(a) or len(np.unique(b)) != len(b):
            raise ValueError("Repeated indices inside a fold.")
        if set(a) & set(b) or len(set(a) | set(b)) != len(train):
            raise ValueError("Each fold must partition the complete training cohort.")
        for column in ["district_id", "FPrimary"]:
            if set(train.iloc[a][column]) & set(train.iloc[b][column]):
                raise ValueError(f"Training/validation overlap in {column}.")
        if train.iloc[a][TARGET].nunique() != 2 or train.iloc[b][TARGET].nunique() != 2:
            raise ValueError("Both classes must occur in each fold partition.")
        seen[b] += 1
    if not np.all(seen == 1):
        raise ValueError("Each child must have exactly one held-out prediction per model.")
    X, y = train.drop(columns=TARGET), train[TARGET]
    keys = [json.dumps([str(h), str(p)], separators=(',', ':'))
            for h, p in zip(train.FPrimary, train.hhmid)]
    models = tuple(models)
    if not models or len(set(models)) != len(models) or not set(models).issubset(MODEL_NAMES):
        raise ValueError("Specify unique supported model names.")
    rows, predictions = [], []
    for model in models:
        for fold, (a, b) in enumerate(splits, start=1):
            yfit = y.iloc[a]
            class_ratio = float(yfit.eq(0).sum() / yfit.eq(1).sum())
            estimator = comparison_estimator(model, class_ratio)
            preprocessing = "lr" if model in ("lr", "elasticnet") else "xgb"
            pipe = build_pipeline(preprocessing, estimator, include_missing_flags=include_missing_flags, include_geography=include_geography, include_region_predictor=include_region_predictor)
            pipe.fit(X.iloc[a], yfit)
            classes = list(pipe.named_steps['model'].classes_)
            prob = pipe.predict_proba(X.iloc[b])[:, classes.index(1)]
            ids = [keys[i] for i in b]
            scores = evaluate_binary_risk(y.iloc[b], prob, ids, capacity)
            matrix = scores.pop('confusion_matrix')
            scores.update(tn=matrix[0][0], fp=matrix[0][1], fn=matrix[1][0], tp=matrix[1][1])
            scores.update(model=model, fold=fold, training_n=len(a),
                          training_class_ratio=class_ratio,
                          grade_knn_k=int(pipe.named_steps['grade_knn'].n_neighbors_),
                          probability_status='uncalibrated', setting='fixed_untuned_baseline',
                          estimator_parameters=json.dumps(estimator.get_params(), sort_keys=True, default=str))
            rows.append(scores)
            selected = select_top_risk(prob, ids, capacity)
            # Pseudonymous row key permits pairing without exporting raw household/person IDs.
            import hashlib
            for position, probability, chosen, key in zip(b, prob, selected, ids):
                predictions.append(dict(model=model, fold=fold, training_row=int(position),
                    child_key_sha256=hashlib.sha256(key.encode('utf8')).hexdigest(),
                    y_true=int(y.iloc[position]), probability=float(probability),
                    selected_top20_within_fold=bool(chosen)))
            if progress:
                print(f"{model} fold {fold}/5: AP={scores['average_precision']:.4f}, "
                      f"recall@capacity={scores['recall_at_capacity']:.4f}", flush=True)
    metrics, oof = pd.DataFrame(rows), pd.DataFrame(predictions)
    if oof.duplicated(['model', 'training_row']).any() or len(oof) != len(models) * len(train):
        raise RuntimeError("Incomplete or repeated out-of-fold predictions.")
    return metrics, oof


MODEL_NAMES = ("lr", "elasticnet", "xgb", "lightgbm", "catboost", "random_forest", "extra_trees")


def comparison_estimator(model, class_ratio, random_state=42):
    """Fixed exploratory settings, declared before scoring; no outer-fold tuning.

    All trees use the same existing one-hot/imputed representation. CatBoost
    native categorical handling is not tested in this controlled comparison.
    Shallow boosting limits interactions; forests use five-child leaves to
    temper memorisation in this small cohort. No early stopping uses validation.
    """
    if model in ("lr", "xgb"):
        return default_estimator(model, class_ratio, random_state)
    if model == "elasticnet":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(solver="saga", l1_ratio=.5, C=1.0,
                                  class_weight="balanced", max_iter=10000,
                                  random_state=random_state, tol=1e-4)
    if model == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=7,
                              max_depth=3, min_child_samples=20,
                              scale_pos_weight=class_ratio, random_state=random_state,
                              n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1)
    if model == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=300, depth=3, learning_rate=.05,
                                  loss_function="Logloss", class_weights=[1., class_ratio],
                                  random_seed=random_state, thread_count=1,
                                  verbose=False, allow_writing_files=False)
    if model in ("random_forest", "extra_trees"):
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
        cls = RandomForestClassifier if model == "random_forest" else ExtraTreesClassifier
        return cls(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                   class_weight="balanced", random_state=random_state, n_jobs=1)
    raise ValueError(f"Unknown model: {model}")


SENSITIVITY_VARIANTS = {
    "full": (True, True),
    "no_missing_flags": (False, True),
    "no_geography": (True, False),
    "neither": (False, False),
}


def evaluate_sensitivity(train, splits, capacity=0.20, progress=True):
    """Four prespecified variants, seven fixed configurations, identical folds.

    Missingness ablation removes the four explicit flags only; imputation and
    survey-status categories remain. Geography means region plus urban/rural,
    removed from both downstream inputs and grade-KNN matching. District IDs
    still define validation groups. No temporal observations are used.
    """
    splits = list(splits)
    results, predictions = [], []
    for variant, (flags, geography) in SENSITIVITY_VARIANTS.items():
        if progress:
            print(f"Variant: {variant}", flush=True)
        metrics, oof = evaluate_baselines(train, splits, capacity, progress,
            models=MODEL_NAMES, include_missing_flags=flags, include_geography=geography)
        metrics.insert(0, 'variant', variant)
        oof.insert(0, 'variant', variant)
        results.append(metrics)
        predictions.append(oof)
    return pd.concat(results, ignore_index=True), pd.concat(predictions, ignore_index=True)


def evaluate_region_only(train, splits, capacity=0.20, progress=True):
    """Paired fixed-model comparison removing only downstream region indicators.

    Urban/rural, missingness flags, grade-KNN matching and district folds stay
    unchanged. Region remains available to KNN, so this is not a test of complete
    geographic independence. No tuning, calibration or temporal scoring occurs.
    """
    splits = list(splits)
    results, predictions = [], []
    for variant, include_region in [("full", True), ("no_region_predictor", False)]:
        if progress:
            print(f"Region-only variant: {variant}", flush=True)
        metrics, oof = evaluate_baselines(train, splits, capacity, progress,
            models=MODEL_NAMES, include_region_predictor=include_region)
        metrics.insert(0, "variant", variant)
        oof.insert(0, "variant", variant)
        results.append(metrics)
        predictions.append(oof)
    return pd.concat(results, ignore_index=True), pd.concat(predictions, ignore_index=True)
