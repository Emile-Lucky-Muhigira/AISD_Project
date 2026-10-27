"""Nested district-grouped tuning, full features, training data only.

Four prespecified candidates per family. Selection uses mean inner-fold AP;
outer-fold labels are never passed to the search. All probabilities remain
uncalibrated. Previously explored outer folds limit confirmatory interpretation.
"""
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from .pipeline_builder import TARGET, build_pipeline
from .model_evaluation import comparison_estimator
from .evaluation_metrics import evaluate_binary_risk, select_top_risk


TUNING_GRIDS = {
    'elasticnet': [{'C': c, 'l1_ratio': ratio} for c in (.1, 1.) for ratio in (.5, 1.)],
    'xgb': [{'max_depth': d, 'min_child_weight': w} for d in (2, 3) for w in (1, 5)],
    'lightgbm': [{'max_depth': d, 'num_leaves': leaves, 'min_child_samples': n}
                 for d, leaves in [(2, 3), (3, 7)] for n in (10, 20)],
    'catboost': [{'depth': d, 'l2_leaf_reg': r} for d in (2, 3) for r in (3., 10.)],
    'random_forest': [{'min_samples_leaf': n} for n in (3, 5, 10, 20)],
    'extra_trees': [{'min_samples_leaf': n} for n in (3, 5, 10, 20)],
}


def _validate_frame(frame):
    if frame.empty or frame[['FPrimary', 'hhmid', 'district_id', TARGET]].isna().any().any():
        raise ValueError('Nonempty data with known identity, district and outcome required.')
    if frame.duplicated(['FPrimary', 'hhmid']).any() or not frame[TARGET].isin([0, 1]).all():
        raise ValueError('Unique children and binary labels required.')


def _validate_splits(frame, splits, expected):
    splits = [(np.asarray(a), np.asarray(b)) for a, b in splits]
    if len(splits) != expected:
        raise ValueError(f'Expected {expected} folds.')
    seen = np.zeros(len(frame), dtype=int)
    for a, b in splits:
        for idx in [a, b]:
            if (idx.ndim != 1 or idx.dtype.kind not in 'iu' or not len(idx)
                    or idx.min() < 0 or idx.max() >= len(frame)
                    or len(np.unique(idx)) != len(idx)):
                raise ValueError('Invalid fold positions.')
        if set(a) & set(b) or len(set(a) | set(b)) != len(frame):
            raise ValueError('Each fold must partition the supplied training data.')
        for col in ['district_id', 'FPrimary']:
            if set(frame.iloc[a][col]) & set(frame.iloc[b][col]):
                raise ValueError(f'Group overlap: {col}')
        if frame.iloc[a][TARGET].nunique() != 2 or frame.iloc[b][TARGET].nunique() != 2:
            raise ValueError('Both classes required in every training/validation partition.')
        seen[b] += 1
    if not np.all(seen == 1):
        raise ValueError('Each row must be validated exactly once.')
    return splits


def _fit_estimator(model, parameters, X, y):
    ratio = float((y == 0).sum() / (y == 1).sum())
    estimator = comparison_estimator(model, ratio).set_params(**parameters)
    # More deeply nested calibration partitions can require more solver steps.
    # Retry numerical convergence only; preserve the objective, penalty mix, tolerance and data.
    fit_kwargs = {}
    estimator.numerical_weight_scale_ = 1.0
    for attempt in range(2):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                estimator.fit(X, y, **fit_kwargs)
            return estimator, ratio
        except ConvergenceWarning:
            params = estimator.get_params() if hasattr(estimator, 'get_params') else {}
            if model != 'elasticnet' or 'max_iter' not in params or attempt == 1:
                raise
            # Balanced SAGA weights can exceed one. Scaling weights by M and
            # C by M preserves sum(w*loss) + penalty/C up to factor 1/M,
            # while bounding each observation's weight for the numerical solver.
            weights = np.where(np.asarray(y) == 1, len(y)/(2*(y == 1).sum()),
                               len(y)/(2*(y == 0).sum()))
            scale = float(weights.max())
            estimator.set_params(max_iter=params['max_iter'] * 5,
                                 class_weight=None, C=params['C'] * scale)
            estimator.numerical_weight_scale_ = scale
            fit_kwargs = {'sample_weight': weights / scale}
    raise RuntimeError('Unreachable convergence retry state')


def _probability(estimator, X):
    p = estimator.predict_proba(X)[:, list(estimator.classes_).index(1)]
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid predicted probabilities.')
    return p


def tune_on_training(frame, model, candidates=None, inner_splits=3, random_state=142):
    """Search only the supplied training partition; return parameters + audit.

    Preprocessing is fitted once per inner-training partition and reused across
    candidates because only downstream estimator parameters vary. It is never
    fitted across validation boundaries. Grade KNN performs its own internal
    grouped selection within each preprocessing fit.
    """
    _validate_frame(frame)
    if model not in TUNING_GRIDS:
        raise ValueError('Unsupported tuning family.')
    candidates = TUNING_GRIDS[model] if candidates is None else candidates
    if not candidates or any(not isinstance(c, dict) for c in candidates):
        raise ValueError('At least one parameter dictionary required.')
    # Only the declared hyperparameters may vary; no IDs, preprocessing, weights or seeds.
    allowed = set().union(*(set(c) for c in TUNING_GRIDS[model]))
    if any(set(c) - allowed for c in candidates):
        raise ValueError('Candidate contains an undeclared hyperparameter.')
    X, y = frame.drop(columns=TARGET), frame[TARGET]
    splitter = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
    splits = _validate_splits(frame, splitter.split(X, y, frame.district_id), inner_splits)
    kind = 'lr' if model == 'elasticnet' else 'xgb'
    rows = []
    for inner_fold, (a, b) in enumerate(splits, start=1):
        prep = build_pipeline(kind).fit(X.iloc[a])
        fit_X, valid_X = prep.transform(X.iloc[a]), prep.transform(X.iloc[b])
        for candidate_id, parameters in enumerate(candidates):
            ratio = float((y.iloc[a] == 0).sum() / (y.iloc[a] == 1).sum())
            status = 'converged'
            try:
                estimator, ratio = _fit_estimator(model, parameters, fit_X, y.iloc[a])
                ap = float(average_precision_score(y.iloc[b], _probability(estimator, valid_X)))
            except ConvergenceWarning:
                # An unconverged candidate is never scored or selected.
                ap, status = np.nan, 'convergence_failure'
            rows.append(dict(model=model, candidate_id=candidate_id, inner_fold=inner_fold,
                parameters=json.dumps(parameters, sort_keys=True), inner_average_precision=ap, fit_status=status,
                numerical_weight_scale=float(getattr(estimator, 'numerical_weight_scale_', 1.0)) if status == 'converged' else np.nan,
                training_n=len(a), validation_n=len(b), training_events=int(y.iloc[a].sum()),
                validation_events=int(y.iloc[b].sum()), training_class_ratio=ratio,
                grade_knn_k=int(prep.named_steps['grade_knn'].n_neighbors_),
                shared_districts=0, shared_households=0))
    audit = pd.DataFrame(rows)
    eligible = audit.groupby('candidate_id').inner_average_precision.count().eq(inner_splits)
    means = audit.groupby('candidate_id').inner_average_precision.mean().loc[eligible]
    if means.empty:
        raise RuntimeError(f'No {model} candidate converged in every inner fold.')
    audit['eligible'] = audit.candidate_id.map(eligible)
    # Stable candidate order resolves exact ties; no outer performance enters selection.
    best = min(means.index, key=lambda i: (-means.loc[i], i))
    audit['mean_inner_ap'] = audit.candidate_id.map(means)
    audit['selected'] = audit.candidate_id.eq(best)
    return dict(candidates[best]), audit


def evaluate_nested_tuning(train, outer_splits, capacity=.20, progress=True):
    """Six family-specific nested estimates; no overall winner is selected here."""
    _validate_frame(train)
    splits = _validate_splits(train, outer_splits, expected=5)
    X, y = train.drop(columns=TARGET), train[TARGET]
    keys = [json.dumps([str(h), str(p)], separators=(',', ':'))
            for h, p in zip(train.FPrimary, train.hhmid)]
    searches, metrics, predictions = [], [], []
    for fold, (a, b) in enumerate(splits, start=1):
        for model in TUNING_GRIDS:
            best, audit = tune_on_training(train.iloc[a].copy(), model, random_state=142 + fold)
            audit.insert(0, 'outer_fold', fold)
            searches.append(audit)
            # Refit selected settings on the entire outer training partition.
            kind = 'lr' if model == 'elasticnet' else 'xgb'
            prep = build_pipeline(kind).fit(X.iloc[a])
            estimator, ratio = _fit_estimator(model, best, prep.transform(X.iloc[a]), y.iloc[a])
            prob = _probability(estimator, prep.transform(X.iloc[b]))
            ids = [keys[i] for i in b]
            result = evaluate_binary_risk(y.iloc[b], prob, ids, capacity)
            cm = result.pop('confusion_matrix')
            result.update(model=model, fold=fold, best_parameters=json.dumps(best, sort_keys=True),
                mean_inner_ap=float(audit.loc[audit.selected, 'mean_inner_ap'].iloc[0]),
                training_n=len(a), training_class_ratio=ratio,
                estimator_parameters=json.dumps(estimator.get_params(), sort_keys=True, default=str),
                grade_knn_k=int(prep.named_steps['grade_knn'].n_neighbors_),
                probability_status='uncalibrated', feature_variant='full',
                numerical_weight_scale=float(getattr(estimator, 'numerical_weight_scale_', 1.0)),
                tn=cm[0][0], fp=cm[0][1], fn=cm[1][0], tp=cm[1][1])
            metrics.append(result)
            selected = select_top_risk(prob, ids, capacity)
            for pos, p, chosen, key in zip(b, prob, selected, ids):
                predictions.append(dict(model=model, fold=fold, training_row=int(pos),
                    child_key_sha256=hashlib.sha256(key.encode('utf8')).hexdigest(),
                    y_true=int(y.iloc[pos]), probability=float(p), selected_top20_within_fold=bool(chosen)))
            if progress:
                print(f"Outer fold {fold}/5 {model}: selected {best}; AP={result['average_precision']:.4f}", flush=True)
    oof = pd.DataFrame(predictions)
    if len(oof) != len(train)*len(TUNING_GRIDS) or oof.duplicated(['model', 'training_row']).any():
        raise RuntimeError('Incomplete out-of-fold predictions.')
    return pd.concat(searches, ignore_index=True), pd.DataFrame(metrics), oof
