"""Training-only OOF calibration, assessed on untouched outer validation folds.

Sigmoid (logistic recalibration of clipped log-odds) is prespecified primary;
isotonic is a small-sample sensitivity. Each calibration-OOF base model is tuned
inside its own training subset, excluding its calibration validation labels.
No temporal outcomes or previously saved outer OOF scores fit calibrators.
"""
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from .pipeline_builder import TARGET, build_pipeline
from .model_tuning import (TUNING_GRIDS, _validate_frame, _validate_splits,
                           tune_on_training, _fit_estimator, _probability)
from .evaluation_metrics import evaluate_binary_risk, select_top_risk, _probabilities


def _labels(y, p):
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != len(p) or not np.isin(y, [0, 1]).all():
        raise ValueError('Aligned binary labels required.')
    return y.astype(int)


def _log_odds(p):
    p = np.clip(_probabilities(p), 1e-6, 1-1e-6)
    return np.log(p/(1-p)).reshape(-1, 1)


def fit_calibrators(y, p):
    """Fit unweighted mappings using training OOF probabilities only."""
    p = _probabilities(p)
    y = _labels(y, p)
    if len(np.unique(y)) != 2:
        raise ValueError('Both classes required for calibration.')
    sigmoid = LogisticRegression(C=np.inf, solver='lbfgs', max_iter=2000)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        sigmoid.fit(_log_odds(p), y)
    isotonic = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(p, y)
    return {'sigmoid': sigmoid, 'isotonic': isotonic}


def apply_calibrator(method, calibrator, p):
    p = _probabilities(p)
    if method == 'sigmoid':
        return calibrator.predict_proba(_log_odds(p))[:, 1]
    if method == 'isotonic':
        return calibrator.predict(p)
    raise ValueError('Unknown calibration method.')


def reliability_bins(y, p, n_bins=10):
    """Equal-width [0,1] bins, counts and ECE; right endpoint 1 is in last bin.

    ECE = sum(n_bin/N * abs(mean_probability - observed_rate)). Empty bins have
    zero count and undefined means. ECE is bin-dependent, not a guarantee of
    calibrated individual risks. Counts support interpretation of sparse bins.
    """
    p = _probabilities(p)
    y = _labels(y, p)
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError('Positive integer bin count required.')
    bins = np.minimum((p*n_bins).astype(int), n_bins-1)
    rows, ece = [], 0.
    for i in range(n_bins):
        mask = bins == i
        count = int(mask.sum())
        mean_p = float(p[mask].mean()) if count else np.nan
        rate = float(y[mask].mean()) if count else np.nan
        if count: ece += count/len(y)*abs(mean_p-rate)
        rows.append(dict(bin=i+1, lower=i/n_bins, upper=(i+1)/n_bins,
                         n=count, events=int(y[mask].sum()), mean_probability=mean_p,
                         observed_rate=rate))
    return pd.DataFrame(rows), float(ece)


def calibration_training_scores(frame, model, outer_fold):
    """Independent OOF scores for fitting the calibration mapping.

    Hyperparameters are reselected within EACH calibration training partition,
    so calibration targets cannot enter even indirect base-model selection.
    """
    _validate_frame(frame)
    X, y = frame.drop(columns=TARGET), frame[TARGET]
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=342+outer_fold)
    splits = _validate_splits(frame, cv.split(X, y, frame.district_id), 3)
    result = np.full(len(frame), np.nan)
    details = []
    kind = 'lr' if model == 'elasticnet' else 'xgb'
    for inner_fold, (a, b) in enumerate(splits, start=1):
        best, search = tune_on_training(frame.iloc[a].copy(), model,
                                  random_state=1000+10*outer_fold+inner_fold)
        prep = build_pipeline(kind).fit(X.iloc[a])
        estimator, ratio = _fit_estimator(model, best, prep.transform(X.iloc[a]), y.iloc[a])
        result[b] = _probability(estimator, prep.transform(X.iloc[b]))
        details.append(dict(calibration_fold=inner_fold, fit_n=len(a), predict_n=len(b),
            fit_events=int(y.iloc[a].sum()), predict_events=int(y.iloc[b].sum()),
            training_class_ratio=ratio, best_parameters=best,
            numerical_weight_scale=float(getattr(estimator, "numerical_weight_scale_", 1.0)),
            failed_candidate_fits=search.loc[search.fit_status == "convergence_failure", ["candidate_id", "inner_fold", "parameters"]].to_dict("records"),
            shared_districts=0, shared_households=0))
    if not np.isfinite(result).all():
        raise RuntimeError('Incomplete calibration training predictions.')
    return result, details


def evaluate_calibration(train, outer_splits, capacity=.20, progress=True):
    """Return outer metrics, held-out probabilities and reliability bins.

    Compare uncalibrated/sigmoid/isotonic for each family. No winner is selected
    using outer results. A final model/calibrator is not fitted on the full data.
    """
    _validate_frame(train)
    splits = _validate_splits(train, outer_splits, 5)
    X, y = train.drop(columns=TARGET), train[TARGET]
    keys = [json.dumps([str(h), str(p)], separators=(',', ':'))
            for h, p in zip(train.FPrimary, train.hhmid)]
    metrics, predictions, curves = [], [], []
    for fold, (a, b) in enumerate(splits, start=1):
        for model in TUNING_GRIDS:
            train_part = train.iloc[a].copy()
            scores, details = calibration_training_scores(train_part, model, fold)
            calibrators = fit_calibrators(y.iloc[a], scores)
            # Reproduce the tuned outer baseline, without outer-label selection.
            best, _ = tune_on_training(train_part, model, random_state=142+fold)
            kind = 'lr' if model == 'elasticnet' else 'xgb'
            prep = build_pipeline(kind).fit(X.iloc[a])
            estimator, _ = _fit_estimator(model, best, prep.transform(X.iloc[a]), y.iloc[a])
            raw = _probability(estimator, prep.transform(X.iloc[b]))
            methods = {'uncalibrated': raw}
            methods.update({method: apply_calibrator(method, cal, raw) for method, cal in calibrators.items()})
            ids = [keys[i] for i in b]
            for method, prob in methods.items():
                result = evaluate_binary_risk(y.iloc[b], prob, ids, capacity)
                cm = result.pop('confusion_matrix')
                bins, ece = reliability_bins(y.iloc[b], prob)
                bins['model'], bins['method'], bins['fold'] = model, method, str(fold)
                curves.append(bins)
                result.update(model=model, method=method, fold=fold, ece_10_equal_width=ece,
                    mean_probability=float(prob.mean()), calibration_training_n=len(a),
                    calibration_training_events=int(y.iloc[a].sum()),
                    calibration_training_details=json.dumps(details, sort_keys=True),
                    best_parameters=json.dumps(best, sort_keys=True),
                    sigmoid_slope=float(calibrators['sigmoid'].coef_[0,0]),
                    sigmoid_intercept=float(calibrators['sigmoid'].intercept_[0]),
                    isotonic_knots=len(calibrators['isotonic'].X_thresholds_),
                    tn=cm[0][0], fp=cm[0][1], fn=cm[1][0], tp=cm[1][1])
                metrics.append(result)
                selected = select_top_risk(prob, ids, capacity)
                for pos, p, chosen, key in zip(b, prob, selected, ids):
                    predictions.append(dict(model=model, method=method, fold=fold, training_row=int(pos),
                        child_key_sha256=hashlib.sha256(key.encode('utf8')).hexdigest(),
                        y_true=int(y.iloc[pos]), probability=float(p), selected_top20_within_fold=bool(chosen)))
            if progress:
                print(f"Calibration outer fold {fold}/5 {model} completed", flush=True)
    oof = pd.DataFrame(predictions)
    if len(oof) != len(train)*len(TUNING_GRIDS)*3 or oof.duplicated(['model','method','training_row']).any():
        raise RuntimeError('Incomplete outer calibration predictions.')
    for (model, method), group in oof.groupby(['model','method']):
        bins, _ = reliability_bins(group.y_true, group.probability)
        bins['model'], bins['method'], bins['fold'] = model, method, 'pooled_oof'
        curves.append(bins)
    return pd.DataFrame(metrics), oof, pd.concat(curves, ignore_index=True)
