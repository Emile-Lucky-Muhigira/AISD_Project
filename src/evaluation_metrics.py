"""Binary dropout metrics and a fixed-capacity intervention policy.

No fitting or threshold tuning occurs here. Probabilities must be out-of-sample
for performance evaluation. PR-AUC is reported as average precision (AP), not
trapezoidal interpolation of the precision-recall curve.
"""
import hashlib
import math
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix


def _probabilities(probabilities):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("Probabilities must be a nonempty one-dimensional array.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite values between zero and one.")
    return p


def select_top_risk(probabilities, row_ids, capacity=0.20):
    """Select floor(capacity * N) children, never exceeding available capacity.

    Supply unique, stable, nonempty string keys (household + person ID), never
    outcome labels. Descending probability determines priority; SHA-256 of the
    key breaks exact ties reproducibly without relying on input row order.
    IDs are used for ties only, not as model predictors. The key itself resolves
    the theoretical hash-collision case. Zero available places selects nobody.
    """
    p = _probabilities(probabilities)
    if not np.isfinite(capacity) or not 0 < capacity <= 1:
        raise ValueError("Capacity must be greater than zero and at most one.")
    keys = list(row_ids)
    if len(keys) != len(p) or any(not isinstance(k, str) or not k for k in keys):
        raise ValueError("Provide one nonempty string ID per observation.")
    if len(set(keys)) != len(keys):
        raise ValueError("Row IDs must be unique.")
    count = math.floor(capacity * len(p))
    hashes = [hashlib.sha256(k.encode('utf-8')).hexdigest() for k in keys]
    order = sorted(range(len(p)), key=lambda i: (-p[i], hashes[i], keys[i]))
    chosen = np.zeros(len(p), dtype=bool)
    chosen[order[:count]] = True
    return chosen


def evaluate_binary_risk(y_true, probabilities, row_ids, capacity=0.20):
    """Return discrimination, probability error and capacity-policy metrics.

    Confusion matrix rows are actual [0, 1], columns selected [0, 1]. Undefined
    recall/AP with no positives, or precision with zero places, are NaN rather
    than misleading zeroes. Report cohort size and event counts with all scores.
    A cohort-specific top fraction is a ranking policy, not a fixed probability
    cutoff. Its realized cutoff can differ by cohort and can contain ties.
    """
    p = _probabilities(probabilities)
    y = np.asarray(y_true)
    if y.ndim != 1 or len(y) != len(p) or not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be aligned, one-dimensional binary 0/1 values.")
    y = y.astype(int)
    chosen = select_top_risk(p, row_ids, capacity)
    n, events, places = len(y), int(y.sum()), int(chosen.sum())
    tp = int(y[chosen].sum())
    precision = tp / places if places else float('nan')
    recall = tp / events if events else float('nan')
    prevalence = events / n
    return {
        'n': n, 'events': events, 'prevalence': prevalence,
        'average_precision': float(average_precision_score(y, p)) if events else float('nan'),
        'brier_score': float(brier_score_loss(y, p)),
        'requested_capacity': capacity, 'selected_n': places,
        'selected_fraction': places / n,
        'precision_at_capacity': precision, 'recall_at_capacity': recall,
        'lift_at_capacity': precision / prevalence if events and places else float('nan'),
        'random_expected_precision': prevalence if places else float('nan'),
        'random_expected_recall': places / n if events else float('nan'),
        'lowest_selected_probability': float(p[chosen].min()) if places else float('nan'),
        'confusion_matrix': confusion_matrix(y, chosen.astype(int), labels=[0, 1]).tolist(),
    }
