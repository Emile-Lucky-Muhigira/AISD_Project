"""Training-only, nested district-grouped validation of school-grade imputation.

Known validation grades are hidden together with their derived over-age values.
This evaluates recovery of observed grades, not the unknowable true values of
the genuinely missing grades. Dropout and the temporal test are never inputs.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import GroupKFold

from .transformers import GradeKNNImputer


def _legacy_prediction(fit, query):
    """Frozen legacy numeric-code k=5 reference; comparison only."""
    by = ["age", "sex", "urbrur", "regioncode", "asset_count"]
    cols = by + ["grade_level"]
    means = fit[by].mean()
    stds = fit[by].std().replace(0, 1)

    def scaled(frame):
        z = frame[cols].astype(float).copy()
        z[by] = ((z[by] - means) / stds).fillna(0)
        return z

    model = KNNImputer(n_neighbors=5).fit(scaled(fit))
    return np.round(model.transform(scaled(query))[:, -1])


def evaluate_grade_knn(train, output_dir):
    """Write reproducible validation evidence using Wave 1 training rows only."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Exclude dropout even if callers supplied it; grouping is not a predictor.
    features = ["age", "sex", "urbrur", "regioncode", "asset_count",
                "grade_level", "over_age", "district_id"]
    data = train[features].copy()
    rows, errors, inner_rows = [], {}, []
    for fold, (a, b) in enumerate(GroupKFold(n_splits=5).split(data, groups=data.district_id), 1):
        fit = data.iloc[a]
        observed = data.iloc[b].loc[data.iloc[b].grade_level.notna()]
        truth = observed.grade_level.to_numpy(dtype=float)
        hidden = observed.copy()
        hidden[["grade_level", "over_age"]] = np.nan
        assert not set(fit.district_id) & set(hidden.district_id)
        model = GradeKNNImputer().fit(fit)
        predictions = {
            "Training-median baseline": np.repeat(np.rint(fit.grade_level.median()), len(hidden)),
            "Legacy numeric-code KNN (k=5)": _legacy_prediction(fit, hidden),
            "Mixed-distance KNN (inner-selected k)": model.transform(hidden).grade_level.to_numpy(dtype=float),
        }
        for record in model.selection_results_:
            inner_rows.append({"outer fold": fold, **record})
        for name, predicted in predictions.items():
            error = predicted - truth
            errors.setdefault(name, []).extend(error.tolist())
            rows.append({"outer fold": fold, "method": name,
                "training children": len(fit), "training donors": int(fit.grade_level.notna().sum()),
                "held-out observed grades": len(truth), "shared districts": 0,
                "selected k": model.n_neighbors_ if name.startswith("Mixed") else (5 if name.startswith("Legacy") else np.nan),
                "MAE (grades)": float(np.abs(error).mean()),
                "RMSE (grades)": float(np.sqrt(np.mean(error**2))),
                "exact grade %": float((error == 0).mean()*100),
                "within one grade %": float((np.abs(error) <= 1).mean()*100)})
    validation = pd.DataFrame(rows)
    validation.to_csv(output_dir / "B1_knn_validation.csv", index=False)
    pd.DataFrame(inner_rows).to_csv(output_dir / "B1_knn_inner_folds.csv", index=False)
    summary_rows = []
    for name, values in errors.items():
        error = np.asarray(values)
        summary_rows.append({"method": name, "held-out observed grades": len(error),
            "MAE (grades)": float(np.abs(error).mean()),
            "RMSE (grades)": float(np.sqrt(np.mean(error**2))),
            "exact grade %": float((error == 0).mean()*100),
            "within one grade %": float((np.abs(error) <= 1).mean()*100),
            "fold MAE SD": float(validation.loc[validation.method == name, "MAE (grades)"].std(ddof=1))})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "B1_knn_summary.csv", index=False)
    final = GradeKNNImputer().fit(data)
    selection = pd.DataFrame([{"k": k, "inner grade MAE": mae, "selected": k == final.n_neighbors_}
                              for k, mae in final.k_scores_.items()])
    selection.to_csv(output_dir / "B1_knn_k_selection.csv", index=False)
    return {"summary": summary, "outer_folds": validation,
            "full_training_k": final.n_neighbors_, "selection": selection}
