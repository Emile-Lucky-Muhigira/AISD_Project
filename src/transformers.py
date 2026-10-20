"""
Custom scikit-learn transformers for the ElimuSentry preprocessing pipeline.

Every class follows the scikit-learn fit/transform contract:
  - fit(X) learns settings from the training rows only (X_train, or the
    building part of a cross-validation fold);
  - transform(X) applies those stored settings to any rows (checking or test)
    without learning anything new.
This is what keeps the pipeline free of leakage (Phase 1 report, Section C2e).

Each class refers to the Phase 1 report section that planned it.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupKFold
from sklearn.utils.validation import check_is_fitted


# ---------------------------------------------------------------------------
# Fixed rule applied OUTSIDE the pipeline (nothing is learned from data)
# ---------------------------------------------------------------------------
# Inflation between the two survey visits (Phase 1, C1 "Money and inflation").
# World Bank annual average consumer price inflation for Ghana, index 2009 = 100,
# weighted by the months in which interviews actually took place:
#   2009/10 visit (Nov 2009 - Apr 2010)  -> 107.5
#   "2013/14" visit (Mar 2014 - mid 2015) -> 174.0
CPI_FACTOR_W2_TO_W1 = 174.0 / 107.5      # about 1.62
MONEY_COLS = ["school_spend", "durables_value"]


def deflate_to_2009_prices(df, factor=CPI_FACTOR_W2_TO_W1, cols=MONEY_COLS):
    """Convert 2013/14 cedi amounts to 2009/10 prices (a fixed rule)."""
    out = df.copy()
    for c in cols:
        out[c] = out[c] / factor
    return out


# ---------------------------------------------------------------------------
# Row-wise steps that must run in order, before the ColumnTransformer
# ---------------------------------------------------------------------------
class MissedHoursRule(BaseEstimator, TransformerMixin):
    """B1: assume blank missed hours mean zero when attendance was recorded.
    This is an imputation assumption, not a recovered survey answer. Compare
    against median filling inside CV; otherwise use the later median fill.
    Learns nothing."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        rule = X["missed_hours"].isna() & X["attend_hours"].notna()
        X.loc[rule, "missed_hours"] = 0.0
        return X


class MissingFlagger(BaseEstimator, TransformerMixin):
    """B1: add a yes/no 'was missing' flag for the four features whose gap
    carries information (questionnaire design, family situation, or a link
    with dropout). Must run before any filling. Learns nothing."""

    def __init__(self, columns=("teacher_absent_days", "attend_hours",
                                "father_edu", "mother_edu")):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.columns:
            X[f"{c}_missing"] = X[c].isna().astype(int)
        return X


class GradeKNNImputer(BaseEstimator, TransformerMixin):
    """Active grade imputation with a mixed numeric/categorical distance.

    Each matching variable has equal weight. Numeric age/assets contribute
    absolute differences divided by a range fitted on donors only (bounded
    at 1); sex, urban/rural and region contribute 0 for a match, 1 otherwise.
    Missing comparisons are omitted rather than replaced with a category or
    a fictitious mean. All ties at the kth distance are retained, so donor
    row order cannot break ties. Grades are averaged, rounded, and bounded
    to the valid 0--9 scale. Observed grades are preserved.

    With n_neighbors='auto', k is selected by grade MAE in district-grouped
    inner folds of fit(X) only. Neither dropout nor over_age is a matching
    variable; a query's grade never enters its distance. Group IDs must be
    retained in X for tuning, but never enter the distance or model output.
    """

    def __init__(self, n_neighbors="auto",
                 by=("age", "sex", "urbrur", "regioncode", "asset_count"),
                 numeric=("age", "asset_count"),
                 neighbor_grid=(3, 5, 9, 15, 25), selection_splits=3,
                 group_column="district_id"):
        self.n_neighbors = n_neighbors
        self.by = by
        self.numeric = numeric
        self.neighbor_grid = neighbor_grid
        self.selection_splits = selection_splits
        self.group_column = group_column

    def fit(self, X, y=None):
        forbidden = {"grade_level", "over_age", "dropout", "FPrimary", "hhmid",
                     "district_id", "eacode", "hhweight3", self.group_column}
        if forbidden & set(self.by):
            raise ValueError("Grade, grade-derived fields, dropout and group IDs cannot define neighbours")
        if not self.by or not set(self.numeric).issubset(self.by):
            raise ValueError("numeric must be a subset of nonempty by")
        grade = pd.to_numeric(X["grade_level"], errors="coerce")
        if (X["grade_level"].notna() & grade.isna()).any():
            raise ValueError("Observed grades must be numeric on the 0--9 school-grade scale")
        known = grade.notna()
        if not known.any():
            raise ValueError("Cannot impute school grade without any observed training grades")
        if not grade[known].isin(range(10)).all():
            raise ValueError("Observed grades must be integers on the 0--9 school-grade scale")
        donors = X.loc[known, list(self.by)].reset_index(drop=True)
        grades = grade[known].to_numpy(dtype=float)
        self.selection_results_ = []
        if self.n_neighbors == "auto":
            candidates = sorted(set(self.neighbor_grid))
            if not candidates or any(not isinstance(k, (int, np.integer)) or k < 1 for k in candidates):
                raise ValueError("neighbor_grid must contain positive integers")
            if self.group_column not in X:
                raise ValueError(f"Automatic k selection requires {self.group_column}; retain the grouping ID in X")
            groups = X.loc[known, self.group_column].reset_index(drop=True)
            if groups.isna().any():
                raise ValueError("Automatic k selection requires nonmissing training group IDs")
            if self.selection_splits < 2:
                raise ValueError("selection_splits must be at least 2")
            n_splits = min(self.selection_splits, groups.nunique())
            if n_splits < 2:
                self.n_neighbors_ = min(5, len(donors))
                self.selection_status_ = "Insufficient groups: fixed fallback k=min(5, donor count)"
            else:
                errors = {k: [] for k in candidates}
                for fold, (a, b) in enumerate(GroupKFold(n_splits=n_splits).split(donors, groups=groups), 1):
                    fit_donors, query = donors.iloc[a], donors.iloc[b]
                    ranges = self._ranges(fit_donors)
                    distances = self._distances(query, fit_donors, ranges)
                    fallback = float(np.rint(np.median(grades[a])))
                    for k in candidates:
                        pred = self._predict(distances, grades[a], k, fallback)
                        absolute = np.abs(pred - grades[b])
                        errors[k].extend(absolute.tolist())
                        self.selection_results_.append({"fold": fold, "k": k,
                            "fit donors": len(a), "validation grades": len(b),
                            "shared districts": len(set(groups.iloc[a]) & set(groups.iloc[b])),
                            "grade MAE": float(absolute.mean())})
                # Weighted by the number of observed held-out grades, not fold size.
                self.k_scores_ = {k: float(np.mean(v)) for k, v in errors.items()}
                best = min(self.k_scores_.values())
                # Prefer the smoother estimate only when scores tie exactly.
                self.n_neighbors_ = max(k for k, score in self.k_scores_.items()
                                        if abs(score - best) <= 1e-12)
                self.selection_status_ = f"Selected by {n_splits}-fold district-grouped grade MAE"
        else:
            if not isinstance(self.n_neighbors, (int, np.integer)) or self.n_neighbors < 1:
                raise ValueError("n_neighbors must be 'auto' or a positive integer")
            self.n_neighbors_ = int(self.n_neighbors)
            self.selection_status_ = "Fixed k supplied by caller"
        self.donors_ = donors.copy(deep=True)
        self.donor_grades_ = grades.copy()
        self.ranges_ = self._ranges(donors)
        self.fallback_grade_ = float(np.rint(np.median(grades)))
        return self

    def _ranges(self, donors):
        ranges = {}
        for col in self.numeric:
            x = pd.to_numeric(donors[col], errors="coerce")
            width = x.max() - x.min()
            ranges[col] = float(width) if pd.notna(width) and width > 0 else 1.0
        return ranges

    def _distances(self, query, donors, ranges):
        shape = (len(query), len(donors))
        total = np.zeros(shape, dtype=float)
        count = np.zeros(shape, dtype=int)
        for col in self.by:
            if col in self.numeric:
                a = pd.to_numeric(query[col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                b = pd.to_numeric(donors[col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                valid = np.isfinite(a[:, None]) & np.isfinite(b[None, :])
                delta = np.minimum(np.abs(a[:, None] - b[None, :]) / ranges[col], 1.0)
            else:
                a = query[col].to_numpy(dtype=object)
                b = donors[col].to_numpy(dtype=object)
                valid = pd.notna(a[:, None]) & pd.notna(b[None, :])
                # Only equality matters: labels are never subtracted or ranked.
                a = np.where(pd.notna(a), a, "__missing_query__")
                b = np.where(pd.notna(b), b, "__missing_donor__")
                delta = (a[:, None] != b[None, :]).astype(float)
            total += np.where(valid, delta, 0.0)
            count += valid
        return np.divide(total, count, out=np.full(shape, np.inf), where=count > 0)

    @staticmethod
    def _predict(distances, grades, k, fallback):
        kth = min(k, len(grades)) - 1
        cutoff = np.partition(distances, kth, axis=1)[:, kth]
        chosen = np.isfinite(distances) & (distances <= cutoff[:, None] + 1e-12)
        n = chosen.sum(axis=1)
        average = np.divide(chosen @ grades, n, out=np.full(len(distances), fallback), where=n > 0)
        return np.clip(np.rint(average), 0, 9)

    def transform(self, X):
        check_is_fitted(self, "donors_")
        X = X.copy()
        missing = X["grade_level"].isna()
        if missing.any():
            distances = self._distances(X.loc[missing, list(self.by)], self.donors_, self.ranges_)
            X.loc[missing, "grade_level"] = self._predict(distances, self.donor_grades_,
                                                         self.n_neighbors_, self.fallback_grade_)
        recompute = (missing | X["over_age"].isna()) & X["grade_level"].notna()
        X.loc[recompute, "over_age"] = (X.loc[recompute, "age"]
                                        - X.loc[recompute, "grade_level"] - 5)
        return X


class ColumnDropper(BaseEstimator, TransformerMixin):
    """B4b: remove the three features that repeat others (33 -> 30)."""

    def __init__(self, columns=("persons_per_room", "orphan", "grade_level")):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.columns if c in X.columns])


class CategoryRecoder(BaseEstimator, TransformerMixin):
    """C1: fixed recodings that need no learning.
    - yes/no items coded 1/2 become 0/1 (sex: 1 = female; urbrur: 1 = rural;
      feeding_program: 1 = in a programme);
    - has_textbooks is reversed so that a higher number means more books
      (none 0, some 1, all 2);
    - relationship to head is grouped into own child / grandchild / other."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["sex"] = X["sex"].map({1: 0, 2: 1})                 # 1 = female
        X["urbrur"] = X["urbrur"].map({1: 0, 2: 1})           # 1 = rural
        X["feeding_program"] = X["feeding_program"].map({1: 1, 2: 0})
        X["has_textbooks"] = X["has_textbooks"].map({1: 2, 2: 1, 3: 0})
        X["relationship_to_head"] = self._to_labels(
            X["relationship_to_head"],
            lambda v: "own_child" if v == 3 else ("grandchild" if v == 4 else "other"))
        X["school_type"] = self._to_labels(X["school_type"], lambda v: f"type_{int(v)}")
        X["regioncode"] = self._to_labels(X["regioncode"], lambda v: f"region_{int(v)}")
        for c in ("math_status", "english_status"):
            X[c] = self._to_labels(X[c], str)
        return X

    @staticmethod
    def _to_labels(series, rule):
        """Apply a labelling rule, keeping missing values as np.nan (object dtype)."""
        return pd.Series([np.nan if pd.isna(v) else rule(v) for v in series],
                         index=series.index, dtype=object)


# ---------------------------------------------------------------------------
# Column-wise steps used inside the ColumnTransformer
# ---------------------------------------------------------------------------
class LogIQRCapper(BaseEstimator, TransformerMixin):
    """B2/B3: cap the high end at the IQR fence computed on log1p(x) of the
    training data, converted back to original units. Optionally return the
    log1p of the capped value (used for logistic regression and PCA only).
    No row is ever deleted."""

    def __init__(self, cap=True, log=False):
        self.cap = cap
        self.log = log

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        lx = np.log1p(X)
        q1 = np.nanpercentile(lx, 25, axis=0)
        q3 = np.nanpercentile(lx, 75, axis=0)
        self.upper_ = np.expm1(q3 + 1.5 * (q3 - q1))
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        if self.cap:
            X = np.minimum(X, self.upper_)
        if self.log:
            X = np.log1p(X)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class TrainingRank(BaseEstimator, TransformerMixin):
    """C1: turn a money amount into its rank (0 = lowest, 1 = highest) among
    the training values seen in fit(). New values are placed within that
    stored reference, so the test children never shape the scale."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.reference_ = [np.sort(col[~np.isnan(col)]) for col in X.T]
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        out = np.empty_like(X)
        for j, ref in enumerate(self.reference_):
            col = X[:, j]
            lo = np.searchsorted(ref, col, side="left")
            hi = np.searchsorted(ref, col, side="right")
            out[:, j] = (lo + hi) / 2.0 / len(ref)       # mid-rank, 0..1
            out[np.isnan(col), j] = np.nan
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)
