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
from sklearn.impute import KNNImputer


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
    """B1: on the 2009/10 paper forms a blank 'hours missed' most likely meant
    0 hours. Set it to 0 when attendance was recorded; leave it missing
    otherwise (it is filled with the median later). Learns nothing."""

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
    """B1: estimate a missing school grade from the 5 most similar children
    (age, sex, urban/rural, region, key assets), then recalculate years
    over-age = age - grade - 5. The neighbours are searched only among the
    training rows seen in fit()."""

    def __init__(self, n_neighbors=5,
                 by=("age", "sex", "urbrur", "regioncode", "asset_count")):
        self.n_neighbors = n_neighbors
        self.by = by

    def fit(self, X, y=None):
        cols = list(self.by) + ["grade_level"]
        Z = X[cols].astype(float)
        # scale the matching features so no single one dominates the distance
        self.means_ = Z[list(self.by)].mean()
        self.stds_ = Z[list(self.by)].std().replace(0, 1)
        self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors)
        self.imputer_.fit(self._scaled(Z))
        return self

    def _scaled(self, Z):
        Z = Z.copy()
        by = list(self.by)
        Z[by] = (Z[by] - self.means_) / self.stds_
        # matching features may themselves have rare gaps; use the training mean (0)
        Z[by] = Z[by].fillna(0.0)
        return Z

    def transform(self, X):
        X = X.copy()
        cols = list(self.by) + ["grade_level"]
        filled = self.imputer_.transform(self._scaled(X[cols].astype(float)))
        grade = np.round(filled[:, -1])
        missing = X["grade_level"].isna()
        X.loc[missing, "grade_level"] = grade[missing.to_numpy()]
        recompute = X["over_age"].isna() & X["grade_level"].notna()
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