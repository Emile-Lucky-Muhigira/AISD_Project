"""
Builds the ElimuSentry preprocessing pipeline (Phase 1 report, Sections B-C).

Two versions share every step except the last ones:
  model="xgb"  -> no log transform, no scaling, all one-hot levels kept   (53 columns)
  model="lr"   -> log1p on long tails, StandardScaler, one level per
                  label dropped as the reference                          (48 columns)

Order of steps (Phase 1, "Order of steps that prevents leakage"):
  1. fixed recording-error rules            -> already applied in 01_build_dataset
  2. convert 2013/14 money to 2009/10 prices -> transformers.deflate_to_2009_prices
     (both are fixed rules, applied before the pipeline; nothing is learned)
  3. inside the pipeline, learned from the building part only:
     hours-missed rule -> 'was missing' flags -> grade by KNN (+ over-age)
     -> drop 3 repeating features -> recode categories -> ColumnTransformer
     (fill gaps, rank money, cap/log, encode, scale) -> model

Grade KNN selects k by inner district-grouped grade validation inside each fit.
Retain district_id as grouping metadata in X. It is excluded from the distance
and the final model inputs. Nominal category codes have no numerical ordering.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from .transformers import (MissedHoursRule, MissingFlagger, GradeKNNImputer,
                           ColumnDropper, CategoryRecoder, LogIQRCapper,
                           TrainingRank)

ID_COLS = ["FPrimary", "hhmid", "district_id", "eacode", "hhweight3"]
TARGET = "dropout"

# --- feature groups after the three repeating features are removed (30) ---
MEAN_NUM = ["english_score", "math_score"]                       # |skew| < 0.5
MEDIAN_NUM = ["age", "over_age", "attend_hours", "teacher_absent_days",
              "asset_count"]                                     # |skew| >= 0.5 or counts
CAP_NUM = ["travel_minutes", "hh_size", "rooms"]                 # log-IQR cap (B3)
LOG_ONLY = ["missed_hours"]                                      # log only (B3)
MONEY = ["school_spend", "durables_value"]                       # rank (C1)
ORDINAL = ["father_edu", "mother_edu", "has_textbooks"]          # mode
BINARY = ["sex", "urbrur", "on_vacation", "feeding_program", "father_alive",
          "mother_alive", "father_in_hh", "mother_in_hh", "electricity"]  # mode
NOMINAL = ["regioncode", "relationship_to_head", "math_status",
           "english_status"]                                     # mode, then one-hot
NOMINAL_MISSING = ["school_type"]                                # 'missing' option
FLAGS = ["teacher_absent_days_missing", "attend_hours_missing",
         "father_edu_missing", "mother_edu_missing"]

FEATURES_30 = (MEAN_NUM + MEDIAN_NUM + CAP_NUM + LOG_ONLY + MONEY + ORDINAL
               + BINARY + NOMINAL + NOMINAL_MISSING)


def _scaler(model):
    return StandardScaler() if model == "lr" else "passthrough"


def build_preprocessor(model="xgb"):
    """ColumnTransformer: every step is fitted on X_train only."""
    lr = model == "lr"
    sc = _scaler(model)
    drop = "first" if lr else None

    num_mean = Pipeline([("fill", SimpleImputer(strategy="mean")), ("scale", sc)])
    num_median = Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", sc)])
    num_cap = Pipeline([("fill", SimpleImputer(strategy="median")),
                        ("cap", LogIQRCapper(cap=True, log=lr)), ("scale", sc)])
    num_log = Pipeline([("fill", SimpleImputer(strategy="median")),
                        ("log", LogIQRCapper(cap=False, log=lr)), ("scale", sc)])
    money = Pipeline([("fill", SimpleImputer(strategy="median")),
                      ("rank", TrainingRank()), ("scale", sc)])
    ordinal = Pipeline([("fill", SimpleImputer(strategy="most_frequent")), ("scale", sc)])
    binary = SimpleImputer(strategy="most_frequent")
    nominal = Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(drop=drop, handle_unknown="ignore", sparse_output=False))])
    nominal_missing = Pipeline([("fill", SimpleImputer(strategy="constant",
                                                       fill_value="missing")),
                                ("onehot", OneHotEncoder(drop=drop, handle_unknown="ignore", sparse_output=False))])

    return ColumnTransformer([
        ("mean", num_mean, MEAN_NUM),
        ("median", num_median, MEDIAN_NUM),
        ("cap", num_cap, CAP_NUM),
        ("log", num_log, LOG_ONLY),
        ("money", money, MONEY),
        ("ordinal", ordinal, ORDINAL),
        ("binary", binary, BINARY),
        ("nominal", nominal, NOMINAL),
        ("school_type", nominal_missing, NOMINAL_MISSING),
        ("flags", "passthrough", FLAGS),
    ], remainder="drop", verbose_feature_names_out=False)


def build_pipeline(model="xgb", estimator=None):
    """Full pipeline: row-wise rules, then the ColumnTransformer, then the model.
    Pass estimator=None to get the preprocessing only (e.g. for PCA)."""
    steps = [
        ("missed_hours_rule", MissedHoursRule()),
        ("missing_flags", MissingFlagger()),
        ("grade_knn", GradeKNNImputer()),
        ("drop_repeating", ColumnDropper()),
        ("recode", CategoryRecoder()),
        ("prep", build_preprocessor(model)),
    ]
    if estimator is not None:
        steps.append(("model", estimator))
    return Pipeline(steps)


def default_estimator(model, scale_pos_weight=1.0, random_state=42):
    """The two models used only to check that the pipeline works end to end."""
    if model == "lr":
        return LogisticRegression(class_weight="balanced", max_iter=2000)
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
                         random_state=random_state, n_jobs=1)