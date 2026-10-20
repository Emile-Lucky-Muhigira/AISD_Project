"""Fixed identity-based temporal evaluation cohorts; targets never define masks."""
import hashlib
import json
from pathlib import Path

import pandas as pd


def temporal_masks(train, test):
    keys = set(zip(train["FPrimary"], train["hhmid"]))
    repeated = pd.Series([key in keys for key in zip(test["FPrimary"], test["hhmid"])], index=test.index)
    known_household = test["FPrimary"].isin(set(train["FPrimary"]))
    return pd.DataFrame({"in_both_tables": repeated,
                         "in_training_household": known_household,
                         "in_primary_test": ~known_household & ~repeated,
                         "in_secondary_test": ~repeated}, index=test.index)


def cohort_audit(train, test):
    masks = temporal_masks(train, test)
    people = set(zip(train.FPrimary, train.hhmid));households = set(train.FPrimary)
    rows = []
    cohorts = [("All temporal records (descriptive)", test),
               ("Primary: new households", test.loc[masks.in_primary_test]),
               ("Secondary: new people", test.loc[masks.in_secondary_test])]
    for label, frame in cohorts:
        positives = int(frame.dropout.sum()) if "dropout" in frame else None
        rows.append({"cohort": label, "rows": len(frame), "households": frame.FPrimary.nunique(),
                     "dropouts": positives,
                     "dropout %": 100*positives/len(frame) if positives is not None and len(frame) else None,
                     "shared people": len(people & set(zip(frame.FPrimary, frame.hhmid))),
                     "shared households": len(households & set(frame.FPrimary)),
                     "shared districts": len(set(train.district_id) & set(frame.district_id)),
                     "train % of effective train+test": 100*len(train)/(len(train)+len(frame)),
                     "test % of effective train+test": 100*len(frame)/(len(train)+len(frame))})
    assert rows[1]["shared people"] == rows[1]["shared households"] == 0
    assert rows[2]["shared people"] == 0
    return pd.DataFrame(rows)


def write_evaluation_protocol(root, train, test):
    root = Path(root)
    audit = cohort_audit(train, test)
    audit.to_csv(root/"phase1_strategy/C2_temporal_cohorts.csv", index=False)
    protocol = {
        "status": "Retrospective temporal evaluation; aggregates and outcomes were inspected during development",
        "primary": "Wave 2->3 children from households absent from Wave 1->2 training; no repeated people",
        "secondary": "Wave 2->3 children absent as people from training; shared households explicitly allowed",
        "geographic_scope": "Later children in partly familiar districts; not a new-district external test",
        "mask_inputs": ["FPrimary", "hhmid"],
        "outer_training_validation": "5-fold StratifiedGroupKFold, district_id, shuffle=True, random_state=42",
        "grade_imputation": "Mixed numeric/categorical distance; k in 3,5,9,15,25 selected by 3 inner district-grouped folds within each fit",
        "future_modelling": "Select model settings, calibration and threshold within training folds; freeze before scoring primary and secondary temporal cohorts; never select methods on temporal scores",
        "further_confirmation": "A genuinely untouched future/external cohort is needed for confirmatory performance claims",
        "cohorts": audit.to_dict(orient="records"),
        "data_sha256": {name: hashlib.sha256((root/"data/processed"/name).read_bytes()).hexdigest()
                        for name in ["train_w1_w2_raw.csv", "test_w2_w3_raw.csv"]},
    }
    (root/"phase1_strategy/C2_evaluation_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return audit
