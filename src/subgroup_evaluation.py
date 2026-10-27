"""Descriptive subgroup audit of existing nested, calibrated OOF predictions.

Intervals resample districts with predictions fixed, not model fits; they do not
include training/tuning uncertainty or support population/causal inference.
"""
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from .evaluation_metrics import select_top_risk
from .model_calibration import reliability_bins


def subgroup_labels(train):
    """Mutually exclusive categories; unknown parent answers stay unknown."""
    for col in ["father_in_hh", "mother_in_hh"]:
        if not train[col].dropna().isin([0, 1]).all():
            raise ValueError("Parent co-residence must be 0/1 or missing.")
    f, m = train.father_in_hh, train.mother_in_hh
    family = np.select([f.eq(1)&m.eq(1), f.eq(1)&m.eq(0),
                        f.eq(0)&m.eq(1), f.eq(0)&m.eq(0)],
                       ["both_parents", "father_only", "mother_only", "neither_parent"],
                       default="unknown")
    region = train.regioncode.map(lambda v: "unknown" if pd.isna(v) else f"region_{int(v)}")
    return pd.DataFrame({"region": region.to_numpy(), "parent_co_residence": family})


def _scores(y, p, selected, weights=None):
    w = np.ones(len(y)) if weights is None else np.asarray(weights)
    n, events = w.sum(), np.dot(w, y)
    places, tp = np.dot(w, selected), np.dot(w, y*selected)
    return {
        "average_precision": float(average_precision_score(y, p, sample_weight=w)) if 0 < events < n else np.nan,
        "brier_score": float(np.dot(w, (p-y)**2)/n),
        "recall_under_fold_policy": float(tp/events) if events else np.nan,
        "precision_under_fold_policy": float(tp/places) if places else np.nan,
    }


def evaluate_subgroups(train, oof, bootstrap_reps=500, random_state=42):
    """Pool each child's one held-out sigmoid prediction, retaining fold policy.

    Never reallocate 20% capacity independently within subgroups. Report all
    observed groups; an analyst-defined caution flag (n<50, events<10,
    non-events<10, or districts<5) is not a rubric requirement. Bootstrap CIs
    are suppressed for these sparse groups and when fewer than 90% of draws
    have a defined metric. Region labels preserve raw survey codes.
    """
    if not isinstance(bootstrap_reps, int) or bootstrap_reps < 0:
        raise ValueError("bootstrap_reps must be a nonnegative integer.")
    train = train.reset_index(drop=True)
    if train[["FPrimary", "hhmid", "district_id", "dropout"]].isna().any().any():
        raise ValueError("Identity, groups and labels must be present.")
    if train.duplicated(["FPrimary", "hhmid"]).any() or not train.dropout.isin([0,1]).all():
        raise ValueError("Unique children and binary labels are required.")
    labels = subgroup_labels(train)
    keys = [json.dumps([str(h),str(p)],separators=(',', ':')) for h,p in zip(train.FPrimary,train.hhmid)]
    hashes = np.array([hashlib.sha256(k.encode('utf8')).hexdigest() for k in keys])
    source = oof[oof.method.eq("sigmoid")].copy()
    if source.empty:
        raise ValueError("Sigmoid OOF predictions are required.")
    rows=[]
    expected_folds=None
    for model, data in source.groupby("model", sort=True):
        data=data.sort_values("training_row").reset_index(drop=True)
        if len(data)!=len(train) or not np.array_equal(data.training_row, np.arange(len(train))):
            raise ValueError("Exactly one prediction per training child is required.")
        if not np.array_equal(data.child_key_sha256,hashes) or not np.array_equal(data.y_true,train.dropout):
            raise ValueError("Prediction identities or labels do not match training rows.")
        p=data.probability.to_numpy(float); y=data.y_true.to_numpy(int)
        if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():
            raise ValueError("Invalid probabilities.")
        if not data.selected_top20_within_fold.isin([True,False]).all():
            raise ValueError("Selection must contain booleans.")
        selected=data.selected_top20_within_fold.to_numpy(bool)
        if set(data.fold)!={1,2,3,4,5}:
            raise ValueError("Five held-out folds are required.")
        if expected_folds is None: expected_folds=data.fold.to_numpy()
        if not np.array_equal(expected_folds,data.fold):
            raise ValueError("Models must share the same folds.")
        for column in ["district_id", "FPrimary"]:
            if pd.DataFrame({"group":train[column],"fold":data.fold}).groupby("group").fold.nunique().max()!=1:
                raise ValueError("A group spans multiple held-out folds.")
        for fold,g in data.groupby("fold"):
            idx=g.index.to_numpy()
            if not np.array_equal(selected[idx],select_top_risk(p[idx],[keys[i] for i in idx],.2)):
                raise ValueError("Stored selections do not match the original fold-level policy.")
        groups=[("overall","all",np.ones(len(train),dtype=bool))]
        for dimension in labels:
            groups.extend((dimension,group,labels[dimension].eq(group).to_numpy()) for group in sorted(labels[dimension].unique()))
        for dimension,group,mask in groups:
            gy,gp,gs=y[mask],p[mask],selected[mask]
            districts=train.loc[mask,"district_id"].to_numpy()
            unique,inverse=np.unique(districts,return_inverse=True)
            n,events,places=len(gy),int(gy.sum()),int(gs.sum())
            tp=int((gy*gs).sum()); fp=places-tp; fn=events-tp; tn=n-events-fp
            sparse=n<50 or events<10 or n-events<10 or len(unique)<5
            metrics=_scores(gy,gp,gs)
            _,ece=reliability_bins(gy,gp)
            row=dict(model=model,method="sigmoid",dimension=dimension,subgroup=group,
                     n=n,events=events,non_events=n-events,districts=len(unique),folds=int(data.loc[mask,"fold"].nunique()),
                     prevalence=events/n,selected_n=places,selected_fraction=places/n,
                     tp=tp,fp=fp,fn=fn,tn=tn,mean_probability=float(gp.mean()),ece_10_equal_width=ece,
                     sparse_group=sparse,interval_scope="fixed_predictions_district_bootstrap",
                     bootstrap_reps=bootstrap_reps,**metrics)
            draws={k:[] for k in metrics}
            if not sparse and bootstrap_reps:
                # Same seed per subgroup/model pairs yields matched resamples.
                rng=np.random.default_rng(random_state)
                for _ in range(bootstrap_reps):
                    counts=rng.multinomial(len(unique),np.ones(len(unique))/len(unique))
                    values=_scores(gy,gp,gs,counts[inverse])
                    for key,value in values.items():
                        if np.isfinite(value): draws[key].append(value)
            for key,values in draws.items():
                row[key+"_bootstrap_valid"]=len(values)
                lo,hi=np.quantile(values,[.025,.975]) if len(values)>=max(1,.9*bootstrap_reps) else (np.nan,np.nan)
                row[key+"_ci_low"]=lo;row[key+"_ci_high"]=hi
            rows.append(row)
    return pd.DataFrame(rows)
