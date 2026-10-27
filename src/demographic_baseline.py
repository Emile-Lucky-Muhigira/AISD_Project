"""Prespecified age/sex logistic benchmark; training-only sigmoid calibration.

No geography, family variables, survey missingness flags, or identifiers enter
the estimator. This is a simple comparator, not every possible demographic rule.
"""
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from .model_tuning import _validate_frame, _validate_splits, _probability
from .model_calibration import fit_calibrators, apply_calibrator, reliability_bins
from .evaluation_metrics import evaluate_binary_risk, select_top_risk
from .high_recall_evaluation import policy_metrics
from .subgroup_evaluation import subgroup_labels


def demographic_pipeline():
    """Fixed C=1 L2, unweighted; all learned preprocessing stays inside folds."""
    age=Pipeline([('fill',SimpleImputer(strategy='median')),('scale',StandardScaler())])
    sex=Pipeline([('fill',SimpleImputer(strategy='most_frequent')),
                  ('encode',OneHotEncoder(categories=[[1,2]],drop='first',handle_unknown='error',sparse_output=False))])
    return Pipeline([('prep',ColumnTransformer([('age',age,['age']),('sex',sex,['sex'])],remainder='drop')),
                     ('model',LogisticRegression(C=1.,solver='lbfgs',max_iter=2000,random_state=42))])


def evaluate_demographic(train,splits):
    _validate_frame(train);splits=_validate_splits(train,splits,5)
    if not train.sex.dropna().isin([1,2]).all():raise ValueError('Sex must be survey codes 1/2 or missing.')
    X=train.drop(columns='dropout');y=train.dropout.to_numpy()
    keys=[json.dumps([str(h),str(p)],separators=(',',':')) for h,p in zip(train.FPrimary,train.hhmid)]
    records=[];predictions=[];curves=[]
    for fold,(a,b) in enumerate(splits,1):
        part=train.iloc[a].reset_index(drop=True)
        cv=StratifiedGroupKFold(3,shuffle=True,random_state=1242+fold)
        inner=_validate_splits(part,cv.split(part,part.dropout,part.district_id),3)
        scores=np.full(len(a),np.nan)
        for ia,ib in inner:
            pipe=demographic_pipeline().fit(X.iloc[a[ia]],y[a[ia]])
            scores[ib]=_probability(pipe,X.iloc[a[ib]])
        cal=fit_calibrators(y[a],scores)['sigmoid']
        pipe=demographic_pipeline().fit(X.iloc[a],y[a])
        raw=_probability(pipe,X.iloc[b])
        calibrated=apply_calibrator('sigmoid',cal,raw)
        for method,p in [('uncalibrated',raw),('sigmoid',calibrated)]:
            r=evaluate_binary_risk(y[b],p,[keys[i] for i in b],.2)
            cm=r.pop('confusion_matrix');bins,ece=reliability_bins(y[b],p)
            r.update(model='demographic_age_sex',method=method,fold=fold,tn=cm[0][0],fp=cm[0][1],fn=cm[1][0],tp=cm[1][1],
                     ece_10_equal_width=ece,training_n=len(a),shared_districts=0,shared_households=0,
                     sigmoid_slope=float(cal.coef_[0,0]))
            records.append(r);curves.append(bins.assign(model='demographic_age_sex',method=method,fold=fold))
            chosen=select_top_risk(p,[keys[i] for i in b],.2)
            for pos,q,s in zip(b,p,chosen):
                predictions.append(dict(model='demographic_age_sex',method=method,fold=fold,training_row=int(pos),
                    child_key_sha256=hashlib.sha256(keys[pos].encode()).hexdigest(),y_true=int(y[pos]),
                    probability=float(q),selected_top20_within_fold=bool(s)))
    return pd.DataFrame(records),pd.DataFrame(predictions),pd.concat(curves,ignore_index=True)


def compare_demographic(train,demographic_oof,existing_oof):
    """Compare existing tuned sigmoid models without refitting or selecting them."""
    keys=[json.dumps([str(h),str(p)],separators=(',',':')) for h,p in zip(train.FPrimary,train.hhmid)]
    expected=[hashlib.sha256(k.encode()).hexdigest() for k in keys]
    sources=pd.concat([demographic_oof[demographic_oof.method.eq('sigmoid')],
                       existing_oof[existing_oof.method.eq('sigmoid')]],ignore_index=True)
    labels=subgroup_labels(train.reset_index(drop=True));rows=[];reference=None
    for model,g in sources.groupby('model'):
        g=g.sort_values('training_row').reset_index(drop=True)
        if (not np.array_equal(g.training_row,np.arange(len(train))) or not np.array_equal(g.y_true,train.dropout)
            or not np.array_equal(g.child_key_sha256,expected)):
            raise ValueError('Comparison identity/label mismatch.')
        if reference is None:reference=g.fold.to_numpy()
        if not np.array_equal(reference,g.fold):raise ValueError('Comparison must use identical folds.')
        for _,h in g.groupby('fold'):
            chosen=select_top_risk(h.probability,[keys[i] for i in h.training_row],.2)
            if not np.array_equal(chosen,h.selected_top20_within_fold):raise ValueError('Different intervention policy.')
        groups=[('overall','all',np.ones(len(train),bool))]
        for dimension in labels:
            groups.extend((dimension,name,labels[dimension].eq(name).to_numpy()) for name in sorted(labels[dimension].unique()))
        for dimension,name,mask in groups:
            r=policy_metrics(g.y_true[mask],g.probability[mask],g.selected_top20_within_fold[mask])
            r.pop('average_precision') # report mean fold AP separately
            r.update(model=model,dimension=dimension,subgroup=name,
                     sparse_group=bool(r['n']<50 or r['events']<10 or r['n']-r['events']<10 or train.loc[mask,'district_id'].nunique()<5))
            rows.append(r)
    return pd.DataFrame(rows)
