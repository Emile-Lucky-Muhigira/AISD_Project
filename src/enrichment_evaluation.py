"""Prespecified paired feature experiment; fixed models, nested F2 thresholds.

Baseline / literacy+funding / literacy+funding+routed teacher questions.
Six families, identical district folds. No temporal data or model selection.
Sigmoid calibration and F2 threshold use inner OOF scores only. F2 is a
recall-focused experimental objective, not an assumed monetary cost ratio.
"""
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, brier_score_loss
from .feature_enrichment import LITERACY
from .pipeline_builder import build_pipeline, TARGET
from .high_recall_evaluation import FAMILIES, policy_metrics
from .model_tuning import _validate_frame, _validate_splits, _fit_estimator, _probability
from .model_calibration import fit_calibrators, apply_calibrator, reliability_bins
from .evaluation_metrics import select_top_risk
from .subgroup_evaluation import subgroup_labels

VARIANTS=('baseline','literacy_funding','literacy_funding_teacher')


class RouteTeacherAbsence(BaseEstimator, TransformerMixin):
    def fit(self,X,y=None):
        return self

    def transform(self,X):
        result=X.copy()
        result['teacher_absent_days']=result['teacher_absent_days_routed']
        return result


def enrichment_preprocessor(model,variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown feature experiment.')
    prep=build_pipeline('lr' if model=='elasticnet' else 'xgb')
    if variant=='baseline':
        return prep
    ct=prep.named_steps['prep']
    ct.transformers=list(ct.transformers)+[
        ('literacy',SimpleImputer(strategy='most_frequent',keep_empty_features=True),LITERACY),
        ('school_funding',OneHotEncoder(handle_unknown='ignore',sparse_output=False,
                                       drop='first' if model=='elasticnet' else None),['school_expense_payer'])]
    if variant.endswith('_teacher'):
        prep.steps.insert(0,('route_teacher',RouteTeacherAbsence()))
        ct.transformers.append(('teacher_arrangement',OneHotEncoder(handle_unknown='ignore',sparse_output=False,
                                drop='first' if model=='elasticnet' else None),['teaching_arrangement']))
    return prep


def threshold_for_f2(y,scores):
    y=np.asarray(y);p=np.asarray(scores,float)
    if (y.ndim!=1 or not len(y) or p.shape!=y.shape or not np.isin(y,[0,1]).all()
        or not y.sum() or not np.isfinite(p).all() or ((p<0)|(p>1)).any()):
        raise ValueError('Aligned finite scores and binary events required.')
    order=np.argsort(-p,kind='stable');ranked=p[order]
    ends=np.r_[np.flatnonzero(ranked[:-1]!=ranked[1:]),len(p)-1]
    tp=np.cumsum(y[order])[ends];places=ends+1
    f2=5*tp/(4*y.sum()+places)
    best=int(np.argmax(f2)) # ties choose fewer alerts; equal scores never split
    return float(ranked[ends[best]]),float(f2[best])


def evaluate_enrichment(train,outer_splits,progress=True):
    _validate_frame(train);splits=_validate_splits(train,outer_splits,5)
    X=train.drop(columns=TARGET);y=train[TARGET].to_numpy()
    keys=[json.dumps([str(h),str(p)],separators=(',',':')) for h,p in zip(train.FPrimary,train.hhmid)]
    records=[];predictions=[];curves=[]
    for fold,(a,b) in enumerate(splits,1):
        part=train.iloc[a].reset_index(drop=True)
        cv=StratifiedGroupKFold(3,shuffle=True,random_state=842+fold)
        inner=_validate_splits(part,cv.split(part,part[TARGET],part.district_id),3)
        for variant in VARIANTS:
            if progress:print(f'Enrichment fold {fold}/5: {variant}',flush=True)
            inner_scores={m:np.full(len(a),np.nan) for m in FAMILIES}
            for ia,ib in inner:
                cache={}
                for model in FAMILIES:
                    kind=model=='elasticnet'
                    if kind not in cache:
                        prep=enrichment_preprocessor(model,variant).fit(X.iloc[a[ia]])
                        cache[kind]=(prep.transform(X.iloc[a[ia]]),prep.transform(X.iloc[a[ib]]))
                    estimator,_=_fit_estimator(model,{},cache[kind][0],y[a[ia]])
                    inner_scores[model][ib]=_probability(estimator,cache[kind][1])
            cache={}
            for model in FAMILIES:
                inner_p=inner_scores[model]
                threshold,inner_f2=threshold_for_f2(y[a],inner_p)
                calibrator=fit_calibrators(y[a],inner_p)['sigmoid']
                kind=model=='elasticnet'
                if kind not in cache:
                    prep=enrichment_preprocessor(model,variant).fit(X.iloc[a])
                    cache[kind]=(prep.transform(X.iloc[a]),prep.transform(X.iloc[b]),int(prep.named_steps['grade_knn'].n_neighbors_))
                fit_x,valid_x,k=cache[kind]
                estimator,_=_fit_estimator(model,{},fit_x,y[a])
                raw=_probability(estimator,valid_x)
                calibrated=apply_calibrator('sigmoid',calibrator,raw)
                bins,ece=reliability_bins(y[b],calibrated)
                curves.append(bins.assign(variant=variant,model=model,fold=fold))
                top=select_top_risk(raw,[keys[i] for i in b],.2)
                flags=raw>=threshold
                result=policy_metrics(y[b],raw,flags)
                result.update(variant=variant,model=model,fold=fold,threshold=threshold,inner_f2=inner_f2,
                    f2=5*result['tp']/(5*result['tp']+4*result['fn']+result['fp']),
                    top20_tp=int(y[b][top].sum()),top20_n=int(top.sum()),
                    top20_recall=float(y[b][top].sum()/y[b].sum()),top20_precision=float(y[b][top].mean()),
                    calibrated_brier=float(brier_score_loss(y[b],calibrated)),calibrated_ece=ece,
                    calibrated_ap=float(average_precision_score(y[b],calibrated)),grade_knn_k=k,
                    training_n=len(a),shared_districts=0,shared_households=0)
                records.append(result)
                for pos,score,prob,flag,t in zip(b,raw,calibrated,flags,top):
                    predictions.append(dict(variant=variant,model=model,fold=fold,training_row=int(pos),
                        child_key_sha256=hashlib.sha256(keys[pos].encode()).hexdigest(),y_true=int(y[pos]),
                        score=float(score),probability=float(prob),threshold=threshold,selected_f2=bool(flag),selected_top20=bool(t)))
            if progress:print('Finished six families, inner thresholds and sigmoid calibration',flush=True)
    oof=pd.DataFrame(predictions).sort_values(['variant','model','training_row']).reset_index(drop=True)
    labels=subgroup_labels(train.reset_index(drop=True));subgroups=[]
    for (variant,model),g in oof.groupby(['variant','model'],sort=False):
        g=g.sort_values('training_row').reset_index(drop=True)
        if not np.array_equal(g.training_row,np.arange(len(train))):
            raise RuntimeError('Incomplete held-out predictions.')
        groups=[('overall','all',np.ones(len(train),bool))]
        for dimension in labels:
            groups.extend((dimension,name,labels[dimension].eq(name).to_numpy()) for name in sorted(labels[dimension].unique()))
        for dimension,name,mask in groups:
            for policy,column in [('F2','selected_f2'),('top20','selected_top20')]:
                result=policy_metrics(g.y_true[mask],g.score[mask],g[column][mask])
                result.pop('average_precision') # aggregate scores may differ across refits
                result.update(variant=variant,model=model,dimension=dimension,subgroup=name,policy=policy,
                    sparse_group=bool(result['n']<50 or result['events']<10 or train.loc[mask,'district_id'].nunique()<5))
                subgroups.append(result)
    return pd.DataFrame(records),oof,pd.DataFrame(subgroups),pd.concat(curves,ignore_index=True)
