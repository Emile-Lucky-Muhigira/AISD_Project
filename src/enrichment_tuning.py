"""Nested AP tuning of literacy/funding, compared at a training recall target.

Six original grids; no temporal data. Calibration/threshold OOF predictions
come from models whose hyperparameters were selected without those labels.
The target is an inner empirical constraint, NOT guaranteed outer recall.
"""
import hashlib
import json
import math
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from .enrichment_evaluation import enrichment_preprocessor
from .model_tuning import TUNING_GRIDS, _validate_frame, _validate_splits, _fit_estimator, _probability
from .model_calibration import fit_calibrators, apply_calibrator, reliability_bins
from .evaluation_metrics import select_top_risk
from .high_recall_evaluation import policy_metrics
from .subgroup_evaluation import subgroup_labels
from .pipeline_builder import TARGET


def recall_threshold(y,scores,target=.80):
    """Highest inclusive threshold catching ceil(target * events) inner events."""
    y=np.asarray(y);p=np.asarray(scores,float)
    if (y.ndim!=1 or not len(y) or p.shape!=y.shape or not np.isin(y,[0,1]).all()
        or not y.sum() or not np.isfinite(p).all() or ((p<0)|(p>1)).any() or not 0<target<=1):
        raise ValueError('Aligned binary events, finite scores and valid target required.')
    return float(np.sort(p[y==1])[::-1][math.ceil(target*y.sum())-1])


def _preprocessor(model):
    return enrichment_preprocessor(model,'literacy_funding')


def search_models(frame,seed):
    """AP selection for all six families, reusing only fold-local preprocessing."""
    _validate_frame(frame)
    X=frame.drop(columns=TARGET);y=frame[TARGET].to_numpy()
    cv=StratifiedGroupKFold(3,shuffle=True,random_state=seed)
    splits=_validate_splits(frame,cv.split(X,y,frame.district_id),3)
    rows=[]
    for fold,(a,b) in enumerate(splits,1):
        cache={}
        for model,candidates in TUNING_GRIDS.items():
            kind=model=='elasticnet'
            if kind not in cache:
                prep=_preprocessor(model).fit(X.iloc[a])
                cache[kind]=(prep.transform(X.iloc[a]),prep.transform(X.iloc[b]),int(prep.named_steps['grade_knn'].n_neighbors_))
            fit_x,valid_x,k=cache[kind]
            for i,params in enumerate(candidates):
                status='converged';ap=np.nan
                try:
                    estimator,_=_fit_estimator(model,params,fit_x,y[a])
                    ap=float(average_precision_score(y[b],_probability(estimator,valid_x)))
                except ConvergenceWarning:
                    status='convergence_failure'
                rows.append(dict(model=model,candidate_id=i,search_fold=fold,parameters=json.dumps(params,sort_keys=True),
                    average_precision=ap,fit_status=status,training_n=len(a),validation_n=len(b),
                    training_events=int(y[a].sum()),validation_events=int(y[b].sum()),grade_knn_k=k,
                    shared_districts=0,shared_households=0))
    audit=pd.DataFrame(rows);best={};audit['selected']=False
    for model,candidates in TUNING_GRIDS.items():
        g=audit[audit.model.eq(model)]
        means=g.groupby('candidate_id').average_precision.agg(['count','mean'])
        eligible=means[means['count'].eq(3)]
        if eligible.empty:raise RuntimeError(f'No converged candidate for {model}.')
        i=min(eligible.index,key=lambda i:(-eligible.loc[i,'mean'],i))
        best[model]=dict(candidates[i])
        audit.loc[audit.model.eq(model)&audit.candidate_id.eq(i),'selected']=True
    return best,audit


def training_scores(frame,outer_fold,progress=False):
    """OOF scores for fixed and tuned procedures; retune within each fit subset."""
    X=frame.drop(columns=TARGET);y=frame[TARGET].to_numpy()
    cv=StratifiedGroupKFold(3,shuffle=True,random_state=942+outer_fold)
    splits=_validate_splits(frame,cv.split(X,y,frame.district_id),3)
    scores={(setting,model):np.full(len(frame),np.nan) for setting in ['fixed','tuned'] for model in TUNING_GRIDS}
    searches=[]
    for fold,(a,b) in enumerate(splits,1):
        if progress:print(f'  threshold/calibration fold {fold}/3: nested AP search',flush=True)
        best,audit=search_models(frame.iloc[a].copy(),1942+10*outer_fold+fold)
        searches.append(audit.assign(calibration_fold=fold))
        cache={}
        for model in TUNING_GRIDS:
            kind=model=='elasticnet'
            if kind not in cache:
                prep=_preprocessor(model).fit(X.iloc[a])
                cache[kind]=(prep.transform(X.iloc[a]),prep.transform(X.iloc[b]))
            fit_x,valid_x=cache[kind]
            for setting,params in [('fixed',{}),('tuned',best[model])]:
                estimator,_=_fit_estimator(model,params,fit_x,y[a])
                scores[setting,model][b]=_probability(estimator,valid_x)
    if not all(np.isfinite(p).all() for p in scores.values()):
        raise RuntimeError('Incomplete inner predictions.')
    return scores,pd.concat(searches,ignore_index=True)


def evaluate_enrichment_tuning(train,outer_splits,target=.80,progress=True):
    _validate_frame(train);splits=_validate_splits(train,outer_splits,5)
    # Validate the target before spending time fitting models.
    recall_threshold([0,1],[.1,.9],target)
    X=train.drop(columns=TARGET);y=train[TARGET].to_numpy()
    keys=[json.dumps([str(h),str(p)],separators=(',',':')) for h,p in zip(train.FPrimary,train.hhmid)]
    searches=[];metrics=[];predictions=[];curves=[]
    for fold,(a,b) in enumerate(splits,1):
        if progress:print(f'Enriched tuning: outer fold {fold}/5',flush=True)
        part=train.iloc[a].reset_index(drop=True)
        inner,audit=training_scores(part,fold,progress)
        searches.append(audit.assign(outer_fold=fold))
        best,audit=search_models(part,142+fold)
        searches.append(audit.assign(calibration_fold=0,outer_fold=fold))
        cache={}
        for model in TUNING_GRIDS:
            kind=model=='elasticnet'
            if kind not in cache:
                prep=_preprocessor(model).fit(X.iloc[a])
                cache[kind]=(prep.transform(X.iloc[a]),prep.transform(X.iloc[b]),int(prep.named_steps['grade_knn'].n_neighbors_))
            fit_x,valid_x,k=cache[kind]
            for setting,params in [('fixed',{}),('tuned',best[model])]:
                inner_p=inner[setting,model]
                threshold=recall_threshold(y[a],inner_p,target)
                cal=fit_calibrators(y[a],inner_p)['sigmoid']
                estimator,_=_fit_estimator(model,params,fit_x,y[a])
                raw=_probability(estimator,valid_x)
                prob=apply_calibrator('sigmoid',cal,raw)
                bins,ece=reliability_bins(y[b],prob)
                curves.append(bins.assign(model=model,setting=setting,fold=fold))
                selected=raw>=threshold;top=select_top_risk(raw,[keys[i] for i in b],.2)
                r=policy_metrics(y[b],raw,selected)
                r.update(model=model,setting=setting,fold=fold,parameters=json.dumps(params,sort_keys=True),
                    recall_target=target,threshold=threshold,inner_recall=float(y[a][inner_p>=threshold].sum()/y[a].sum()),
                    inner_flagged_fraction=float((inner_p>=threshold).mean()),
                    top20_tp=int(y[b][top].sum()),top20_n=int(top.sum()),
                    top20_recall=float(y[b][top].sum()/y[b].sum()),top20_precision=float(y[b][top].mean()),
                    calibrated_brier=float(brier_score_loss(y[b],prob)),calibrated_ece=ece,
                    grade_knn_k=k,shared_districts=0,shared_households=0)
                metrics.append(r)
                for pos,p,q,s,t in zip(b,raw,prob,selected,top):
                    predictions.append(dict(model=model,setting=setting,fold=fold,training_row=int(pos),
                        child_key_sha256=hashlib.sha256(keys[pos].encode()).hexdigest(),y_true=int(y[pos]),
                        score=float(p),probability=float(q),threshold=threshold,selected_recall80=bool(s),selected_top20=bool(t)))
            if progress:print(f'  {model}: outer fits complete',flush=True)
    oof=pd.DataFrame(predictions).sort_values(['setting','model','training_row']).reset_index(drop=True)
    labels=subgroup_labels(train.reset_index(drop=True));groups=[]
    for (setting,model),g in oof.groupby(['setting','model']):
        g=g.sort_values('training_row').reset_index(drop=True)
        if not np.array_equal(g.training_row,np.arange(len(train))):raise RuntimeError('Incomplete outer coverage.')
        masks=[('overall','all',np.ones(len(train),bool))]
        for dimension in labels:
            masks.extend((dimension,name,labels[dimension].eq(name).to_numpy()) for name in sorted(labels[dimension].unique()))
        for dimension,name,mask in masks:
            for policy,column in [('recall80','selected_recall80'),('top20','selected_top20')]:
                r=policy_metrics(g.y_true[mask],g.score[mask],g[column][mask]);r.pop('average_precision')
                r.update(setting=setting,model=model,dimension=dimension,subgroup=name,policy=policy,
                    sparse_group=bool(r['n']<50 or r['events']<10 or r['n']-r['events']<10 or train.loc[mask,'district_id'].nunique()<5))
                groups.append(r)
    return pd.concat(searches,ignore_index=True),pd.DataFrame(metrics),oof,pd.DataFrame(groups),pd.concat(curves,ignore_index=True)
