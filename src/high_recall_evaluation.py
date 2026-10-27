"""Exploratory nested grouped selection for high recall; no temporal data.

Raw model scores drive thresholds, not calibrated risk estimates. A universal
threshold is learned only on inner OOF scores, then frozen for outer evaluation.
"""
import hashlib
import json
import math
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from .pipeline_builder import build_pipeline, TARGET
from .model_evaluation import comparison_estimator
from .model_tuning import _validate_frame, _validate_splits, _probability
from .subgroup_evaluation import subgroup_labels
from .evaluation_metrics import select_top_risk

FAMILIES = ('elasticnet', 'xgb', 'lightgbm', 'catboost', 'random_forest', 'extra_trees')
CANDIDATES = [dict(model=m, positive_multiplier=w, include_parent_indicators=keep)
              for m in FAMILIES for w in (1., 2.) for keep in (True, False)]


def threshold_for_recall(y, scores, groups, target=.95, min_group_events=10):
    """Highest inclusive threshold meeting empirical training recall constraints.

    Tied scores are all included. No per-person or group-specific threshold is
    used. Sparse groups are reported but do not add a threshold constraint.
    """
    y=np.asarray(y); p=np.asarray(scores,float); groups=np.asarray(groups)
    if (y.ndim!=1 or not len(y) or p.shape!=y.shape or groups.shape!=y.shape
        or not np.isin(y,[0,1]).all() or not np.isfinite(p).all()
        or ((p<0)|(p>1)).any() or not y.sum() or not 0<target<=1
        or not isinstance(min_group_events,int) or min_group_events<1):
        raise ValueError('Valid aligned scores, labels, groups and recall target required.')
    constraints=[('overall',np.ones(len(y),bool))]
    constraints += [(str(g),groups==g) for g in sorted(set(groups))
                    if y[groups==g].sum()>=min_group_events]
    thresholds=[]
    for _,mask in constraints:
        positives=np.sort(p[mask & (y==1)])[::-1]
        required=math.ceil(target*len(positives))
        thresholds.append(float(positives[required-1]))
    threshold=min(thresholds)
    selected=p>=threshold
    details={name:dict(events=int(y[mask].sum()), recall=float(y[mask & selected].sum()/y[mask].sum()))
             for name,mask in constraints}
    return threshold, details


def make_preprocessor(candidate):
    prep=build_pipeline('lr' if candidate['model']=='elasticnet' else 'xgb')
    if not candidate['include_parent_indicators']:
        columns=prep.named_steps['prep']
        columns.transformers=[(name,transformer,[c for c in cols if c not in ('father_in_hh','mother_in_hh')])
                              for name,transformer,cols in columns.transformers]
    return prep


def fit_candidate(candidate, X, y):
    """Compute weights from this fit partition only; reject nonconvergence."""
    y=np.asarray(y); model=candidate['model']; multiplier=candidate['positive_multiplier']
    n0=int((y==0).sum()); n1=int((y==1).sum())
    ratio=n0/n1; estimator=comparison_estimator(model,ratio)
    weights={0:len(y)/(2*n0), 1:len(y)/(2*n1)*multiplier}
    if model in ('elasticnet','random_forest','extra_trees'):
        estimator.set_params(class_weight=weights)
    elif model in ('xgb','lightgbm'):
        estimator.set_params(scale_pos_weight=ratio*multiplier)
    else:
        estimator.set_params(class_weights=[1.,ratio*multiplier])
    fit_kwargs={}; scale=1.
    for attempt in range(2):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                estimator.fit(X,y,**fit_kwargs)
            return estimator,dict(training_n=len(y),training_events=n1,training_class_ratio=ratio,
                                  positive_multiplier=multiplier,numerical_weight_scale=scale)
        except ConvergenceWarning:
            if model!='elasticnet' or attempt==1: raise
            # Preserve the actual candidate weights, including doubled positives.
            sample_weights=np.where(y==1,weights[1],weights[0]);scale=float(sample_weights.max())
            estimator.set_params(class_weight=None,C=estimator.C*scale,max_iter=estimator.max_iter*5)
            fit_kwargs={'sample_weight':sample_weights/scale}


def score_candidates_on_training(frame, outer_fold, candidates=None):
    """Shared inner OOF fits; never accepts outer validation data."""
    _validate_frame(frame)
    candidates=CANDIDATES if candidates is None else candidates
    X=frame.drop(columns=TARGET); y=frame[TARGET].to_numpy()
    cv=StratifiedGroupKFold(3,shuffle=True,random_state=642+outer_fold)
    splits=_validate_splits(frame,cv.split(X,y,frame.district_id),3)
    scores=np.full((len(candidates),len(frame)),np.nan);diagnostics=[[] for _ in candidates]
    for inner_fold,(a,b) in enumerate(splits,1):
        cache={}
        for candidate_id,candidate in enumerate(candidates):
            key=(candidate['model']=='elasticnet',candidate['include_parent_indicators'])
            if key not in cache:
                prep=make_preprocessor(candidate).fit(X.iloc[a])
                cache[key]=(prep.transform(X.iloc[a]),prep.transform(X.iloc[b]),int(prep.named_steps['grade_knn'].n_neighbors_))
            fit_X,valid_X,k=cache[key]
            try:
                estimator,detail=fit_candidate(candidate,fit_X,y[a])
                scores[candidate_id,b]=_probability(estimator,valid_X)
                detail['status']='converged'
            except ConvergenceWarning:
                detail={'status':'convergence_failure','training_n':len(a),'training_events':int(y[a].sum())}
            detail.update(inner_fold=inner_fold,validation_n=len(b),grade_knn_k=k,
                          shared_districts=0,shared_households=0)
            diagnostics[candidate_id].append(detail)
    return scores, diagnostics


def select_on_training(frame, outer_fold, target=.95, candidates=None):
    """Select among 24 fixed configurations using three inner grouped folds.

    Other estimator settings stay at the declared comparison defaults. Select
    minimum alert fraction subject to recall constraints; break ties by higher
    pooled inner AP, then candidate order. Inner scores are selection evidence,
    not independent estimates of selected-model performance.
    """
    candidates=CANDIDATES if candidates is None else candidates
    scores,diagnostics=score_candidates_on_training(frame,outer_fold,candidates)
    y=frame[TARGET].to_numpy()
    groups=subgroup_labels(frame.reset_index(drop=True)).parent_co_residence.to_numpy()
    records=[]
    for i,candidate in enumerate(candidates):
        eligible=bool(np.isfinite(scores[i]).all())
        threshold,details=threshold_for_recall(y,scores[i],groups,target) if eligible else (np.nan,{})
        chosen=scores[i]>=threshold
        records.append(dict(candidate_id=i,**candidate,eligible=eligible,threshold=threshold,
            inner_selected_fraction=float(chosen.mean()) if eligible else np.nan,
            inner_recall=float(y[chosen].sum()/y.sum()) if eligible else np.nan,
            inner_average_precision=float(average_precision_score(y,scores[i])) if eligible else np.nan,
            inner_group_constraints=json.dumps(details,sort_keys=True),fit_details=json.dumps(diagnostics[i],sort_keys=True)))
    audit=pd.DataFrame(records)
    eligible=audit[audit.eligible]
    if eligible.empty: raise RuntimeError('No candidate converged in all inner folds.')
    best=eligible.sort_values(['inner_selected_fraction','inner_average_precision','candidate_id'],ascending=[True,False,True]).iloc[0]
    audit['selected']=audit.candidate_id.eq(best.candidate_id)
    return dict(candidates[int(best.candidate_id)]),float(best.threshold),audit


def policy_metrics(y, scores, selected):
    y=np.asarray(y); p=np.asarray(scores);selected=np.asarray(selected,bool)
    n=len(y);events=int(y.sum());places=int(selected.sum());tp=int(y[selected].sum())
    return dict(n=n,events=events,selected_n=places,selected_fraction=places/n,
                tp=tp,fn=events-tp,fp=places-tp,tn=n-events-places+tp,
                recall=tp/events if events else np.nan,precision=tp/places if places else np.nan,
                average_precision=float(average_precision_score(y,p)) if 0<events<n else np.nan)


def evaluate_high_recall(train, outer_splits, target=.95, progress=True):
    _validate_frame(train); splits=_validate_splits(train,outer_splits,5)
    X=train.drop(columns=TARGET);y=train[TARGET].to_numpy()
    keys=[json.dumps([str(h),str(p)],separators=(',',':')) for h,p in zip(train.FPrimary,train.hhmid)]
    searches=[];metrics=[];predictions=[]
    for fold,(a,b) in enumerate(splits,1):
        if progress: print(f'High-recall outer fold {fold}/5: selecting on training only',flush=True)
        candidate,threshold,audit=select_on_training(train.iloc[a].copy(),fold,target)
        audit.insert(0,'outer_fold',fold); searches.append(audit)
        prep=make_preprocessor(candidate).fit(X.iloc[a])
        estimator,detail=fit_candidate(candidate,prep.transform(X.iloc[a]),y[a])
        p=_probability(estimator,prep.transform(X.iloc[b])); selected=p>=threshold
        top20=select_top_risk(p,[keys[i] for i in b],.2)
        result=policy_metrics(y[b],p,selected)
        result.update(fold=fold,**candidate,threshold=threshold,target_recall=target,
                      target_met=bool(result['recall']>=target),refit_details=json.dumps(detail,sort_keys=True),
                      top20_recall=float(y[b][top20].sum()/y[b].sum()),top20_selected_n=int(top20.sum()))
        metrics.append(result)
        for pos,score,flag,top in zip(b,p,selected,top20):
            predictions.append(dict(training_row=int(pos),fold=fold,child_key_sha256=hashlib.sha256(keys[pos].encode()).hexdigest(),
                y_true=int(y[pos]),score=float(score),threshold=threshold,selected_high_recall=bool(flag),selected_top20=bool(top),**candidate))
        if progress: print(f"Selected {candidate}; held-out recall {result['recall']:.3f}, flagged {result['selected_fraction']:.3f}",flush=True)
    oof=pd.DataFrame(predictions).sort_values('training_row').reset_index(drop=True)
    if not np.array_equal(oof.training_row,np.arange(len(train))): raise RuntimeError('OOF coverage failure.')
    labels=subgroup_labels(train.reset_index(drop=True)); subgroup=[]
    groups=[('overall','all',np.ones(len(train),bool))]
    for dimension in labels:
        groups.extend((dimension,g,labels[dimension].eq(g).to_numpy()) for g in sorted(labels[dimension].unique()))
    for dimension,g,mask in groups:
        r=policy_metrics(oof.y_true[mask],oof.score[mask],oof.selected_high_recall[mask])
        # Scores from different selected model families are not on a common scale.
        r.pop('average_precision')
        r.update(dimension=dimension,subgroup=g,target_recall=target,target_met=bool(r['recall']>=target),
                 sparse_group=bool(r['n']<50 or r['events']<10 or r['n']-r['events']<10 or train.loc[mask,'district_id'].nunique()<5))
        subgroup.append(r)
    return pd.concat(searches,ignore_index=True),pd.DataFrame(metrics),oof,pd.DataFrame(subgroup)
