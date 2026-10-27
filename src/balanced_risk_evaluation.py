"""Nested selection balancing false negatives and false positives using F1.

Exploratory on previously inspected outer folds; never consumes temporal data.
Raw decision scores are not calibrated probabilities. Existing models stay intact.
"""
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from .high_recall_evaluation import (FAMILIES,score_candidates_on_training,
    make_preprocessor,fit_candidate,policy_metrics)
from .model_tuning import _validate_frame,_validate_splits,_probability
from .pipeline_builder import TARGET
from .subgroup_evaluation import subgroup_labels
from .evaluation_metrics import select_top_risk

BALANCED_CANDIDATES=[dict(model=m,positive_multiplier=w,include_parent_indicators=keep)
                     for m in FAMILIES for w in (.5,1.,2.) for keep in (True,False)]


def threshold_for_f1(y,scores):
    """Exact best empirical F1 over inclusive thresholds; ties flag fewer rows.

    F1=2TP/(2TP+FP+FN). Tied scores are never split with outcome labels.
    An all-flag policy may win on uninformative data; report that, never disguise it.
    """
    y=np.asarray(y);p=np.asarray(scores,float)
    if (y.ndim!=1 or not len(y) or p.shape!=y.shape or not np.isin(y,[0,1]).all()
        or not y.sum() or not np.isfinite(p).all() or ((p<0)|(p>1)).any()):
        raise ValueError('Aligned finite scores and binary labels with events required.')
    order=np.argsort(-p,kind='stable');ordered=p[order]
    ends=np.r_[np.flatnonzero(ordered[:-1]!=ordered[1:]),len(p)-1]
    caught=np.cumsum(y[order])[ends];places=ends+1
    f1=2*caught/(places+y.sum())
    best=int(np.argmax(f1))
    return float(ordered[ends[best]]),float(f1[best])


def balanced_metrics(y,scores,selected):
    result=policy_metrics(y,scores,selected)
    denominator=2*result['tp']+result['fp']+result['fn']
    result['f1']=2*result['tp']/denominator if denominator else 0.
    result['false_positive_rate']=result['fp']/(result['fp']+result['tn']) if result['fp']+result['tn'] else np.nan
    return result


def select_balanced_on_training(frame,outer_fold,candidates=None):
    candidates=BALANCED_CANDIDATES if candidates is None else candidates
    scores,details=score_candidates_on_training(frame,outer_fold,candidates)
    y=frame[TARGET].to_numpy();groups=subgroup_labels(frame.reset_index(drop=True)).parent_co_residence
    rows=[]
    for i,candidate in enumerate(candidates):
        eligible=bool(np.isfinite(scores[i]).all())
        row=dict(candidate_id=i,**candidate,eligible=eligible,fit_details=json.dumps(details[i],sort_keys=True))
        if eligible:
            threshold,f1=threshold_for_f1(y,scores[i]);selected=scores[i]>=threshold
            metrics=balanced_metrics(y,scores[i],selected)
            subgroup={g:balanced_metrics(y[groups.eq(g)],scores[i][groups.eq(g)],selected[groups.eq(g)]) for g in sorted(groups.unique())}
            # Convert undefined precision/AP to JSON null rather than nonstandard NaN.
            subgroup={g:{k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in vals.items()} for g,vals in subgroup.items()}
            row.update(threshold=threshold,inner_f1=f1,inner_precision=metrics['precision'],inner_recall=metrics['recall'],
                       inner_selected_fraction=metrics['selected_fraction'],inner_average_precision=metrics['average_precision'],
                       inner_subgroup_metrics=json.dumps(subgroup,sort_keys=True,allow_nan=False))
        else:
            row.update(threshold=np.nan,inner_f1=np.nan,inner_precision=np.nan,inner_recall=np.nan,
                       inner_selected_fraction=np.nan,inner_average_precision=np.nan,inner_subgroup_metrics='{}')
        rows.append(row)
    audit=pd.DataFrame(rows);eligible=audit[audit.eligible]
    if eligible.empty:raise RuntimeError('No candidate converged in all training folds.')
    # Deterministic ties prefer fewer alerts, then higher AP and candidate order.
    best=eligible.sort_values(['inner_f1','inner_selected_fraction','inner_average_precision','candidate_id'],
                             ascending=[False,True,False,True]).iloc[0]
    audit['selected']=audit.candidate_id.eq(best.candidate_id)
    return dict(candidates[int(best.candidate_id)]),float(best.threshold),audit


def evaluate_balanced_risk(train,outer_splits,progress=True):
    _validate_frame(train);splits=_validate_splits(train,outer_splits,5)
    X=train.drop(columns=TARGET);y=train[TARGET].to_numpy()
    keys=[json.dumps([str(h),str(p)],separators=(',',':')) for h,p in zip(train.FPrimary,train.hhmid)]
    searches=[];metrics=[];predictions=[]
    for fold,(a,b) in enumerate(splits,1):
        if progress:print(f'Balanced outer fold {fold}/5: training-only F1 selection',flush=True)
        candidate,threshold,audit=select_balanced_on_training(train.iloc[a].copy(),fold)
        audit.insert(0,'outer_fold',fold);searches.append(audit)
        prep=make_preprocessor(candidate).fit(X.iloc[a])
        estimator,detail=fit_candidate(candidate,prep.transform(X.iloc[a]),y[a])
        p=_probability(estimator,prep.transform(X.iloc[b]));selected=p>=threshold
        top20=select_top_risk(p,[keys[i] for i in b],.2)
        result=balanced_metrics(y[b],p,selected)
        result.update(fold=fold,**candidate,threshold=threshold,refit_details=json.dumps(detail,sort_keys=True),
                      top20_recall=float(y[b][top20].sum()/y[b].sum()),top20_selected_n=int(top20.sum()))
        metrics.append(result)
        for pos,score,flag,top in zip(b,p,selected,top20):
            predictions.append(dict(training_row=int(pos),fold=fold,child_key_sha256=hashlib.sha256(keys[pos].encode()).hexdigest(),
                y_true=int(y[pos]),score=float(score),threshold=threshold,selected_balanced=bool(flag),selected_top20=bool(top),**candidate))
        if progress:print(f"Selected {candidate}; precision {result['precision']:.3f}, recall {result['recall']:.3f}, flagged {result['selected_fraction']:.3f}",flush=True)
    oof=pd.DataFrame(predictions).sort_values('training_row').reset_index(drop=True)
    if not np.array_equal(oof.training_row,np.arange(len(train))):raise RuntimeError('Incomplete held-out coverage.')
    labels=subgroup_labels(train.reset_index(drop=True))
    groups=[('overall','all',np.ones(len(train),bool))]
    for dimension in labels:
        groups.extend((dimension,g,labels[dimension].eq(g).to_numpy()) for g in sorted(labels[dimension].unique()))
    subgroups=[]
    for dimension,group,mask in groups:
        result=balanced_metrics(oof.y_true[mask],oof.score[mask],oof.selected_balanced[mask])
        result.pop('average_precision') # Different selected families have incomparable score scales.
        result.update(dimension=dimension,subgroup=group,sparse_group=bool(result['n']<50 or result['events']<10
            or result['n']-result['events']<10 or train.loc[mask,'district_id'].nunique()<5))
        subgroups.append(result)
    return pd.concat(searches,ignore_index=True),pd.DataFrame(metrics),oof,pd.DataFrame(subgroups)


def compare_policies(train,balanced_oof,calibration_oof,high_recall_oof):
    """Descriptive comparison only; never used to choose candidates/thresholds."""
    rows=[]
    sources=[('balanced_F1',balanced_oof,'selected_balanced'),('high_recall',high_recall_oof,'selected_high_recall')]
    sources += [(model+'_top20',g,'selected_top20_within_fold') for model,g in calibration_oof[calibration_oof.method.eq('sigmoid')].groupby('model')]
    expected_keys=[hashlib.sha256(json.dumps([str(h),str(p)],separators=(',',':')).encode()).hexdigest() for h,p in zip(train.FPrimary,train.hhmid)]
    for name,data,column in sources:
        data=data.sort_values('training_row')
        if (not np.array_equal(data.training_row,np.arange(len(train))) or not np.array_equal(data.y_true,train[TARGET])
            or not np.array_equal(data.child_key_sha256,expected_keys)):
            raise ValueError('Policy benchmark identity/label mismatch.')
        result=balanced_metrics(data.y_true,np.zeros(len(data)),data[column])
        result.pop('average_precision');rows.append(dict(policy=name,**result))
    result=balanced_metrics(train[TARGET],np.zeros(len(train)),np.ones(len(train),bool));result.pop('average_precision')
    rows.append(dict(policy='flag_everyone',**result))
    return pd.DataFrame(rows)
