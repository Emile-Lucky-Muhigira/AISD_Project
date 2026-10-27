"""Frozen retrospective evaluation. Model fitting accepts training data only."""
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from .pipeline_builder import build_pipeline
from .model_tuning import _validate_frame, _validate_splits, tune_on_training, _fit_estimator, _probability
from .model_calibration import calibration_training_scores, fit_calibrators, apply_calibrator, reliability_bins
from .demographic_baseline import demographic_pipeline
from .evaluation_split import temporal_masks, cohort_audit
from .evaluation_metrics import evaluate_binary_risk, select_top_risk
from .subgroup_evaluation import subgroup_labels, _scores


def identity_keys(frame):
    return [json.dumps([str(h),str(p)],separators=(',',':')) for h,p in zip(frame.FPrimary,frame.hhmid)]


def region_strata(train):
    """Freeze geography strata from earlier-period outcomes only."""
    return {float(k):'high' if v>train.dropout.mean() else 'low'
            for k,v in train.groupby('regioncode').dropout.mean().items()}


def fit_temporal_models(train, protocol):
    _validate_frame(train)
    X=train.drop(columns='dropout'); y=train.dropout.to_numpy()
    fits={}; audits=[]; searches=[]; calibration_scores=[]
    for model in protocol['families']+['demographic_age_sex']:
        print(f'Temporal training-only fit: {model}',flush=True)
        if model=='demographic_age_sex':
            cv=StratifiedGroupKFold(3,shuffle=True,random_state=1242)
            splits=_validate_splits(train,cv.split(X,y,train.district_id),3)
            scores=np.full(len(train),np.nan); details=[]
            for a,b in splits:
                pipe=demographic_pipeline().fit(X.iloc[a],y[a])
                scores[b]=_probability(pipe,X.iloc[b])
                details.append(dict(fit_n=len(a),predict_n=len(b),shared_households=0,shared_districts=0))
            prep=None; estimator=demographic_pipeline().fit(X,y); best={}; ratio=None; k=None
        else:
            scores,details=calibration_training_scores(train,model,protocol['fit']['calibration_outer_fold_argument'])
            best,search=tune_on_training(train,model,random_state=protocol['fit']['tuning_seed'])
            searches.append(search)
            prep=build_pipeline('lr' if model=='elasticnet' else 'xgb').fit(X)
            estimator,ratio=_fit_estimator(model,best,prep.transform(X),y)
            k=int(prep.named_steps['grade_knn'].n_neighbors_)
        calibrators=fit_calibrators(y,scores)
        fits[model]=(prep,estimator,calibrators)
        audits.append(dict(model=model,training_n=len(train),training_events=int(y.sum()),
            parameters=json.dumps(best,sort_keys=True),training_class_ratio=ratio,grade_knn_k=k,
            sigmoid_slope=float(calibrators['sigmoid'].coef_[0,0]),
            sigmoid_intercept=float(calibrators['sigmoid'].intercept_[0]),
            calibration_details=json.dumps(details,sort_keys=True)))
        calibration_scores.extend(dict(model=model,training_row=i,probability=float(p),y_true=int(y[i])) for i,p in enumerate(scores))
    return fits,pd.DataFrame(audits),pd.concat(searches,ignore_index=True),pd.DataFrame(calibration_scores)


def predict_temporal(fits, features):
    """Reject outcome-bearing input: no prediction-time fit or recalibration."""
    if 'dropout' in features: raise ValueError('Remove outcomes before prediction.')
    result={}
    for model,(prep,estimator,calibrators) in fits.items():
        raw=_probability(estimator,prep.transform(features) if prep is not None else features)
        result[(model,'uncalibrated')]=raw
        for method in ('sigmoid','isotonic'):
            # The established demographic comparator reports raw and sigmoid.
            if model=='demographic_age_sex' and method=='isotonic':continue
            result[(model,method)]=apply_calibrator(method,calibrators[method],raw)
    return result


def fixed_policy_summary(y,p,selected,districts,bootstrap_reps=1000):
    y=np.asarray(y,int);p=np.asarray(p,float);selected=np.asarray(selected,bool)
    unique,inverse=np.unique(districts,return_inverse=True)
    n=len(y); events=int(y.sum()); places=int(selected.sum());tp=int(y[selected].sum())
    metrics=_scores(y,p,selected)
    metrics['recall_at_capacity']=metrics.pop('recall_under_fold_policy')
    metrics['precision_at_capacity']=metrics.pop('precision_under_fold_policy')
    _,ece=reliability_bins(y,p)
    row=dict(n=n,events=events,selected_n=places,tp=tp,fp=places-tp,fn=events-tp,
             tn=n-events-places+tp,prevalence=events/n,selected_fraction=places/n,
             districts=len(unique),few_clusters=len(unique)<5,sparse_group=n<30 or events<5,
             mean_probability=float(p.mean()),ece_10_equal_width=ece,**metrics)
    draws={k:[] for k in _scores(y,p,selected)}; zero_events=0
    if bootstrap_reps and len(unique)>1:
        rng=np.random.default_rng(20261025)
        # All replicates share score order and ties. Compute weighted AP at
        # distinct score thresholds in batches, exactly as the looped metric.
        counts=rng.multinomial(len(unique),np.ones(len(unique))/len(unique),size=bootstrap_reps)
        weights=counts[:,inverse]
        total=weights.sum(axis=1);positive=weights@y;places=weights@selected.astype(int)
        tp=weights@(y*selected);zero_events=int((positive==0).sum())
        order=np.argsort(-p,kind='stable')
        ends=np.r_[np.flatnonzero(np.diff(p[order])!=0),len(p)-1]
        cumulative_positive=np.cumsum(weights[:,order]*y[order],axis=1)[:,ends]
        cumulative_total=np.cumsum(weights[:,order],axis=1)[:,ends]
        precision=np.divide(cumulative_positive,cumulative_total,out=np.zeros_like(cumulative_positive,dtype=float),where=cumulative_total>0)
        increments=np.diff(cumulative_positive,axis=1,prepend=0)
        with np.errstate(divide='ignore',invalid='ignore'):
            ap=(precision*increments).sum(axis=1)/positive
            ap[(positive==0)|(positive==total)]=np.nan
            arrays={'average_precision':ap,'brier_score':weights@((p-y)**2)/total,
                    'recall_under_fold_policy':tp/positive,'precision_under_fold_policy':tp/places}
        draws={key:array[np.isfinite(array)].tolist() for key,array in arrays.items()}
    row.update(bootstrap_reps=bootstrap_reps,bootstrap_zero_event_reps=zero_events)
    for key,values in draws.items():
        key=key.replace('under_fold_policy','at_capacity')
        row[key+'_bootstrap_valid']=len(values)
        lo,hi=np.quantile(values,[.025,.975]) if len(values)>=max(1,.9*bootstrap_reps) else (np.nan,np.nan)
        row[key+'_ci_low']=lo;row[key+'_ci_high']=hi
    return row


def evaluate_temporal(train,test,predictions,protocol,bootstrap_reps=1000):
    _validate_frame(train);_validate_frame(test)
    masks=temporal_masks(train,test);strata=region_strata(train)
    records=[];curves=[];subgroups=[];rows=[]
    for cohort,mask in [('primary',masks.in_primary_test),('secondary',masks.in_secondary_test)]:
        part=test.loc[mask].reset_index(drop=True);positions=np.flatnonzero(mask)
        keys=identity_keys(part); y=part.dropout.to_numpy(int)
        labels=subgroup_labels(part)
        labels['sex']=part.sex.map({1:'male',2:'female'}).fillna('unknown')
        labels['urban_rural']=part.urbrur.map({1:'urban',2:'rural'}).fillna('unknown')
        labels['training_region_risk']=part.regioncode.map(strata).fillna('unseen_or_missing')
        for (model,method),all_p in predictions.items():
            print(f'Temporal reporting: {cohort} / {model} / {method}',flush=True)
            if len(all_p)!=len(test):raise ValueError('Prediction length mismatch.')
            p=np.asarray(all_p)[positions]
            meta=dict(cohort=cohort,model=model,method=method)
            for capacity in sorted([protocol['policy']['primary_capacity']]+protocol['policy']['capacity_sensitivities']):
                r=evaluate_binary_risk(y,p,keys,capacity);r.pop('confusion_matrix')
                selected=select_top_risk(p,keys,capacity)
                r.update(fixed_policy_summary(y,p,selected,part.district_id,bootstrap_reps if capacity==.2 else 0))
                records.append(dict(**meta,capacity=capacity,**r))
            selected=select_top_risk(p,keys,.2)
            bins,_=reliability_bins(y,p);curves.append(bins.assign(**meta))
            for j,pos in enumerate(positions):
                rows.append(dict(**meta,temporal_row=int(pos),child_key_sha256=hashlib.sha256(keys[j].encode()).hexdigest(),
                                 probability=float(p[j]),selected_top20=bool(selected[j]),y_true=int(y[j])))
            if method=='sigmoid':
                for dimension in labels:
                    for name in sorted(labels[dimension].unique()):
                        keep=labels[dimension].eq(name).to_numpy()
                        result=fixed_policy_summary(y[keep],p[keep],selected[keep],part.loc[keep,'district_id'],bootstrap_reps)
                        subgroups.append(dict(**meta,dimension=dimension,subgroup=name,**result))
        print(f'Temporal scoring complete: {cohort}, n={len(part)}, events={int(y.sum())}',flush=True)
    return dict(metrics=pd.DataFrame(records),predictions=pd.DataFrame(rows),
                calibration_curves=pd.concat(curves,ignore_index=True),subgroups=pd.DataFrame(subgroups),
                cohorts=cohort_audit(train,test),region_strata=pd.DataFrame([dict(regioncode=k,risk_stratum=v) for k,v in strata.items()]))


def run_temporal_evaluation(root):
    """Reproduce the frozen fit and evaluation; never overwrite earlier results."""
    from pathlib import Path
    from .transformers import deflate_to_2009_prices
    root=Path(root);out=root/'phase2_implementation'
    protocol=json.loads((out/'temporal_evaluation_protocol.json').read_text())
    # Frozen manifest has the original machine's paths; resolve repo-relative.
    for original,expected in protocol['source_sha256'].items():
        relative=original.replace('\\','/').split('/AISD_Project/',1)[1]
        if hashlib.sha256((root/relative).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Frozen source changed: {relative}; review before rerunning.')
    train=pd.read_csv(root/'data/processed/train_w1_w2_raw.csv')
    test=deflate_to_2009_prices(pd.read_csv(root/'data/processed/test_w2_w3_raw.csv'))
    fits,audit,search,scores=fit_temporal_models(train,protocol)
    outputs=evaluate_temporal(train,test,predict_temporal(fits,test.drop(columns='dropout')),protocol)
    outputs.update(fit_audit=audit,tuning_search=search,calibration_training_scores=scores)
    for name,frame in outputs.items():frame.to_csv(out/f'temporal_{name}.csv',index=False)
    return outputs
