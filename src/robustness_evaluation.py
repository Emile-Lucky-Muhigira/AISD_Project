"""Prespecified corrected-data robustness study; frozen baseline is never overwritten.

All candidate selection and calibration exclude outer outcomes. Candidate family,
feature variant and class weight are selected together INSIDE each training fold.
Calibration cross-fitting repeats that selection inside each calibration fold.
Later outcomes are used for reporting only, after saving the selected protocol.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score
from .pipeline_builder import build_pipeline
from .model_evaluation import comparison_estimator, SENSITIVITY_VARIANTS
from .model_tuning import _validate_frame, _validate_splits, _probability
from .model_calibration import fit_calibrators, apply_calibrator, reliability_bins
from .evaluation_metrics import evaluate_binary_risk, select_top_risk
from .temporal_evaluation import identity_keys, fixed_policy_summary
from .evaluation_split import temporal_masks
from .subgroup_evaluation import subgroup_labels
from .transformers import deflate_to_2009_prices

FAMILIES = ('xgb','lightgbm','catboost','elasticnet','random_forest','extra_trees')
WEIGHTS = ('none','moderate','balanced')
CANDIDATES = tuple(dict(model=m,variant=v,weight=w) for m in FAMILIES
                   for v in SENSITIVITY_VARIANTS for w in WEIGHTS)


def candidate_id(c):
    return ':'.join(c[k] for k in ('model','variant','weight'))


def splits(frame, seed):
    return _validate_splits(frame, StratifiedGroupKFold(3,shuffle=True,random_state=seed)
        .split(frame,frame.dropout,frame.district_id),3)


def make_preprocessor(c):
    flags,geo=SENSITIVITY_VARIANTS[c['variant']]
    return build_pipeline('lr' if c['model']=='elasticnet' else 'xgb',
                          include_missing_flags=flags,include_geography=geo)


def fit_candidate(c, X, y):
    y=np.asarray(y); ratio=float((y==0).sum()/(y==1).sum())
    strength={'none':1.,'moderate':np.sqrt(ratio),'balanced':ratio}[c['weight']]
    est=comparison_estimator(c['model'],ratio)
    if c['model'] in ('xgb','lightgbm'): est.set_params(scale_pos_weight=strength)
    elif c['model']=='catboost': est.set_params(class_weights=[1.,strength])
    else:
        # Normalise class weights to mean one so regularisation is comparable.
        norm=len(y)/((y==0).sum()+strength*(y==1).sum())
        est.set_params(class_weight={0:norm,1:norm*strength})
        if c['model']=='elasticnet': est.set_params(max_iter=50000)
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        try:
            est.fit(X,y)
        except ConvergenceWarning:
            if c['model']!='elasticnet':raise
            # Preserve weighted regularised objective while bounding SAGA weights.
            weights=np.array([est.class_weight[int(v)] for v in y])
            scale=float(weights.max())
            est.set_params(class_weight=None,C=est.C*scale,max_iter=100000)
            est.fit(X,y,sample_weight=weights/scale)
    return est


def bank(frame, seed):
    """OOF scores for fixed candidates; preprocessing shared only within a split."""
    _validate_frame(frame)
    frame=frame.reset_index(drop=True); y=frame.dropout.to_numpy(); X=frame.drop(columns='dropout')
    scores=np.full((len(frame),len(CANDIDATES)),np.nan); selected=np.zeros_like(scores,dtype=bool)
    keys=identity_keys(frame); folds=np.zeros(len(frame),int)
    for fold,(a,b) in enumerate(splits(frame,seed),1):
        prepared={}; folds[b]=fold
        for j,c in enumerate(CANDIDATES):
            key=(c['variant'],'lr' if c['model']=='elasticnet' else 'xgb')
            if key not in prepared:
                prep=make_preprocessor(c).fit(X.iloc[a])
                prepared[key]=(prep.transform(X.iloc[a]),prep.transform(X.iloc[b]))
            xa,xb=prepared[key]; est=fit_candidate(c,xa,y[a]); p=_probability(est,xb)
            scores[b,j]=p; selected[b,j]=select_top_risk(p,[keys[i] for i in b],.2)
    if not np.isfinite(scores).all(): raise RuntimeError('Incomplete candidate bank')
    return scores,selected,folds


def select_candidate(y,scores,selected,folds,objective):
    """Pooled event recall across fold-specific fixed-capacity allocations."""
    y=np.asarray(y); rows=[]
    for j,c in enumerate(CANDIDATES):
        ap=float(np.mean([average_precision_score(y[folds==f],scores[folds==f,j]) for f in np.unique(folds)]))
        rows.append(dict(candidate=j,**c,inner_recall=float(y[selected[:,j]].sum()/y.sum()),inner_ap=ap))
    audit=pd.DataFrame(rows)
    columns=['inner_recall','inner_ap','candidate'] if objective=='capacity' else ['inner_ap','inner_recall','candidate']
    if objective not in ('capacity','ap'):raise ValueError('Unknown objective')
    winner=int(audit.sort_values(columns,ascending=[False,False,True]).iloc[0].candidate)
    return winner,audit


def raw_predict(frame, target_features, candidate):
    if 'dropout' in target_features:raise ValueError('Prediction features must exclude outcomes')
    prep=make_preprocessor(candidate).fit(frame.drop(columns='dropout'))
    est=fit_candidate(candidate,prep.transform(frame.drop(columns='dropout')),frame.dropout)
    return _probability(est,prep.transform(target_features))


def fit_and_predict(frame, target_features, cached_bank, seed, include_fixed=True):
    """Return predictions before any target outcomes are available to this function."""
    frame=frame.reset_index(drop=True); y=frame.dropout.to_numpy()
    scores,chosen,folds=cached_bank(frame,seed)
    winners={};search=[];cal_scores={o:np.full(len(frame),np.nan) for o in ('ap','capacity')}
    for objective in cal_scores:
        winner,audit=select_candidate(y,scores,chosen,folds,objective)
        winners[objective]=winner;search.append(audit.assign(objective=objective))
    # Re-select entire procedure within each calibration-training subset.
    for a,b in splits(frame,seed+100):
        part=frame.iloc[a].reset_index(drop=True)
        ss,cc,ff=cached_bank(part,seed+200)
        for objective in cal_scores:
            j,_=select_candidate(part.dropout,ss,cc,ff,objective)
            cal_scores[objective][b]=raw_predict(part,frame.iloc[b].drop(columns='dropout'),CANDIDATES[j])
    predictions={};fits=[]
    for objective,j in winners.items():
        raw=raw_predict(frame,target_features,CANDIDATES[j])
        cal=fit_calibrators(y,cal_scores[objective])['sigmoid']
        name='selected_'+objective
        predictions[name]=apply_calibrator('sigmoid',cal,raw)
        fits.append(dict(configuration=name,**CANDIDATES[j],sigmoid_slope=float(cal.coef_[0,0])))
    if include_fixed:
        for j,c in enumerate(CANDIDATES):
            if c['weight']!='balanced':continue
            raw=raw_predict(frame,target_features,c)
            cal=fit_calibrators(y,scores[:,j])['sigmoid']
            name=candidate_id(c);predictions[name]=apply_calibrator('sigmoid',cal,raw)
            fits.append(dict(configuration=name,**c,sigmoid_slope=float(cal.coef_[0,0])))
    return predictions,pd.DataFrame(fits),pd.concat(search,ignore_index=True)


def metrics_and_rows(frame, predictions, scope, fold):
    metrics=[];rows=[];curves=[];groups=[];keys=identity_keys(frame)
    for name,p in predictions.items():
        r=evaluate_binary_risk(frame.dropout,p,keys,.2);cm=r.pop('confusion_matrix')
        bins,ece=reliability_bins(frame.dropout,p)
        r.update(scope=scope,fold=fold,configuration=name,tp=cm[1][1],fp=cm[0][1],fn=cm[1][0],tn=cm[0][0],ece=ece)
        metrics.append(r);curves.append(bins.assign(scope=scope,fold=fold,configuration=name))
        chosen=select_top_risk(p,keys,.2)
        rows.extend(dict(scope=scope,fold=fold,configuration=name,child_key_sha256=hashlib.sha256(k.encode()).hexdigest(),
                         district_id=int(d),y_true=int(y),probability=float(q),selected=bool(z))
                    for k,d,y,q,z in zip(keys,frame.district_id,frame.dropout,p,chosen))
        labels=subgroup_labels(frame)
        for dimension in labels:
            for label in sorted(labels[dimension].unique()):
                mask=labels[dimension].eq(label).to_numpy()
                rr=fixed_policy_summary(frame.dropout.to_numpy()[mask],p[mask],chosen[mask],frame.district_id.to_numpy()[mask],0)
                groups.append(dict(scope=scope,fold=fold,configuration=name,dimension=dimension,subgroup=label,**rr))
    return [pd.DataFrame(metrics),pd.DataFrame(rows),pd.concat(curves),pd.DataFrame(groups)]


def run_robustness(root):
    root=Path(root); out=root/'phase2_implementation'; start=time.time()
    source=Path(__file__).read_bytes(); digest=hashlib.sha256(source).hexdigest()
    cache=Memory(root/'data/processed/robustness_cache'/digest[:12],verbose=0).cache(bank)
    train=pd.read_csv(root/'data/processed/train_w1_w2_corrected.csv')
    original=pd.read_csv(root/'data/processed/train_w1_w2_raw.csv')
    pd.testing.assert_frame_equal(train[['FPrimary','hhmid','district_id','dropout']],original[['FPrimary','hhmid','district_id','dropout']])
    assert train.columns[-1]=='dropout'
    protocol=dict(date='2026-10-28',source_sha256=digest,training_sha256=hashlib.sha256((root/'data/processed/train_w1_w2_corrected.csv').read_bytes()).hexdigest(),
        candidates=list(CANDIDATES),capacity=.2,primary_procedure='selected_capacity',comparator='selected_ap',
        fixed_ablations='Six families x four variants; balanced weights; sigmoid calibration from independent OOF scores',
        selection='inner pooled recall at 20%; tie mean fold AP then fixed candidate order; AP comparator reverses the first two criteria',
        calibration='Nested cross-fitting repeats candidate selection; no outer labels used',
        interpretation='Exploratory retrospective extension; no independent confirmatory cohort; no temporal-outcome-driven iteration')
    (out/'robustness_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    _validate_frame(train)
    outer=_validate_splits(train,StratifiedGroupKFold(5,shuffle=True,random_state=42).split(train,train.dropout,train.district_id),5)
    accum=[[],[],[],[]];audits=[];searches=[]
    for fold,(a,b) in enumerate(outer,1):
        print(f'Robustness outer fold {fold}/5 starting',flush=True)
        pred,fit,search=fit_and_predict(train.iloc[a],train.iloc[b].drop(columns='dropout'),cache,4200+fold)
        for dest,table in zip(accum,metrics_and_rows(train.iloc[b],pred,'training_oof',fold)):dest.append(table)
        audits.append(fit.assign(scope='training_oof',fold=fold));searches.append(search.assign(scope='training_oof',fold=fold))
        for name,parts in zip(('metrics','predictions','calibration_curves','subgroups'),accum):pd.concat(parts,ignore_index=True).to_csv(out/f'robustness_{name}.csv',index=False)
        print(f'Robustness fold {fold} complete',flush=True)
    print('Training-only final selection and calibration',flush=True)
    # Only later FEATURES reach fitting/prediction. Outcomes enter reporting below.
    test=deflate_to_2009_prices(pd.read_csv(root/'data/processed/test_w2_w3_corrected.csv'))
    base_test=pd.read_csv(root/'data/processed/test_w2_w3_raw.csv')
    pd.testing.assert_frame_equal(test[['FPrimary','hhmid','district_id','dropout']],base_test[['FPrimary','hhmid','district_id','dropout']])
    pred,fit,search=fit_and_predict(train,test.drop(columns='dropout'),cache,5200)
    audits.append(fit.assign(scope='final_training_fit',fold=0));searches.append(search.assign(scope='final_training_fit',fold=0))
    pd.concat(audits,ignore_index=True).to_csv(out/'robustness_fit_audit.csv',index=False)
    pd.concat(searches,ignore_index=True).to_csv(out/'robustness_search.csv',index=False)
    protocol['final_training_selected']=fit[fit.configuration.str.startswith('selected_')].to_dict('records')
    (out/'robustness_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    masks=temporal_masks(original,base_test)
    for scope,mask in [('primary',masks.in_primary_test),('secondary',masks.in_secondary_test)]:
        for dest,table in zip(accum,metrics_and_rows(test.loc[mask],{k:v[mask] for k,v in pred.items()},scope,0)):dest.append(table)
    for name,parts in zip(('metrics','predictions','calibration_curves','subgroups'),accum):pd.concat(parts,ignore_index=True).to_csv(out/f'robustness_{name}.csv',index=False)
    m=pd.concat(accum[0],ignore_index=True)
    summary=m.groupby(['scope','configuration']).agg(mean_ap=('average_precision','mean'),mean_brier=('brier_score','mean'),tp=('tp','sum'),fp=('fp','sum'),fn=('fn','sum'),selected_n=('selected_n','sum')).reset_index()
    summary['recall']=summary.tp/(summary.tp+summary.fn);summary['precision']=summary.tp/summary.selected_n
    summary.to_csv(out/'robustness_summary.csv',index=False)
    (out/'robustness_execution.json').write_text(json.dumps(dict(success=True,seconds=time.time()-start,protocol=protocol,outer_folds=5),indent=2)+'\n')
    return summary
