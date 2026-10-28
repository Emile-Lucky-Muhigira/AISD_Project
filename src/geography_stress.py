"""Adaptive exploratory extension: region-held-out selection inside district CV.

Motivated by the preceding retrospective failure. No later outcomes enter fits
or selection, but this is NOT a new independent confirmation experiment.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.model_selection import StratifiedGroupKFold
from .robustness_evaluation import (CANDIDATES,make_preprocessor,fit_candidate,
    fit_and_predict,metrics_and_rows)
from .model_tuning import _validate_splits,_validate_frame,_probability
from .evaluation_metrics import select_top_risk
from .temporal_evaluation import identity_keys
from .evaluation_split import temporal_masks
from .transformers import deflate_to_2009_prices


def region_splits(frame,seed):
    groups=frame.regioncode.fillna(-1)
    folds=_validate_splits(frame,StratifiedGroupKFold(3,shuffle=True,random_state=seed)
                         .split(frame,frame.dropout,groups),3)
    for a,b in folds:
        if set(groups.iloc[a])&set(groups.iloc[b]):raise ValueError('Region leakage')
    return folds


def region_bank(frame,seed):
    _validate_frame(frame)
    frame=frame.reset_index(drop=True);X=frame.drop(columns='dropout');y=frame.dropout.to_numpy()
    scores=np.full((len(frame),len(CANDIDATES)),np.nan);chosen=np.zeros_like(scores,dtype=bool)
    fold_ids=np.zeros(len(frame),int);keys=identity_keys(frame)
    for fold,(a,b) in enumerate(region_splits(frame,seed),1):
        prepared={};fold_ids[b]=fold
        for j,c in enumerate(CANDIDATES):
            key=(c['variant'],'lr' if c['model']=='elasticnet' else 'xgb')
            if key not in prepared:
                prep=make_preprocessor(c).fit(X.iloc[a])
                prepared[key]=(prep.transform(X.iloc[a]),prep.transform(X.iloc[b]))
            xa,xb=prepared[key];model=fit_candidate(c,xa,y[a]);p=_probability(model,xb)
            scores[b,j]=p;chosen[b,j]=select_top_risk(p,[keys[i] for i in b],.2)
    if not np.isfinite(scores).all():raise RuntimeError('Incomplete regional scores')
    return scores,chosen,fold_ids


def run_region_stage(root,stage):
    root=Path(root);out=root/'phase2_implementation'
    digest=hashlib.sha256(Path(__file__).read_bytes()+(root/'src/robustness_evaluation.py').read_bytes()).hexdigest()
    cached=Memory(root/'data/processed/region_stress_cache'/digest[:12],verbose=0).cache(region_bank)
    train=pd.read_csv(root/'data/processed/train_w1_w2_corrected.csv')
    if stage!='final':
        fold=int(stage)
        outer=_validate_splits(train,StratifiedGroupKFold(5,shuffle=True,random_state=42).split(train,train.dropout,train.district_id),5)
        a,b=outer[fold-1];part=train.iloc[a];target=train.iloc[b];seed=8200+fold
    else:
        fold=0;part=train;target=deflate_to_2009_prices(pd.read_csv(root/'data/processed/test_w2_w3_corrected.csv'));seed=9200
    print('Region stress stage',stage,'starting',flush=True)
    predictions,fit,search=fit_and_predict(part,target.drop(columns='dropout'),cached,seed,include_fixed=False)
    predictions={'region_'+k:v for k,v in predictions.items()}
    fit['configuration']='region_'+fit.configuration
    # Save training-selected configuration before later-outcome reporting.
    fit.assign(stage=stage).to_csv(out/f'region_stress_fit_{stage}.csv',index=False)
    search.assign(stage=stage).to_csv(out/f'region_stress_search_{stage}.csv',index=False)
    if stage=='final':
        masks=temporal_masks(train,target);tables=[[],[],[],[]]
        for scope,mask in [('primary',masks.in_primary_test),('secondary',masks.in_secondary_test)]:
            for parts,t in zip(tables,metrics_and_rows(target.loc[mask],{k:v[mask] for k,v in predictions.items()},scope,0)):parts.append(t)
        outputs=[pd.concat(t,ignore_index=True) for t in tables]
    else:outputs=metrics_and_rows(target,predictions,'training_oof',fold)
    for name,t in zip(['metrics','predictions','calibration_curves','subgroups'],outputs):t.to_csv(out/f'region_stress_{name}_{stage}.csv',index=False)
    print('Region stress stage',stage,'COMPLETE',flush=True)


def finalise_region_stress(root):
    root=Path(root);out=root/'phase2_implementation';stages=['1','2','3','4','5','final']
    for name in ['metrics','predictions','calibration_curves','subgroups','fit','search']:
        parts=[pd.read_csv(out/f'region_stress_{name}_{stage}.csv') for stage in stages]
        pd.concat(parts,ignore_index=True).to_csv(out/f'region_stress_{name}.csv',index=False)
    m=pd.read_csv(out/'region_stress_metrics.csv')
    summary=m.groupby(['scope','configuration']).agg(mean_ap=('average_precision','mean'),mean_brier=('brier_score','mean'),tp=('tp','sum'),fp=('fp','sum'),fn=('fn','sum'),selected_n=('selected_n','sum')).reset_index()
    summary['recall']=summary.tp/(summary.tp+summary.fn);summary['precision']=summary.tp/summary.selected_n
    summary.to_csv(out/'region_stress_summary.csv',index=False)
    protocol=json.loads((out/'region_stress_protocol.json').read_text());protocol['status']='completed'
    protocol['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'region_stress_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    return summary
