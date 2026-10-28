"""Exploratory IP-follow-up sensitivity under MAR of full follow-up status.
Known completers enter the observation model but not the binary dropout fit.
Every observation model excludes outcome-validation districts and households.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from .pipeline_builder import build_pipeline
from .followup_diagnostics import observation_indicator
from .model_calibration import _log_odds
from .robustness_evaluation import metrics_and_rows,splits
from .evaluation_split import temporal_masks
from .transformers import deflate_to_2009_prices


def observation_weights(baseline, frame, excluded):
    """Cross-fit propensities on allowed baseline groups, then align by identity."""
    base=baseline[~baseline.district_id.isin(excluded)].reset_index(drop=True)
    if set(frame.district_id)&set(excluded):raise ValueError('Excluded prediction district in training')
    y=observation_indicator(base.status).to_numpy();X=base.drop(columns='status');p=np.full(len(base),np.nan)
    for a,b in StratifiedGroupKFold(3,shuffle=True,random_state=7242).split(X,y,base.district_id):
        for col in ['district_id','FPrimary']:
            if set(base.iloc[a][col])&set(base.iloc[b][col]):raise ValueError('Observation-model group leakage')
        pipe=build_pipeline('lr',LogisticRegression(C=.1,max_iter=10000)).fit(X.iloc[a],y[a])
        p[b]=pipe.predict_proba(X.iloc[b])[:,1]
    probs=base[['FPrimary','hhmid']].assign(observation_probability=p)
    aligned=frame[['FPrimary','hhmid']].merge(probs,on=['FPrimary','hhmid'],how='left',validate='one_to_one')
    if aligned.observation_probability.isna().any():raise ValueError('Missing cross-fit propensity')
    weights=np.minimum(y.mean()/aligned.observation_probability.to_numpy(),5.)
    return weights


def fit_predict(train,features,weights):
    if 'dropout' in features:raise ValueError('Prediction outcomes not permitted')
    prep=build_pipeline('xgb').fit(train.drop(columns='dropout'))
    est=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=5,max_features='sqrt',class_weight='balanced',random_state=42,n_jobs=1)
    est.fit(prep.transform(train.drop(columns='dropout')),train.dropout,sample_weight=weights)
    return est.predict_proba(prep.transform(features))[:,1]


def calibrated_prediction(baseline,train,features,excluded):
    train=train.reset_index(drop=True);weights=observation_weights(baseline,train,excluded)
    scores=np.full(len(train),np.nan)
    for a,b in splits(train,7342):
        inner=train.iloc[a];blocked=set(excluded)|set(train.iloc[b].district_id)
        w=observation_weights(baseline,inner,blocked)
        scores[b]=fit_predict(inner,train.iloc[b].drop(columns='dropout'),w)
    cal=LogisticRegression(C=np.inf,max_iter=10000).fit(_log_odds(scores),train.dropout,sample_weight=weights)
    raw=fit_predict(train,features,weights)
    return cal.predict_proba(_log_odds(raw))[:,1],dict(max_weight=float(weights.max()),effective_n=float(weights.sum()**2/(weights@weights)),sigmoid_slope=float(cal.coef_[0,0]))


def run_followup_weighted(root):
    root=Path(root);out=root/'phase2_implementation'
    train=pd.read_csv(root/'data/processed/train_w1_w2_corrected.csv')
    baseline=pd.read_csv(root/'data/processed/w1_baseline_all_statuses.csv')
    tables=[[],[],[],[]];audits=[]
    cv=StratifiedGroupKFold(5,shuffle=True,random_state=42)
    for fold,(a,b) in enumerate(cv.split(train,train.dropout,train.district_id),1):
        p,audit=calibrated_prediction(baseline,train.iloc[a],train.iloc[b].drop(columns='dropout'),set(train.iloc[b].district_id))
        for parts,t in zip(tables,metrics_and_rows(train.iloc[b],{'ip_followup_extra_trees':p},'training_oof',fold)):parts.append(t)
        audits.append(dict(scope='training_oof',fold=fold,**audit));print(f'Follow-up-weighted fold {fold}/5 complete',flush=True)
    test=deflate_to_2009_prices(pd.read_csv(root/'data/processed/test_w2_w3_corrected.csv'))
    p,audit=calibrated_prediction(baseline,train,test.drop(columns='dropout'),set())
    audits.append(dict(scope='final_training_fit',fold=0,**audit))
    masks=temporal_masks(train,test)
    for scope,mask in [('primary',masks.in_primary_test),('secondary',masks.in_secondary_test)]:
        for parts,t in zip(tables,metrics_and_rows(test.loc[mask],{'ip_followup_extra_trees':p[mask]},scope,0)):parts.append(t)
    for name,parts in zip(['metrics','predictions','calibration_curves','subgroups'],tables):pd.concat(parts,ignore_index=True).to_csv(out/f'followup_weighted_{name}.csv',index=False)
    pd.DataFrame(audits).to_csv(out/'followup_weighted_fit_audit.csv',index=False)
    protocol=dict(status='completed_exploratory_sensitivity',observation_model='C=0.1 logistic; 3-fold district-grouped cross-fitting',
        estimand='Dropout among baseline children who would not complete basic education by follow-up; existing binary outcome unchanged',
        assumptions='Full follow-up status independent of observation conditional on baseline predictors; positivity; adequate propensity specification. Assumptions not established by these data.',
        weighting='Stabilised observation weights capped at 5, in tree fitting and calibration. Known completers included as observed in propensity model, excluded from binary outcome fit.',
        evaluation='Unweighted performance on observed held-out children; not an estimate of performance on unobserved outcomes.',
        adoption='Sensitivity only; no later-outcome-driven weight, cap, model or threshold tuning.')
    (out/'followup_weighted_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    return pd.concat(tables[0],ignore_index=True)
