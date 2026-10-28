"""Assess follow-up weighting feasibility without changing the binary outcome."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score,brier_score_loss
from .pipeline_builder import build_pipeline
from .model_calibration import reliability_bins

KNOWN = ('enrolled','dropout','completed_basic')
UNKNOWN = ('moved_out','household_not_found','id_mismatch','not_in_w2_roster','status_unknown')


def observation_indicator(status):
    if not status.isin(KNOWN+UNKNOWN).all():raise ValueError('Unrecognised follow-up status')
    return status.isin(KNOWN).astype(int)


def weight_summary(p, observed, cap=None):
    p=np.asarray(p,float);observed=np.asarray(observed,bool)
    if not np.isfinite(p).all() or (p<=0).any() or (p>=1).any():raise ValueError('Invalid observation probabilities')
    w=observed.mean()/p[observed]
    if cap is not None:w=np.minimum(w,cap)
    return w,dict(n_observed=int(observed.sum()),min_weight=float(w.min()),max_weight=float(w.max()),
                  effective_sample_size=float(w.sum()**2/(w*w).sum()),mean_weight=float(w.mean()))


def run_followup_diagnostics(root):
    root=Path(root);out=root/'phase2_implementation'
    d=pd.read_csv(root/'data/processed/w1_baseline_all_statuses.csv')
    y=observation_indicator(d.status).to_numpy();X=d.drop(columns='status')
    p=np.full(len(d),np.nan);folds=[]
    for fold,(a,b) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=6242).split(X,y,d.district_id),1):
        for col in ['district_id','FPrimary']:
            if set(d.iloc[a][col])&set(d.iloc[b][col]):raise ValueError('Group overlap')
        pipe=build_pipeline('lr',LogisticRegression(C=.1,max_iter=10000))
        pipe.fit(X.iloc[a],y[a]);p[b]=pipe.predict_proba(X.iloc[b])[:,1]
        folds.append(dict(fold=fold,fit_n=len(a),validation_n=len(b),shared_households=0,shared_districts=0))
    bins,ece=reliability_bins(y,p);bins.to_csv(out/'followup_calibration.csv',index=False)
    records=[];balance=[]
    for cap in [None,5.,10.]:
        w,summary=weight_summary(p,y,cap);label='none' if cap is None else str(cap)
        records.append(dict(weight_cap=label,**summary))
        for col in ['age','sex','father_in_hh','mother_in_hh','asset_count','urbrur']:
            v=pd.to_numeric(d[col],errors='coerce');obs=v[y==1].to_numpy();ok=np.isfinite(obs)
            balance.append(dict(weight_cap=label,feature=col,baseline_mean=float(v.mean()),observed_mean=float(np.nanmean(obs)),
                                weighted_observed_mean=float(np.average(obs[ok],weights=w[ok]))))
    pd.DataFrame(records).to_csv(out/'followup_weight_diagnostics.csv',index=False)
    pd.DataFrame(balance).to_csv(out/'followup_balance.csv',index=False)
    result=dict(n=len(d),known=int(y.sum()),unknown=int((1-y).sum()),completed_basic=int(d.status.eq('completed_basic').sum()),
        followup_auc=float(roc_auc_score(y,p)),followup_brier=float(brier_score_loss(y,p)),ece=ece,
        probability_quantiles={str(q):float(np.quantile(p,q)) for q in [0,.01,.05,.5,.95,.99,1]},
        folds=folds,dropout_weighting_applied_in_this_diagnostic=False,separate_weighted_sensitivity='followup_weighted_protocol.json',
        reason='Observed completers are known follow-up, not attrition. Binary dropout analysis conditions on non-completion, which is unknown among lost children. A weighted sensitivity is identifiable only under a baseline-conditional missing-at-random assumption for the full follow-up status (including completion), positivity and adequate observation-model specification. These assumptions are not verified by weight stability; this diagnostic alone is not a corrected population dropout estimate.',
        limitations='Observation model uses baseline predictors only and grouped OOF validation. MAR/positivity cannot be established from observed data; missing outcomes are not imputed. Weight caps are sensitivity diagnostics, not selected for prediction gains.')
    (out/'followup_feasibility.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
