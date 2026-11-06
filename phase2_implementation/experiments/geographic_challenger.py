"""Fixed LightGBM challenger; historical town mapping, training-only calibration."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.pipeline_builder import build_pipeline
from src.model_evaluation import comparison_estimator
from src.model_calibration import fit_calibrators,apply_calibrator
from src.transformers import deflate_to_2009_prices
from src.evaluation_split import temporal_masks
from src.temporal_evaluation import identity_keys,fixed_policy_summary
from src.evaluation_metrics import select_top_risk

def run():
 out=ROOT/'phase2_implementation/experiments/geographic_grouping_results'
 counts=pd.read_csv(out/'district_town_support.csv');links={}
 for district,g in counts.groupby('district_id'):
  top=g[g.households==g.households.max()]
  if len(top)==1 and top.households.iloc[0]>=5:links[district]=str(int(top.town.iloc[0]))
 train=pd.read_csv(ROOT/'data/processed/train_w1_w2_raw.csv');test=deflate_to_2009_prices(pd.read_csv(ROOT/'data/processed/test_w2_w3_raw.csv'))
 for d in [train,test]:d['work_town']=d.district_id.map(links).fillna('0')
 X=train.drop(columns='dropout');y=train.dropout.to_numpy();rows=[];predictions=[]
 masks=temporal_masks(train,test)
 for variant in ['official_regions','official_regions_plus_work_town']:
  def make(labels):
   p=build_pipeline('xgb',comparison_estimator('lightgbm',float((labels==0).sum()/(labels==1).sum())))
   if variant.endswith('work_town'):p.named_steps['prep'].transformers.append(('work_town',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['work_town']))
   return p
  oof=np.full(len(train),np.nan)
  for a,b in StratifiedGroupKFold(5,shuffle=True,random_state=42).split(X,y,train.district_id):
   assert not set(train.iloc[a].district_id)&set(train.iloc[b].district_id)
   oof[b]=make(y[a]).fit(X.iloc[a],y[a]).predict_proba(X.iloc[b])[:,1]
  cal=fit_calibrators(y,oof)['sigmoid'];assert cal.coef_[0,0]>0
  fit=make(y).fit(X,y);prob=apply_calibrator('sigmoid',cal,fit.predict_proba(test.drop(columns='dropout'))[:,1])
  for cohort,mask in [('primary',masks.in_primary_test),('secondary',masks.in_secondary_test)]:
   d=test.loc[mask];p=prob[mask];sel=select_top_risk(p,identity_keys(d),.2);m=fixed_policy_summary(d.dropout.to_numpy(),p,sel,d.district_id.to_numpy(),bootstrap_reps=1000);m.update(variant=variant,cohort=cohort,town_covered=int(d.work_town.ne('0').sum()));rows.append(m)
   predictions.extend(dict(variant=variant,cohort=cohort,child_hash=hashlib.sha256(key.encode()).hexdigest(),y=int(yy),probability=float(pp),selected=bool(ss)) for key,yy,pp,ss in zip(identity_keys(d),d.dropout,p,sel))
 pd.DataFrame(rows).to_csv(out/'challenger_temporal_metrics.csv',index=False);pd.DataFrame(predictions).to_csv(out/'challenger_temporal_predictions.csv',index=False)
 print(pd.DataFrame(rows)[['variant','cohort','tp','fp','fn','precision_at_capacity','recall_at_capacity','town_covered']].to_string(index=False))
 return pd.DataFrame(rows)
if __name__=='__main__':run()
