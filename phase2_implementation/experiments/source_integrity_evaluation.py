"""Paired training-only sensitivity to source-conflicting duplicate values."""
from pathlib import Path
import sys,json
import pandas as pd,numpy as np
from sklearn.model_selection import StratifiedGroupKFold
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.model_evaluation import evaluate_baselines
E=ROOT/'phase2_implementation/experiments/source_integrity_results'
LOCAL=ROOT/'data/processed/source_integrity_audit'
base=pd.read_csv(ROOT/'data/processed/train_w1_w2_raw.csv');a=pd.read_csv(LOCAL/'train_baseline.csv');b=pd.read_csv(LOCAL/'train_conservative.csv')
keys=['FPrimary','hhmid'];a=a.set_index(keys);b=b.set_index(keys);updated=base.set_index(keys).copy()
changes=~(a.eq(b)|(a.isna()&b.isna()))
assert not changes['dropout'].any()
for c in changes.columns:
 ids=changes.index[changes[c]]
 if len(ids):updated.loc[ids,c]=b.loc[ids,c]
updated=updated.reset_index()[base.columns]
models=('elasticnet','xgb','lightgbm','catboost','random_forest','extra_trees');results=[]
for seed in [42,137,293]:
 splits=list(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(base,base.dropout,base.district_id))
 for name,data in [('original',base),('duplicate_corrected',updated)]:
  metrics,_=evaluate_baselines(data,splits,models=models,progress=False);metrics['variant']=name;metrics['seed']=seed;results.append(metrics)
  print(seed,name,'complete',flush=True)
m=pd.concat(results,ignore_index=True);m.to_csv(E/'paired_metrics.csv',index=False)
s=m.groupby(['variant','model','seed']).agg(tp=('tp','sum'),fp=('fp','sum'),fn=('fn','sum'),ap=('average_precision','mean')).reset_index();s['precision']=s.tp/(s.tp+s.fp);s['recall']=s.tp/(s.tp+s.fn);s['f1']=2*s.tp/(2*s.tp+s.fp+s.fn);s.to_csv(E/'paired_summary.csv',index=False)
print(s.groupby(['variant','model'])[['tp','fp','precision','recall','f1','ap']].mean().to_string())
