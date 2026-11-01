from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
import sys,json,warnings
from pathlib import Path
sys.path.insert(0,str(ROOT))
import pandas as pd,numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score
from src.pipeline_builder import build_pipeline
from src.model_tuning import _fit_estimator,_validate_splits
from src.evaluation_metrics import select_top_risk
from src.temporal_evaluation import identity_keys
out=Path("phase2_implementation/experiments/nested_cleaning_results");out.mkdir(exist_ok=True)
d=pd.read_csv(str(ROOT / 'data/processed/train_w1_w2_raw.csv'));X=d.drop(columns="dropout");y=d.dropout.to_numpy();keys=identity_keys(d)
models=["xgb","lightgbm","catboost","elasticnet","random_forest","extra_trees"]
valid=d[["school_type","grade_level"]].notna().all(axis=1).to_numpy()
def prepared(a,b,rule,kind):
 fit=a if rule=="all" else a[valid[a]]
 prep=build_pipeline(kind).fit(X.iloc[fit]);return fit,prep.transform(X.iloc[fit]),prep.transform(X.iloc[b])
def score(y,p,ids):
 sel=select_top_risk(p,ids,.2);tp=int(y[sel].sum());return dict(ap=float(average_precision_score(y,p)),tp=tp,fp=int(sel.sum())-tp,fn=int(y.sum())-tp)
rows=[];search=[];preds=[]
outer=_validate_splits(d,StratifiedGroupKFold(5,shuffle=True,random_state=42).split(X,y,d.district_id),5)
for fold,(a,b) in enumerate(outer):
 part=d.iloc[a].reset_index(drop=True);inner=_validate_splits(part,StratifiedGroupKFold(3,shuffle=True,random_state=9042+fold).split(part,part.dropout,part.district_id),3);scores=[]
 for j,(ia,ib) in enumerate(inner):
  aa=a[ia];bb=a[ib]
  for rule in ["all","known_school_grade"]:
   cache={}
   for model in models:
    kind="lr" if model=="elasticnet" else "xgb"
    if kind not in cache:cache[kind]=prepared(aa,bb,rule,kind)
    fit,A,B=cache[kind];est,_=_fit_estimator(model,{},A,y[fit]);p=est.predict_proba(B)[:,1];r=score(y[bb],p,[keys[i] for i in bb]);r.update(outer=fold,inner=j,rule=rule,model=model);scores.append(r)
 search.extend(scores);s=pd.DataFrame(scores).groupby(["rule","model"]).agg(tp=("tp","sum"),ap=("ap","mean")).reset_index().sort_values(["tp","ap","rule","model"],ascending=[False,False,True,True]);chosen=s.iloc[0];base=s[s.rule=="all"].iloc[0]
 for policy,c in [("nested_cleaning",chosen),("nested_all_rows",base)]:
  fit,A,B=prepared(a,b,c.rule,"lr" if c.model=="elasticnet" else "xgb");est,_=_fit_estimator(c.model,{},A,y[fit]);p=est.predict_proba(B)[:,1];r=score(y[b],p,[keys[i] for i in b]);r.update(fold=fold,policy=policy,rule=c.rule,model=c.model,retained=len(fit));rows.append(r)
  preds.extend(dict(policy=policy,fold=fold,row=int(i),y=int(y[i]),score=float(q)) for i,q in zip(b,p))
 pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False);pd.DataFrame(search).to_csv(out/"search.csv",index=False);pd.DataFrame(preds).to_csv(out/"predictions.csv",index=False);print("outer",fold+1,"complete",chosen.rule,chosen.model,flush=True)
r=pd.DataFrame(rows);print(r.to_string(index=False));print(r.groupby("policy").agg(ap=("ap","mean"),tp=("tp","sum"),fp=("fp","sum"),fn=("fn","sum")).to_string())
