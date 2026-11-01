from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
import sys,warnings
from pathlib import Path
sys.path.insert(0,str(ROOT));warnings.filterwarnings("ignore")
import pandas as pd,numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score
from src.pipeline_builder import build_pipeline
from src.model_evaluation import comparison_estimator
from src.evaluation_metrics import select_top_risk
from src.temporal_evaluation import identity_keys
out=Path("phase2_implementation/experiments/anomaly_cleaning_results");out.mkdir(exist_ok=True)
d=pd.read_csv(str(ROOT / 'data/processed/train_w1_w2_raw.csv'));X=d.drop(columns="dropout");y=d.dropout.to_numpy();keys=identity_keys(d);rows=[];audits=[]
for seed in [42,137,293]:
 for fold,(a,b) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(X,y,d.district_id)):
  detector_prep=build_pipeline().fit(X.iloc[a]);A=detector_prep.transform(X.iloc[a]);det=IsolationForest(n_estimators=200,random_state=42,n_jobs=1).fit(A);scores=det.score_samples(A)
  for fraction in [0,.01,.03,.05]:
   remove=np.argsort(scores,kind="stable")[:int(len(a)*fraction)];keep=np.ones(len(a),bool);keep[remove]=False;fit=a[keep]
   audits.append(dict(seed=seed,fold=fold,fraction=fraction,removed=len(remove),removed_events=int(y[a[remove]].sum()),removed_mother_only=int(((d.iloc[a[remove]].mother_in_hh==1)&(d.iloc[a[remove]].father_in_hh==0)).sum())))
   prep=detector_prep if fraction==0 else build_pipeline().fit(X.iloc[fit]);AA=prep.transform(X.iloc[fit]);B=prep.transform(X.iloc[b])
   for model in ["random_forest","extra_trees","xgb"]:
    est=comparison_estimator(model,float((y[fit]==0).sum()/y[fit].sum())).fit(AA,y[fit]);p=est.predict_proba(B)[:,1];sel=select_top_risk(p,[keys[i] for i in b],.2);tp=int(y[b][sel].sum())
    rows.append(dict(seed=seed,fold=fold,fraction=fraction,model=model,ap=average_precision_score(y[b],p),tp=tp,fp=int(sel.sum())-tp,fn=int(y[b].sum())-tp))
  pd.DataFrame(rows).to_csv(out/"fold_metrics.csv",index=False);pd.DataFrame(audits).to_csv(out/"removal_audit.csv",index=False);print(seed,fold,"complete",flush=True)
s=pd.DataFrame(rows).groupby(["seed","fraction","model"]).agg(ap=("ap","mean"),tp=("tp","sum"),fp=("fp","sum"),fn=("fn","sum")).reset_index();s["precision"]=s.tp/(s.tp+s.fp);s["recall"]=s.tp/(s.tp+s.fn);s["f1"]=2*s.tp/(2*s.tp+s.fp+s.fn);s.to_csv(out/"summary.csv",index=False);print(s.groupby(["fraction","model"])[["ap","tp","fp","precision","recall","f1"]].mean().round(4).to_string(),flush=True)
