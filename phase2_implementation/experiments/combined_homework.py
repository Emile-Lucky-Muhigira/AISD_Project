from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
import sys
sys.path.insert(0,str(ROOT))
import pandas as pd,numpy as np,pyreadstat
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score
from src.pipeline_builder import build_pipeline
from src.model_evaluation import comparison_estimator
from src.evaluation_metrics import select_top_risk
from src.temporal_evaluation import identity_keys
from src.data_cleaning import align
out=Path("phase2_implementation/experiments/combined_homework_results");out.mkdir(exist_ok=True)
d=pd.read_csv(str(ROOT / 'data/processed/train_w1_w2_raw.csv'));raw,_=pyreadstat.read_dta(str(ROOT / 'data/raw/Ghana_Panel_Survey/Data/Wave 1/s1fi.dta'));e=align(d,raw,"s1fi_hhmid");h=pd.to_numeric(e.s1fi_12i,errors="coerce");m=pd.to_numeric(e.s1fi_12ii,errors="coerce");holiday=h.eq(99)|m.eq(99);h=h.where(h.between(0,98));m=m.where(m.between(0,59));v=(h.fillna(0)+m.fillna(0)/60).where(h.notna()|m.notna()).mask(holiday)
X=d.drop(columns="dropout").copy();X["homework_hours"]=v;y=d.dropout.to_numpy();keys=identity_keys(d);print("homework observed",int(v.notna().sum()),"holiday",int(holiday.sum()),flush=True);rows=[]
for seed in [42,137,293]:
 for fold,(a,b) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(X,y,d.district_id)):
  for variant in ["baseline","homework_clean","homework_nogeo","homework_clean_nogeo"]:
   fit=a
   if "clean" in variant:fit=a[d.iloc[a][["school_type","grade_level"]].notna().all(axis=1).to_numpy()]
   prep=build_pipeline(include_geography="nogeo" not in variant)
   if variant!="baseline":prep.named_steps["prep"].transformers=list(prep.named_steps["prep"].transformers)+[("homework",SimpleImputer(strategy="median",keep_empty_features=True),["homework_hours"])]
   prep.fit(X.iloc[fit]);A=prep.transform(X.iloc[fit]);B=prep.transform(X.iloc[b])
   for model in ["xgb","random_forest","extra_trees"]:
    est=comparison_estimator(model,float((y[fit]==0).sum()/y[fit].sum())).fit(A,y[fit]);p=est.predict_proba(B)[:,1];sel=select_top_risk(p,[keys[i] for i in b],.2);tp=int(y[b][sel].sum());rows.append(dict(seed=seed,fold=fold,variant=variant,model=model,ap=average_precision_score(y[b],p),tp=tp,fp=int(sel.sum())-tp,fn=int(y[b].sum())-tp))
  pd.DataFrame(rows).to_csv(out/"fold_metrics.csv",index=False);print(seed,fold,"complete",flush=True)
s=pd.DataFrame(rows).groupby(["seed","variant","model"]).agg(ap=("ap","mean"),tp=("tp","sum"),fp=("fp","sum"),fn=("fn","sum")).reset_index();s.to_csv(out/"summary.csv",index=False);print(s.to_string(index=False));print(s.groupby(["variant","model"])[["ap","tp","fp","fn"]].mean().to_string())
