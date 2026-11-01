from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
import sys,json,warnings,itertools
from pathlib import Path
sys.path.insert(0,str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score
from src.pipeline_builder import build_pipeline
from src.model_evaluation import comparison_estimator
from src.evaluation_metrics import select_top_risk
from src.temporal_evaluation import identity_keys
out=Path("phase2_implementation/experiments/selective_cleaning_results");out.mkdir(exist_ok=True)
d=pd.read_csv(str(ROOT / 'data/processed/train_w1_w2_raw.csv'));X=d.drop(columns="dropout");y=d.dropout.to_numpy();keys=identity_keys(d)
cols=["school_type","grade_level","attend_hours","travel_minutes","father_edu","mother_edu","has_textbooks"]
rules={"all":np.ones(len(d),bool)}
for c in cols:rules["known_"+c]=d[c].notna().to_numpy()
for a,b in itertools.combinations(["school_type","grade_level","attend_hours","travel_minutes"],2):rules["known_"+a+"_"+b]=d[[a,b]].notna().all(axis=1).to_numpy()
for n in [1,2,3]:rules["core_missing_at_most_"+str(n)]=(d[cols].isna().sum(axis=1)<=n).to_numpy()
rules["one_learning_score"]=d[["math_score","english_score"]].notna().any(axis=1).to_numpy()
rows=pd.read_csv(out/"fold_metrics.csv").to_dict("records") if (out/"fold_metrics.csv").exists() else []
done={(r["seed"],r["rule"]) for r in rows if sum(t["seed"]==r["seed"] and t["rule"]==r["rule"] for t in rows)==15}
for seed in [42,137,293]:
 splits=list(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(X,y,d.district_id))
 for rule,mask in rules.items():
  if (seed,rule) in done:continue
  for fold,(a,b) in enumerate(splits):
   fit=a[mask[a]];prep=build_pipeline().fit(X.iloc[fit]);A=prep.transform(X.iloc[fit]);B=prep.transform(X.iloc[b])
   for model in ["random_forest","extra_trees","xgb"]:
    est=comparison_estimator(model,float((y[fit]==0).sum()/y[fit].sum())).fit(A,y[fit]);p=est.predict_proba(B)[:,1];chosen=select_top_risk(p,[keys[i] for i in b],.2)
    rows.append(dict(seed=seed,rule=rule,fold=fold,model=model,retained=len(fit),ap=average_precision_score(y[b],p),tp=int(y[b][chosen].sum()),fp=int(chosen.sum()-y[b][chosen].sum()),fn=int(y[b].sum()-y[b][chosen].sum())))
  pd.DataFrame(rows).to_csv(out/"fold_metrics.csv",index=False)
  print(seed,rule,"finished",flush=True)
r=pd.DataFrame(rows);s=r.groupby(["seed","rule","model"]).agg(ap=("ap","mean"),tp=("tp","sum"),fp=("fp","sum"),fn=("fn","sum")).reset_index();s.to_csv(out/"summary.csv",index=False)
print(s.groupby(["rule","model"])[["ap","tp","fp"]].mean().sort_values("tp",ascending=False).head(15).to_string(),flush=True)
