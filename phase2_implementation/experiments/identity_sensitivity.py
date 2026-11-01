from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
exec(open("phase2_implementation/experiments/label_audit.py").read().split("results=[]")[0])
import sys
sys.path.insert(0,str(ROOT))
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score
from src.pipeline_builder import build_pipeline
from src.model_evaluation import comparison_estimator
from src.evaluation_metrics import select_top_risk
from src.temporal_evaluation import identity_keys
a=norm(load(1,"s1d.dta"),["FPrimary","hhmid"]);b=norm(load(2,"01b2_roster.dta"),["FPrimary","hhmid"])
a["birth1"]=pd.to_numeric(a.s1d_3iii,errors="coerce");b["birth2"]=pd.to_numeric(b.yearofbirth,errors="coerce").fillna(pd.to_datetime(b.dateofbirth,errors="coerce").dt.year)
d=pd.read_csv(root/"data/processed/train_w1_w2_raw.csv");z=d.merge(a[["FPrimary","hhmid","birth1"]],on=["FPrimary","hhmid"],validate="one_to_one").merge(b[["FPrimary","hhmid","birth2"]],on=["FPrimary","hhmid"],validate="one_to_one");known=z.birth1.between(1990,2010)&z.birth2.between(1990,2010);bad=(known&(z.birth1-z.birth2).abs().gt(2)).to_numpy();Path("phase2_implementation/experiments/linkage_birth_audit.json").write_text(json.dumps(dict(known=int(known.sum()),over2=int(bad.sum()),positive_over2=int(d.loc[bad,"dropout"].sum()))));X=d.drop(columns="dropout");y=d.dropout.to_numpy();keys=identity_keys(d);rows=[]
for seed in [42,137,293]:
 for fold,(aa,bb) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(X,y,d.district_id)):
  fit=aa[~bad[aa]];prep=build_pipeline().fit(X.iloc[fit]);A=prep.transform(X.iloc[fit]);B=prep.transform(X.iloc[bb])
  for model in ["extra_trees","random_forest","xgb"]:
   est=comparison_estimator(model,float((y[fit]==0).sum()/y[fit].sum())).fit(A,y[fit]);p=est.predict_proba(B)[:,1];sel=select_top_risk(p,[keys[i] for i in bb],.2);tp=int(y[bb][sel].sum());rows.append(dict(seed=seed,fold=fold,model=model,ap=average_precision_score(y[bb],p),tp=tp,fp=int(sel.sum())-tp,fn=int(y[bb].sum())-tp))
 print(seed,"complete",flush=True)
r=pd.DataFrame(rows);r.to_csv("phase2_implementation/experiments/identity_sensitivity_metrics.csv",index=False);s=r.groupby(["seed","model"]).agg(ap=("ap","mean"),tp=("tp","sum"),fp=("fp","sum"),fn=("fn","sum")).reset_index();s.to_csv("phase2_implementation/experiments/identity_sensitivity_summary.csv",index=False);print(s.to_string(index=False));print(s.groupby("model")[["ap","tp","fp","fn"]].mean().to_string())
