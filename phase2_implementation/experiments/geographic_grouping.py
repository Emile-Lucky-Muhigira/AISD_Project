"""Outcome-blind regional and historical work-town geography sensitivity."""
from pathlib import Path
import sys,json
import pandas as pd,numpy as np,pyreadstat
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.pipeline_builder import build_pipeline
from src.model_evaluation import comparison_estimator
from src.evaluation_metrics import select_top_risk
from sklearn.metrics import average_precision_score
OUT=ROOT/'phase2_implementation/experiments/geographic_grouping_results';OUT.mkdir(exist_ok=True)
raw=ROOT/'data/raw/Ghana_Panel_Survey/Data/Wave 1'
d,meta=pyreadstat.read_dta(raw/'s1gii.dta',encoding='latin1');k,_=pyreadstat.read_dta(raw/'key_hhld_info.dta',encoding='latin1')
for frame,col in [(d,'hhno'),(k,'FPrimary')]:frame[col]=pd.to_numeric(frame[col],errors='coerce')
d=d.merge(k[['FPrimary','regioncode','districtcode']],left_on='hhno',right_on='FPrimary',how='left',validate='many_to_one');d['district_id']=d.regioncode*1000+d.districtcode
rows=[]
for suffix in ['i','ii','iii','iv']:
 town=pd.to_numeric(d['s1gii_16'+suffix],errors='coerce');reason=pd.to_numeric(d['s1gii_21'+suffix],errors='coerce')
 ok=town.between(1,10)&reason.isin([1,2,3,4,14]);z=d.loc[ok,['hhno','district_id']].copy();z['town']=town[ok];rows.append(z)
trips=pd.concat(rows).drop_duplicates();counts=trips.groupby(['district_id','town']).hhno.nunique().reset_index(name='households')
# Require at least five distinct households and a unique most-common named town.
links={}
for district,g in counts.groupby('district_id'):
 top=g[g.households==g.households.max()]
 if len(top)==1 and top.households.iloc[0]>=5:links[district]=int(top.town.iloc[0])
counts.to_csv(OUT/'district_town_support.csv',index=False)
train=pd.read_csv(ROOT/'data/processed/train_w1_w2_raw.csv');train['work_town']=train.district_id.map(links).fillna(0).astype(str)
train=train[[c for c in train if c!='dropout']+['dropout']]
protocol={'variants':['no_geography','region_removed_only','official_regions','official_regions_plus_work_town'],'seeds':[42,137,293],'models':['elasticnet','xgb','lightgbm','catboost','random_forest','extra_trees'],'capacity':0.2,'mapping':'Unique modal named work-related town, at least 5 distinct baseline households per district; unmatched=unknown; no outcomes used. Includes all baseline survey households as fixed contextual covariates, including held-out districts.','limitations':'Past travel by current district residents does not establish origin-to-destination migration flows, neighbouring districts, commuting catchments, or child migration. Other urban destinations excluded rather than guessed. Same children reused; exploratory.','mapped_districts':len(links),'covered_children':int(train.district_id.isin(links).sum()),'total_children':len(train)}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
records=pd.read_csv(OUT/'fold_metrics.csv').to_dict('records') if (OUT/'fold_metrics.csv').exists() else []
for seed in protocol['seeds']:
 splits=list(StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(train,train.dropout,train.district_id))
 for variant in protocol['variants']:
  if sum(z['seed']==seed and z['variant']==variant for z in records)==30:continue
  for model in protocol['models']:
   for fold,(a,b) in enumerate(splits):
    assert not set(train.iloc[a].district_id)&set(train.iloc[b].district_id)
    X=train.drop(columns='dropout');y=train.dropout
    pipe=build_pipeline('lr' if model=='elasticnet' else 'xgb',comparison_estimator(model,float(y.iloc[a].eq(0).sum()/y.iloc[a].eq(1).sum())),include_geography=variant!='no_geography',include_region_predictor=variant!='region_removed_only')
    if variant.endswith('work_town'):
     pipe.named_steps['prep'].transformers.append(('work_town',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['work_town']))
    pipe.fit(X.iloc[a],y.iloc[a]);p=pipe.predict_proba(X.iloc[b])[:,1];sel=select_top_risk(p,[str(i) for i in b],.2);tp=int(y.iloc[b].to_numpy()[sel].sum());fp=int(sel.sum())-tp;fn=int(y.iloc[b].sum())-tp
    records.append(dict(seed=seed,variant=variant,model=model,fold=fold,tp=tp,fp=fp,fn=fn,ap=average_precision_score(y.iloc[b],p)))
  print(seed,variant,'complete',flush=True)
  pd.DataFrame(records).to_csv(OUT/'fold_metrics.csv',index=False)
m=pd.DataFrame(records);s=m.groupby(['seed','variant','model']).agg(tp=('tp','sum'),fp=('fp','sum'),fn=('fn','sum'),ap=('ap','mean')).reset_index();s['precision']=s.tp/(s.tp+s.fp);s['recall']=s.tp/(s.tp+s.fn);s['f1']=2*s.tp/(2*s.tp+s.fp+s.fn);s.to_csv(OUT/'summary.csv',index=False);print(s.groupby(['variant','model'])[['tp','fp','precision','recall','f1']].mean().to_string());print(protocol)
