from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
import pandas as pd,numpy as np,pyreadstat,json
from pathlib import Path
root=Path(str(ROOT));raw=root/"data/raw/Ghana_Panel_Survey/Data"
def load(w,f):
 try:return pyreadstat.read_dta(raw/f"Wave {w}"/f)[0]
 except UnicodeDecodeError:return pyreadstat.read_dta(raw/f"Wave {w}"/f,encoding="latin1")[0]
def norm(d,cols):
 for c in cols:d[c]=pd.to_numeric(d[c],errors="coerce")
 return d
results=[]
for w,prefix in [(2,"train_w1_w2"),(3,"test_w2_w3")]:
 base=pd.read_csv(root/f"data/processed/{prefix}_raw.csv");e=load(w,"01fi_generaleducation.dta" if w==2 else "01gi_generaleducation.dta");norm(e,["FPrimary","hhmid","highestgrade","attendingstill","attendedlast12"])
 if w==2:linked=e
 else:
  r=load(3,"01b2_roster.dta");h=load(3,"00_hh_info.dta");r=r.merge(h[["FPrimary","FPrimary_original"]],on="FPrimary",how="left",validate="many_to_one");norm(r,["FPrimary","FPrimary_original","hhmid","hhmid_original"]);split=r.hhmid_original.notna()&r.FPrimary_original.notna();r["prev_hh"]=r.FPrimary.where(~split,r.FPrimary_original);r["prev_id"]=r.hhmid.where(~split,r.hhmid_original);linked=r[["FPrimary","hhmid","prev_hh","prev_id"]].merge(e,on=["FPrimary","hhmid"],how="left");linked=linked.drop(columns=["FPrimary","hhmid"]).rename(columns={"prev_hh":"FPrimary","prev_id":"hhmid"})
 counts=linked.groupby(["FPrimary","hhmid"]).size();z=base.merge(linked,on=["FPrimary","hhmid"],how="left");dup=base.set_index(["FPrimary","hhmid"]).index.isin(counts[counts>1].index)
 knownlow=z.highestgrade.isin([0,1,11,12,13,14,15,16,17,18]);unknown=~z.highestgrade.isin([0,1]+list(range(11,32 if w==2 else 35)))
 res=dict(wave=w,n=len(base),matched_rows=len(z),duplicate_link_children=int(dup.sum()),positive_unknown_completion=int(((z.dropout==1)&unknown).sum()),positive_unknown_grade_values=z.loc[(z.dropout==1)&unknown,"highestgrade"].value_counts(dropna=False).to_dict(),contradictory_enrolled_no_last12=int(((z.attendingstill==1)&(z.attendedlast12==5)).sum()),positive_known_incomplete=int(((z.dropout==1)&knownlow).sum()))
 results.append(res)
print(json.dumps(results,indent=2,default=str));Path("phase2_implementation/experiments/label_audit_results.json").write_text(json.dumps(results,indent=2,default=str))
