"""Audit completion codes and conservative duplicate resolution without overwriting data."""
from pathlib import Path
import os, json, contextlib, io
import pandas as pd, numpy as np, nbformat
ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT)
OUT=ROOT/'phase2_implementation/experiments/source_integrity_results'
LOCAL=ROOT/'data/processed/source_integrity_audit';LOCAL.mkdir(exist_ok=True)
nb=nbformat.read(ROOT/'notebooks/01_build_dataset.ipynb',4)
original_drop=pd.DataFrame.drop_duplicates
original_save=pd.DataFrame.to_csv
records=[]
def run(conservative=False):
 env={};stage=[0]
 def dedup(self,subset=None,keep='first',inplace=False,ignore_index=False):
  keys=list(self.columns) if subset is None else ([subset] if isinstance(subset,str) else list(subset))
  dup=self.duplicated(keys,keep=False); data=self.copy(); conflicts=0;fields=set()
  if dup.any():
   for _,g in self.loc[dup].groupby(keys,dropna=False):
    for c in self.columns.difference(keys):
     if g[c].nunique(dropna=False)>1:
      conflicts+=1;fields.add(c)
      if conservative:data.loc[g.index,c]=np.nan
   records.append(dict(conservative=conservative,cell=stage[0],duplicate_rows=int(dup.sum()),conflicting_fields=sorted(fields),conflicting_group_fields=conflicts))
  return original_drop(data,subset=subset,keep=keep,inplace=inplace,ignore_index=ignore_index)
 pd.DataFrame.drop_duplicates=dedup
 pd.DataFrame.to_csv=lambda *a,**k:None
 try:
  with contextlib.redirect_stdout(io.StringIO()):
   for i,c in enumerate(nb.cells):
    if i>27:break
    if c.cell_type=='code':stage[0]=i;exec(compile(c.source,str(i),'exec'),env)
 finally:pd.DataFrame.drop_duplicates=original_drop;pd.DataFrame.to_csv=original_save
 return env
baseline=run();alternative=run(True)
summary={}
for name,frame,gcol,maxcode in [('training',baseline['base'],'highest_grade_w2',31),('later',baseline['base2'],'highest_grade_w3',34)]:
 known=frame[gcol].isin([0,1]+list(range(11,maxcode+1)))
 summary[name]={'baseline_children':len(frame),'unknown_completion_among_dropouts':int((frame.status.eq('dropout')&~known).sum()),'completed_excluded':int(frame.status.eq('completed_basic').sum()),'unknown_completion_values':frame.loc[frame.status.eq('dropout')&~known,gcol].value_counts(dropna=False).to_dict()}
for label,name in [('training','train'),('later','test')]:
 a=baseline[name].copy();b=alternative[name].copy();keys=['FPrimary','hhmid'];a=a.set_index(keys);b=b.set_index(keys);common=a.index.intersection(b.index);aa=a.loc[common];bb=b.loc[common];changed=~(aa.eq(bb)|(aa.isna()&bb.isna()));summary[label].update(original_labelled=len(a),conservative_labelled=len(b),removed=len(a.index.difference(b.index)),added=len(b.index.difference(a.index)),changed_children=int(changed.any(axis=1).sum()),changed_fields=changed.sum()[changed.sum()>0].to_dict())
 b.reset_index().to_csv(LOCAL/(name+'_conservative.csv'),index=False)
 a.reset_index().to_csv(LOCAL/(name+'_baseline.csv'),index=False)
# Actual value labels are retained as aggregate evidence.
labels={}
for w,f in [(2,'01fi_generaleducation.dta'),(3,'01gi_generaleducation.dta')]:
 _,meta=baseline['load'](w,f);labels[str(w)]=meta.variable_value_labels.get('highestgrade',{})
(OUT/'audit.json').write_text(json.dumps({'summary':summary,'duplicate_operations':records,'completion_code_labels':labels},indent=2,default=str))
print(json.dumps(summary,indent=2,default=str))
