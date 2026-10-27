"""Source-justified cleaning variants; originals and outcomes are preserved.

Wave 1 education Q10/Q11 metadata label 99 in either component as vacation.
Conflicting test answers are uncertain, not resolved by file order. An asset
total with any owned item of unknown value is unknown, not a complete total.
Teacher routing and attendance assumptions are separate sensitivity variants.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import pyreadstat

VARIANTS=('original','corrected','corrected_routed','corrected_routed_attendance')


def read_raw(raw,wave,name):
    path=Path(raw)/f'Wave {wave}'/name
    try:d,_=pyreadstat.read_dta(path)
    except UnicodeDecodeError:d,_=pyreadstat.read_dta(path,encoding='latin1')
    return d


def align(base,data,member='hhmid'):
    d=data.rename(columns={member:'hhmid'}).copy()
    for c in ['FPrimary','hhmid']:d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=['FPrimary','hhmid'])
    return base[['FPrimary','hhmid']].merge(d,on=['FPrimary','hhmid'],how='left',validate='one_to_one',sort=False)


def conflicting_test_keys(data,member,columns):
    d=data.rename(columns={member:'hhmid'}).copy()
    for c in ['FPrimary','hhmid']:d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=['FPrimary','hhmid'])
    bad=set()
    for key,g in d[d.duplicated(['FPrimary','hhmid'],keep=False)].groupby(['FPrimary','hhmid']):
        answers=g[columns].fillna('').astype(str).apply(lambda s:s.str.upper().str.strip())
        if answers.drop_duplicates().shape[0]>1:bad.add(key)
    return bad


def incomplete_asset_households(data,wave):
    hh=pd.to_numeric(data['hhno' if wave==1 else 'FPrimary'],errors='coerce')
    qty=pd.to_numeric(data['s3aiii_a' if wave==1 else 'quantity'],errors='coerce')
    if wave==1:
        a=pd.to_numeric(data.s3aiii_c1,errors='coerce');b=pd.to_numeric(data.s3aiii_c2,errors='coerce')
        value=(a.fillna(0)+b.fillna(0)/100).where(a.notna()|b.notna())
    else:value=pd.to_numeric(data.currentvalue,errors='coerce')
    bad=qty.gt(0)&(value.isna()|value.lt(0))
    return set(hh[bad].dropna())


def build_cleaning_variants(base,raw,wave):
    if wave not in (1,2):raise ValueError('Expected baseline wave 1 or 2.')
    if base.columns[-1]!='dropout' or base[['FPrimary','hhmid']].isna().any().any() or base.duplicated(['FPrimary','hhmid']).any():
        raise ValueError('Unique known keys and target-last schema required.')
    base=base.reset_index(drop=True);clean=base.copy(deep=True)
    edu=align(base,read_raw(raw,wave,'s1fi.dta' if wave==1 else '01fi_generaleducation.dta'),'s1fi_hhmid' if wave==1 else 'hhmid')
    if wave==1:
        explicit=edu[['s1fi_10i','s1fi_10ii','s1fi_11i','s1fi_11ii']].apply(pd.to_numeric,errors='coerce').eq(99).any(axis=1)
        clean.loc[explicit,'on_vacation']=1
    specs=([('math','s9d.dta','s9d_id',[f's9d_{i}' for i in range(1,9)]),
            ('english','s9e.dta','s9d_ie',[f's9e_{i}' for i in range(9,16)])] if wave==1 else
           [('math','09d_math.dta','hhmid',[f'd_{i:02d}' for i in range(1,9)]),
            ('english','09e_english.dta','hhmid',[f'e_{i:02d}' for i in range(9,16)])])
    keys=list(zip(base.FPrimary,base.hhmid))
    for subject,file,member,columns in specs:
        bad=conflicting_test_keys(read_raw(raw,wave,file),member,columns)
        mask=np.array([key in bad for key in keys])
        clean.loc[mask,subject+'_score']=np.nan
        clean.loc[mask,subject+'_status']='conflicting_records'
    assets=read_raw(raw,wave,'s3aiii.dta' if wave==1 else '03aiii_durablegoodquestions.dta')
    clean.loc[clean.FPrimary.isin(incomplete_asset_households(assets,wave)),'durables_value']=np.nan
    routed=clean.copy(deep=True)
    cols=['s1fi_32','s1fi_34','s1fi_36','s1fi_38'] if wave==1 else ['oneteacher','teacherabsent','englangteacherabsent','mathteacherabsent']
    t=edu[cols].apply(pd.to_numeric,errors='coerce')
    days=t[cols[1:]].where(lambda x:x.ge(0)&x.le(22))
    one=t[cols[0]].eq(1);many=t[cols[0]].eq(2 if wave==1 else 5)
    routed['teacher_absent_days']=np.nan
    routed.loc[one,'teacher_absent_days']=days.loc[one,cols[1]]
    # Maximum observed subject absence is a declared proxy, not total lost days.
    routed.loc[many,'teacher_absent_days']=days.loc[many,cols[2:]].max(axis=1)
    attendance=routed.copy(deep=True)
    if wave==2:
        h=edu[['classhours','missedhours']].apply(pd.to_numeric,errors='coerce')
        m=edu[['classminutes','missedminutes']].apply(pd.to_numeric,errors='coerce')
        attended=(h.classhours.fillna(0)+m.classminutes.fillna(0)/60).where(h.classhours.notna()|m.classminutes.notna())
        missed=(h.missedhours.fillna(0)+m.missedminutes.fillna(0)/60).where(h.missedhours.notna()|m.missedminutes.notna())
        inferred=attended.eq(0)&missed.eq(0)
        attendance.loc[inferred,['attend_hours','missed_hours']]=0.
        # Wave 2 has no explicit holiday evidence in these four fields.
        attendance['on_vacation']=np.nan
    variants=dict(zip(VARIANTS,[base,clean,routed,attendance]));changes=[]
    for variant,frame in variants.items():
        pd.testing.assert_frame_equal(frame[['FPrimary','hhmid','district_id','dropout']],base[['FPrimary','hhmid','district_id','dropout']])
        assert frame.columns.tolist()==base.columns.tolist()
        for c in base:
            equal=frame[c].eq(base[c])|(frame[c].isna()&base[c].isna())
            if not equal.all():changes.append(dict(wave=wave,variant=variant,column=c,changed_rows=int((~equal).sum()),missing_before=int(base[c].isna().sum()),missing_after=int(frame[c].isna().sum())))
    return variants,pd.DataFrame(changes)


def export_cleaning_variants(root):
    root=Path(root);raw=root/'data/raw/Ghana_Panel_Survey/Data';proc=root/'data/processed';audits=[]
    for wave,prefix in [(1,'train_w1_w2'),(2,'test_w2_w3')]:
        base=pd.read_csv(proc/f'{prefix}_raw.csv')
        variants,audit=build_cleaning_variants(base,raw,wave);audits.append(audit)
        for name,frame in variants.items():
            if name!='original':frame.to_csv(proc/f'{prefix}_{name}.csv',index=False)
    result=pd.concat(audits,ignore_index=True)
    result.to_csv(root/'phase2_implementation/recleaning_changes.csv',index=False)
    return result


def evaluate_cleaning(root):
    """Fixed six-family models and nested training-only sigmoid calibration.

    No hyperparameter search. Compare like with like on the existing five
    district folds; these are exploratory, repeatedly inspected training folds.
    Later-wave feature tables are never loaded by this evaluation.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    from .pipeline_builder import build_pipeline
    from .model_tuning import _validate_splits,_fit_estimator,_probability,TUNING_GRIDS
    from .model_calibration import fit_calibrators,apply_calibrator,reliability_bins
    from .evaluation_metrics import evaluate_binary_risk,select_top_risk
    root=Path(root);proc=root/'data/processed';out=root/'phase2_implementation'
    original=pd.read_csv(proc/'train_w1_w2_raw.csv');y=original.dropout.to_numpy()
    splits=_validate_splits(original,StratifiedGroupKFold(5,shuffle=True,random_state=42).split(original,y,original.district_id),5)
    records=[];predictions=[]
    keys=[json.dumps([str(h),str(p)],separators=(',',':')) for h,p in zip(original.FPrimary,original.hhmid)]
    for variant in VARIANTS:
        frame=original if variant=='original' else pd.read_csv(proc/f'train_w1_w2_{variant}.csv')
        pd.testing.assert_frame_equal(frame[['FPrimary','hhmid','district_id','dropout']],original[['FPrimary','hhmid','district_id','dropout']])
        X=frame.drop(columns='dropout')
        for fold,(a,b) in enumerate(splits,1):
            part=frame.iloc[a].reset_index(drop=True)
            inner=_validate_splits(part,StratifiedGroupKFold(3,shuffle=True,random_state=2242+fold).split(part,part.dropout,part.district_id),3)
            prepared={}
            for model in TUNING_GRIDS:
                kind='lr' if model=='elasticnet' else 'xgb'
                def fit_predict(fit_idx,predict_idx):
                    # Tree families share preprocessing only for the identical
                    # training/validation partition, never across fold borders.
                    key=(kind,tuple(fit_idx),tuple(predict_idx))
                    if key not in prepared:
                        prep=build_pipeline(kind)
                        if variant=='corrected_routed_attendance':prep.set_params(missed_hours_rule='passthrough')
                        prep.fit(X.iloc[fit_idx])
                        prepared[key]=(prep.transform(X.iloc[fit_idx]),prep.transform(X.iloc[predict_idx]))
                    fit_x,predict_x=prepared[key]
                    est,_=_fit_estimator(model,{},fit_x,y[fit_idx])
                    return _probability(est,predict_x)
                scores=np.full(len(a),np.nan)
                for ia,ib in inner:scores[ib]=fit_predict(a[ia],a[ib])
                calibrator=fit_calibrators(y[a],scores)['sigmoid']
                raw=fit_predict(a,b);p=apply_calibrator('sigmoid',calibrator,raw)
                ids=[keys[i] for i in b];r=evaluate_binary_risk(y[b],p,ids,.2);cm=r.pop('confusion_matrix')
                _,ece=reliability_bins(y[b],p)
                r.update(variant=variant,model=model,fold=fold,method='sigmoid',setting='fixed_untuned_training_only',
                         tp=cm[1][1],fp=cm[0][1],fn=cm[1][0],tn=cm[0][0],ece_10_equal_width=ece,
                         sigmoid_slope=float(calibrator.coef_[0,0]),shared_households=0,shared_districts=0)
                records.append(r);selected=select_top_risk(p,ids,.2)
                predictions.extend(dict(variant=variant,model=model,fold=fold,training_row=int(i),child_key_sha256=hashlib.sha256(keys[i].encode()).hexdigest(),y_true=int(y[i]),probability=float(q),selected_top20=bool(s)) for i,q,s in zip(b,p,selected))
            print(f'Cleaning comparison: {variant}, fold {fold}/5',flush=True)
    metrics=pd.DataFrame(records);oof=pd.DataFrame(predictions)
    summary=metrics.groupby(['variant','model']).agg(mean_ap=('average_precision','mean'),mean_brier=('brier_score','mean'),tp=('tp','sum'),fp=('fp','sum'),fn=('fn','sum'),selected_n=('selected_n','sum')).reset_index()
    summary['recall']=summary.tp/y.sum();summary['precision']=summary.tp/summary.selected_n
    for name,data in [('metrics',metrics),('oof_predictions',oof),('summary',summary)]:data.to_csv(out/f'recleaning_{name}.csv',index=False)
    return metrics,oof,summary
