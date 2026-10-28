"""Paired fixed-policy uncertainty and figures for the robustness extension."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from .robustness_evaluation import metrics_and_rows, FAMILIES
from .temporal_evaluation import identity_keys
import hashlib


def paired_comparison(d,reps=1000):
    """District resampling with original selected sets held fixed, not a refit CI."""
    y=d.y_true.to_numpy();new=d.selected.to_numpy(bool);old=d.baseline_selected.to_numpy(bool)
    _,inverse=np.unique(d.district_id,return_inverse=True);g=inverse.max()+1
    rng=np.random.default_rng(20261028);counts=rng.multinomial(g,np.ones(g)/g,size=reps);w=counts[:,inverse]
    events=w@y;newn=w@new;oldn=w@old
    ntp=w@(y*new);otp=w@(y*old)
    valid=(events>0)&(newn>0)&(oldn>0)
    rd=(ntp[valid]-otp[valid])/events[valid];pdiff=ntp[valid]/newn[valid]-otp[valid]/oldn[valid]
    aps=[]
    for weights in w:
        if 0<(weights@y)<weights.sum():
            aps.append(average_precision_score(y,d.probability,sample_weight=weights)-average_precision_score(y,d.baseline_probability,sample_weight=weights))
    result=dict(n=len(d),events=int(y.sum()),new_tp=int((y*new).sum()),baseline_tp=int((y*old).sum()),
        delta_recall=float((y*new).sum()/y.sum()-(y*old).sum()/y.sum()),
        delta_precision=float((y*new).sum()/new.sum()-(y*old).sum()/old.sum()),
        delta_ap=float(average_precision_score(y,d.probability)-average_precision_score(y,d.baseline_probability)),bootstrap_reps=reps)
    for key,values in [('recall',rd),('precision',pdiff),('ap',aps)]:
        lo,hi=np.quantile(values,[.025,.975]);result['delta_'+key+'_ci_low']=float(lo);result['delta_'+key+'_ci_high']=float(hi)
    return result


def report_robustness(root):
    root=Path(root);out=root/'phase2_implementation';fig=root/'figures'
    new=pd.read_csv(out/'robustness_predictions.csv');old=pd.read_csv(out/'temporal_predictions.csv')
    old=old[(old.model=='extra_trees')&(old.method=='sigmoid')]
    # Fixed equal-weight ensemble declared separately before new temporal scoring.
    ensemble_parts=[];ensemble_metrics=[];ensemble_curves=[];ensemble_groups=[]
    for (scope,fold),group in new.groupby(['scope','fold']):
        base=group[group.configuration.isin([m+':full:balanced' for m in FAMILIES])]
        if base.groupby('child_key_sha256').size().ne(6).any():raise ValueError('Incomplete ensemble')
        means=base.groupby('child_key_sha256').probability.mean()
        filename='train_w1_w2_corrected.csv' if scope=='training_oof' else 'test_w2_w3_corrected.csv'
        frame=pd.read_csv(root/'data/processed'/filename)
        hashes=[hashlib.sha256(k.encode()).hexdigest() for k in identity_keys(frame)]
        frame=frame.assign(_hash=hashes);frame=frame[frame._hash.isin(means.index)].copy()
        probabilities=frame._hash.map(means).to_numpy();frame=frame.drop(columns='_hash')
        tables=metrics_and_rows(frame,{'equal_weight_ensemble':probabilities},scope,fold)
        for dest,t in zip([ensemble_metrics,ensemble_parts,ensemble_curves,ensemble_groups],tables):dest.append(t)
    for name,parts in zip(['metrics','predictions','calibration_curves','subgroups'],[ensemble_metrics,ensemble_parts,ensemble_curves,ensemble_groups]):
        pd.concat(parts,ignore_index=True).to_csv(out/f'robustness_ensemble_{name}.csv',index=False)
    new=pd.concat([new,*ensemble_parts,pd.read_csv(out/'followup_weighted_predictions.csv')],ignore_index=True)
    configurations=['selected_capacity','selected_ap','equal_weight_ensemble','ip_followup_extra_trees']
    if (out/'region_stress_predictions.csv').exists():
        new=pd.concat([new,pd.read_csv(out/'region_stress_predictions.csv')],ignore_index=True)
        configurations+=['region_selected_capacity','region_selected_ap']
    comparisons=[]
    for scope in ['primary','secondary']:
        b=old[old.cohort==scope][['child_key_sha256','y_true','probability','selected_top20']].rename(columns={'probability':'baseline_probability','selected_top20':'baseline_selected'})
        for configuration in configurations:
            a=new[(new.scope==scope)&(new.configuration==configuration)]
            merged=a.merge(b,on=['child_key_sha256','y_true'],validate='one_to_one')
            if len(merged)!=len(a) or len(merged)!=len(b):raise ValueError('Paired cohort mismatch')
            comparisons.append(dict(scope=scope,configuration=configuration,**paired_comparison(merged)))
    result=pd.DataFrame(comparisons);result.to_csv(out/'robustness_paired_comparison.csv',index=False)
    summary=pd.read_csv(out/'robustness_summary.csv')
    em=pd.concat(ensemble_metrics,ignore_index=True)
    es=em.groupby(['scope','configuration']).agg(mean_ap=('average_precision','mean'),mean_brier=('brier_score','mean'),tp=('tp','sum'),fp=('fp','sum'),fn=('fn','sum'),selected_n=('selected_n','sum')).reset_index()
    es['recall']=es.tp/(es.tp+es.fn);es['precision']=es.tp/es.selected_n
    es.to_csv(out/'robustness_ensemble_summary.csv',index=False)
    summary=pd.concat([summary,es],ignore_index=True)
    configs=['selected_capacity','selected_ap','equal_weight_ensemble','extra_trees:full:balanced','extra_trees:no_missing_flags:balanced','extra_trees:no_geography:balanced','extra_trees:neither:balanced']
    names=['Capacity selection','AP selection','Equal-weight ensemble','Extra Trees: full','Extra Trees: no flags','Extra Trees: no geography','Extra Trees: neither']
    if (out/'region_stress_summary.csv').exists():
        summary=pd.concat([summary,pd.read_csv(out/'region_stress_summary.csv')],ignore_index=True)
        configs+=['region_selected_capacity','region_selected_ap']
        names+=['Region-stress capacity','Region-stress AP']
    f,axes=plt.subplots(1,2,figsize=(11,5.5),layout='constrained')
    for ax,scope in zip(axes,['training_oof','primary']):
        s=summary[summary.scope==scope].set_index('configuration').loc[configs]
        ax.barh(names,s.recall,color='#267dad');ax.invert_yaxis();ax.set_xlim(0,.65)
        ax.set_xlabel('Recall at 20% capacity');ax.set_title('Grouped training validation' if scope=='training_oof' else 'Primary retrospective cohort')
        for j,row in enumerate(s.itertuples()):ax.text(row.recall+.01,j,f'{row.tp} TP / {row.fp} FP',va='center',fontsize=8)
    f.savefig(fig/'fig_robustness_capacity.png',dpi=300);plt.close(f)
    curves=pd.concat([pd.read_csv(out/'robustness_calibration_curves.csv'),*ensemble_curves],ignore_index=True)
    if (out/'region_stress_calibration_curves.csv').exists():
        curves=pd.concat([curves,pd.read_csv(out/'region_stress_calibration_curves.csv')],ignore_index=True)
    f,ax=plt.subplots(figsize=(6,4.5),layout='constrained')
    for name in configs[:2]+configs[2:]:
        d=curves[(curves.scope=='primary')&(curves.configuration==name)&(curves.n>=10)]
        ax.plot(d.mean_probability,d.observed_rate,'o-',label=names[configs.index(name)],markersize=3)
    ax.plot([0,.5],[0,.5],'k--',linewidth=1);ax.set(xlim=(0,.5),ylim=(0,.5),xlabel='Mean predicted risk',ylabel='Observed dropout fraction',title='Primary calibration (bins with at least 10 children)')
    ax.legend(fontsize=7);f.savefig(fig/'fig_robustness_calibration.png',dpi=300);plt.close(f)
    return result


def sync_robustness_decisions(root):
    p=Path(root)/'phase2_implementation/decision_log.csv'
    d=pd.read_csv(p);cols=list(d.columns)
    rows=[
      ['Calibrated feature ablations','Professor: compare missingness and geographic effects','Corrected data; six families; full/no flags/no geography/neither; grouped OOF sigmoid calibration and temporal evaluation','Completed exploratory sensitivity; frozen baseline retained'],
      ['Capacity-focused selection','Professor: intervention capacity; user: fewer misses and false alarms','72 fixed candidates; inner pooled recall at 20%; AP tie-break; independent nested calibration; AP-selected control','Completed; later outcomes never choose settings'],
      ['Follow-up weighting sensitivity','Professor: consider inverse probability of follow-up','Known completers treated as observed; cross-fitted baseline observation model; capped stabilised weights; weighted Extra Trees and sigmoid calibration','Completed sensitivity under explicit MAR/positivity assumptions; unknown outcomes not recovered'],
      ['Equal-weight ensemble','Additional stability comparison','Arithmetic mean of six full-feature calibrated models; fixed equal weights declared before new temporal results','Exploratory challenger; no temporal weight tuning'],
      ['Paired uncertainty and provenance','Report improvements without overstating evidence','District bootstrap of paired fixed-policy differences; original frozen outputs preserved; extension executed separately','Intervals condition on fitted models; retrospective evidence, not new external confirmation']]
    if d[cols[0]].eq('Attrition').any():
        d.loc[d[cols[0]].eq('Attrition'),cols[-1]]='Completed descriptive audit and separate MAR-dependent weighting sensitivity'
    rows.append(['Region-held-out selection','Additional response to failed temporal improvement','Same candidate grid; region-disjoint inner selection, district outer folds; independent nested calibration','Adaptive exploratory extension; no independent temporal confirmation'])
    d=d[~d[cols[0]].isin([r[0] for r in rows])]
    d=pd.concat([d,pd.DataFrame(rows,columns=cols)],ignore_index=True)
    d.to_csv(p,index=False)
    return len(d)
