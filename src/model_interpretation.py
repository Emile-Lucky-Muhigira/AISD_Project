"""Held-out native XGBoost TreeSHAP and penalized logistic score odds ratios.

Refit saved TRAINING-selected settings, then require reproduction of prior OOF
predictions. SHAP explains the uncalibrated raw margin, not probability points.
No temporal inputs, feature selection or final deployed model are produced.
"""
import hashlib,json
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from xgboost import DMatrix
from .pipeline_builder import build_pipeline,FEATURES_30,FLAGS
from .model_tuning import _validate_frame,_validate_splits,_fit_estimator,_probability


def feature_group(name):
    for source in sorted(FEATURES_30+FLAGS,key=len,reverse=True):
        if name==source or name.startswith(source+'_'):return source
    raise ValueError(f'Unmapped model feature: {name}')


def group_contributions(values,names):
    # Native TreeSHAP returns float32; aggregate in float64 to avoid order-
    # dependent rounding when regrouping many encoded columns.
    values=np.asarray(values,dtype=np.float64)
    if values.ndim!=2 or values.shape[1]!=len(names):raise ValueError('Feature alignment mismatch.')
    groups=[feature_group(n) for n in names]
    result=pd.DataFrame(values,columns=groups).T.groupby(level=0,sort=True).sum().T
    np.testing.assert_allclose(result.sum(axis=1),values.sum(axis=1),atol=1e-6)
    return result


def coefficient_units(prep):
    """Map each encoded coefficient to its actual fit-fold contrast/unit."""
    ct=prep.named_steps['prep'];result={}
    for block,transformer,columns in ct.transformers_:
        if block=='remainder' or not len(columns):continue
        names=ct.get_feature_names_out()[ct.output_indices_[block]]
        if block in ('nominal','school_type'):
            enc=transformer.named_steps['onehot'];i=0
            for source,cats,drop in zip(columns,enc.categories_,enc.drop_idx_):
                for j,cat in enumerate(cats):
                    if j==drop:continue
                    result[names[i]]=f'{source}: {cat} versus reference {cats[drop]}';i+=1
        elif block=='binary':
            contrasts={'sex':'female versus male','urbrur':'rural versus urban','feeding_program':'yes versus no',
                       'father_alive':'alive versus deceased','mother_alive':'alive versus deceased',
                       'father_in_hh':'present versus absent','mother_in_hh':'present versus absent',
                       'electricity':'yes versus no','on_vacation':'vacation versus not vacation'}
            result.update({n:contrasts[n] for n in names})
        elif block=='flags':result.update({n:'missing versus observed' for n in names})
        else:
            transform={'cap':'capped log1p value','log':'log1p value','money':'training-reference rank'}.get(block,'imputed value')
            result.update({n:f'+1 training-fold SD of {transform}' for n in names})
    return result


def interpret_models(train,splits,calibration_metrics,calibration_oof):
    _validate_frame(train);splits=_validate_splits(train,splits,5)
    X=train.drop(columns='dropout');y=train.dropout.to_numpy()
    keys=[hashlib.sha256(json.dumps([str(h),str(p)],separators=(',',':')).encode()).hexdigest() for h,p in zip(train.FPrimary,train.hhmid)]
    checks=[];contributions=[];children=[];coefficients=[]
    for fold,(a,b) in enumerate(splits,1):
        for model in ['xgb','elasticnet']:
            rows=calibration_metrics[calibration_metrics.model.eq(model)&calibration_metrics.method.eq('sigmoid')&calibration_metrics.fold.eq(fold)]
            if len(rows)!=1:raise ValueError('Exactly one saved fit specification required.')
            spec=rows.iloc[0];params=json.loads(spec.best_parameters)
            prep=build_pipeline('lr' if model=='elasticnet' else 'xgb').fit(X.iloc[a])
            fit_x,valid_x=prep.transform(X.iloc[a]),prep.transform(X.iloc[b])
            estimator,_=_fit_estimator(model,params,fit_x,y[a]);raw=_probability(estimator,valid_x)
            expected=calibration_oof[calibration_oof.model.eq(model)&calibration_oof.method.eq('sigmoid')&calibration_oof.fold.eq(fold)].set_index('training_row')
            if set(expected.index)!=set(b):raise ValueError('Saved fold differs from current fold.')
            expected=expected.loc[b]
            if not np.array_equal(expected.y_true,y[b]) or not np.array_equal(expected.child_key_sha256,[keys[i] for i in b]):
                raise ValueError('Saved identity or label mismatch.')
            calibrated=expit(spec.sigmoid_slope*logit(np.clip(raw,1e-6,1-1e-6))+spec.sigmoid_intercept)
            error=float(np.max(np.abs(calibrated-expected.probability.to_numpy())))
            np.testing.assert_allclose(calibrated,expected.probability,rtol=1e-6,atol=1e-7)
            names=prep.named_steps['prep'].get_feature_names_out().tolist()
            check=dict(model=model,fold=fold,training_n=len(a),validation_n=len(b),parameters=spec.best_parameters,
                       prediction_max_error=error,shared_districts=0,shared_households=0)
            if model=='xgb':
                booster=estimator.get_booster();matrix=DMatrix(valid_x)
                shap=booster.predict(matrix,pred_contribs=True,approx_contribs=False)
                margin=booster.predict(matrix,output_margin=True)
                np.testing.assert_allclose(shap.sum(axis=1),margin,rtol=1e-5,atol=1e-5)
                np.testing.assert_allclose(expit(margin),raw,atol=1e-6)
                check['shap_additivity_max_error']=float(np.max(np.abs(shap.sum(axis=1)-margin)))
                grouped=group_contributions(shap[:,:-1],names)
                for j,pos in enumerate(b):
                    flag=bool(expected.iloc[j].selected_top20_within_fold)
                    outcome=('TP' if y[pos] else 'FP') if flag else ('FN' if y[pos] else 'TN')
                    children.append(dict(training_row=int(pos),fold=fold,child_key_sha256=keys[pos],y_true=int(y[pos]),
                        selected_top20=flag,outcome=outcome,raw_margin=float(margin[j]),base_margin=float(shap[j,-1]),
                        raw_probability=float(raw[j]),calibrated_probability=float(calibrated[j])))
                    for name,value in grouped.iloc[j].items():
                        contributions.append(dict(training_row=int(pos),fold=fold,child_key_sha256=keys[pos],feature=name,shap_log_odds=float(value)))
            else:
                units=coefficient_units(prep)
                np.testing.assert_allclose(valid_x@estimator.coef_[0]+estimator.intercept_[0],estimator.decision_function(valid_x),atol=1e-8)
                for name,beta in zip(names,estimator.coef_[0]):
                    coefficients.append(dict(fold=fold,feature=name,source_feature=feature_group(name),contrast=units[name],
                        coefficient=float(beta),score_odds_ratio=float(np.exp(beta)),nonzero=bool(abs(beta)>1e-10),
                        intercept=float(estimator.intercept_[0]),parameters=spec.best_parameters))
            checks.append(check)
        print(f'Explanation fold {fold}/5: reproduced XGBoost and logistic predictions',flush=True)
    values=pd.DataFrame(contributions);people=pd.DataFrame(children).sort_values('training_row').reset_index(drop=True)
    if not np.array_equal(people.training_row,np.arange(len(train))):raise ValueError('Incomplete held-out explanations.')
    global_summary=values.assign(abs_shap=values.shap_log_odds.abs()).groupby('feature').agg(
        mean_abs_shap=('abs_shap','mean'),mean_signed_shap=('shap_log_odds','mean'),n=('training_row','count')).reset_index().sort_values('mean_abs_shap',ascending=False)
    # Predetermined selection rule: one pseudonymous example of each error type.
    examples=people.sort_values('child_key_sha256').groupby('outcome',sort=True).head(1)
    local=values.merge(examples,on=['training_row','fold','child_key_sha256'],validate='many_to_one')
    odds=pd.DataFrame(coefficients)
    odds_summary=odds.groupby(['feature','contrast']).agg(folds_present=('fold','nunique'),
        median_or=('score_odds_ratio','median'),min_or=('score_odds_ratio','min'),max_or=('score_odds_ratio','max'),
        nonzero_folds=('nonzero','sum')).reset_index()
    return pd.DataFrame(checks),values,people,global_summary,local,odds,odds_summary
