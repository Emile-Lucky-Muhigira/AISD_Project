"""Retrospective diagnostics of the frozen Extra Trees fit; no model selection."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import ks_2samp

from .pipeline_builder import build_pipeline, ID_COLS, NOMINAL, NOMINAL_MISSING, BINARY, ORDINAL
from .model_tuning import _fit_estimator, _probability
from .model_interpretation import group_contributions
from .transformers import deflate_to_2009_prices
from .temporal_evaluation import identity_keys


def positive_tree_shap(estimator, matrix):
    """Explain uncalibrated positive-class probabilities, verifying the scale."""
    import shap
    explainer = shap.TreeExplainer(estimator, feature_perturbation='tree_path_dependent', model_output='raw')
    values = np.asarray(explainer.shap_values(matrix, check_additivity=True))
    positive = list(estimator.classes_).index(1)
    if values.ndim != 3 or values.shape[:2] != matrix.shape:
        raise ValueError('Unexpected classifier SHAP dimensions.')
    values = values[:, :, positive]
    base = float(np.asarray(explainer.expected_value)[positive])
    probability = _probability(estimator, matrix)
    np.testing.assert_allclose(base + values.sum(axis=1), probability, atol=1e-7, rtol=1e-7)
    return values, base, float(np.max(np.abs(base + values.sum(axis=1) - probability)))


def distribution_drift(reference, later, categorical=False):
    """Descriptive effect sizes; missingness separate from observed-value shift.

    TV includes missing as a category. KS uses observed values only, without
    an iid p-value (children are clustered and cohorts differ in composition).
    """
    result = dict(training_n=len(reference), temporal_n=len(later),
                  training_missing=float(reference.isna().mean()), temporal_missing=float(later.isna().mean()))
    result['missing_change_pp'] = 100 * (result['temporal_missing'] - result['training_missing'])
    if categorical:
        # Tagged values prevent literal strings from colliding with missing.
        encode = lambda s: s.map(lambda v: ('missing', '') if pd.isna(v) else ('observed', str(v)))
        a, b = encode(reference).value_counts(normalize=True), encode(later).value_counts(normalize=True)
        levels = a.index.union(b.index)
        result['total_variation'] = float(np.abs(a.reindex(levels, fill_value=0) - b.reindex(levels, fill_value=0)).sum() / 2)
    else:
        a, b = pd.to_numeric(reference.dropna()), pd.to_numeric(later.dropna())
        result.update(training_median=float(a.median()), temporal_median=float(b.median()),
                      ks_statistic=float(ks_2samp(a, b).statistic) if len(a) and len(b) else np.nan)
    return result


def run_temporal_diagnostics(root):
    root = Path(root); out = root / 'phase2_implementation'
    train = pd.read_csv(root / 'data/processed/train_w1_w2_raw.csv')
    later = deflate_to_2009_prices(pd.read_csv(root / 'data/processed/test_w2_w3_raw.csv'))
    spec = pd.read_csv(out / 'temporal_fit_audit.csv').set_index('model').loc['extra_trees']
    saved = pd.read_csv(out / 'temporal_predictions.csv')
    saved = saved[saved.model.eq('extra_trees') & saved.method.eq('sigmoid')].copy()
    X = train.drop(columns='dropout'); Z = later.drop(columns='dropout')
    prep = build_pipeline('xgb').fit(X)
    estimator, _ = _fit_estimator('extra_trees', json.loads(spec.parameters), prep.transform(X), train.dropout)
    matrix = prep.transform(Z); raw = _probability(estimator, matrix)
    calibrated = expit(spec.sigmoid_slope * logit(np.clip(raw, 1e-6, 1-1e-6)) + spec.sigmoid_intercept)
    keys = [hashlib.sha256(k.encode()).hexdigest() for k in identity_keys(later)]
    for row in saved.itertuples():
        if keys[row.temporal_row] != row.child_key_sha256 or later.dropout.iloc[row.temporal_row] != row.y_true:
            raise ValueError('Recorded temporal identities or labels differ.')
    np.testing.assert_allclose(calibrated[saved.temporal_row], saved.probability, atol=1e-8, rtol=1e-7)
    positions = np.sort(saved.temporal_row.unique())
    values, base, shap_error = positive_tree_shap(estimator, matrix[positions])
    grouped = group_contributions(values, prep.named_steps['prep'].get_feature_names_out())
    grouped.index = positions
    globals_, locals_, children, drift, regions = [], [], [], [], []
    categories = set(NOMINAL + NOMINAL_MISSING + BINARY + ORDINAL + ['orphan'])
    for cohort, records in saved.groupby('cohort'):
        ids = records.temporal_row.to_numpy(); part = later.iloc[ids]
        contributions = grouped.loc[ids]
        for feature in contributions:
            globals_.append(dict(cohort=cohort, feature=feature,
                mean_absolute_shap=float(contributions[feature].abs().mean()), mean_signed_shap=float(contributions[feature].mean())))
        records = records.copy()
        records['outcome'] = np.where(records.selected_top20, np.where(records.y_true.eq(1), 'TP', 'FP'), np.where(records.y_true.eq(1), 'FN', 'TN'))
        for record in records.itertuples():
            children.append(dict(cohort=cohort, temporal_row=record.temporal_row, child_key_sha256=record.child_key_sha256,
                outcome=record.outcome, y_true=record.y_true, selected_top20=record.selected_top20,
                base_raw_probability=base, raw_probability=raw[record.temporal_row], calibrated_probability=record.probability))
        # A fixed identity ordering, not largest or most attractive explanations.
        examples = records.sort_values('child_key_sha256').groupby('outcome', sort=True).head(1)
        for record in examples.itertuples():
            for feature, value in grouped.loc[record.temporal_row].items():
                locals_.append(dict(cohort=cohort, outcome=record.outcome, child_key_sha256=record.child_key_sha256,
                    temporal_row=record.temporal_row, feature=feature, shap_probability=float(value)))
        for feature in X.columns:
            if feature in ID_COLS: continue
            drift.append(dict(cohort=cohort, feature=feature, representation='raw, money in 2009 prices',
                **distribution_drift(X[feature], part[feature], feature in categories)))
        for region in sorted(set(train.regioncode.dropna()) | set(part.regioncode.dropna())):
            a = train[train.regioncode.eq(region)]; b = part[part.regioncode.eq(region)]
            region_records = records[records.temporal_row.isin(b.index)]
            flag = region_records.selected_top20.to_numpy(bool); y = region_records.y_true.to_numpy()
            regions.append(dict(cohort=cohort, regioncode=region, training_n=len(a), temporal_n=len(b),
                training_prevalence=float(a.dropout.mean()), temporal_prevalence=float(b.dropout.mean()),
                training_share=len(a)/len(train), temporal_share=len(b)/len(part),
                mean_probability=float(region_records.probability.mean()), selected_n=int(flag.sum()),
                tp=int(y[flag].sum()), fp=int(flag.sum()-y[flag].sum()), fn=int(y[~flag].sum()),
                mean_region_shap=float(grouped.loc[b.index, 'regioncode'].mean()) if len(b) else np.nan,
                sparse_group=len(b)<30 or b.dropout.sum()<5))
    tables = dict(global_shap=pd.DataFrame(globals_), local_shap=pd.DataFrame(locals_),
                  children=pd.DataFrame(children), feature_drift=pd.DataFrame(drift), region_drift=pd.DataFrame(regions))
    tables['global_shap'] = tables['global_shap'].sort_values(['cohort', 'mean_absolute_shap'], ascending=[True,False])
    for name, frame in tables.items(): frame.to_csv(out / f'temporal_diagnostic_{name}.csv', index=False)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for ax, cohort in zip(axes, ['primary', 'secondary']):
        top = tables['global_shap'].query('cohort == @cohort').head(10).iloc[::-1]
        ax.barh(top.feature, top.mean_absolute_shap)
        ax.set(title=f'{cohort.title()} temporal cohort', xlabel='Mean |grouped SHAP| (raw probability)')
    fig.savefig(root / 'figures/fig_temporal_extratrees_shap.png', dpi=160); plt.close(fig)
    manifest = dict(model='extra_trees', selection_changed=False, calibration_refitted=False,
        prediction_max_error=float(np.max(np.abs(calibrated[saved.temporal_row]-saved.probability))),
        shap_additivity_max_error=shap_error, shap_scale='uncalibrated positive-class probability',
        background='training tree path counts', local_selection='smallest SHA256 identity per outcome and cohort',
        interpretation='Retrospective associations, not causal effects; cohort composition and survey routing can explain distribution shifts.',
        sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            root/'data/processed/train_w1_w2_raw.csv', root/'data/processed/test_w2_w3_raw.csv', out/'temporal_fit_audit.csv', out/'temporal_predictions.csv']})
    (out/'temporal_diagnostic_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    return tables, manifest

