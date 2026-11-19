"""Compact logistic model comparison and selected-model training.

Run: python -m src.phase3_readiness
All regional summaries are predictor-only and fitted within each training fold.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import average_precision_score
from .evaluation_split import temporal_masks
from .evaluation_metrics import evaluate_binary_risk, select_top_risk
from .model_calibration import fit_calibrators, apply_calibrator
from .transformers import deflate_to_2009_prices
from .temporal_evaluation import fixed_policy_summary

# Fixed substantive specification, not selected by outcome association.
CORE = ['age', 'sex', 'urbrur', 'over_age', 'attend_hours', 'missed_hours',
        'teacher_absent_days', 'travel_minutes', 'school_spend', 'asset_count',
        'father_edu', 'mother_edu', 'hh_size', 'electricity']
CONTEXT = ['school_spend', 'asset_count', 'electricity']
VARIANTS = ('compact_no_region', 'compact_regional_context')
SELECTED_VARIANT = 'compact_no_region'


class RegionalContext(BaseEstimator, TransformerMixin):
    """Training-child regional medians; unseen regions use global medians.

    These are sample context proxies, not representative regional statistics.
    No outcome, regional prevalence, test feature or explicit missing flag enters.
    """
    def __init__(self, regional=True):
        self.regional = regional

    def fit(self, X, y=None):
        if self.regional:
            self.global_ = X[CONTEXT].median()
            self.regions_ = X.groupby('regioncode')[CONTEXT].median()
        return self

    def transform(self, X):
        out = X[CORE].copy()
        if self.regional:
            for c in CONTEXT:
                out['regional_' + c] = X.regioncode.map(self.regions_[c]).fillna(self.global_[c])
        return out.replace([np.inf, -np.inf], np.nan)


def make_pipeline(variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown fixed variant')
    return Pipeline([('context', RegionalContext(variant == VARIANTS[1])),
                     ('fill', SimpleImputer(strategy='median', keep_empty_features=True)),
                     ('scale', StandardScaler()),
                     ('model', LogisticRegression(C=0.1, max_iter=3000))])


def locked_split(frame, seed=20261113):
    """Reserve 20% of district IDs independently of labels, once, with fixed seed."""
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed)
                .split(frame, groups=frame.district_id))
    for c in ('district_id', 'FPrimary'):
        if set(frame.iloc[a][c]) & set(frame.iloc[b][c]):
            raise ValueError('Group overlap: ' + c)
    return a, b


def attrition_bounds(total, observed, events):
    if not 0 <= events <= observed <= total or total <= 0:
        raise ValueError('Invalid cohort counts')
    return {'lower': events / total, 'upper': (events + total - observed) / total}


def run(root):
    root = Path(root)
    out = root / 'phase2_implementation/phase3_readiness'
    out.mkdir(exist_ok=True)
    paths = [root / 'data/processed' / n for n in
             ('train_w1_w2_corrected.csv', 'test_w2_w3_corrected.csv')]
    train, later = [pd.read_csv(p) for p in paths]
    later = deflate_to_2009_prices(later)
    a, b = locked_split(train)
    development = train.iloc[a].reset_index(drop=True)
    validation = train.iloc[b].reset_index(drop=True)
    protocol = dict(status='Retrospective model comparison and selection; independent confirmation remains required',
                    seed=20261113, locked_district_fraction=.2, core_features=CORE,
                    regional_features=CONTEXT, variants=VARIANTS,
                    estimator='LogisticRegression(C=0.1), no class weighting, fixed settings',
                    selection='Final compact no-region specification chosen after retrospective temporal comparison; not an untouched test',
                    selected_variant=SELECTED_VARIANT,
                    selection_rationale='Highest primary temporal recall, precision, F1 and AP of the two compact specifications; simpler representation and less regional concentration',
                    fit_scope='Development districts only (1049 children); no refit on reserved or temporal outcomes',
                    calibration='Sigmoid fitted on development-only district OOF probabilities',
                    context='Training-only unweighted regional medians of child-sample predictors; no population claim',
                    missingness='No explicit flags or assessment-status features; median imputation can still reveal missingness',
                    temporal='Existing Wave 2->3 cohort, households absent from ALL original Wave 1->2 training',
                    chronology='Wave 3 supplies existing test labels; adding Wave 2->3 labels is not training before Wave 3',
                    input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    # Export only hashed identities and group membership, not raw household IDs.
    manifest = []
    for i, r in train.iterrows():
        key = f'{r.FPrimary}:{r.hhmid}'
        manifest.append(dict(child_sha256=hashlib.sha256(key.encode()).hexdigest(),
                             district_id=r.district_id, partition='development' if i in set(a) else 'locked_validation'))
    manifest_frame = pd.DataFrame(manifest)
    manifest_path = out / 'partition_manifest.csv'
    if manifest_path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(manifest_path), manifest_frame, check_dtype=False)
    manifest_frame.to_csv(manifest_path, index=False)
    cv = list(StratifiedGroupKFold(5, shuffle=True, random_state=20261113).split(
        development, development.dropout, development.district_id))
    masks = temporal_masks(train, later)
    cohorts = [('development', development), ('locked_validation', validation),
               ('primary_temporal', later.loc[masks.in_primary_test]),
               ('secondary_temporal', later.loc[masks.in_secondary_test])]
    counts = pd.DataFrame([dict(cohort=n, rows=len(f), events=int(f.dropout.sum()),
                               districts=f.district_id.nunique()) for n, f in cohorts])
    counts.to_csv(out / 'cohort_counts.csv', index=False)
    rows, predictions, fits, fitted = [], [], [], {}
    for variant in VARIANTS:
        oof = np.full(len(development), np.nan)
        fold_ap = []
        for fold, (tr, va) in enumerate(cv, 1):
            model = make_pipeline(variant).fit(development.iloc[tr], development.dropout.iloc[tr])
            oof[va] = model.predict_proba(development.iloc[va])[:, 1]
            fold_ap.append(average_precision_score(development.dropout.iloc[va], oof[va]))
        fits.append(dict(variant=variant, mean_development_ap=float(np.mean(fold_ap))))
        print(variant, 'development AP', np.mean(fold_ap), flush=True)
        # Only development labels enter fitting and recalibration.
        cal = fit_calibrators(development.dropout, oof)['sigmoid']
        model = make_pipeline(variant).fit(development, development.dropout)
        fitted[variant] = (model, cal)
    search = pd.DataFrame(fits)
    selected = search.sort_values('mean_development_ap', ascending=False, kind='stable').iloc[0].variant
    protocol['development_ap_leader'] = selected
    # The development AP leader is distinct from the retrospective project choice.
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    search.to_csv(out / 'development_selection.csv', index=False)
    for variant, (model, cal) in fitted.items():
        width = model[:-1].transform(development).shape[1]
        for scope, frame in cohorts[1:]:
            raw = model.predict_proba(frame)[:, 1]
            prob = apply_calibrator('sigmoid', cal, raw)
            keys = [f'{h}:{p}' for h, p in zip(frame.FPrimary, frame.hhmid)]
            result = evaluate_binary_risk(frame.dropout, prob, keys, .2)
            cm = result.pop('confusion_matrix')
            chosen = select_top_risk(prob, keys, .2)
            uncertainty = fixed_policy_summary(frame.dropout, prob, chosen, frame.district_id, 1000)
            result.update({k: v for k, v in uncertainty.items() if '_ci_' in k or 'bootstrap' in k})
            regional = frame.regioncode.isin([8, 10]).to_numpy()
            result.update(variant=variant, cohort=scope, columns=width,
                          f1=2*cm[1][1]/(2*cm[1][1]+cm[0][1]+cm[1][0]),
                          development_events_per_column=float(development.dropout.sum()/width),
                          tp=cm[1][1], fp=cm[0][1], fn=cm[1][0], tn=cm[0][0],
                          alerts_regions_8_10=int((chosen & regional).sum()),
                          dropouts_other_regions=int(frame.dropout.to_numpy()[~regional].sum()),
                          caught_other_regions=int(frame.dropout.to_numpy()[chosen & ~regional].sum()))
            rows.append(result)
            for k, d, y, p, z in zip(keys, frame.district_id, frame.dropout, prob, chosen):
                predictions.append(dict(child_sha256=hashlib.sha256(k.encode()).hexdigest(),
                                        cohort=scope, variant=variant, district_id=d,
                                        y_true=int(y), probability=float(p), selected=bool(z)))
    protocol['logistic_slope_epv_caveat'] = 'Events/columns is descriptive, not a universal stability criterion or tree-ensemble EPV rule; intercept excluded'
    # Worst-case bounds for the baseline cohort; includes known completers.
    flow = pd.read_csv(root / 'phase2_implementation/attrition_comparison.csv')
    baseline = flow[flow.wave.eq('Wave 1 to 2')]
    verified = baseline[(baseline.comparison == 'cohort_flow') & (baseline.group == 'Verified outcome (including completers)')].iloc[0]
    events = baseline[(baseline.comparison == 'status_breakdown') & (baseline.level == 'dropout')].iloc[0]
    if (int(verified.n_total), int(verified['count']), int(events['count'])) != (2532, 1374, 188):
        raise ValueError('Baseline cohort counts changed; recompute attrition bounds')
    # Counts independently checked against the saved cohort-flow audit.
    protocol['baseline_attrition_bounds'] = dict(total=2532, unobserved=1158,
        observed=1374, events=188, **attrition_bounds(2532, 1374, 188),
        interpretation='Worst-case baseline dropout bounds; 67 known completers count as observed non-events; source: attrition_comparison.csv')
    pd.DataFrame([dict(assumed_unobserved_dropout_rate=q,
                       implied_baseline_dropout_rate=(188 + 1158*q)/2532)
                  for q in (0., .05, .10, .20, .30, .50, 1.)]).to_csv(out / 'attrition_sensitivity.csv', index=False)
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    search.to_csv(out / 'development_selection.csv', index=False)
    pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
    pd.DataFrame(predictions).to_csv(out / 'predictions.csv', index=False)
    # Persist exactly the fitted model used for the reported results.
    # This intentionally does not refit on all 1307 children after evaluation.
    import joblib
    from .selected_model import predict_selected, export_selected_evidence
    selected_pipe, selected_cal = fitted[SELECTED_VARIANT]
    artifact = dict(pipeline=selected_pipe, calibrator=selected_cal,
                    metadata=protocol, schema_version=1)
    artifact_path = root / 'data/processed/selected_model.joblib'
    joblib.dump(artifact, artifact_path)
    loaded = joblib.load(artifact_path)
    for scope, frame in cohorts[1:]:
        expected = pd.DataFrame(predictions).query('variant == @SELECTED_VARIANT and cohort == @scope')
        actual = predict_selected(loaded, frame.drop(columns='dropout'), money_basis='2009_10')
        np.testing.assert_allclose(actual.probability, expected.probability, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(actual.selected, expected.selected)
    selected_metrics = pd.DataFrame(rows).query('variant == @SELECTED_VARIANT')
    selected_metrics.to_csv(root / 'phase2_implementation/selected_model_metrics.csv', index=False)
    selected_metadata = dict(**protocol, artifact_path='data/processed/selected_model.joblib',
                             artifact_sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                             artifact_prediction_roundtrip_verified=True)
    (root / 'phase2_implementation/selected_model.json').write_text(json.dumps(selected_metadata, indent=2)+'\n')
    export_selected_evidence(root, loaded, cohorts, pd.DataFrame(rows))
    print(counts.to_string(index=False), flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == '__main__':
    # Use the canonical module name so joblib artifacts load in fresh processes.
    from src.phase3_readiness import run as run_training
    run_training(Path(__file__).resolve().parents[1])
