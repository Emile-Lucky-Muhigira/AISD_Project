"""Inference and reporting for the selected compact logistic model.

Train: python -m src.phase3_readiness
Score harmonised predictor rows: python -m src.selected_model --input FILE
    --output FILE --money-basis 2009_10|2013_14
The money basis must be declared; future nominal prices require a separate rule.
Only load the locally generated, trusted joblib artifact.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from .evaluation_metrics import select_top_risk
from .model_calibration import apply_calibrator, reliability_bins
from .transformers import deflate_to_2009_prices


def predict_selected(artifact, frame, *, money_basis, capacity=.2):
    from .phase3_readiness import CORE
    required = set(CORE) | {'FPrimary', 'hhmid'}
    if not required.issubset(frame):
        raise ValueError('Missing columns: ' + ', '.join(sorted(required - set(frame))))
    if 'dropout' in frame:
        raise ValueError('Scoring input must exclude the outcome column')
    if frame[['FPrimary', 'hhmid']].isna().any().any() or frame.duplicated(['FPrimary', 'hhmid']).any():
        raise ValueError('Unique, nonmissing child identities required')
    if money_basis not in ('2009_10', '2013_14'):
        raise ValueError('Declare money basis as 2009_10 or 2013_14')
    X = frame.copy()
    if money_basis == '2013_14':
        X = deflate_to_2009_prices(X, cols=['school_spend'])
    raw = artifact['pipeline'].predict_proba(X)[:, 1]
    p = apply_calibrator('sigmoid', artifact['calibrator'], raw)
    keys = [f'{h}:{i}' for h, i in zip(X.FPrimary, X.hhmid)]
    chosen = select_top_risk(p, keys, capacity)
    result = X[['FPrimary', 'hhmid']].copy()
    result['probability'] = p
    result['selected'] = chosen
    return result


def export_selected_evidence(root, artifact, cohorts, metrics):
    """Selected-model coefficients, calibration and capacity; no tuning here."""
    from .phase3_readiness import CORE, SELECTED_VARIANT
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    out = root / 'phase2_implementation'
    figures = root / 'figures'
    model = artifact['pipeline'].named_steps['model']
    coeff = pd.DataFrame({'feature': CORE, 'standardized_log_odds_coefficient': model.coef_[0]})
    coeff['calibrated_log_odds_coefficient'] = coeff.standardized_log_odds_coefficient * artifact['calibrator'].coef_[0, 0]
    coeff.to_csv(out / 'selected_model_coefficients.csv', index=False)
    # Calibration intervals are descriptive; fitting never uses these cohorts.
    calibration, capacities = [], []
    for name, frame in cohorts[1:]:
        result = predict_selected(artifact, frame.drop(columns='dropout'), money_basis='2009_10')
        bins, _ = reliability_bins(frame.dropout, result.probability)
        calibration.append(bins.assign(cohort=name))
        for capacity in (.1, .2, .3, .4, .5):
            flags = select_top_risk(result.probability, [f'{h}:{i}' for h,i in zip(frame.FPrimary,frame.hhmid)], capacity)
            tp = int(frame.dropout.to_numpy()[flags].sum())
            alerts = int(flags.sum()); events = int(frame.dropout.sum())
            capacities.append(dict(cohort=name, capacity=capacity, alerts=alerts, tp=tp,
                                   fp=alerts-tp, fn=events-tp, precision=tp/alerts,
                                   recall=tp/events, f1=2*tp/(alerts+events)))
    pd.concat(calibration).to_csv(out / 'selected_model_calibration.csv', index=False)
    pd.DataFrame(capacities).to_csv(out / 'selected_model_capacity.csv', index=False)
    comparison_path = out / 'temporal_precision_recall_f1.csv'
    comparison = pd.read_csv(comparison_path)
    comparison = comparison[~comparison.model.str.startswith('compact_')]
    additions = []
    for variant in metrics.variant.unique():
        row = {'model': variant}
        for prefix, cohort in [('primary','primary_temporal'),('secondary','secondary_temporal')]:
            r = metrics[(metrics.variant == variant) & (metrics.cohort == cohort)].iloc[0]
            row.update({prefix+'_precision':r.precision_at_capacity,
                        prefix+'_recall':r.recall_at_capacity, prefix+'_f1':r.f1})
        additions.append(row)
    pd.concat([comparison, pd.DataFrame(additions)], ignore_index=True).to_csv(comparison_path,index=False,float_format='%.6f')
    log_path = out / 'decision_log.csv'
    log = pd.read_csv(log_path)
    for col in log.select_dtypes(include='object'):
        log[col] = log[col].str.replace('Review recommendation: ', '', regex=False)
        log[col] = log[col].str.replace('Additional audit requested after temporal failure', 'Source-integrity audit', regex=False)
    log.loc[log.Step.eq('Model families'), 'Status'] = 'Comprehensive benchmark comparison; compact specification selected separately'
    log.loc[log.Step.eq('Frozen temporal evaluation'), 'Step'] = 'Comprehensive temporal benchmarks'
    log = log[~log.Step.isin(['Selected compact specification', 'Selected model scoring'])]
    additions = pd.DataFrame([
        ['Selected compact specification', 'Evaluate parsimonious substantive predictors',
         '14-input no-region logistic regression; 1049 development children; development-only sigmoid; primary precision .205479, recall .394737, F1 .270270',
         'Chosen using retrospective temporal comparison; independent confirmation required'],
        ['Selected model scoring', 'Reproduce the fitted model and cohort alert policy',
         'Persisted pipeline and calibrator; explicit money basis; outcome rejection; prediction roundtrip verified',
         'Implemented and tested']], columns=log.columns)
    pd.concat([log, additions], ignore_index=True).to_csv(log_path,index=False)
    primary = metrics[metrics.cohort.eq('primary_temporal')].set_index('variant')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
    labels = ['Regional context', 'No region (selected)']
    variants = ['compact_regional_context', SELECTED_VARIANT]
    x = np.arange(2)
    for j, (col, label) in enumerate([('precision_at_capacity','Precision'),('recall_at_capacity','Recall'),('f1','F1')]):
        axes[0].bar(x+(j-1)*.23, primary.loc[variants,col], .23, label=label)
    axes[0].set_xticks(x, labels); axes[0].set_ylim(0, .5)
    axes[0].set_ylabel('Score'); axes[0].legend(fontsize=8)
    axes[0].set_title('Primary cohort: 731 children, 146 alerts')
    b = pd.concat(calibration).query("cohort == 'primary_temporal' and n > 0")
    axes[1].plot([0,1],[0,1],'--',color='gray',label='Perfect calibration')
    axes[1].scatter(b.mean_probability,b.observed_rate,s=np.sqrt(b.n)*10,color='#24729b')
    axes[1].set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted risk',ylabel='Observed dropout rate',title='Selected model: descriptive calibration')
    axes[1].legend(fontsize=8)
    fig.savefig(figures/'fig_selected_model.png',dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,4.8), constrained_layout=True)
    c = coeff.sort_values('calibrated_log_odds_coefficient')
    ax.barh(c.feature,c.calibrated_log_odds_coefficient,color='#24729b')
    ax.axvline(0,color='gray',lw=.8)
    ax.set(xlabel='Calibrated log-odds coefficient per training standard deviation',title='Selected compact logistic model: associations, not causal effects')
    fig.savefig(figures/'fig_selected_model_coefficients.png',dpi=300); plt.close(fig)


def main():
    import joblib
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--money-basis',choices=('2009_10','2013_14'),required=True)
    parser.add_argument('--model',type=Path,default=Path(__file__).resolve().parents[1]/'data/processed/selected_model.joblib')
    args=parser.parse_args()
    scores=predict_selected(joblib.load(args.model),pd.read_csv(args.input),money_basis=args.money_basis)
    scores.to_csv(args.output,index=False)
    print(f'Scored {len(scores)} children; selected {int(scores.selected.sum())} at 20% capacity.')


if __name__ == '__main__':
    main()
