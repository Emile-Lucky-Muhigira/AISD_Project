"""Alert-capacity sensitivity for the frozen Extra Trees policy.

Training estimates use grouped out-of-fold calibrated scores and apply the
same top-capacity rule separately inside each held-out fold. Later-cohort
results are retrospective descriptions only and must not select capacity.
"""
import numpy as np
import pandas as pd
from .evaluation_metrics import select_top_risk

CAPACITIES = (0.10, 0.20, 0.30, 0.40, 0.50)

def _metrics(y, selected):
    y = np.asarray(y, dtype=int)
    selected = np.asarray(selected, dtype=bool)
    tp = int(y[selected].sum())
    alerts = int(selected.sum())
    fp = alerts - tp
    fn = int(y.sum()) - tp
    precision = tp / alerts if alerts else float("nan")
    recall = tp / int(y.sum()) if y.sum() else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    return dict(n=len(y), events=int(y.sum()), alerts=alerts, tp=tp, fp=fp, fn=fn,
                precision=precision, recall=recall, f1=f1)

def analyze_capacity_policy(training_oof, temporal, capacities=CAPACITIES):
    """Return a workload/capture curve; do not choose a deployment budget.

    Inputs are the existing calibrated Extra Trees OOF and primary temporal
    prediction rows. Training selection is fold-local; temporal outcomes are
    used only to describe historical performance at each hypothetical budget.
    """
    required_train={"fold","child_key_sha256","y_true","probability"}
    required_test={"child_key_sha256","y_true","probability"}
    if not required_train.issubset(training_oof) or not required_test.issubset(temporal):
        raise ValueError("Prediction inputs are missing required columns.")
    if training_oof.child_key_sha256.duplicated().any() or temporal.child_key_sha256.duplicated().any():
        raise ValueError("Each child must appear once in each evaluation scope.")
    if not np.isin(training_oof.y_true,[0,1]).all() or not np.isin(temporal.y_true,[0,1]).all():
        raise ValueError("Outcomes must be binary.")
    rows=[]
    for capacity in capacities:
        selected=np.zeros(len(training_oof),dtype=bool)
        for _,fold in training_oof.groupby("fold",sort=True):
            positions=training_oof.index.get_indexer(fold.index)
            selected[positions]=select_top_risk(fold.probability.to_numpy(),
                fold.child_key_sha256.astype(str).tolist(),float(capacity))
        tr=_metrics(training_oof.y_true,selected)
        te_selected=select_top_risk(temporal.probability.to_numpy(),
            temporal.child_key_sha256.astype(str).tolist(),float(capacity))
        te=_metrics(temporal.y_true,te_selected)
        rows.append(dict(capacity=capacity,**{f"train_{k}":v for k,v in tr.items()},
                         **{f"primary_later_{k}":v for k,v in te.items()}))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from pathlib import Path
    import json
    root=Path(__file__).resolve().parents[1]
    out=root/"phase2_implementation"
    train=pd.read_csv(out/"calibration_oof_predictions.csv")
    train=train[(train.model=="extra_trees")&(train.method=="sigmoid")].reset_index(drop=True)
    later=pd.read_csv(out/"temporal_predictions.csv")
    later=later[(later.cohort=="primary")&(later.model=="extra_trees")&
                (later.method=="sigmoid")].reset_index(drop=True)
    result=analyze_capacity_policy(train,later)
    result.to_csv(out/"capacity_policy_analysis.csv",index=False,float_format="%.6f")
    protocol={"status":"completed",
      "model":"frozen Extra Trees",
      "training":"five-fold district-grouped out-of-fold sigmoid-calibrated predictions; capacity applied within each held-out fold",
      "temporal":"primary later cohort; retrospective descriptive analysis only",
      "capacities":list(CAPACITIES),
      "selection":"rank by predicted risk; deterministic pseudonymous-key tie breaking; floor of capacity times cohort/fold size",
      "threshold_selection":"No probability threshold or deployment capacity selected from later outcomes.",
      "limitations":"Training OOF reflects model selection and repeated project exploration; later cohort was previously inspected and is not independent prospective confirmation. Capacity trades off recall, precision and staff workload; the appropriate budget requires operational costs and pilot evidence."}
    (out/"capacity_policy_protocol.json").write_text(json.dumps(protocol,indent=2)+"\n",encoding="utf-8")
    print(result.to_string(index=False,float_format=lambda x:f"{x:.3f}"))
