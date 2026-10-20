"""Reproducible exploratory tests with sparse-table and multiplicity checks.

These row-level tests do not constitute survey-weighted or cluster-robust
population inference. They describe the training sample, not causal effects.
"""
import numpy as np
import pandas as pd
from scipy import stats


def adjust_pvalues(values, method="bh"):
    """Adjust one declared test family, preserving unestimable entries as NaN."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if ((values[finite] < 0) | (values[finite] > 1)).any():
        raise ValueError("p-values must be between zero and one")
    adjusted = np.full(values.shape, np.nan)
    if finite.any():
        adjusted[finite] = stats.false_discovery_control(values[finite], method=method)
    return adjusted


def categorical_test(x, y, *, seed=42, resamples=99999):
    """Pearson test for adequate cells; Fisher exact/conditional MC if sparse.

    Cramer's V always uses uncorrected Pearson chi-square as an effect-size
    definition, irrespective of which test supplies the p-value.
    """
    pairs = pd.DataFrame({"x": np.asarray(x), "y": np.asarray(y)}).dropna()
    table = pd.crosstab(pairs.x, pairs.y)
    table = table.loc[table.sum(axis=1) > 0, table.sum(axis=0) > 0]
    if min(table.shape, default=0) < 2:
        return {"test": "Not estimable", "p-value": np.nan, "effect": 0.0,
                "Pearson chi2": np.nan, "df": 0, "minimum expected": np.nan,
                "cells expected <5": 0, "observed n": len(pairs), "MC standard error": np.nan}
    chi2, asymptotic_p, dof, expected = stats.chi2_contingency(table, correction=False)
    sparse = bool((expected < 5).any())
    mc_se = np.nan
    if sparse and table.shape == (2, 2):
        name = "Fisher exact (2x2)"
        p = float(stats.fisher_exact(table.to_numpy(), alternative="two-sided").pvalue)
    elif sparse:
        name = f"Fisher Monte Carlo (RxC; {resamples})"
        method = stats.MonteCarloMethod(n_resamples=resamples, rng=np.random.default_rng(seed))
        p = float(stats.fisher_exact(table.to_numpy(), method=method).pvalue)
        mc_se = float(np.sqrt(p*(1-p)/(resamples+1)))
    else:
        name, p = "Pearson chi-square", float(asymptotic_p)
    v = float(np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape)-1))))
    return {"test": name, "p-value": p, "effect": v, "Pearson chi2": float(chi2),
            "df": int(dof), "minimum expected": float(expected.min()),
            "cells expected <5": int((expected < 5).sum()), "observed n": len(pairs),
            "MC standard error": mc_se}
