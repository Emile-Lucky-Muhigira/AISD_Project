import unittest
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from src.temporal_diagnostics import distribution_drift, positive_tree_shap


class TemporalDiagnosticTests(unittest.TestCase):
    def test_tree_shap_reconstructs_probability_not_log_odds(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(80, 3)); y = (x[:, 0] + x[:, 1] > 0).astype(int)
        fit = ExtraTreesClassifier(n_estimators=12, max_depth=3, class_weight='balanced', random_state=42).fit(x, y)
        values, base, error = positive_tree_shap(fit, x[:7])
        np.testing.assert_allclose(base + values.sum(axis=1), fit.predict_proba(x[:7])[:, 1], atol=1e-8)
        self.assertLess(error, 1e-8)

    def test_category_shift_includes_missing_and_unseen_levels(self):
        result = distribution_drift(pd.Series(['a', 'a', None, 'missing']), pd.Series(['b', 'b', None, 'missing']), True)
        self.assertEqual(result['total_variation'], .5)
        self.assertEqual(result['missing_change_pp'], 0)

    def test_numeric_shift_separates_missingness(self):
        result = distribution_drift(pd.Series([1., 2., np.nan, np.nan]), pd.Series([3., 4., 5., np.nan]))
        self.assertEqual(result['ks_statistic'], 1.)
        self.assertEqual(result['missing_change_pp'], -25.)
        empty = distribution_drift(pd.Series([np.nan]), pd.Series([1.]))
        self.assertTrue(np.isnan(empty['ks_statistic']))


if __name__ == '__main__': unittest.main()
