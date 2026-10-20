import unittest
import numpy as np
import pandas as pd
from scipy import stats
from src.association_tests import categorical_test, adjust_pvalues
from src.evaluation_split import temporal_masks, cohort_audit


def observations(table):
    x, y = [], []
    for i, row in enumerate(table):
        for j, count in enumerate(row):
            x.extend([i]*count);y.extend([j]*count)
    return x, y


class AssociationTests(unittest.TestCase):
    def test_sparse_2x2_matches_fisher_and_records_expected_counts(self):
        table = [[8, 2], [1, 5]]
        x, y = observations(table)
        result = categorical_test(x, y)
        self.assertIn("Fisher exact", result["test"])
        self.assertAlmostEqual(result["p-value"], stats.fisher_exact(table).pvalue)
        self.assertGreater(result["cells expected <5"], 0)

    def test_sparse_rxc_is_seeded_and_dense_uses_pearson(self):
        x, y = observations([[8, 2], [1, 5], [1, 1]])
        a = categorical_test(x, y, resamples=999)
        b = categorical_test(x, y, resamples=999)
        self.assertIn("Monte Carlo", a["test"]);self.assertEqual(a["p-value"], b["p-value"])
        x, y = observations([[80, 20], [20, 80]])
        self.assertEqual(categorical_test(x, y)["test"], "Pearson chi-square")

    def test_multiple_comparisons_preserve_nan_and_by_is_more_conservative(self):
        raw = np.array([0.001, 0.04, np.nan, 0.02, 0.5])
        bh, by = adjust_pvalues(raw), adjust_pvalues(raw, method="by")
        self.assertTrue(np.isnan(bh[2]))
        self.assertTrue(np.all(bh[np.isfinite(raw)] >= raw[np.isfinite(raw)]))
        self.assertTrue(np.all(by[np.isfinite(raw)] >= bh[np.isfinite(raw)]))

    def test_degenerate_categories_are_not_estimable(self):
        self.assertTrue(np.isnan(categorical_test([1, 1, 1], [0, 1, 0])["p-value"]))


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({"FPrimary": [10, 10, 20], "hhmid": [1, 2, 1],
                                   "district_id": [1, 1, 2], "dropout": [0, 1, 0]})
        self.test = pd.DataFrame({"FPrimary": [10, 10, 30], "hhmid": [1, 3, 1],
                                  "district_id": [1, 1, 2], "dropout": [1, 0, 1]})

    def test_primary_disjoint_households_secondary_disjoint_people(self):
        masks = temporal_masks(self.train, self.test)
        self.assertEqual(masks.in_primary_test.tolist(), [False, False, True])
        self.assertEqual(masks.in_secondary_test.tolist(), [False, True, True])
        audit = cohort_audit(self.train, self.test)
        self.assertEqual(audit.iloc[1]["shared households"], 0)

    def test_masks_do_not_use_or_need_outcomes(self):
        a = temporal_masks(self.train, self.test)
        b = temporal_masks(self.train.drop(columns="dropout"), self.test.drop(columns="dropout"))
        pd.testing.assert_frame_equal(a, b)


if __name__ == "__main__":unittest.main()
