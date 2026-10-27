"""Ablation behaviour on synthetic data; no private data dependencies."""
import unittest
import numpy as np
import pandas as pd
from src.pipeline_builder import build_pipeline, FEATURES_30, FLAGS, ID_COLS


class PipelineVariantTests(unittest.TestCase):
    def setUp(self):
        n = 36
        self.x = pd.DataFrame({c: np.ones(n) for c in FEATURES_30})
        self.x['age'] = np.tile([8, 9, 10, 11, 12, 13], 6)
        self.x['grade_level'] = self.x.age - 7.
        self.x.loc[::7, 'grade_level'] = np.nan
        self.x['over_age'] = self.x.age - self.x.grade_level - 5
        self.x['district_id'] = np.repeat(np.arange(6), 6)
        self.x['FPrimary'] = np.arange(n)
        self.x['hhmid'] = 1
        self.x['regioncode'] = np.tile([1, 2, 3], 12)
        self.x['urbrur'] = np.tile([1, 2], 18)
        self.x['sex'] = np.tile([1, 2], 18)
        self.x['asset_count'] = np.arange(n) % 5
        self.x['orphan'] = 0
        self.x['persons_per_room'] = 1
        for c in ['teacher_absent_days', 'attend_hours', 'father_edu', 'mother_edu']:
            self.x.loc[::4, c] = np.nan

    def test_default_and_explicit_full_match(self):
        a = build_pipeline().fit(self.x)
        b = build_pipeline(include_missing_flags=True, include_geography=True).fit(self.x)
        np.testing.assert_allclose(a.transform(self.x), b.transform(self.x))

    def test_flag_removal_drops_exactly_four_columns(self):
        for model in ['lr', 'xgb']:
            a = build_pipeline(model).fit(self.x)
            b = build_pipeline(model, include_missing_flags=False).fit(self.x)
            an = a.named_steps['prep'].get_feature_names_out()
            bn = b.named_steps['prep'].get_feature_names_out()
            self.assertEqual(set(an) - set(bn), set(FLAGS))
            np.testing.assert_allclose(a.transform(self.x)[:, [i for i,c in enumerate(an) if c in bn]], b.transform(self.x))

    def test_region_only_retains_urban_knn_and_all_other_columns(self):
        for model in ["lr", "xgb"]:
            full = build_pipeline(model).fit(self.x)
            ablated = build_pipeline(model, include_region_predictor=False).fit(self.x)
            names = full.named_steps["prep"].get_feature_names_out()
            kept = ablated.named_steps["prep"].get_feature_names_out()
            removed = set(names) - set(kept)
            self.assertTrue(removed)
            self.assertTrue(all(c.startswith("regioncode_") for c in removed))
            self.assertIn("urbrur", kept)
            self.assertTrue(set(FLAGS).issubset(kept))
            self.assertEqual(full.named_steps["grade_knn"].by, ablated.named_steps["grade_knn"].by)
            self.assertIn("regioncode", ablated.named_steps["grade_knn"].by)
            self.assertEqual(full.named_steps["grade_knn"].n_neighbors_, ablated.named_steps["grade_knn"].n_neighbors_)
            np.testing.assert_allclose(full.transform(self.x)[:, [i for i,c in enumerate(names) if c in kept]], ablated.transform(self.x))

    def test_geography_cannot_change_ablated_fit_or_transform(self):
        changed = self.x.copy()
        changed.regioncode = np.arange(len(changed)) + 20
        changed.urbrur = 3 - changed.urbrur
        for flags in [True, False]:
            a = build_pipeline(include_geography=False, include_missing_flags=flags).fit(self.x)
            b = build_pipeline(include_geography=False, include_missing_flags=flags).fit(changed)
            np.testing.assert_allclose(a.transform(self.x), a.transform(changed))
            np.testing.assert_allclose(a.transform(self.x), b.transform(changed))
            self.assertEqual(a.named_steps['grade_knn'].n_neighbors_, b.named_steps['grade_knn'].n_neighbors_)
            names = a.named_steps['prep'].get_feature_names_out()
            self.assertFalse(any('regioncode' in c or c == 'urbrur' or c in ID_COLS for c in names))


if __name__ == '__main__':
    unittest.main()
