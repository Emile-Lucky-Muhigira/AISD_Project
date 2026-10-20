"""Behavioural checks for grade imputation; no private survey data required."""
import unittest
import numpy as np
import pandas as pd
from sklearn.base import clone
from src.transformers import GradeKNNImputer
from src.pipeline_builder import build_pipeline


class ActiveKNNTests(unittest.TestCase):
    def test_pipeline_routes_to_mixed_distance_with_train_only_selection(self):
        x = pd.DataFrame({"age": [8, 9, 10, 11, 12, 13],
            "asset_count": [0, 1, 2, 3, 4, 5], "sex": [1, 2, 1, 2, 1, 2],
            "urbrur": [1, 2, 1, 2, 1, 2], "regioncode": [1, 3, 2, 1, 3, 2],
            "grade_level": [1., 2., 3., 4., 5., 6.], "over_age": [2.] * 6,
            "district_id": [1, 2, 3, 1, 2, 3]})
        step = build_pipeline().named_steps["grade_knn"]
        self.assertIsInstance(step, GradeKNNImputer)
        self.assertEqual(step.n_neighbors, "auto")
        step.fit(x)
        self.assertIn(step.n_neighbors_, (3, 5, 9, 15, 25))
        self.assertTrue(all(r["shared districts"] == 0 for r in step.selection_results_))


class GradeKNNTests(unittest.TestCase):
    def setUp(self):
        self.x = pd.DataFrame({
            "age": [8, 9, 10, 11, 12, 13, 10, 11],
            "asset_count": [0, 1, 2, 3, 4, 5, 2, 3],
            "sex": [1, 2, 1, 2, 1, 2, 1, 2],
            "urbrur": [1, 2, 1, 2, 1, 2, 1, 2],
            "regioncode": [1, 2, 3, 1, 2, 3, 3, 1],
            "grade_level": [1., 2., 3., 4., 5., 6., np.nan, np.nan],
            "over_age": [2., 2., 2., 2., 2., 2., np.nan, np.nan],
            "district_id": [1, 2, 3, 1, 2, 3, 3, 1],
            "dropout": [0, 1, 0, 1, 0, 1, 0, 1],
        })

    def test_region_relabelling_cannot_change_predictions_or_k(self):
        recoded = self.x.copy()
        recoded.regioncode = recoded.regioncode.map({1: "z", 2: "a", 3: "q"})
        a = GradeKNNImputer(neighbor_grid=(1, 3, 5)).fit(self.x)
        b = GradeKNNImputer(neighbor_grid=(1, 3, 5)).fit(recoded)
        self.assertEqual(a.n_neighbors_, b.n_neighbors_)
        np.testing.assert_array_equal(a.transform(self.x).grade_level, b.transform(recoded).grade_level)

    def test_donor_row_order_and_boundary_ties(self):
        a = GradeKNNImputer(n_neighbors=1).fit(self.x)
        b = GradeKNNImputer(n_neighbors=1).fit(self.x.iloc[::-1])
        np.testing.assert_array_equal(a.transform(self.x).grade_level, b.transform(self.x).grade_level)
        same = self.x.iloc[:2].copy()
        same.loc[:, list(a.by)] = same.iloc[0][list(a.by)].to_numpy()
        same.grade_level = [1., 5.]
        q = same.iloc[:1].copy();q.grade_level = np.nan
        self.assertEqual(GradeKNNImputer(n_neighbors=1).fit(same).transform(q).grade_level.iloc[0], 3.)

    def test_observed_grades_preserved_and_over_age_recomputed(self):
        out = GradeKNNImputer(n_neighbors=3).fit_transform(self.x)
        known = self.x.grade_level.notna()
        np.testing.assert_array_equal(out.loc[known, "grade_level"], self.x.loc[known, "grade_level"])
        self.assertTrue(out.grade_level.isin(range(10)).all())
        np.testing.assert_array_equal(out.over_age, out.age - out.grade_level - 5)

    def test_transform_never_learns_from_query_or_dropout(self):
        model = GradeKNNImputer(neighbor_grid=(1, 3, 5)).fit(self.x)
        ranges = model.ranges_.copy();donors = model.donors_.copy();k = model.n_neighbors_
        query = self.x.iloc[-2:].copy();query.dropout = 999;query.over_age = -999
        before = model.transform(query).grade_level
        query.age = query.age + 100
        model.transform(query)
        self.assertEqual(model.ranges_, ranges);self.assertEqual(model.n_neighbors_, k)
        pd.testing.assert_frame_equal(model.donors_, donors)
        changed_labels = self.x.copy();changed_labels.dropout = 1 - changed_labels.dropout
        other = GradeKNNImputer(neighbor_grid=(1, 3, 5)).fit(changed_labels)
        np.testing.assert_array_equal(before, other.transform(self.x.iloc[-2:]).grade_level)

    def test_missing_predictors_unseen_categories_and_too_few_donors(self):
        model = GradeKNNImputer(n_neighbors=25).fit(self.x.iloc[:2])
        q = self.x.iloc[-2:].copy()
        q = q.assign(**{col: np.nan for col in model.by})
        out = model.transform(q)
        self.assertTrue((out.grade_level == np.rint(self.x.iloc[:2].grade_level.median())).all())
        q = self.x.iloc[-2:].copy();q.regioncode = 999
        self.assertTrue(model.transform(q).grade_level.isin(range(10)).all())

    def test_auto_selection_groups_and_clear_failures(self):
        model = clone(GradeKNNImputer(neighbor_grid=(1, 3, 5))).fit(self.x)
        self.assertTrue(all(r["shared districts"] == 0 for r in model.selection_results_))
        with self.assertRaisesRegex(ValueError, "district_id"):
            GradeKNNImputer().fit(self.x.drop(columns="district_id"))
        with self.assertRaisesRegex(ValueError, "without any observed"):
            GradeKNNImputer().fit(self.x.assign(grade_level=np.nan))
        with self.assertRaisesRegex(ValueError, "cannot define neighbours"):
            GradeKNNImputer(by=("age", "over_age"), numeric=("age",)).fit(self.x)


if __name__ == "__main__":
    unittest.main()
