import unittest
import numpy as np
import pandas as pd
from src.phase3_readiness import RegionalContext, CORE, CONTEXT, locked_split, attrition_bounds
from src.phase3_readiness import make_pipeline
from src.selected_model import predict_selected
from src.model_calibration import fit_calibrators


class ReadinessTests(unittest.TestCase):
    def test_context_ignores_targets_and_unseen_region_uses_training(self):
        train = pd.DataFrame({c: [1., 3., 9., 11.] for c in CORE})
        train['regioncode'] = [1, 1, 2, 2]
        train['dropout'] = [0, 1, 0, 1]
        model = RegionalContext().fit(train)
        target = train.iloc[:2].copy()
        target['regioncode'] = [1, 999]
        target[CONTEXT] = 100000.
        result = model.transform(target)
        for c in CONTEXT:
            self.assertEqual(result['regional_' + c].tolist(), [2., 6.])
        self.assertEqual(result.shape[1], 17)
        self.assertNotIn('regioncode', result)
        self.assertNotIn('dropout', result)
        train['dropout'] = 1 - train.dropout
        pd.testing.assert_frame_equal(model.transform(target), RegionalContext().fit(train).transform(target))

    def test_lock_independent_of_labels_and_group_disjoint(self):
        f = pd.DataFrame(dict(district_id=np.repeat(np.arange(20), 4),
                              FPrimary=np.repeat(np.arange(40), 2), dropout=np.arange(80) % 2))
        a, b = locked_split(f)
        self.assertFalse(set(f.iloc[a].district_id) & set(f.iloc[b].district_id))
        f.dropout = 1 - f.dropout
        aa, bb = locked_split(f)
        np.testing.assert_array_equal(a, aa)
        np.testing.assert_array_equal(b, bb)

    def test_attrition_extremes(self):
        self.assertEqual(attrition_bounds(100, 60, 10), {'lower': .1, 'upper': .5})
        with self.assertRaises(ValueError):
            attrition_bounds(100, 60, 70)

    def test_selected_model_has_no_region_dependency(self):
        rng = np.random.default_rng(7)
        x = pd.DataFrame(rng.normal(size=(40, len(CORE))), columns=CORE)
        y = np.arange(40) % 2
        pipe = make_pipeline('compact_no_region').fit(x, y)
        expected = pipe.predict_proba(x)
        changed = x.assign(regioncode=999, district_id=888, teacher_absent_days_missing=1)
        np.testing.assert_allclose(expected, pipe.predict_proba(changed))

    def test_scoring_price_basis_and_outcome_guard(self):
        rng = np.random.default_rng(8)
        x = pd.DataFrame(rng.normal(size=(40, len(CORE))), columns=CORE)
        x['FPrimary'] = np.arange(40)
        x['hhmid'] = 1
        y = np.arange(40) % 2
        pipe = make_pipeline('compact_no_region').fit(x, y)
        artifact = dict(pipeline=pipe, calibrator=fit_calibrators(y, pipe.predict_proba(x)[:, 1])['sigmoid'])
        first = predict_selected(artifact, x, money_basis='2009_10')
        nominal = x.copy()
        nominal['school_spend'] *= 174.0 / 107.5
        second = predict_selected(artifact, nominal, money_basis='2013_14')
        np.testing.assert_allclose(first.probability, second.probability)
        np.testing.assert_array_equal(first.selected, second.selected)
        self.assertEqual(int(first.selected.sum()), 8)
        with self.assertRaises(ValueError):
            predict_selected(artifact, x.assign(dropout=y), money_basis='2009_10')
        with self.assertRaises(ValueError):
            predict_selected(artifact, x, money_basis='unknown')
        with self.assertRaises(ValueError):
            predict_selected(artifact, pd.concat([x, x.iloc[:1]]), money_basis='2009_10')


if __name__ == '__main__':
    unittest.main()
