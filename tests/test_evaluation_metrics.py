import unittest
import numpy as np
from src.evaluation_metrics import evaluate_binary_risk, select_top_risk


class EvaluationMetricsTests(unittest.TestCase):
    def test_hand_calculated_metrics(self):
        y = [1, 0, 1, 0, 0]
        p = [.9, .8, .7, .2, .1]
        r = evaluate_binary_risk(y, p, list('abcde'), capacity=.4)
        self.assertAlmostEqual(r['average_precision'], (1 + 2/3)/2)
        self.assertAlmostEqual(r['brier_score'], .79/5)
        self.assertEqual(r['precision_at_capacity'], .5)
        self.assertEqual(r['recall_at_capacity'], .5)
        self.assertEqual(r['confusion_matrix'], [[2, 1], [1, 1]])
        self.assertEqual(r['lift_at_capacity'], 1.25)

    def test_ties_are_order_invariant_and_label_independent(self):
        ids = np.array(list('abcdefghij'))
        p = np.ones(10) * .5
        chosen = select_top_risk(p, ids)
        perm = np.array([9, 3, 1, 7, 2, 0, 8, 4, 6, 5])
        other = select_top_risk(p[perm], ids[perm])
        self.assertEqual(set(ids[chosen]), set(ids[perm][other]))
        self.assertEqual(chosen.sum(), 2)

    def test_capacity_rounding_and_small_cohort(self):
        self.assertEqual(select_top_risk(np.arange(7)/7, list('abcdefg')).sum(), 1)
        r = evaluate_binary_risk([0, 1, 0], [.1, .9, .2], list('abc'))
        self.assertEqual(r['selected_n'], 0)
        self.assertTrue(np.isnan(r['precision_at_capacity']))
        self.assertEqual(r['recall_at_capacity'], 0)
        self.assertEqual(r['confusion_matrix'], [[2, 0], [1, 0]])

    def test_single_class_remains_explicit(self):
        r = evaluate_binary_risk([0]*5, [.2]*5, list('abcde'))
        self.assertTrue(np.isnan(r['average_precision']))
        self.assertTrue(np.isnan(r['recall_at_capacity']))
        self.assertEqual(r['confusion_matrix'], [[4, 1], [0, 0]])
        r = evaluate_binary_risk([1]*5, [.2]*5, list('abcde'))
        self.assertEqual(r['average_precision'], 1)

    def test_invalid_inputs_fail(self):
        for p in [[], [float('nan')], [-.1], [1.1], [[.2]]]:
            with self.subTest(p=p), self.assertRaises(ValueError):
                select_top_risk(p, ['a'])
        for cap in [0, -1, 1.1, float('nan')]:
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                select_top_risk([.1], ['a'], cap)
        for ids in [['a', 'a'], ['a'], ['', 'b'], [None, 'b']]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                select_top_risk([.1, .2], ids)
        for y in [[2, 0], [1], [float('nan'), 0]]:
            with self.subTest(y=y), self.assertRaises(ValueError):
                evaluate_binary_risk(y, [.1, .2], ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
