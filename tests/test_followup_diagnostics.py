import unittest
import numpy as np
import pandas as pd
from src.followup_diagnostics import observation_indicator,weight_summary

class FollowupTests(unittest.TestCase):
    def test_completion_is_observed_not_attrition(self):
        self.assertEqual(observation_indicator(pd.Series(['enrolled','dropout','completed_basic','moved_out'])).tolist(),[1,1,1,0])
    def test_unknown_status_rejected(self):
        with self.assertRaises(ValueError):observation_indicator(pd.Series(['new_code']))
    def test_constant_probabilities_keep_effective_sample_size(self):
        w,s=weight_summary([.5]*4,[1,0,1,0])
        np.testing.assert_allclose(w,[1,1]);self.assertEqual(s['effective_sample_size'],2)
    def test_weight_cap_limits_extreme_observation_weights(self):
        w,_=weight_summary([.01,.5,.5],[1,1,0],5)
        self.assertEqual(w.max(),5)
    def test_zero_probability_rejected(self):
        with self.assertRaises(ValueError):weight_summary([0,.5],[1,0])
