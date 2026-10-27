import unittest
from unittest.mock import Mock
import numpy as np
import pandas as pd
from src.temporal_evaluation import predict_temporal, region_strata, fixed_policy_summary
from src.evaluation_split import temporal_masks


class TemporalTests(unittest.TestCase):
    def test_outcomes_rejected_before_predict(self):
        estimator=Mock()
        with self.assertRaises(ValueError):
            predict_temporal({'xgb':(None,estimator,{})},pd.DataFrame({'dropout':[0]}))
        estimator.predict_proba.assert_not_called()

    def test_cohorts_ignore_outcomes(self):
        train=pd.DataFrame({'FPrimary':[1,2],'hhmid':[1,1],'dropout':[0,1]})
        test=pd.DataFrame({'FPrimary':[1,1,3],'hhmid':[1,2,1],'dropout':[1,0,1]})
        a=temporal_masks(train,test)
        b=temporal_masks(train.assign(dropout=1-train.dropout),test.assign(dropout=1-test.dropout))
        pd.testing.assert_frame_equal(a,b)
        self.assertEqual(a.in_primary_test.tolist(),[False,False,True])
        self.assertEqual(a.in_secondary_test.tolist(),[False,True,True])

    def test_strata_fixed_from_training(self):
        train=pd.DataFrame({'regioncode':[1,1,2,2],'dropout':[1,1,0,0]})
        self.assertEqual(region_strata(train),{1.:'high',2.:'low'})
        self.assertTrue(pd.isna(pd.Series([3]).map(region_strata(train)).iloc[0]))

    def test_fixed_subgroup_allocation_not_reranked(self):
        r=fixed_policy_summary([1,0,1],[.9,.8,.1],[False,False,False],[1,1,2],20)
        self.assertEqual((r['tp'],r['fp'],r['fn'],r['tn']),(0,0,2,1))
        self.assertTrue(np.isnan(r['precision_at_capacity']))
        self.assertEqual(r['recall_at_capacity'],0)
        self.assertEqual(r['precision_at_capacity_bootstrap_valid'],0)

    def test_no_event_interval_is_undefined(self):
        r=fixed_policy_summary([0,0],[.2,.1],[True,False],[1,2],20)
        self.assertTrue(np.isnan(r['average_precision']))
        self.assertEqual(r['bootstrap_zero_event_reps'],20)
        self.assertTrue(np.isnan(r['recall_at_capacity_ci_low']))

    def test_batched_bootstrap_matches_independent_loop_with_ties(self):
        from src.subgroup_evaluation import _scores
        y=np.array([1,0,0,1,0,1]);p=np.array([.8,.8,.3,.3,.1,.1])
        s=np.array([True,True,False,False,False,False]);districts=np.array([1,1,2,2,3,3])
        r=fixed_policy_summary(y,p,s,districts,100)
        rng=np.random.default_rng(20261025);draws=[]
        for _ in range(100):
            weights=rng.multinomial(3,[1/3]*3)[districts-1]
            draws.append(_scores(y,p,s,weights))
        for key in draws[0]:
            v=np.array([d[key] for d in draws]);v=v[np.isfinite(v)]
            name=key.replace('under_fold_policy','at_capacity')
            self.assertEqual(r[name+'_bootstrap_valid'],len(v))
            if len(v)>=90:
                np.testing.assert_allclose([r[name+'_ci_low'],r[name+'_ci_high']],np.quantile(v,[.025,.975]),atol=1e-12)

if __name__=='__main__':unittest.main()
