import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from src.balanced_risk_evaluation import (threshold_for_f1,balanced_metrics,
    select_balanced_on_training,evaluate_balanced_risk)


class BalancedRiskTests(unittest.TestCase):
    def test_threshold_matches_exhaustive_search_with_ties(self):
        rng=np.random.default_rng(17)
        for _ in range(30):
            y=np.r_[1,rng.integers(0,2,29)];p=rng.integers(0,6,30)/5
            threshold,f1=threshold_for_f1(y,p)
            choices=sorted(set(p),reverse=True)
            expected=max(choices,key=lambda t:f1_score(y,p>=t))
            self.assertEqual(threshold,expected)
            self.assertAlmostEqual(f1,f1_score(y,p>=threshold))

    def test_f1_penalizes_flagging_everyone(self):
        y=np.array([1,1,0,0,0,0]);p=np.array([.9,.8,.7,.6,.5,.4])
        threshold,score=threshold_for_f1(y,p)
        self.assertEqual(threshold,.8);self.assertEqual(score,1.)
        self.assertGreater(score,f1_score(y,np.ones(6)))

    def test_invalid_scores_and_no_events(self):
        for p in [[np.nan,.2],[1.1,.2]]:
            with self.assertRaises(ValueError):threshold_for_f1([1,0],p)
        with self.assertRaises(ValueError):threshold_for_f1([0,0],[.1,.2])

    def test_selection_uses_f1_not_recall_alone_and_rejects_failure(self):
        frame=pd.DataFrame({'dropout':[1,1,0,0,0,0],'father_in_hh':1,'mother_in_hh':1,'regioncode':1})
        candidates=[dict(model='random_forest',positive_multiplier=w,include_parent_indicators=True) for w in [1.,2.]]
        scores=np.array([[.5]*6,[.9,.8,.7,.6,.5,.4]])
        with patch('src.balanced_risk_evaluation.score_candidates_on_training',return_value=(scores,[[],[]])):
            chosen,threshold,audit=select_balanced_on_training(frame,1,candidates)
        self.assertEqual(chosen['positive_multiplier'],2.)
        self.assertEqual(threshold,.8)
        self.assertEqual(audit.selected.tolist(),[False,True])
        with patch('src.balanced_risk_evaluation.score_candidates_on_training',return_value=(scores*np.nan,[[],[]])):
            with self.assertRaises(RuntimeError):select_balanced_on_training(frame,1,candidates)

    def test_outer_rows_are_excluded_from_selection_and_refit(self):
        n=100
        frame=pd.DataFrame(dict(FPrimary=np.arange(n),hhmid=1,district_id=np.repeat(np.arange(10),10),
            dropout=np.tile([0,1],50),father_in_hh=1,mother_in_hh=1,regioncode=1))
        splits=[(np.flatnonzero(np.arange(n)//20!=i),np.flatnonzero(np.arange(n)//20==i)) for i in range(5)]
        class Prep:
            def fit(self,X):return self
            def transform(self,X):return X
        class Estimator:
            classes_=np.array([0,1])
            def __init__(self,X):self.ids=set(X.FPrimary)
            def predict_proba(self,X):
                assert not self.ids.intersection(X.FPrimary)
                p=np.where(X.FPrimary%2,.8,.2)
                return np.c_[1-p,p]
        candidate=dict(model='random_forest',positive_multiplier=1.,include_parent_indicators=True)
        def select(part,fold):
            self.assertFalse(set(part.FPrimary).intersection(splits[fold-1][1]))
            return candidate,.8,pd.DataFrame([dict(candidate_id=0,selected=True)])
        with patch('src.balanced_risk_evaluation.select_balanced_on_training',side_effect=select),patch(
            'src.balanced_risk_evaluation.make_preprocessor',return_value=Prep()),patch(
            'src.balanced_risk_evaluation.fit_candidate',side_effect=lambda c,X,y:(Estimator(X),{})):
            _,metrics,oof,groups=evaluate_balanced_risk(frame,splits,progress=False)
        self.assertTrue(metrics.f1.eq(1).all());self.assertEqual(len(oof),n)
        self.assertEqual(groups[groups.dimension=='overall'].iloc[0]['fp'],0)

    def test_confusion_counts_and_f1(self):
        r=balanced_metrics([0,1,1],[.3,.7,.1],[True,True,False])
        self.assertEqual((r['tp'],r['fp'],r['fn'],r['tn']),(1,1,1,0))
        self.assertEqual(r['f1'],.5)


if __name__=='__main__':unittest.main()
