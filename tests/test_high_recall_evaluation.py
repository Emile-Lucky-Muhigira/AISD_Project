import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.high_recall_evaluation import (threshold_for_recall, make_preprocessor,
    select_on_training, evaluate_high_recall, policy_metrics, fit_candidate)


class IdentityPrep:
    def __init__(self):
        self.named_steps={'grade_knn':type('K',(),{'n_neighbors_':3})()}
    def fit(self,X):
        self.fit_ids=set(X.FPrimary);return self
    def transform(self,X):return X


class HighRecallTests(unittest.TestCase):
    def test_highest_threshold_and_ties(self):
        y=np.array([1,1,1,1,0,0]);p=np.array([.9,.8,.8,.1,.8,.05])
        threshold,_=threshold_for_recall(y,p,['a']*6,.75)
        self.assertEqual(threshold,.8)
        self.assertEqual(int((p>=threshold).sum()),4)
        self.assertLess(y[p>threshold].sum()/y.sum(),.75)

    def test_group_constraint_lowers_universal_threshold(self):
        y=np.ones(20);p=np.r_[np.linspace(.5,.9,10),np.linspace(.05,.4,10)]
        threshold,details=threshold_for_recall(y,p,['a']*10+['b']*10,.95)
        self.assertAlmostEqual(threshold,.05)
        self.assertGreaterEqual(details['b']['recall'],.95)

    def test_invalid_and_sparse_inputs(self):
        for p in [[np.nan,.2],[1.1,.2]]:
            with self.assertRaises(ValueError):threshold_for_recall([1,0],p,['a','b'])
        with self.assertRaises(ValueError):threshold_for_recall([0,0],[.1,.2],['a','b'])
        _,details=threshold_for_recall([1,0],[.1,.2],['a','b'])
        self.assertEqual(set(details),{'overall'})

    def test_parent_ablation_changes_only_two_predictors(self):
        c=dict(model='random_forest',positive_multiplier=1.,include_parent_indicators=True)
        full=make_preprocessor(c);ablated=make_preprocessor({**c,'include_parent_indicators':False})
        a=[col for _,_,cols in full.named_steps['prep'].transformers for col in cols]
        b=[col for _,_,cols in ablated.named_steps['prep'].transformers for col in cols]
        self.assertEqual(set(a)-set(b),{'father_in_hh','mother_in_hh'})
        self.assertEqual(full.named_steps['grade_knn'].get_params(),ablated.named_steps['grade_knn'].get_params())

    def test_fit_weights_use_only_fit_labels(self):
        class Fake:
            def set_params(self,**kw):self.params=kw;return self
            def fit(self,X,y,**kw):return self
        with patch('src.high_recall_evaluation.comparison_estimator',return_value=Fake()):
            c=dict(model='random_forest',positive_multiplier=2.,include_parent_indicators=True)
            estimator,details=fit_candidate(c,np.zeros((4,1)),np.array([0,0,0,1]))
            self.assertEqual(estimator.params['class_weight'],{0:4/6,1:4.})
            self.assertEqual(details['training_class_ratio'],3.)

    def test_nested_fit_and_prediction_groups_do_not_overlap(self):
        n=100
        frame=pd.DataFrame(dict(FPrimary=np.arange(n),hhmid=1,district_id=np.repeat(np.arange(10),10),
            dropout=np.tile([0,1],50),father_in_hh=1,mother_in_hh=1,regioncode=1))
        class FakeEstimator:
            classes_=np.array([0,1])
            def __init__(self,X):self.ids=set(X.FPrimary)
            def predict_proba(self,X):
                assert not self.ids.intersection(X.FPrimary)
                p=np.where(X.FPrimary%2,.8,.2)
                return np.c_[1-p,p]
        def fit(c,X,y):return FakeEstimator(X),{'training_n':len(y)}
        candidate=dict(model='random_forest',positive_multiplier=1.,include_parent_indicators=True)
        with patch('src.high_recall_evaluation.make_preprocessor',side_effect=lambda c:IdentityPrep()),patch('src.high_recall_evaluation.fit_candidate',side_effect=fit):
            _,threshold,audit=select_on_training(frame,1,candidates=[candidate])
            self.assertEqual(threshold,.8)
            self.assertTrue(audit.selected.all())
            splits=[(np.flatnonzero(np.arange(n)//20!=i),np.flatnonzero(np.arange(n)//20==i)) for i in range(5)]
            def select(part,fold,target):
                self.assertFalse(set(part.FPrimary).intersection(splits[fold-1][1]))
                return candidate,.8,pd.DataFrame([dict(candidate_id=0,selected=True)])
            with patch('src.high_recall_evaluation.select_on_training',side_effect=select):
                _,metrics,oof,groups=evaluate_high_recall(frame,splits,progress=False)
            self.assertEqual(len(oof),n)
            self.assertTrue(metrics.target_met.all())
            self.assertEqual(groups[groups.dimension=='overall'].iloc[0]['fn'],0)

    def test_policy_counts(self):
        r=policy_metrics([0,1,1],[.3,.7,.1],[True,True,False])
        self.assertEqual((r['tp'],r['fp'],r['fn'],r['tn']),(1,1,1,0))
        self.assertEqual(r['recall'],.5)

    def test_logistic_retry_preserves_doubled_weights(self):
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        class Retry:
            C=1.;max_iter=10
            def set_params(self,**kw):
                for key,value in kw.items():setattr(self,key,value)
                return self
            def fit(self,X,y,**kw):
                if self.max_iter==10:warnings.warn('retry',ConvergenceWarning)
                else:self.final_weights=kw['sample_weight']
                return self
        c=dict(model='elasticnet',positive_multiplier=2.,include_parent_indicators=True)
        with patch('src.high_recall_evaluation.comparison_estimator',return_value=Retry()):
            model,detail=fit_candidate(c,np.zeros((4,1)),np.array([0,0,0,1]))
        np.testing.assert_allclose(model.final_weights*detail['numerical_weight_scale'],[2/3,2/3,2/3,4])
        self.assertEqual(model.C/detail['numerical_weight_scale'],1.)

    def test_failed_candidates_cannot_supply_thresholds(self):
        from sklearn.exceptions import ConvergenceWarning
        n=60
        f=pd.DataFrame(dict(FPrimary=np.arange(n),hhmid=1,district_id=np.repeat(np.arange(6),10),
            dropout=np.tile([0,1],30),father_in_hh=1,mother_in_hh=1,regioncode=1))
        c=dict(model='elasticnet',positive_multiplier=1.,include_parent_indicators=True)
        with patch('src.high_recall_evaluation.make_preprocessor',return_value=IdentityPrep()),patch(
            'src.high_recall_evaluation.fit_candidate',side_effect=ConvergenceWarning('failed')):
            with self.assertRaises(RuntimeError):select_on_training(f,1,candidates=[c])


if __name__=='__main__':unittest.main()
