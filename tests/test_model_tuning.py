"""Verify tuning boundaries and selection without expensive tree fits."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.model_tuning import tune_on_training, _validate_splits, _fit_estimator


class TuningTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({'FPrimary': np.arange(60), 'hhmid': 1,
            'district_id': np.repeat(np.arange(10), 6),
            'dropout': np.tile([0, 0, 0, 0, 1, 1], 10),
            'signal': np.tile([.1, .2, .1, .2, .8, .9], 10)})

    def test_inner_selection_fit_boundaries_and_class_weights(self):
        frame = self.frame
        events = []
        preprocessors = []
        class Prep:
            def fit(self, X):
                self.seen = set(X.index)
                self.named_steps = {'grade_knn': SimpleNamespace(n_neighbors_=3)}
                preprocessors.append(self.seen)
                return self
            def transform(self, X):
                # Preserve index so the fake estimator can check boundary separation.
                return X[['signal']].copy()
        class Estimator:
            classes_ = np.array([0, 1])
            def __init__(self, ratio): self.ratio = ratio
            def set_params(self, **params): self.params=params; return self
            def fit(self, X, y):
                self.seen = set(X.index)
                events.append((self.seen, self.ratio))
                return self
            def predict_proba(self, X):
                if self.seen & set(X.index): raise AssertionError('Validation leaked into fit')
                p = X.signal.to_numpy()
                if self.params['C'] == .1: p=1-p
                return np.column_stack([1-p,p])
        with patch('src.model_tuning.build_pipeline', side_effect=lambda kind: Prep()), patch(
                'src.model_tuning.comparison_estimator', side_effect=lambda model,ratio: Estimator(ratio)):
            best, audit = tune_on_training(frame, 'elasticnet', [{'C': .1}, {'C': 1.}])
        self.assertEqual(best, {'C': 1.})
        self.assertEqual(len(preprocessors), 3)
        self.assertEqual(len(events), 6)
        for indices, ratio in events:
            y = frame.loc[list(indices), 'dropout']
            self.assertAlmostEqual(ratio, (y==0).sum()/(y==1).sum())
            self.assertIn(indices, preprocessors)
            self.assertLess(len(indices), len(frame))
        self.assertTrue(audit.loc[audit.selected, 'inner_average_precision'].eq(1).all())
        self.assertTrue(audit[['shared_districts','shared_households']].eq(0).all().all())

    def test_reject_overlapping_outer_groups_before_fitting(self):
        f=self.frame
        splits=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(f,f.dropout,groups=f.district_id))
        _validate_splits(f,splits,5)
        a,b=splits[0];bad=list(splits);bad[0]=(np.append(a,b[0]),b)
        with self.assertRaises(ValueError): _validate_splits(f,bad,5)
        changed=f.copy();changed.loc[b[0],'FPrimary']=changed.loc[a[0],'FPrimary']
        with self.assertRaises(ValueError): _validate_splits(changed,splits,5)

    def test_forbid_weight_or_preprocessing_search(self):
        with self.assertRaises(ValueError):
            tune_on_training(self.frame,'xgb',[{'scale_pos_weight':99}])

    def test_unconverged_candidates_are_ineligible(self):
        from sklearn.exceptions import ConvergenceWarning
        class Prep:
            named_steps={'grade_knn':SimpleNamespace(n_neighbors_=3)}
            def fit(self,X):return self
            def transform(self,X):return X
        def fit(model,params,X,y):
            if params['C']==.1:raise ConvergenceWarning('candidate failed')
            return object(),2.
        with patch('src.model_tuning.build_pipeline',return_value=Prep()),patch(
            'src.model_tuning._fit_estimator',side_effect=fit),patch(
            'src.model_tuning._probability',side_effect=lambda estimator,X:X.signal.to_numpy()):
            best,audit=tune_on_training(self.frame,'elasticnet',[{'C':.1},{'C':1.}])
            self.assertEqual(best,{'C':1.})
            self.assertFalse(audit.loc[audit.candidate_id==0,'eligible'].any())
            self.assertTrue(audit.loc[audit.candidate_id==0,'inner_average_precision'].isna().all())
            with self.assertRaises(RuntimeError):
                tune_on_training(self.frame,'elasticnet',[{'C':.1}])

    def test_weight_rescaling_preserves_logistic_objective(self):
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        X,y=make_classification(n_samples=300,n_features=8,weights=[.8,.2],random_state=1)
        w=np.where(y==1,len(y)/(2*(y==1).sum()),len(y)/(2*(y==0).sum()))
        scale=w.max()
        a=LogisticRegression(solver='saga',l1_ratio=.5,C=.1,class_weight='balanced',tol=1e-8,max_iter=50000,random_state=42).fit(X,y)
        b=LogisticRegression(solver='saga',l1_ratio=.5,C=.1*scale,tol=1e-8,max_iter=50000,random_state=42).fit(X,y,sample_weight=w/scale)
        np.testing.assert_allclose(a.predict_proba(X),b.predict_proba(X),atol=1e-6)

    def test_convergence_retry_changes_only_iteration_budget(self):
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        class RetryEstimator:
            def __init__(self): self.params={'max_iter':10, 'C':.1}; self.budgets=[]
            def get_params(self): return self.params.copy()
            def set_params(self,**kw): self.params.update(kw); return self
            def fit(self,X,y,**kwargs):
                self.budgets.append(self.params['max_iter'])
                if self.params['max_iter']<50: warnings.warn('not converged',ConvergenceWarning)
                return self
        estimator=RetryEstimator()
        with patch('src.model_tuning.comparison_estimator',return_value=estimator):
            fitted,_=_fit_estimator('elasticnet',{},self.frame[['signal']],self.frame.dropout)
        self.assertEqual(fitted.budgets,[10,50])
        self.assertAlmostEqual(fitted.params['C']/fitted.numerical_weight_scale_,.1)

    def test_convergence_warning_is_a_failure(self):
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        class Nonconverging:
            def set_params(self,**kw): return self
            def fit(self,X,y): warnings.warn('not converged',ConvergenceWarning)
        with patch('src.model_tuning.comparison_estimator',return_value=Nonconverging()):
            with self.assertRaises(ConvergenceWarning):
                _fit_estimator('elasticnet',{},self.frame[['signal']],self.frame.dropout)


if __name__=='__main__': unittest.main()
