import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.model_calibration import (fit_calibrators, apply_calibrator,
                                   reliability_bins, calibration_training_scores)


class CalibrationTests(unittest.TestCase):
    def test_reliability_bins_and_endpoints(self):
        bins,ece=reliability_bins([0,1],[.1,.9],n_bins=2)
        self.assertAlmostEqual(ece,.1)
        self.assertEqual(bins.n.tolist(),[1,1])
        bins,ece=reliability_bins([0,1],[0.,1.])
        self.assertEqual(ece,0)
        self.assertEqual(bins.n.sum(),2)
        self.assertEqual(bins.iloc[-1]['n'],1)
        self.assertTrue(bins.loc[bins.n==0,'mean_probability'].isna().all())

    def test_constant_scores_recover_unweighted_event_rate(self):
        y=np.tile([0,0,0,1],20)
        maps=fit_calibrators(y,np.full(len(y),.5))
        for method,cal in maps.items():
            p=apply_calibrator(method,cal,[.5,.5])
            np.testing.assert_allclose(p,.25,atol=1e-3)

    def test_boundary_probabilities_are_finite(self):
        maps=fit_calibrators([0,1,0,1,0,1],[0,.2,.4,.6,.8,1])
        for method,cal in maps.items():
            p=apply_calibrator(method,cal,[0,.1,.5,.9,1])
            self.assertTrue(np.isfinite(p).all())
            self.assertTrue(((p>=0)&(p<=1)).all())
        with self.assertRaises(ValueError):fit_calibrators([0,0],[.1,.2])

    def test_calibration_scores_exclude_validation_from_tuning_and_fit(self):
        frame=pd.DataFrame({'FPrimary':np.arange(60),'hhmid':1,
            'district_id':np.repeat(np.arange(10),6),
            'dropout':np.tile([0,0,0,0,1,1],10),'signal':np.linspace(.1,.9,60)})
        tuned=[];fitted=[]
        def tune(part,model,**kw):
            tuned.append(set(part.index));return {},pd.DataFrame(columns=['fit_status','candidate_id','inner_fold','parameters'])
        class Prep:
            def fit(self,X):
                if 'dropout' in X:raise AssertionError('Outcome entered preprocessing')
                self.seen=set(X.index)
                self.assert_seen=self.seen==tuned[-1]
                if not self.assert_seen:raise AssertionError('Fit boundary differs from tuning')
                return self
            def transform(self,X):return X[['signal']]
        class Est:
            classes_=np.array([0,1])
            def __init__(self,seen):self.seen=seen
            def predict_proba(self,X):
                if self.seen & set(X.index):raise AssertionError('Calibration row used in fit')
                p=X.signal.to_numpy();return np.column_stack([1-p,p])
        def fit(model,params,X,y):
            if set(X.index)!=tuned[-1]:raise AssertionError('Calibration label entered tuning')
            fitted.append(set(X.index));return Est(set(X.index)),2.
        with patch('src.model_calibration.tune_on_training',side_effect=tune),patch(
            'src.model_calibration.build_pipeline',side_effect=lambda kind:Prep()),patch(
            'src.model_calibration._fit_estimator',side_effect=fit):
            scores,details=calibration_training_scores(frame,'elasticnet',1)
        np.testing.assert_allclose(scores,frame.signal)
        self.assertEqual(len(tuned),3)
        self.assertEqual(tuned,fitted)
        self.assertEqual(sum(d['predict_n'] for d in details),60)
        self.assertTrue(all(d['shared_districts']==d['shared_households']==0 for d in details))


if __name__=='__main__':unittest.main()
