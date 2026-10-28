import unittest
from unittest.mock import patch
import numpy as np
from src import robustness_evaluation as r

class RobustnessTests(unittest.TestCase):
    def test_capacity_selection_counts_events_not_mean_fold_recall(self):
        y=np.array([1,0,1,1,1,0]);fold=np.array([1,1,2,2,2,2])
        scores=np.array([[.9,.1],[.1,.9],[.8,.9],[.7,.8],[.6,.7],[.1,.1]])
        chosen=np.array([[1,0],[0,0],[0,1],[0,1],[0,0],[0,0]],bool)
        with patch.object(r,'CANDIDATES',r.CANDIDATES[:2]):
            winner,audit=r.select_candidate(y,scores,chosen,fold,'capacity')
        self.assertEqual(winner,1)
        self.assertAlmostEqual(audit.iloc[1].inner_recall,.5)

    def test_geography_variant_propagates_to_knn(self):
        p=r.make_preprocessor(dict(model='xgb',variant='no_geography',weight='balanced'))
        self.assertNotIn('regioncode',p.named_steps['grade_knn'].by)
        self.assertNotIn('urbrur',p.named_steps['grade_knn'].by)
        cols={name:list(cols) for name,_,cols in p.named_steps['prep'].transformers}
        self.assertNotIn('regioncode',cols['nominal'])
        self.assertNotIn('urbrur',cols['binary'])

    def test_no_flags_variant_removes_creation_and_inputs(self):
        p=r.make_preprocessor(dict(model='xgb',variant='no_missing_flags',weight='balanced'))
        self.assertEqual(p.named_steps['missing_flags'],'passthrough')
        cols={name:list(cols) for name,_,cols in p.named_steps['prep'].transformers}
        self.assertEqual(cols['flags'],[])

    def test_prediction_rejects_target_bearing_frame(self):
        import pandas as pd
        with self.assertRaises(ValueError):r.raw_predict(None,pd.DataFrame({'dropout':[1]}),{})

    def test_weight_strengths_are_fold_local(self):
        X=np.arange(24).reshape(12,2); y=np.array([0]*9+[1]*3)
        for weight,ratio in [('none',1),('moderate',np.sqrt(3)),('balanced',3)]:
            e=r.fit_candidate(dict(model='extra_trees',variant='full',weight=weight),X,y)
            self.assertAlmostEqual(e.class_weight[1]/e.class_weight[0],ratio)
            self.assertAlmostEqual(np.mean([e.class_weight[int(a)] for a in y]),1)

    def test_calibration_reselects_without_prediction_groups(self):
        import pandas as pd
        frame=pd.DataFrame({'FPrimary':np.arange(60),'hhmid':1,'district_id':np.repeat(np.arange(30),2),'dropout':np.tile([0,1],30)})
        target=pd.DataFrame({'FPrimary':[100,101],'hhmid':1,'district_id':[100,101]})
        bank_sizes=[];fit_sizes=[]
        def fake_bank(part,seed):
            bank_sizes.append(len(part)); scores=np.tile(np.linspace(.1,.9,len(part))[:,None],(1,2))
            return scores,scores>.7,np.tile([1,2,3],int(np.ceil(len(part)/3)))[:len(part)]
        def fake_predict(part,features,c):
            self.assertNotIn('dropout',features)
            self.assertFalse(set(part.district_id)&set(features.district_id))
            fit_sizes.append(len(part));return np.linspace(.2,.8,len(features))
        with patch.object(r,'CANDIDATES',r.CANDIDATES[:2]),patch.object(r,'raw_predict',side_effect=fake_predict):
            predictions,_,_=r.fit_and_predict(frame,target,fake_bank,42,include_fixed=False)
        self.assertEqual(len(bank_sizes),4)
        self.assertEqual(bank_sizes[0],60)
        self.assertTrue(all(n<60 for n in bank_sizes[1:]))
        self.assertEqual(set(predictions),{'selected_ap','selected_capacity'})
        self.assertTrue(all(len(p)==2 for p in predictions.values()))

    def test_convergence_retry_preserves_weighted_objective(self):
        from sklearn.exceptions import ConvergenceWarning
        class Estimator:
            C=1.
            def set_params(self,**kw):
                self.__dict__.update(kw);return self
            def fit(self,X,y,**kw):
                if not kw:
                    self.original_weights=np.array([self.class_weight[int(v)] for v in y])
                    raise ConvergenceWarning('simulated')
                self.retry_weights=kw['sample_weight'];return self
        estimator=Estimator();y=np.array([0]*9+[1]*3)
        with patch.object(r,'comparison_estimator',return_value=estimator):
            result=r.fit_candidate(dict(model='elasticnet',variant='full',weight='balanced'),np.ones((12,2)),y)
        np.testing.assert_allclose(result.C*result.retry_weights,result.original_weights)
        self.assertLessEqual(result.retry_weights.max(),1.)

if __name__=='__main__':unittest.main()
