import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from src.feature_enrichment import questionnaire_features, enrich_frame
from src.enrichment_evaluation import threshold_for_f2, enrichment_preprocessor, evaluate_enrichment


class EnrichmentTests(unittest.TestCase):
    def test_wave_codes_routing_and_invalid_answers(self):
        for wave,no in [(1,2),(2,5)]:
            cols=(['s1fi_22','s1fi_32','s1fi_34','s1fi_36','s1fi_38'] if wave==1 else
                  ['whopaid','oneteacher','teacherabsent','englangteacherabsent','mathteacherabsent'])
            edu=pd.DataFrame([[8 if wave==1 else 95,1,3,9,8],
                              [2,no,12,4,7],[99,no,0,-8,5],[1,no,2,99,-8],
                              [3,99,3,2,2]],columns=cols)
            litcols=['s1fiii_55','s1fiii_56','s1fiii_58'] if wave==1 else ['readenglish','writeenglish','writtencalculations']
            lit=pd.DataFrame([[1,no,99]]*5,columns=litcols)
            got=questionnaire_features(edu,lit,wave)
            np.testing.assert_equal(got.teacher_absent_days_routed.to_numpy(),[3,7,5,np.nan,np.nan])
            self.assertEqual(got.school_expense_payer.iloc[0],'other')
            self.assertEqual(got.school_expense_payer.iloc[2],'unknown')
            self.assertEqual(got.reads_english.iloc[0],1)
            self.assertEqual(got.writes_english.iloc[0],0)
            self.assertTrue(got.written_calculations.isna().all())

    def test_duplicate_raw_keys_rejected(self):
        base=pd.DataFrame(dict(FPrimary=[1],hhmid=[2],dropout=[1]))
        raw=pd.DataFrame(dict(FPrimary=[1,1],s1fi_hhmid=[2,2]))
        with patch('src.feature_enrichment.pyreadstat.read_dta',return_value=(raw,None)):
            with self.assertRaises(pd.errors.MergeError):enrich_frame(base,'.',1)

    def test_join_preserves_order_and_target_last(self):
        base=pd.DataFrame(dict(FPrimary=[2,1],hhmid=[1,1],age=[10,11],dropout=[0,1]))
        edu=pd.DataFrame(dict(FPrimary=[1,2],s1fi_hhmid=[1,1],s1fi_22=[1,2],s1fi_32=[1,2],
                              s1fi_34=[4,8],s1fi_36=[0,3],s1fi_38=[0,2]))
        lit=pd.DataFrame(dict(FPrimary=[1,2],s1fi_hhmid=[1,1],s1fiii_55=[1,2],s1fiii_56=[1,2],s1fiii_58=[1,2]))
        with patch('src.feature_enrichment.pyreadstat.read_dta',side_effect=[(edu,None),(lit,None)]):
            got=enrich_frame(base,'.',1)
        pd.testing.assert_frame_equal(got[list(base)],base)
        self.assertEqual(got.columns[-1],'dropout')
        self.assertEqual(got.reads_english.tolist(),[0,1])

    def test_f2_matches_exhaustive_threshold_search(self):
        y=np.array([1,0,1,0,1,0,0]);p=np.array([.8,.8,.6,.5,.4,.3,.1])
        threshold,value=threshold_for_f2(y,p)
        options=[(fbeta_score(y,p>=t,beta=2),int((p>=t).sum()),t) for t in np.unique(p)]
        best=sorted(options,key=lambda v:(-v[0],v[1]))[0]
        self.assertAlmostEqual(value,best[0]);self.assertEqual(threshold,best[2])
        with self.assertRaises(ValueError):threshold_for_f2(y,np.full(len(y),np.nan))

    def test_preserves_knn_redundancy_and_baseline_columns(self):
        old=enrichment_preprocessor('xgb','baseline')
        new=enrichment_preprocessor('xgb','literacy_funding_teacher')
        self.assertEqual(old.named_steps['grade_knn'].get_params(),new.named_steps['grade_knn'].get_params())
        self.assertEqual(old.named_steps['drop_repeating'].get_params(),new.named_steps['drop_repeating'].get_params())
        self.assertEqual(repr(old.named_steps['prep'].transformers),repr(new.named_steps['prep'].transformers[:10]))
        self.assertNotIn('route_teacher',old.named_steps)

    def test_nested_training_boundaries_and_prediction_coverage(self):
        n=100
        frame=pd.DataFrame(dict(FPrimary=np.arange(n),hhmid=1,district_id=np.repeat(np.arange(20),5),
            dropout=np.arange(n)%2,father_in_hh=1,mother_in_hh=1,regioncode=1))
        splits=[(np.flatnonzero(np.arange(n)//20!=i),np.flatnonzero(np.arange(n)//20==i)) for i in range(5)]
        fit_sizes=[];threshold_sizes=[]
        class Prep:
            named_steps={'grade_knn':type('KNN',(),{'n_neighbors_':15})()}
            def fit(self,X):
                fit_sizes.append(len(X));return self
            def transform(self,X):return X
        class Estimator:
            classes_=np.array([0,1])
            def __init__(self,X):self.ids=set(X.FPrimary)
            def predict_proba(self,X):
                assert not self.ids.intersection(X.FPrimary)
                p=np.where(X.FPrimary%2,.7,.3)
                return np.c_[1-p,p]
        original_threshold=threshold_for_f2
        def threshold(y,p):
            threshold_sizes.append(len(y));return original_threshold(y,p)
        with patch('src.enrichment_evaluation.FAMILIES',('random_forest',)), \
             patch('src.enrichment_evaluation.enrichment_preprocessor',side_effect=lambda *a:Prep()), \
             patch('src.enrichment_evaluation._fit_estimator',side_effect=lambda m,p,X,y:(Estimator(X),1)), \
             patch('src.enrichment_evaluation.threshold_for_f2',side_effect=threshold):
            metrics,oof,groups,curves=evaluate_enrichment(frame,splits,progress=False)
        self.assertEqual(len(metrics),15);self.assertEqual(len(oof),300)
        self.assertEqual(set(threshold_sizes),{80})
        self.assertTrue(all(size<=80 for size in fit_sizes))
        self.assertTrue(any(size<80 for size in fit_sizes))
        self.assertTrue((metrics.shared_households==0).all())


if __name__=='__main__':unittest.main()
