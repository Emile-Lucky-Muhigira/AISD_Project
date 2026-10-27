import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.enrichment_tuning import recall_threshold, search_models, training_scores, evaluate_enrichment_tuning


def frame():
    n=120
    return pd.DataFrame(dict(FPrimary=np.arange(n),hhmid=1,district_id=np.repeat(np.arange(20),6),
        dropout=np.arange(n)%2,father_in_hh=1,mother_in_hh=1,regioncode=1))


class Prep:
    named_steps={'grade_knn':type('KNN',(),{'n_neighbors_':15})()}
    def fit(self,X):return self
    def transform(self,X):return X


class Estimator:
    classes_=np.array([0,1])
    def __init__(self,X,good=True):self.ids=set(X.FPrimary);self.good=good
    def predict_proba(self,X):
        assert not self.ids.intersection(X.FPrimary), 'Prediction rows entered fitting'
        p=np.where(X.FPrimary%2,.8,.2) if self.good else np.full(len(X),.5)
        return np.c_[1-p,p]


class TuningTests(unittest.TestCase):
    def test_highest_feasible_threshold_including_ties(self):
        y=np.array([1,1,1,1,1,0,0]);p=np.array([.9,.8,.6,.4,.2,.4,.1])
        t=recall_threshold(y,p,.8)
        self.assertEqual(t,.4)
        self.assertEqual(y[p>=t].sum(),4)
        self.assertEqual((p>=t).sum(),5)
        for other in np.unique(p[p>t]):self.assertLess(y[p>=other].sum()/y.sum(),.8)
        for target in [0,1.1,np.nan]:
            with self.assertRaises(ValueError):recall_threshold(y,p,target)
        with self.assertRaises(ValueError):recall_threshold([0,0],[.2,.3])

    def test_search_selects_ap_using_only_inner_predictions(self):
        with patch('src.enrichment_tuning.TUNING_GRIDS',{'random_forest':[{'quality':False},{'quality':True}]}), \
             patch('src.enrichment_tuning._preprocessor',side_effect=lambda m:Prep()), \
             patch('src.enrichment_tuning._fit_estimator',side_effect=lambda m,p,X,y:(Estimator(X,p['quality']),1)):
            best,audit=search_models(frame(),42)
        self.assertEqual(best,{'random_forest':{'quality':True}})
        self.assertEqual(audit.loc[audit.selected,'candidate_id'].tolist(),[1,1,1])

    def test_calibration_oof_retunes_without_its_validation_rows(self):
        searched=[]
        def search(part,seed):
            searched.append(set(part.FPrimary))
            return {'random_forest':{}},pd.DataFrame({'model':['random_forest']})
        def fit(model,params,X,y):
            self.assertEqual(set(X.FPrimary),searched[-1])
            return Estimator(X),1
        with patch('src.enrichment_tuning.TUNING_GRIDS',{'random_forest':[{}]}), \
             patch('src.enrichment_tuning.search_models',side_effect=search), \
             patch('src.enrichment_tuning._preprocessor',side_effect=lambda m:Prep()), \
             patch('src.enrichment_tuning._fit_estimator',side_effect=fit):
            scores,audit=training_scores(frame(),1)
        self.assertEqual(len(searched),3)
        self.assertTrue(all(len(s)<120 for s in searched))
        self.assertTrue(all(np.isfinite(p).all() for p in scores.values()))
        self.assertEqual(set(audit.calibration_fold),{1,2,3})

    def test_outer_rows_never_enter_search_or_threshold_fit(self):
        data=frame();n=len(data);seen=[]
        splits=[(np.flatnonzero(np.arange(n)//24!=i),np.flatnonzero(np.arange(n)//24==i)) for i in range(5)]
        def scores(part,fold,progress):
            self.assertEqual(set(part.FPrimary),set(data.iloc[splits[fold-1][0]].FPrimary))
            seen.append(set(part.FPrimary))
            p=np.where(part.FPrimary%2,.8,.2)
            return {(s,'random_forest'):p for s in ['fixed','tuned']},pd.DataFrame({'model':['random_forest']})
        def search(part,seed):
            self.assertEqual(set(part.FPrimary),seen[-1])
            return {'random_forest':{}},pd.DataFrame({'model':['random_forest']})
        with patch('src.enrichment_tuning.TUNING_GRIDS',{'random_forest':[{}]}), \
             patch('src.enrichment_tuning.training_scores',side_effect=scores), \
             patch('src.enrichment_tuning.search_models',side_effect=search), \
             patch('src.enrichment_tuning._preprocessor',side_effect=lambda m:Prep()), \
             patch('src.enrichment_tuning._fit_estimator',side_effect=lambda m,p,X,y:(Estimator(X),1)):
            audit,metrics,oof,groups,curves=evaluate_enrichment_tuning(data,splits,progress=False)
        self.assertEqual(len(metrics),10);self.assertEqual(len(oof),240)
        self.assertTrue((metrics.inner_recall>=.8).all())
        self.assertEqual(len(seen),5)


if __name__=='__main__':unittest.main()
