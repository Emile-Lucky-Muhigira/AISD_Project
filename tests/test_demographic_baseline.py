import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.demographic_baseline import demographic_pipeline,evaluate_demographic,compare_demographic


class DemographicTests(unittest.TestCase):
    def test_uses_only_age_and_sex_and_training_imputation(self):
        x=pd.DataFrame({'age':[8.,10.,12.,np.nan],'sex':[1,2,1,2],'district_id':[1,2,3,4],'outcome_copy':[0,1,0,1]})
        p=demographic_pipeline().fit(x,[0,1,0,1]);q=x.copy()
        q['district_id']=999;q['outcome_copy']=1-q.outcome_copy
        np.testing.assert_array_equal(p.predict_proba(x),p.predict_proba(q))
        self.assertEqual(p.named_steps['prep'].named_transformers_['age'].named_steps['fill'].statistics_[0],10)
        self.assertEqual(p.named_steps['prep'].transform(x).shape[1],2)

    def test_nested_fitting_excludes_prediction_rows(self):
        n=120
        d=pd.DataFrame(dict(FPrimary=np.arange(n),hhmid=1,district_id=np.repeat(np.arange(20),6),
            age=8+np.arange(n)%6,sex=1+np.arange(n)%2,dropout=np.arange(n)%2,father_in_hh=1,mother_in_hh=1,regioncode=1))
        splits=[(np.flatnonzero(np.arange(n)//24!=i),np.flatnonzero(np.arange(n)//24==i)) for i in range(5)]
        fitted=[]
        class Estimator:
            classes_=np.array([0,1])
            def fit(self,X,y):self.ids=set(X.FPrimary);fitted.append(len(X));return self
            def predict_proba(self,X):
                assert not self.ids.intersection(X.FPrimary)
                p=np.where(X.sex==2,.7,.3);return np.c_[1-p,p]
        with patch('src.demographic_baseline.demographic_pipeline',side_effect=Estimator):
            metrics,oof,curves=evaluate_demographic(d,splits)
        self.assertEqual(len(metrics),10);self.assertEqual(len(oof),240)
        self.assertEqual(len(fitted),20);self.assertLessEqual(max(fitted),96)
        empty=oof.iloc[:0].copy()
        result=compare_demographic(d,oof,empty)
        self.assertEqual(result[result.dimension.eq('overall')].selected_n.iloc[0],20)
        broken=oof.copy();broken.loc[broken.method.eq('sigmoid'),'child_key_sha256']='bad'
        with self.assertRaises(ValueError):compare_demographic(d,broken,empty)


if __name__=='__main__':unittest.main()
