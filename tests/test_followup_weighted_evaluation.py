import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src import followup_weighted_evaluation as f

class WeightedTests(unittest.TestCase):
    def test_prediction_rejects_outcomes(self):
        with self.assertRaises(ValueError):f.fit_predict(None,pd.DataFrame({'dropout':[1]}),None)
    def test_excluded_groups_cannot_enter_weight_training(self):
        d=pd.DataFrame({'FPrimary':[1],'hhmid':[1],'district_id':[1],'status':['enrolled']})
        with self.assertRaises(ValueError):f.observation_weights(d,d,{1})
    def test_nested_weighting_blocks_calibration_and_outer_groups(self):
        d=pd.DataFrame({'FPrimary':np.arange(60),'hhmid':1,'district_id':np.repeat(np.arange(30),2),'dropout':np.tile([0,1],30)})
        target=pd.DataFrame({'FPrimary':[99,100],'hhmid':1,'district_id':[99,100]})
        calls=[]
        def weights(base,train,excluded):
            self.assertTrue({99,100}.issubset(excluded))
            self.assertFalse(set(train.district_id)&set(excluded));calls.append(excluded)
            return np.ones(len(train))
        def predict(train,features,w):
            self.assertFalse(set(train.district_id)&set(features.district_id))
            return np.linspace(.1,.9,len(features))
        with patch.object(f,'observation_weights',side_effect=weights),patch.object(f,'fit_predict',side_effect=predict):
            p,_=f.calibrated_prediction(None,d,target,{99,100})
        self.assertEqual(len(calls),4);self.assertEqual(len(p),2)
        self.assertTrue(all(len(c)>2 for c in calls[1:]))
