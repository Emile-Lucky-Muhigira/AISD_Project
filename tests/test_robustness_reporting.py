import unittest
import pandas as pd
from src.robustness_reporting import paired_comparison

class PairedTests(unittest.TestCase):
    def test_identical_policies_have_exact_zero_differences(self):
        d=pd.DataFrame({'district_id':[1,1,2,2,3,3],'y_true':[1,0,1,0,1,0],
                        'probability':[.8,.2,.7,.1,.6,.3],'selected':[True,False,True,False,False,False]})
        d['baseline_probability']=d.probability;d['baseline_selected']=d.selected
        r=paired_comparison(d,50)
        for k,v in r.items():
            if k.startswith('delta_'):self.assertEqual(v,0)
