import unittest
import pandas as pd
from src.data_cleaning import conflicting_test_keys,incomplete_asset_households,align

class CleaningTests(unittest.TestCase):
    def test_conflict_is_order_independent(self):
        d=pd.DataFrame({'FPrimary':[1,1,2,2],'member':[1,1,1,1],'q':['A','B','A','a ']})
        expected={(1,1)}
        self.assertEqual(conflicting_test_keys(d,'member',['q']),expected)
        self.assertEqual(conflicting_test_keys(d.iloc[::-1],'member',['q']),expected)

    def test_owned_unknown_not_zero_but_known_zero_preserved(self):
        d=pd.DataFrame({'hhno':[1,2,3,4],'s3aiii_a':[1,1,0,1],
                        's3aiii_c1':[None,0,None,None],'s3aiii_c2':[None,None,None,50]})
        self.assertEqual(incomplete_asset_households(d,1),{1})

    def test_wave_two_partial_total_is_uncertain(self):
        d=pd.DataFrame({'FPrimary':[1,1,2],'quantity':[1,1,0],'currentvalue':[20,None,None]})
        self.assertEqual(incomplete_asset_households(d,2),{1})

    def test_alignment_rejects_identified_duplicates(self):
        b=pd.DataFrame({'FPrimary':[1],'hhmid':[1]})
        d=pd.DataFrame({'FPrimary':[1,1],'hhmid':[1,1]})
        with self.assertRaises(pd.errors.MergeError):align(b,d)

    def test_alignment_keeps_base_order_and_ignores_unidentified(self):
        b=pd.DataFrame({'FPrimary':[2,1],'hhmid':[1,1]})
        d=pd.DataFrame({'FPrimary':[1,2,3,3],'hhmid':[1,1,None,None],'v':[4,5,6,7]})
        self.assertEqual(align(b,d).v.tolist(),[5,4])

if __name__=='__main__':unittest.main()
