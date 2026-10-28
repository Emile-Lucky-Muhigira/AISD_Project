import unittest
import numpy as np
import pandas as pd
from src.geography_stress import region_splits

class RegionStressTests(unittest.TestCase):
    def test_regions_districts_and_households_never_cross_inner_boundary(self):
        d=pd.DataFrame({'FPrimary':np.arange(120),'hhmid':1,'district_id':np.repeat(np.arange(30),4),
                        'regioncode':np.repeat(np.arange(10),12),'dropout':np.tile([0,0,1,0],30)})
        seen=np.zeros(len(d),int)
        for a,b in region_splits(d,42):
            for col in ['regioncode','district_id','FPrimary']:
                self.assertFalse(set(d.iloc[a][col])&set(d.iloc[b][col]))
            seen[b]+=1
        np.testing.assert_array_equal(seen,np.ones(len(d),int))
