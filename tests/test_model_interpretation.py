import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier,DMatrix
from src.model_interpretation import feature_group,group_contributions,coefficient_units


class InterpretationTests(unittest.TestCase):
    def test_grouping_preserves_signed_sum_and_missing_flags(self):
        names=['regioncode_region_1','regioncode_region_2','father_edu','father_edu_missing']
        got=group_contributions([[1,-2,3,4]],names)
        self.assertEqual(got.regioncode.iloc[0],-1)
        self.assertEqual(got.father_edu_missing.iloc[0],4)
        self.assertEqual(feature_group('father_edu_missing'),'father_edu_missing')
        with self.assertRaises(ValueError):feature_group('FPrimary')

    def test_grouping_accumulates_float32_values_in_double_precision(self):
        values=np.array([[1e7,.25,-1e7]],dtype=np.float32)
        got=group_contributions(values,['regioncode_1','age','regioncode_2'])
        self.assertEqual(got.sum(axis=1).iloc[0],.25)

    def test_reference_category_is_fit_encoder_reference(self):
        enc=Pipeline([('fill',SimpleImputer(strategy='most_frequent')),
                      ('onehot',OneHotEncoder(drop='first',sparse_output=False))])
        ct=ColumnTransformer([('nominal',enc,['regioncode'])],verbose_feature_names_out=False).fit(
            pd.DataFrame({'regioncode':['region_2','region_1','region_3']}))
        units=coefficient_units(SimpleNamespace(named_steps={'prep':ct}))
        self.assertEqual(units['regioncode_region_2'],'regioncode: region_2 versus reference region_1')

    def test_native_shap_reconstructs_heldout_raw_margin(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(80,3));y=(x[:,0]+x[:,1]>.1).astype(int)
        model=XGBClassifier(n_estimators=10,max_depth=2,n_jobs=1).fit(x[:60],y[:60])
        matrix=DMatrix(x[60:]);booster=model.get_booster()
        shap=booster.predict(matrix,pred_contribs=True,approx_contribs=False)
        np.testing.assert_allclose(shap.sum(axis=1),booster.predict(matrix,output_margin=True),atol=1e-5)


if __name__=='__main__':unittest.main()
