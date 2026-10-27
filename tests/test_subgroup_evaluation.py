import hashlib
import json
import unittest
import numpy as np
import pandas as pd
from src.evaluation_metrics import select_top_risk
from src.subgroup_evaluation import subgroup_labels, evaluate_subgroups, _scores


class SubgroupTests(unittest.TestCase):
    def setUp(self):
        n=100
        self.train=pd.DataFrame({"FPrimary":np.arange(n),"hhmid":1,
            "district_id":np.repeat(np.arange(10),10),"dropout":np.tile([0,1],50),
            "regioncode":np.tile([1,2],50),"father_in_hh":1,"mother_in_hh":1})
        keys=[json.dumps([str(h),str(p)],separators=(',',':')) for h,p in zip(self.train.FPrimary,self.train.hhmid)]
        folds=np.repeat(np.arange(1,6),20)
        # All selections belong to region 1; demonstrates no within-group reranking.
        prob=np.where(self.train.regioncode.eq(1),.8,.2)
        selected=np.zeros(n,dtype=bool)
        for fold in range(1,6):
            idx=np.flatnonzero(folds==fold)
            selected[idx]=select_top_risk(prob[idx],[keys[i] for i in idx],.2)
        self.oof=pd.DataFrame({"model":"example","method":"sigmoid","fold":folds,
            "training_row":np.arange(n),"child_key_sha256":[hashlib.sha256(k.encode()).hexdigest() for k in keys],
            "y_true":self.train.dropout,"probability":prob,"selected_top20_within_fold":selected})

    def test_parent_categories_preserve_unknown(self):
        d=pd.DataFrame({"father_in_hh":[1,1,0,0,np.nan,1],"mother_in_hh":[1,0,1,0,0,np.nan],"regioncode":[1]*6})
        self.assertEqual(subgroup_labels(d).parent_co_residence.tolist(),
            ["both_parents","father_only","mother_only","neither_parent","unknown","unknown"])

    def test_original_policy_and_sparse_single_class_groups(self):
        result=evaluate_subgroups(self.train,self.oof,bootstrap_reps=0)
        regions=result[result.dimension.eq("region")].set_index("subgroup")
        self.assertEqual(regions.loc["region_1","selected_n"],20)
        self.assertEqual(regions.loc["region_2","selected_n"],0)
        self.assertEqual(regions.loc["region_2","recall_under_fold_policy"],0)
        self.assertTrue(np.isnan(regions.loc["region_2","precision_under_fold_policy"]))
        self.assertTrue(regions.sparse_group.all())
        self.assertTrue(regions.average_precision.isna().all())
        self.assertEqual(regions.n.sum(),100)
        self.assertEqual(regions.tp.sum()+regions.fp.sum(),20)

    def test_reject_wrong_identity_duplicate_and_policy(self):
        for mutation in ["identity","duplicate","policy"]:
            bad=self.oof.copy()
            if mutation=="identity": bad.loc[0,"child_key_sha256"]="wrong"
            if mutation=="duplicate": bad.loc[0,"training_row"]=1
            if mutation=="policy": bad.loc[0,"selected_top20_within_fold"]=not bad.loc[0,"selected_top20_within_fold"]
            with self.assertRaises(ValueError): evaluate_subgroups(self.train,bad,0)

    def test_bootstrap_reproducible_and_cluster_weights(self):
        a=evaluate_subgroups(self.train,self.oof,30,42)
        b=evaluate_subgroups(self.train,self.oof,30,42)
        pd.testing.assert_frame_equal(a,b)
        r=a[a.dimension.eq("overall")].iloc[0]
        self.assertEqual(r.brier_score_bootstrap_valid,30)
        self.assertAlmostEqual(r.brier_score_ci_low,.64)
        self.assertAlmostEqual(r.brier_score_ci_high,.64)
        y=np.array([0,1,1]);p=np.array([.1,.8,.4]);s=np.array([0,1,0],bool);w=np.array([2,1,3])
        weighted=_scores(y,p,s,w)
        repeated=_scores(np.repeat(y,w),np.repeat(p,w),np.repeat(s,w))
        for key in weighted: self.assertAlmostEqual(weighted[key],repeated[key])

    def test_groups_cannot_span_folds(self):
        bad=self.train.copy()
        bad.loc[20,"district_id"]=0
        with self.assertRaises(ValueError): evaluate_subgroups(bad,self.oof,0)


if __name__=="__main__": unittest.main()
