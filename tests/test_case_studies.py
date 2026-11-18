"""Small reproducibility and statistical sanity checks."""
import unittest
import numpy as np
from src.bird_naive_bayes import evaluate as bird_eval
from src.recommendation_ab import welch_test, evaluate as reco_eval
from src.conversion_ab import evaluate as conversion_eval, two_proportion_z_test

class CaseStudyTests(unittest.TestCase):
    def test_bird_model_is_reproducible(self):
        a,b=bird_eval(),bird_eval()
        self.assertEqual(a['accuracy'],b['accuracy'])
        self.assertGreater(a['accuracy'],0.75)
    def test_welch_detects_difference(self):
        r=reco_eval()
        self.assertLess(r['p_value'],0.05)
        self.assertLess(r['control_mean'],r['treatment_mean'])
    def test_welch_no_difference(self):
        v=np.arange(20, dtype=float)
        self.assertAlmostEqual(welch_test(v,v)['p_value'],1.0)
    def test_z_test_detects_difference(self):
        r=conversion_eval()
        self.assertTrue(r['reject_null'])
        self.assertLess(r['z_statistic'],0)
    def test_z_test_symmetry(self):
        a=two_proportion_z_test(20,100,30,100)
        b=two_proportion_z_test(30,100,20,100)
        self.assertAlmostEqual(a['z_statistic'],-b['z_statistic'])
        self.assertAlmostEqual(a['p_value'],b['p_value'])

if __name__=='__main__':
    unittest.main()
