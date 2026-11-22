import unittest
import numpy as np
import pandas as pd
from ple_experiment import PLEPreprocessor

class PLETests(unittest.TestCase):
    def test_interpolation_and_validation_never_refits(self):
        p=PLEPreprocessor(['x'],2).fit(pd.DataFrame({'x':[0.,2.,4.]}))
        edges=p.edges[0].copy()
        got=p.transform(pd.DataFrame({'x':[-100.,1.,3.,100.]}))
        np.testing.assert_array_equal(got[:,:2],[[0,0],[.5,0],[1,.5],[1,1]])
        np.testing.assert_array_equal(p.edges[0],edges)

    def test_codes_repeated_edges_constants_and_empty(self):
        f=pd.DataFrame({'x':[0,0,0,4,-7,-8,-9,np.nan], 'z':[np.nan]*8})
        p=PLEPreprocessor(['x','z'],8).fit(f)
        got=p.transform(f)
        assert np.isfinite(got).all() and all(np.diff(p.edges[0])>0)
        # Last nine columns: three code flags per two inputs, nonfinite flags, missing fraction.
        np.testing.assert_array_equal(got[4,-9:-3],[0,0,0,0,1,0])
        np.testing.assert_array_equal(got,p.transform(f[['z','x']]))
        assert p.transform(f.iloc[:0]).shape==(0,got.shape[1])
        with self.assertRaises(ValueError): p.transform(f[['x']])

if __name__=='__main__': unittest.main()
