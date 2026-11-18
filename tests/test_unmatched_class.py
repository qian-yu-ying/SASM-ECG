"""Check that unsupported labels receive the declared per-class penalty."""
import unittest
import numpy as np
from sasm.model import fit_cached_features
from sasm.optimizer import certified_fit,objective

class UnmatchedTests(unittest.TestCase):
    def test_zero_lambda_keeps_alpha_only(self):
        rng=np.random.default_rng(13);x=rng.normal(size=(40,6));y=(rng.random((40,2))>.5).astype(float)
        rw=np.zeros((2,6,2));rb=np.zeros((2,2))
        w,b,_=fit_cached_features(x,y,rw,rb,alpha=.1,lam=10.,lam_vec=np.array([10.,0.]))
        lp,lb,_=fit_cached_features(x,y,method='lp',alpha=.1)
        np.testing.assert_allclose(w[:,1],lp[:,1],atol=2e-4,rtol=0)
        np.testing.assert_allclose(b[1],lb[1],atol=2e-4,rtol=0)
        self.assertGreater(np.max(abs(w[:,0]-lp[:,0])),1e-3)

    def test_class_specific_gradient(self):
        rng=np.random.default_rng(14);x=rng.normal(size=(12,3));y=(rng.random((12,2))>.5).astype(float)
        fun=objective(x,y,.1,np.zeros((3,2)),10.,np.array([10.,0.]))
        t=rng.normal(size=8);eps=1e-6;eye=np.eye(8)
        num=np.array([(fun(t+eps*e)[0]-fun(t-eps*e)[0])/(2*eps) for e in eye])
        np.testing.assert_allclose(fun(t)[1],num,atol=1e-6,rtol=1e-6)
if __name__=='__main__':unittest.main()
