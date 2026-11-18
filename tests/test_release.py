"""Small numerical tests for new portable release functions; no assets needed."""
import unittest
import numpy as np
from sasm.source_solver import solve_cell,make_objective
from sasm.model import fit_cached_features
from sasm.route_b import fit_route_b

class ReleaseTests(unittest.TestCase):
    def test_source_solver_stationarity_and_gradient(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(30,4));y=(rng.random((30,2))>.5).astype(float);mask=np.ones_like(y,bool)
        w,b,d=solve_cell(x,y,mask,1.,'bce_signed',6000,5e-5)
        self.assertTrue(d['converged'])
        fun=make_objective(x,y,mask,1.,'bce');t=np.vstack([w,b]).ravel();_,g=fun(t)
        self.assertLess(np.max(abs(g)),5e-5)
        h=1e-5;idx=3;tp=t.copy();tm=t.copy();tp[idx]+=h;tm[idx]-=h
        self.assertAlmostEqual((fun(tp)[0]-fun(tm)[0])/(2*h),g[idx],places=6)

    def test_native_fit_matches_common_solver(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(25,4));y=(rng.random((25,2))>.5).astype(float)
        rw=rng.normal(size=(1,4,2));rb=np.zeros((1,2));cfg=dict(alpha=.1,lam=10.)
        a,b,_=fit_route_b(x,y,rw,rb,cfg,'native_raw')
        c,d,_=fit_cached_features(x,y,rw,rb,method='sasm-ens',**cfg)
        np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)

    def test_native_constant_label_policy_fails_closed(self):
        with self.assertRaises(ValueError):
            fit_route_b(np.ones((5,3)),np.zeros((5,2)),np.zeros((1,3,2)),np.zeros((1,2)),dict(alpha=.1,lam=10.),'native_raw')

if __name__=='__main__':unittest.main()
