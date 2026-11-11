"""Small algebraic checks, independent of the clinical caches."""
import unittest
import numpy as np
from sasm.anchor import ecgfounder_raw_head,target_coordinate_anchor
from sasm.optimizer import objective


class CoreChecks(unittest.TestCase):
    def test_affine_scores(self):
        r=np.random.default_rng(6)
        features=r.normal(size=(9,5));dense=r.normal(size=(4,5));db=r.normal(size=4)
        w=r.normal(size=(4,3));b=r.normal(size=3);mu=r.normal(size=4);sd=r.uniform(.5,2,size=4)
        mapping=np.eye(3);tm=r.normal(size=5);ts=r.uniform(.5,2,size=5)
        raw,off=ecgfounder_raw_head(dense,db,w,b,mu,sd,mapping)
        anchor,bias=target_coordinate_anchor(raw,off,tm,ts)
        original=((features@dense.T+db-mu)/sd)@w+b
        converted=((features-tm)/ts)@anchor+bias
        np.testing.assert_allclose(original,converted,atol=1e-10)

    def test_gradient(self):
        r=np.random.default_rng(7);x=r.normal(size=(11,4));y=(r.random((11,2))>.5).astype(float)
        fn=objective(x,y,.1,r.normal(size=(4,2)),10.)
        t=r.normal(size=10);eye=np.eye(len(t));eps=1e-6
        numerical=np.array([(fn(t+e*eps)[0]-fn(t-e*eps)[0])/(2*eps) for e in eye])
        np.testing.assert_allclose(fn(t)[1],numerical,atol=1e-6,rtol=1e-6)

    def test_shared_objective_identity(self):
        r=np.random.default_rng(8);anchors=r.normal(size=(10,4,2));w=r.normal(size=(4,2));mean=anchors.mean(0)
        left=np.mean(np.sum((w-anchors)**2,axis=(1,2)))
        right=np.sum((w-mean)**2)+np.mean(np.sum((anchors-mean)**2,axis=(1,2)))
        self.assertAlmostEqual(left,right,places=10)


if __name__=='__main__':unittest.main()
