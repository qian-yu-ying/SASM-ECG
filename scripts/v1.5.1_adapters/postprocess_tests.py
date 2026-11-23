"""Synthetic and actual-data witnesses before expensive postprocessing."""
from bootstrap import *

def main():
    pin(10);validate();errors=[];checks=0
    for seed in range(10):
        rng=np.random.default_rng(seed);n=50;y=rng.integers(0,2,n);s=np.round(rng.normal(size=n),seed%3)
        co=rng.multinomial(20,np.full(20,.05),size=15);inv=np.arange(n)%20
        v=weighted_rank(y,s,co,inv,chunk=7);a=weighted_rank(y,s,co,inv,chunk=7,ap_only=True)
        assert np.allclose(v[:,0],a[:,0],equal_nan=True)
        for i,wc in enumerate(co):
            w=wc[inv];ref=[average_precision_score(y,s,sample_weight=w),roc_auc_score(y,s,sample_weight=w),rocpoint_weighted(y,s,w)[0]]
            errors.append(float(np.max(abs(v[i]-ref))));checks+=3
    for bb in ['ECGFounder','MERL']:
        for ds in ['ningbo','cpsc2018','sph']:
            qi,y=H.labels(ds);pred,_,_,order=load_predictions('main',bb,ds,5,0);assert np.array_equal(qi,order)
            rng=np.random.default_rng(151);n=len(y);co=rng.multinomial(n,np.full(n,1/n),size=8);inv=np.arange(n)
            for c in [0,y.shape[1]-1]:
                z=pred[METHODS[0]][:,c];v=weighted_rank(y[:,c],z,co,inv)
                for i,w in enumerate(co):
                    ref=[average_precision_score(y[:,c],z,sample_weight=w),roc_auc_score(y[:,c],z,sample_weight=w),rocpoint_weighted(y[:,c],z,w)[0]]
                    errors.append(float(np.max(abs(v[i]-ref))));checks+=3
    assert max(errors)<=1e-12,max(errors)
    dump(OUT/'postprocess_unit_tests.json',dict(all_passed=True,synthetic_and_real_checks=checks,max_sklearn_error=max(errors),drop_intermediate_exact=True,zero_weight_groups_removed=True,ap_only_identical=True))
    notice('POSTPROCESS_TESTS_PASSED',max_error=max(errors),checks=checks)

if __name__=='__main__':main()
