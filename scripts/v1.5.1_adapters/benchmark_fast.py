"""Synthetic performance witness; never fits models or reads outcome tables."""
from bootstrap import *

def fast_rank(y,score,counts,inv,chunk=64,ap_only=False):
    order=np.argsort(-score,kind='stable');yy=y[order];ss=score[order]
    ends=np.r_[np.flatnonzero(ss[1:]!=ss[:-1]),len(ss)-1]
    result=np.full((len(counts),3),np.nan)
    for start in range(0,len(counts),chunk):
        w=np.ascontiguousarray(counts[start:start+chunk,inv[order]],dtype=float)
        tp=np.cumsum(w*yy,axis=1)[:,ends];total=np.cumsum(w,axis=1)[:,ends];fp=total-tp
        pp=tp[:,-1];nn=fp[:,-1];dt=np.diff(np.c_[np.zeros(len(w)),tp],axis=1)
        precision=np.divide(tp,total,out=np.zeros_like(tp),where=total>0)
        result[start:start+len(w),0]=np.divide((precision*dt).sum(1),pp,out=np.full(len(w),np.nan),where=pp>0)
        if ap_only:continue
        dfp=np.diff(np.c_[np.zeros(len(w)),fp],axis=1)
        result[start:start+len(w),1]=np.divide(((2*tp-dt)*.5*dfp).sum(1),pp*nn,out=np.full(len(w),np.nan),where=(pp*nn)>0)
        for r in range(len(w)):
            if pp[r]<=0 or nn[r]<=0:continue
            active=np.flatnonzero(np.diff(np.r_[0.,total[r]])>0)
            ff=fp[r,active];j=int(np.searchsorted(ff,nn[r]*(.05+1e-12),side='right'))-1
            # The last eligible ROC point is the required one unless sklearn
            # drops it. Walk back only through dropped points; endpoints stay.
            while 0<j<len(active)-1:
                a,b,c=active[j-1:j+2]
                if fp[r,c]-2*fp[r,b]+fp[r,a]!=0 or tp[r,c]-2*tp[r,b]+tp[r,a]!=0:break
                j-=1
            result[start+r,2]=tp[r,active[j]]/pp[r] if j>=0 else 0.
    return result

def main():
    pin(12);rng=np.random.default_rng(151);n=15628;y=rng.integers(0,2,n).astype(float);z=rng.normal(size=n);co=rng.multinomial(n,np.full(n,1/n),size=2000).astype(np.int16)
    start=time.perf_counter();a=weighted_rank(y,z,co,np.arange(n));old=time.perf_counter()-start
    start=time.perf_counter();b=fast_rank(y,z,co,np.arange(n));new=time.perf_counter()-start
    err=float(np.nanmax(abs(a-b)));assert err<=1e-12
    for seed in range(10):
        rng=np.random.default_rng(seed);n=100;y=rng.integers(0,2,n);z=np.round(rng.normal(size=n),seed%3);co=rng.multinomial(n,np.full(n,1/n),size=100)
        a=weighted_rank(y,z,co,np.arange(n));b=fast_rank(y,z,co,np.arange(n));err=max(err,float(np.nanmax(abs(a-b))));assert err<=1e-12
    report=dict(old_seconds=old,new_seconds=new,max_abs_difference=err,synthetic_only=True,draws=3000)
    dump(RUN/'FAST_BOOTSTRAP_BENCHMARK.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
