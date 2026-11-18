"""Fixed-support, patient-cluster bootstrap of the primary AP contrast."""
from core import *
import pandas as pd,concurrent.futures
def weighted_ap(y,score,weights):
 # Rows of weights represent bootstrap replications; tied scores use group-end precision.
 order=np.argsort(-score,kind='stable');yy=y[order];ss=score[order];w=weights[:,order]
 tp=np.cumsum(w*yy[None,:],axis=1);den=np.cumsum(w,axis=1)
 ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1]
 t=tp[:,ends];d=den[:,ends];increment=np.diff(np.c_[np.zeros(len(w)),t],axis=1)
 precision=np.divide(t,d,out=np.zeros_like(t),where=d>0)
 return np.divide((increment*precision).sum(1),tp[:,-1],out=np.full(len(w),np.nan),where=tp[:,-1]>0)
def one(pair):
 ds,k=pair;v=np.load(R/'vault'/ds/'external.npz',allow_pickle=True);y=v['y'];patients=v['patients'].astype(str);uniq,inv=np.unique(patients,return_inverse=True);n=len(uniq);rng=np.random.default_rng(20261105)
 # Multinomial counts are equivalent to drawing n patient clusters with replacement.
 counts=rng.multinomial(n,np.full(n,1/n),size=2000).astype(np.int16)
 valid=np.ones(2000,bool)
 for j in range(y.shape[1]):
  pos=counts@np.bincount(inv,weights=y[:,j],minlength=n);neg=counts@np.bincount(inv,weights=1-y[:,j],minlength=n);valid&=(pos>0)&(neg>0)
 total=np.zeros(2000,float)
 release=read(R/'release_01.json')
 for method,sign in [('SASM-Ens',1.),('Concat-LP',-1.)]:
  c=release['selected'][method]['candidate']
  for seed in range(20):
   j=dict(phase='external',dataset=ds,method=method,candidate=c,shot=k,seed=seed);pred=np.load(outputs(j).with_suffix('.npz'))['logits']
   value=np.zeros(2000,float)
   for col in range(y.shape[1]):
    for st in range(0,2000,16):
     w=counts[st:st+16,inv].astype(float);value[st:st+16]+=weighted_ap(y[:,col],pred[:,col],w)/y.shape[1]
   total+=sign*value/20
  print('PATIENT_BOOTSTRAP',ds,k,method,flush=True)
 good=valid&np.isfinite(total);interval=np.quantile(total[good],[.025,.975])*100 if good.mean()>=.95 else [None,None]
 return dict(dataset=ds,shot=k,baseline='Concat-LP',metric='ap',replications=2000,valid_replications=int(good.sum()),patient_CI_low_pp=interval[0],patient_CI_high_pp=interval[1],scope='query patient sampling conditional on the fixed twenty support sets; CPSC patient linkage limitation applies')
if __name__=='__main__':
 jobs=[(ds,k) for ds in EXTERNAL_DATASETS for k in [5,10]]
 with concurrent.futures.ProcessPoolExecutor(6) as ex:rows=list(ex.map(one,jobs))
 pd.DataFrame(rows).to_csv(R/'patient_bootstrap_CI.csv',index=False)
