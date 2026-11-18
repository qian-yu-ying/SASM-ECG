import os,sys
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
OLD=BASE
SUP=BASE
sys.path.insert(0,str(OLD));sys.path.insert(0,str(OLD/'source'))
import core as original
import cpu_worker as C
import numpy as np,pandas as pd,time,json,hashlib
from threadpoolctl import threadpool_limits
LIMITER=threadpool_limits(limits=1)
from target_contract import certified_fit,objective
sha=original.sha;read=original.read;dump=original.dump;metric=original.metric
DS=['ningbo','cpsc2018','sph']
VARIANTS=['nonnegative','source_only','target_only','none','perm25','perm50']
def phase(ds):return 'dev' if ds=='chapman' else 'external'
def episode(ds,k,seed):return OLD/'data'/ds/'episodes'/f'{phase(ds)}_shot{k}_seed{seed:02d}.json'
def original_result(backbone,ds,k,seed,method='SASM-Ens'):
 r=OLD if backbone=='ECGFounder' else SUP/'merl';rel=read(r/('release_01.json' if backbone=='ECGFounder' else 'release.json'));c=rel['selected'][method]['candidate']
 return r/'results'/phase(ds)/ds/method/f'c{c:02d}_shot{k}_seed{seed:02d}.json'
def taskkey(j):return f"{j['backbone']}__{j['variant']}__{j['dataset']}__k{j['shot']}__s{j['seed']:02d}"
def output(j):return ROOT/'results'/j['backbone']/j['variant']/j['dataset']/f"shot{j['shot']}_seed{j['seed']:02d}.json"
def jobs():
 for bb,variants in [('ECGFounder',['ensemble',*VARIANTS]),('MERL',['ensemble'])]:
  for variant in variants:
   for ds in DS:
    for k in [5,10]:
     for seed in range(20):yield dict(backbone=bb,variant=variant,dataset=ds,shot=k,seed=seed)
def context(ds,bb='ECGFounder'):
 F=np.load((OLD/'data' if bb=='ECGFounder' else SUP/'merl/data')/ds/'features.npy',mmap_mode='r')
 return F
def input_arrays(j,F):
 ep=read(episode(j['dataset'],j['shot'],j['seed']));si=np.array(ep['support']);qi=np.array(ep['query']);ys=np.asarray(ep['support_y'],float)
 fs=F[si].astype(float);fq=F[qi].astype(float);mu=fs.mean(0);sd=fs.std(0)+1e-6
 return ep,si,qi,ys,mu,sd,(fs-mu)/sd,(fq-mu)/sd
def source_context(ds):
 d=OLD/'data'/ds;q=np.load(d/'Q.npy');dense=np.load(OLD/'assets/dense.npz');dw=dense['Dw'];db=dense['Db'];heads=[]
 for e in range(10):
  z=np.load(OLD/'source/source_heads'/f'ep{e:02d}_bce_signed_model.npz');w=z['weight']@q;b=z['bias']@q;ss=z['scaler_sd'];sm=z['scaler_mean']
  heads.append(dict(raw=dw.T@(w/ss[:,None]),raw_no_source_scale=dw.T@w,offset=b+((db-sm)/ss)@w))
 return heads
def anchors(heads,mu,sd,variant,ds):
 perm=np.array(read(ROOT/'permutations.json')[ds][variant]['mapping']) if variant.startswith('perm') else None
 result=[]
 for h in heads:
  a=sd[:,None]*h['raw'];bp=mu@h['raw']+h['offset']
  if variant=='source_only':a=h['raw'].copy()
  elif variant=='target_only':a=sd[:,None]*h['raw_no_source_scale']
  elif variant=='none':a=h['raw_no_source_scale'].copy()
  elif perm is not None:a=a[:,perm];bp=bp[perm]
  result.append((a,bp))
 return result
def solve(xs,ys,alpha,lam,a,bp,nonnegative=False):
 w,b,diag=certified_fit(xs,ys,alpha,W_anchor=a,lam=lam,nonneg=nonnegative)
 const=np.flatnonzero((ys.sum(0)==0)|(ys.sum(0)==len(ys)))
 if len(const):
  # Constant labels are outside active BCE optimization. Use nearest feasible prior fallback.
  w[:,const]=np.maximum(a[:,const],0) if nonnegative else a[:,const]
  b[const]=bp[const]
 if nonnegative:assert w.min()>=0
 return w,b,diag
def gradient_check(xs,ys,w,b,a,alpha,lam,nonnegative):
 active=np.flatnonzero((ys.sum(0)>0)&(ys.sum(0)<len(ys)));const=np.setdiff1d(np.arange(ys.shape[1]),active)
 if not len(active):return 0.
 fun=objective(xs,ys[:,active],alpha,a[:,active],lam);_,g=fun(np.r_[w[:,active].ravel(),b[active]]);gw=g[:xs.shape[1]*len(active)].reshape(xs.shape[1],len(active)).copy()
 if nonnegative:gw[(w[:,active]<=1e-12)&(gw>0)]=0
 val=max(float(abs(gw).max()),float(abs(g[xs.shape[1]*len(active):]).max()));assert val<=5e-5,val
 return val
