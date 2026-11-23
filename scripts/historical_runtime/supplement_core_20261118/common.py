"""Isolated retrospective supplements. Historical inputs are read-only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,hashlib,importlib.util
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score,roc_auc_score
from threadpoolctl import threadpool_limits
E=Path(__file__).resolve().parent
R=Path('/home/wy/ECGFounder');Q=R/'supplement_queue_20261101_remote'
B=Q/'chapman_dev_native_same_20261103';SW=Q/'sph_dev_native_same_full_20261105';O=Q/'cpsc9_rerun_20261117'
sys.path.insert(0,str(B/'source'))
from target_contract import certified_fit
LIMIT=threadpool_limits(limits=1)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8');t.replace(p)
def emit(stage,**v):
 v=dict(stage=stage,pid=os.getpid(),time=time.time(),**v);print(json.dumps(v),flush=True);dump(E/'status'/f'{stage}.json',v)
def module(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def bind(cpus):os.sched_setaffinity(0,set(cpus));os.nice(8)
def metric(y,z):
 valid=np.flatnonzero((y.sum(0)>0)&(y.sum(0)<len(y)))
 ap=[average_precision_score(y[:,c],z[:,c]) for c in valid];auc=[roc_auc_score(y[:,c],z[:,c]) for c in valid]
 return dict(ap=float(np.mean(ap)),auc=float(np.mean(auc)),classes=len(valid))
def calibration(y,z):
 p=expit(z);ece=[];f1=[]
 for c in range(y.shape[1]):
  iy=np.minimum((p[:,c]*10).astype(int),9);v=0.
  for k in range(10):
   ix=iy==k
   if ix.any():v+=abs(float((y[ix,c]-p[ix,c]).sum()))/len(y)
  ece.append(v);yp=p[:,c]>=.5;tp=float((yp*y[:,c]).sum());den=float(yp.sum()+y[:,c].sum());f1.append(2*tp/den if den else 0.)
 return dict(ece=float(np.mean(ece)),brier=float(((p-y)**2).mean()),f1=float(np.mean(f1)),nll=float(np.mean(np.logaddexp(0,z)-y*z)))
def fit(x,y,alpha,anchor=None,lam=10.,lv=None,prior_bias=None):
 w,b,d=certified_fit(x,y,alpha,W_anchor=anchor,lam=lam if anchor is not None else 0.,lam_vec=lv)
 const=np.array(d['constant_columns'],int)
 if len(const) and anchor is not None:
  w[:,const]=anchor[:,const]
  if prior_bias is not None:b[const]=prior_bias[const]
 return w,b,d
def keep75():
 a=pd.read_csv(B/'source/source_labels.csv').ptbxl_label.astype(str);seen=set();keep=[]
 for i,n in enumerate(a):
  n=' '.join(n.lower().split())
  if n not in seen:keep.append(i);seen.add(n)
 assert len(keep)==71
 return keep
_CTX={}
def context(bb,ds):
 key=(bb,ds)
 if key in _CTX:return _CTX[key]
 if ds=='cpsc2018':
  sys.path.insert(0,str(O));L=module('rerun_linear_supp',O/'linear_rerun.py');f,hs,*_=L.context(bb,'NORM');raw=np.array([h['raw'] for h in hs]);off=np.array([h['off'] for h in hs]);cols=[13,63,19,9,16,49,44,57,-1]
 else:
  f=np.load(B/('data' if bb=='ECGFounder' else 'merl/data')/ds/'features.npy',mmap_mode='r');cols=read(B/'data'/ds/'metadata.json')['cols'];dense=np.load(B/'assets/dense.npz');dw,db=dense['Dw'],dense['Db'];raw=[];off=[];idx=[keep75().index(c) for c in cols]
  for e in range(10):
   p=B/f'source/source_heads/ep{e:02d}_bce_signed_model.npz' if bb=='ECGFounder' else Q/f'merl_source/source_heads/ep{e:02d}.npz';z=np.load(p);w=z['weight'][:,idx];b=z['bias'][idx];mu=z['scaler_mean'];sd=z['scaler_sd']
   raw.append(dw.T@(w/sd[:,None]) if bb=='ECGFounder' else w/sd[:,None]);off.append(b+((db-mu)/sd)@w if bb=='ECGFounder' else b-(mu/sd)@w)
  raw=np.array(raw);off=np.array(off)
 _CTX[key]=(f,raw,off,cols);return _CTX[key]
def episode(ds,k,s,swap=False):
 base=SW if swap else B;phase='dev' if ds==('sph' if swap else 'chapman') else 'external'
 p=O/'episodes'/f'shot{k}_seed{s:02d}.json' if ds=='cpsc2018' else base/'data'/ds/'episodes'/f'{phase}_shot{k}_seed{s:02d}.json';ep=read(p)
 return ep,np.array(ep['support']),np.array(ep['query']),np.array(ep['support_y'],float)
def labels(ds,swap=False):
 if ds=='cpsc2018':z=np.load(O/'labels.npz');return z['query_indices'],z['y'][z['query_indices']].astype(float)
 base=SW if swap else B;dev=ds==('sph' if swap else 'chapman');p=base/'data'/ds/'development_labels.npz' if dev else base/'vault'/ds/'external.npz';z=np.load(p);return z['indices'],z['y'].astype(float)
def arrays(bb,ds,k,s,swap=False):
 ep,si,qi,y=episode(ds,k,s,swap);F,raw,off,cols=context(bb,ds);a=F[si].astype(float);mu=a.mean(0);sd=a.std(0)+1e-6
 assert not set(si)&set(qi)
 return si,qi,y,mu,sd,(a-mu)/sd,(F[qi].astype(float)-mu)/sd,raw,off
def hist(bb,ds,k,s,method,swap=False):
 if ds=='cpsc2018':
  p=O/'linear'/bb/'NORM'/f'shot{k}_seed{s:02d}.npz';z=np.load(p);return z[method],p
 base=SW if swap else B;phase='dev' if ds==('sph' if swap else 'chapman') else 'external';rel=read(base/('release_01.json' if bb=='ECGFounder' else 'merl/release.json'));c=rel['selected'][method]['candidate'];d=base if bb=='ECGFounder' else base/'merl';p=d/'results'/phase/ds/method/f'c{c:02d}_shot{k}_seed{s:02d}.npz';z=np.load(p);return z['logits'],p
