"""Bounded CPU infrastructure; all historical inputs remain read-only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import sys,json,time,hashlib,importlib.util,platform
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import expit,logit
from sklearn.metrics import average_precision_score,roc_auc_score,roc_curve
from threadpoolctl import threadpool_limits
ROOT=Path('/home/wy/ECGFounder/supplement_queue_20261101_remote')
OLD=ROOT/'supplement_core_20261118';DCA=ROOT/'dca_threshold_20261118'
RUN=ROOT/'sasm_v151_adapters_20261122';OUT=RUN/'results/v1.5.1_adapters'
SCRIPTS=RUN/'scripts/v1.5.1_adapters'
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(OLD));import common as H
THREADS=threadpool_limits(limits=1)
METHODS=['Tip-Adapter-ECG/BCE','Tip-Adapter-F-ECG/BCE','TaskRes-ECG/BCE','LP++-ECG/BCE']
CONTROLS=['LP','Direct-LP','SASM-Ens','MeanAnchor']
PTS=[.02,.05,.10,.15,.20]
SPEC_SHA='1ae90e1b4a9d50d015ea787569e053e47b4a4491ff492ff39364f5eff0766e59'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**22),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8');t.replace(p)
def notice(stage,**kw):
    v=dict(stage=stage,time=time.time(),pid=os.getpid(),**kw);print(json.dumps(v),flush=True);dump(RUN/'status'/f'{stage}.json',v)
def pin(core=None):
    core=10+2*(os.getpid()%4) if core is None else core
    os.sched_setaffinity(0,{core})
    try:os.nice(8)
    except OSError:pass
    import torch;torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
def normalize(x):return x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-12)
def metrics(y,z):
    assert y.shape==z.shape and np.isfinite(z).all()
    ap=[];auc=[];ece=[];f1=[];pp=expit(z)
    for c in range(y.shape[1]):
        assert 0<y[:,c].sum()<len(y)
        ap.append(average_precision_score(y[:,c],z[:,c]));auc.append(roc_auc_score(y[:,c],z[:,c]))
        ix=np.minimum((pp[:,c]*10).astype(int),9)
        ece.append(abs(np.bincount(ix,weights=pp[:,c]-y[:,c],minlength=10)).sum()/len(y))
        d=pp[:,c]>=.5;tp=float((d*y[:,c]).sum());den=float(d.sum()+y[:,c].sum());f1.append(2*tp/den if den else 0.)
    return dict(ap=float(np.mean(ap)),auc=float(np.mean(auc)),ece=float(np.mean(ece)),
                brier=float(((pp-y)**2).mean()),nll=float((np.logaddexp(0,z)-y*z).mean()),f1=float(np.mean(f1)))
def macro_ap(y,z):return float(np.mean([average_precision_score(y[:,c],z[:,c]) for c in range(y.shape[1])]))
def confusion(y,d):
    tp=(y*d).sum(0);fp=((1-y)*d).sum(0);pos=y.sum(0);neg=len(y)-pos
    return dict(specificity=(neg-fp)/neg,sensitivity=tp/pos,
                PPV=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=tp+fp>0),
                f1=np.divide(2*tp,pos+tp+fp,out=np.zeros_like(tp),where=pos+tp+fp>0),TP=tp,FP=fp)
def dca_values(y,z,pt):
    d=z>=logit(pt);co=confusion(y,d);nb=co['TP']/len(y)-co['FP']/len(y)*pt/(1-pt)
    return dict(NB=float(nb.mean()),standardized_NB=float((nb/y.mean(0)).mean()))
def archived(bb,ds,k,s,swap=False):
    p=OLD/('swap' if swap and ds!='sph' else 'base/swap' if swap else 'base/main')/bb/ds/f'shot{k}_seed{s:02d}.npz'
    return np.load(p),p

_DATA={}
def dataset(bb,ds,swap=False):
    key=(bb,ds,swap)
    if key not in _DATA:
        f,raw,off,cols=H.context(bb,ds)
        qi,y=H.labels(ds,swap)
        _DATA[key]=dict(F=f,raw=raw.mean(0),off=off.mean(0),cols=cols,qi=qi,y=y)
    return _DATA[key]

def support(bb,ds,k,s,swap=False,need_query=False):
    # This routine never calls the external-query label loader. Query labels are
    # read only by scoring/development selection after a fitted model exists.
    ep,si,qi,y=H.episode(ds,k,s,swap)
    F,raw,off,cols=H.context(bb,ds);raw=raw.mean(0);off=off.mean(0)
    hs=np.asarray(F[si],dtype=np.float64);mu=hs.mean(0);sd=hs.std(0)+1e-6;x=(hs-mu)/sd
    a=sd[:,None]*raw;pb=mu@raw+off;ss=x@a+pb;matched=np.ones(y.shape[1],bool)
    if ds=='cpsc2018':matched[-1]=False
    out=dict(bb=bb,ds=ds,shot=k,seed=s,swap=swap,si=si,qi=qi,y=y,hs=hs,x=x,mu=mu,sd=sd,a=a,pb=pb,ss=ss,matched=matched,raw=raw,off=off)
    if need_query:
        hq=np.asarray(F[qi],dtype=np.float64);out.update(hq=hq,xq=(hq-mu)/sd,sq=hq@raw+off)
    return out

def lp_scores(ep):
    bb,ds,k,s,swap=(ep[k] for k in ['bb','ds','shot','seed','swap'])
    # Reuse actual archived LP coefficients, never fit on query observations.
    if ds=='cpsc2018':
        p=H.O/'linear'/bb/'NORM'/f'shot{k}_seed{s:02d}.weights.npz';z=np.load(p);w=z['LP__w'];b=z['LP__b']
    else:
        base=H.SW if swap else H.B;phase='dev' if ds==('sph' if swap else 'chapman') else 'external'
        rel=read(base/('release_01.json' if bb=='ECGFounder' else 'merl/release.json'));ci=rel['selected']['LP']['candidate'];d=base if bb=='ECGFounder' else base/'merl'
        p=d/'results'/phase/ds/'LP'/f'c{ci:02d}_shot{k}_seed{s:02d}.weights.npz';z=np.load(p);w=z['weight'].mean(0);b=z['bias'].mean(0)
    assert np.max(abs(z['scaler_mean']-ep['mu']))<1e-10 and np.max(abs(z['scaler_sd']-ep['sd']))<1e-10
    return ep['x']@w+b,ep['xq']@w+b if 'xq' in ep else None

def grid(method):
    import itertools
    if method==METHODS[0]:
        return [dict(space=sp,encoding=en,beta=be,alpha=al) for sp,en,be,al in itertools.product(['raw_l2','std_l2'],['signed','positive'],[.5,1.,2.5,5.,10.,20.],[.1,.5,1.,2.,5.,10.])]
    if method==METHODS[1]:return [dict(lr=lr,epochs=ep,wd=wd) for lr,ep,wd in itertools.product([1e-4,5e-4,1e-3,5e-3],[20,50,100],[1e-4,1e-2])]
    if method==METHODS[2]:return [dict(gamma=ga,lambda_res=la,lr=lr,epochs=ep) for ga,la,lr,ep in itertools.product([.2,.5,1.],[0.,.01,.1,1.,10.],[1e-3,5e-3,1e-2],[20,50,100])]
    if method==METHODS[3]:return [dict(space=sp,alpha0=al,lambda_W=la,steps=st) for sp,al,la,st in itertools.product(['std','raw_l2'],[.1,.5,1.,2.],[0.,.01,.1,1.,10.],[50,100,200,300])]
    raise KeyError(method)
