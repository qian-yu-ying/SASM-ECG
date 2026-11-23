"""v1.5.2 isolated CPU infrastructure; historical assets are read-only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import sys,json,time,hashlib,platform,traceback
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import expit,logit
from scipy.optimize import minimize,minimize_scalar
from sklearn.metrics import average_precision_score,roc_auc_score,roc_curve
from threadpoolctl import threadpool_limits
ROOT=Path('/home/wy/ECGFounder/supplement_queue_20261101_remote')
OLD=ROOT/'supplement_core_20261118';DCA=ROOT/'dca_threshold_20261118';V151=ROOT/'sasm_v151_adapters_20261122'
RUN=ROOT/'sasm_v152_bspc_micro_20261123';OUT=RUN/'results/v1.5.2_micro';SCRIPTS=RUN/'scripts/v1.5.2_micro'
SPEC='SASM_v1.5.2_BSPC_MicroRevision_Server_Spec.md';SPEC_SHA='2db81e9bef247f43c9eeae956bb7d67329dc48e9fdcd87a60983abe3587eaaa2'
SPEC151_SHA='1ae90e1b4a9d50d015ea787569e053e47b4a4491ff492ff39364f5eff0766e59'
sys.path.insert(0,str(OLD));import common as H
LIMIT=threadpool_limits(limits=1)
METHODS=['MeanAnchor','SASM-Ens','TaskRes-ECG/BCE','Direct-LP','LP','LP++-ECG/BCE','Direct-LP-Refit-b']
BB=['ECGFounder','MERL'];DS=['ningbo','cpsc2018','sph'];SHOTS=[5,10]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8');tmp.replace(p)
def notice(stage,**kw):
    v=dict(stage=stage,time=time.time(),pid=os.getpid(),**kw);print(json.dumps(v),flush=True);dump(RUN/'status'/f'{stage}.json',v)
def pin(core=None):
    os.sched_setaffinity(0,{10+2*(os.getpid()%4) if core is None else core})
    try:os.nice(8)
    except OSError:pass
    import torch;torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
def guard():assert read(OUT/'gate_v152_check.json')['all_passed'] and sha(RUN/SPEC)==SPEC_SHA
def history_snapshot():
    rows={}
    for base in [OLD,DCA,V151]:
        for p in sorted(base.rglob('*')):
            if p.is_file():
                s=p.stat();rows[str(p)]=[s.st_size,s.st_mtime_ns]
    return rows

def episode(bb,ds,k,s,swap=False,query=True):
    ep,si,qi,y=H.episode(ds,k,s,swap);F,raw,off,cols=H.context(bb,ds)
    hs=np.asarray(F[si],float);mu=hs.mean(0);sd=hs.std(0)+1e-6;x=(hs-mu)/sd
    raw=raw.mean(0);off=off.mean(0);a=sd[:,None]*raw;pb=mu@raw+off
    out=dict(bb=bb,ds=ds,k=k,s=s,swap=swap,si=si,qi=qi,y=y,mu=mu,sd=sd,x=x,a=a,pb=pb)
    if query:out['xq']=(np.asarray(F[qi],float)-mu)/sd
    assert not np.intersect1d(si,qi).size
    return out
def old_prediction(bb,ds,k,s,swap=False):
    phase='swap' if swap and ds!='sph' else 'base/swap' if swap else 'base/main'
    return np.load(OLD/phase/bb/ds/f'shot{k}_seed{s:02d}.npz')
def frozen_predictions(bb,ds,k,s):
    a=old_prediction(bb,ds,k,s);pred={m:a[m] for m in ['MeanAnchor','SASM-Ens','Direct-LP','LP']}
    if 'Direct-LP-Refit-b' in a.files:pred['Direct-LP-Refit-b']=a['Direct-LP-Refit-b']
    p=V151/'fits/main'/bb/ds/f'shot{k}_seed{s:02d}.npz';assert sha(p)==read(p.with_suffix('.json'))['sha256'];z=np.load(p)
    pred['TaskRes-ECG/BCE']=z['m2__query'];pred['LP++-ECG/BCE']=z['m3__query'];pred['Tip-Adapter-F-ECG/BCE']=z['m1__query']
    assert np.array_equal(a['query_indices'],z['query_indices']);return pred
def temperature(bb,m):
    p=V151/'results/v1.5.1_adapters/calibration_thresholds_main.lock.json' if m in ['TaskRes-ECG/BCE','LP++-ECG/BCE','Tip-Adapter-F-ECG/BCE'] else OLD/'temperature.lock.json'
    return read(p)['selected'][bb][m]['T']
def label_names(ds):return np.load(H.O/'labels.npz')['names'].astype(str).tolist() if ds=='cpsc2018' else read(H.B/'data'/ds/'metadata.json')['labels']
def macro_ap(y,z):return float(np.mean([average_precision_score(y[:,c],z[:,c]) for c in range(y.shape[1])]))
def probability_metrics(y,z):
    p=expit(z);ec=[]
    for c in range(y.shape[1]):
        ix=np.minimum((p[:,c]*10).astype(int),9);ec.append(abs(np.bincount(ix,weights=p[:,c]-y[:,c],minlength=10)).sum()/len(y))
    return dict(ece=float(np.mean(ec)),brier=float(((p-y)**2).mean()),nll=float((np.logaddexp(0,z)-y*z).mean()))
def rank_metrics(y,z):
    se=[];pp=[]
    for c in range(y.shape[1]):
        f,t,th=roc_curve(y[:,c],z[:,c],drop_intermediate=True);i=np.flatnonzero(1-f>=.95-1e-12)[-1];d=z[:,c]>=th[i];tp=(y[:,c]*d).sum();fp=((1-y[:,c])*d).sum()
        se.append(t[i]);pp.append(tp/(tp+fp) if tp+fp else 0.)
    return dict(ap=macro_ap(y,z),auc=float(np.mean([roc_auc_score(y[:,c],z[:,c]) for c in range(y.shape[1])])),sensitivity95=float(np.mean(se)),ppv95=float(np.mean(pp)))
