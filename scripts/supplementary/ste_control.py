"""Prespecified STE-only regularization sensitivity; main artifacts are read-only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,hashlib,traceback,concurrent.futures
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score
from scipy.stats import wilcoxon
Q=Path('/home/wy/ECGFounder/supplement_queue_20261101_remote');O=Q/'cpsc9_rerun_20261117';E=Path(__file__).resolve().parent
sys.path.insert(0,str(O));import linear_rerun as L
from target_contract import certified_fit,objective
sha=L.sha;read=L.read;dump=L.dump

def fit(j):
 bb,k,s=j;p=E/'ste/models'/bb/f'shot{k}_seed{s:02d}.json'
 if p.exists():
  m=read(p);assert sha(p.with_suffix('.npz'))==m['prediction_sha256'];return m
 ep=read(O/'episodes'/f'shot{k}_seed{s:02d}.json');si=np.array(ep['support']);qi=np.array(ep['query']);y=np.asarray(ep['support_y'],float)[:,8:9]
 F=np.load(O/f'{bb}_features.npy',mmap_mode='r');xs=F[si].astype(float);mu=xs.mean(0);sd=xs.std(0)+1e-6;x=(xs-mu)/sd
 t=time.monotonic();w,b,d=certified_fit(x,y,10.1)
 val,g=objective(x,y,10.1)(np.r_[w.ravel(),b]);assert abs(g).max()<=5e-5
 # Same objective as alpha=.1 plus a lambda=10 zero-anchor penalty.
 val2,g2=objective(x,y,.1,np.zeros_like(w),10.)(np.r_[w.ravel(),b])
 assert abs(val-val2)<1e-10 and abs(g-g2).max()<1e-10
 z=((F[qi].astype(float)-mu)/sd)@w+b;assert z.shape==(len(qi),1) and np.isfinite(z).all()
 p.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(p.with_suffix('.npz'),logits=z,weight=w,bias=b,scaler_mean=mu,scaler_sd=sd,support_indices=si,query_indices=qi)
 m=dict(backbone=bb,shot=k,seed=s,alpha_effective=10.1,diagnostics=d,recomputed_max_gradient=float(abs(g).max()),seconds=time.monotonic()-t,prediction_sha256=sha(p.with_suffix('.npz')),episode_sha256=sha(O/'episodes'/f'shot{k}_seed{s:02d}.json'),query_labels_loaded_for_fitting=False)
 dump(p,m);return m

def main():
 assert (O/'VERIFIED_COMPLETE.json').exists()
 lock=dict(main_protocol_sha256=sha(O/'protocol.lock.json'),main_verification_sha256=sha(O/'VERIFIED_COMPLETE.json'),code_sha256=sha(__file__),settings=dict(tau=10.1,source_anchor=0,backbones=['ECGFounder','MERL'],shots=[5,10],seeds=list(range(20)),workers=4,selection=False),statistics='four backbone/shot conditions times two contrasts versus current STE or LP; one eight-test Holm family; seed bootstrap conditional on fixed query')
 p=E/'ste/protocol.lock.json'
 if p.exists():assert read(p)==lock
 else:dump(p,lock)
 jobs=[(bb,k,s) for bb in ['ECGFounder','MERL'] for k in [5,10] for s in range(20)]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  fs=[pool.submit(fit,j) for j in jobs]
  for i,f in enumerate(concurrent.futures.as_completed(fs),1):f.result();dump(E/'ste/progress.json',dict(done=i,total=80))
 lab=np.load(O/'labels.npz');qi=lab['query_indices'];y=lab['y'][qi];rows=[];checks=[]
 for bb,k,s in jobs:
  cp=E/'ste/models'/bb/f'shot{k}_seed{s:02d}.npz';a=np.load(cp);assert np.array_equal(a['query_indices'],qi);sc=a['logits'][:,0]
  for norm in ['NORM','SR']:
   base=O/'linear'/bb/norm/f'shot{k}_seed{s:02d}.npz';z=np.load(base)
   other=O/'linear'/bb/('SR' if norm=='NORM' else 'NORM')/base.name
   assert np.array_equal(z['LP'][:,8],np.load(other)['LP'][:,8])
   for m in ['SASM-Ens','MeanAnchor','SASM-Single']:
    for policy,ste in [('current_alpha0.1',z[m][:,8]),('zero_tau10.1',sc),('LP_fallback',z['LP'][:,8])]:
     ap=np.array([average_precision_score(y[:,c],ste if c==8 else z[m][:,c]) for c in range(9)])
     rows.append(dict(backbone=bb,shot=k,seed=s,normal_mapping=norm,method=m,policy=policy,ste_ap=ap[8],all9_ap=ap.mean(),matched8_ap=ap[:8].mean()))
 f=pd.DataFrame(rows);f.to_csv(E/'ste/episode_metrics.csv',index=False)
 group=['backbone','shot','normal_mapping','method','policy']
 f.groupby(group).agg(n=('all9_ap','size'),ste_ap=('ste_ap','mean'),ste_sd=('ste_ap','std'),all9_ap=('all9_ap','mean'),all9_sd=('all9_ap','std'),matched8_ap=('matched8_ap','mean')).reset_index().to_csv(E/'ste/summary.csv',index=False)
 stats=[];rng=np.random.default_rng(2026111701)
 for (bb,k),g in f[(f.normal_mapping=='NORM')&(f.method=='SASM-Ens')].groupby(['backbone','shot']):
  pv=g.pivot(index='seed',columns='policy',values='ste_ap')
  for control in ['current_alpha0.1','LP_fallback']:
   d=(pv['zero_tau10.1']-pv[control]).to_numpy();boot=rng.choice(d,(10000,20),replace=True).mean(1)
   stats.append(dict(backbone=bb,shot=int(k),contrast='zero_tau10.1 minus '+control,delta_pp=float(d.mean()*100),lo_pp=float(np.quantile(boot,.025)*100),hi_pp=float(np.quantile(boot,.975)*100),p=float(wilcoxon(d,method='auto').pvalue) if np.max(abs(d))>1e-12 else 1.))
 st=pd.DataFrame(stats);ix=st.sort_values('p').index;st.loc[ix,'p_holm']=np.minimum(1,np.maximum.accumulate(st.loc[ix,'p'].to_numpy()*(len(st)-np.arange(len(st)))));st.to_csv(E/'ste/paired_effects.csv',index=False)
 dump(E/'ste/COMPLETE.json',dict(time=time.time(),fits=80,protocol_sha256=sha(p),summary_sha256=sha(E/'ste/summary.csv'),main_result_unchanged=True))
 print('STE_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:dump(E/'ste/ERROR.json',dict(error=traceback.format_exc()));raise

