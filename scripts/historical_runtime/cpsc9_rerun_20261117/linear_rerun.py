"""Locked historical Chapman configurations; complete nine-class CPSC rerun."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,traceback,argparse,concurrent.futures
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from audit_prepare import O,R,Q,B,dump,sha,NAMES
sys.path.insert(0,str(B/'source'))
from target_contract import certified_fit,objective
threadpool_limits(limits=1)
COLS=[13,63,19,9,16,49,44,57,-1]
_CONTEXT={}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def seal():
 er=read(B/'release_01.json');mr=read(B/'merl/release.json');review=read(Q/'reviewer_closure_20261107/release.json')
 source_files=[B/f'source/source_heads/ep{e:02d}_bce_signed_model.npz' for e in range(10)]+[Q/f'merl_source/source_heads/ep{e:02d}.npz' for e in range(10)]+[Q/'route_b_fullpool_native_20261116/source'/f'{bb}_fullpool_head.npz' for bb in ['ECGFounder','MERL']]
 cfg={bb:dict(LP=rel['selected']['LP']['config'],Concat=rel['selected']['Concat-LP']['config'],SASM=rel['selected']['SASM-Ens']['config'],Block=review['selected'][bb]['Block-Concat']['config'],DirectLP=review['selected'][bb]['Direct-LP-interpolation']['config'],FullPool=read(Q/'route_b_fullpool_native_20261116/target/fullpool'/bb/'release.json')['selected']['config']) for bb,rel in [('ECGFounder',er),('MERL',mr)]}
 cfg['ECGFounder']['FullFT']=er['selected']['FullFT']['config'];cfg['ECGFounder']['LoRA']=er['selected']['LoRA']['config']
 p=O/'protocol.lock.json';v=dict(date='2026-11-17',scope='CPSC2018 nine-class reconstruction; no other cohort rerun or old worker resumed',physical_gpu=1,shots=[5,10],seeds=list(range(20)),configs=cfg,source_row75=COLS,normal_mapping='NORM primary, SR predefined sensitivity for source-transfer methods; never select between mappings on query AP',STE='unmatched by design; per-class lambda=0 and zero source column; target BCE fit remains active. Direct and Direct-LP use LP score for STE because no direct source scorer exists.',source_assets={str(f):sha(f) for f in source_files},input_hashes={str(f.name):sha(f) for f in [O/'manifest.csv',O/'labels.npz',O/'mapping.json',O/'AUDIT_COMPLETE.json',O/'FEATURES_COMPLETE.json']},episode_hashes={f.name:sha(f) for f in (O/'episodes').glob('*.json')},code_hashes={f.name:sha(f) for f in O.glob('*.py')},retrospective=True,prior_result_exposure=True,hyperparameter_selection='reuse historical Chapman-only locked releases; no CPSC query selection or early stopping',statistics='paired support seeds conditional on fixed query; all9 primary, matched8 and STE secondary; duplicate-group query sensitivity',constant_columns='assert absent in nine-class support episodes',cpu_workers=4)
 if p.exists():assert read(p)==v,'protocol changed'
 else:dump(p,v)
 return v
def context(bb,normal):
 key=(bb,normal)
 if key in _CONTEXT:return _CONTEXT[key]
 cols=COLS.copy();cols[0]=13 if normal=='NORM' else 70
 labels=pd.read_csv(B/'source/source_labels.csv').ptbxl_label.astype(str).tolist();seen=set();keep=[]
 for i,n in enumerate(labels):
  n=' '.join(n.lower().split())
  if n not in seen:keep.append(i);seen.add(n)
 assert len(keep)==71
 q=np.zeros((71,9))
 for j,c in enumerate(cols):
  if c>=0:q[keep.index(c),j]=1
 dense=np.load(B/'assets/dense.npz');dw=dense['Dw'];db=dense['Db'];heads=[]
 def convert(z):
  w=z['weight']@q;b=z['bias']@q;sd=z['scaler_sd'];mu=z['scaler_mean']
  if bb=='ECGFounder':raw=dw.T@(w/sd[:,None]);off=b+((db-mu)/sd)@w;noscale=dw.T@w
  else:raw=w/sd[:,None];off=b-(mu/sd)@w;noscale=w
  assert not raw[:,-1].any() and off[-1]==0
  return dict(raw=raw,off=off,noscale=noscale)
 for e in range(10):
  p=B/f'source/source_heads/ep{e:02d}_bce_signed_model.npz' if bb=='ECGFounder' else Q/f'merl_source/source_heads/ep{e:02d}.npz'
  heads.append(convert(np.load(p)))
 full=convert(np.load(Q/'route_b_fullpool_native_20261116/source'/f'{bb}_fullpool_head.npz'))
 fp=O/f'{bb}_features.npy';assert sha(fp)==read(O/'FEATURES_COMPLETE.json')['files'][fp.name]
 F=np.load(fp,mmap_mode='r');_CONTEXT[key]=(F,heads,full,dw,db,cols);return _CONTEXT[key]
def run_episode(bb,k,s,normal='NORM'):
 p=O/'linear'/bb/normal/f'shot{k}_seed{s:02d}.json'
 if p.exists():
  m=read(p);assert sha(p.with_suffix('.npz'))==m['predictions_sha256'];return m
 start=time.monotonic();proto=read(O/'protocol.lock.json');cfg=proto['configs'][bb];ep=read(O/'episodes'/f'shot{k}_seed{s:02d}.json');si=np.array(ep['support']);qi=np.array(ep['query']);y=np.asarray(ep['support_y'],float)
 assert not set(si)&set(qi) and ((y.sum(0)>0)&(y.sum(0)<len(y))).all()
 F,heads,full,dw,db,cols=context(bb,normal);fs=F[si].astype(float);fq=F[qi].astype(float);mu=fs.mean(0);sd=fs.std(0)+1e-6;x=(fs-mu)/sd;xq=(fq-mu)/sd
 predictions={};models={};diags={};timings={};lamvec=np.r_[np.ones(8),0.]
 def fit(name,xx,xxq,alpha,anchor=None,lam=0.,nonneg=False):
  t=time.monotonic();w,b,d=certified_fit(xx,y,alpha,W_anchor=anchor,lam=lam,lam_vec=lam*lamvec if anchor is not None else None,nonneg=nonneg);assert not d['constant_columns'];pred=xxq@w+b
  timings[name]=time.monotonic()-t;diags[name]=d;models[name+'__w']=w;models[name+'__b']=b;return pred,w,b
 lp,lw,lb=fit('LP',x,xq,cfg['LP']['alpha']);predictions['LP']=lp
 raw=np.mean([h['raw'] for h in heads],0);off=np.mean([h['off'] for h in heads],0);a=sd[:,None]*raw
 direct=fq@raw+off;direct[:,-1]=lp[:,-1];predictions['Direct']=direct;models['Direct__raw']=raw;models['Direct__off']=off
 mix=cfg['DirectLP']['mix'];predictions['Direct-LP']=(1-mix)*lp+mix*direct
 ww=[];bs=[];dd=[];tt=time.monotonic()
 for i,h in enumerate(heads):
  anchor=sd[:,None]*h['raw'];beta=mu@h['raw']+h['off'];assert abs((x@anchor+beta)-(fs@h['raw']+h['off'])).max()<1e-7
  pr,w,b=fit('SASM-head%02d'%i,x,xq,cfg['SASM']['alpha'],anchor,cfg['SASM']['lam']);ww.append(w);bs.append(b);dd.append(diags['SASM-head%02d'%i])
 predictions['SASM-Ens']=xq@np.mean(ww,0)+np.mean(bs,0);predictions['SASM-Single']=xq@ww[0]+bs[0];timings['SASM-Ens']=time.monotonic()-tt
 predictions['MeanAnchor'],mw,mb=fit('MeanAnchor',x,xq,cfg['SASM']['alpha'],a,cfg['SASM']['lam'])
 predictions['FullPool-SASM'],_,_=fit('FullPool-SASM',x,xq,cfg['FullPool']['alpha'],sd[:,None]*full['raw'],cfg['FullPool']['lam'])
 full_direct=fq@full['raw']+full['off'];full_direct[:,-1]=lp[:,-1];predictions['FullPool-Direct']=full_direct
 ss=fs@raw+off;ms=ss.mean(0);ds=ss.std(0)+1e-6;cx=np.c_[x,(ss-ms)/ds];cq=np.c_[xq,(fq@raw+off-ms)/ds]
 predictions['Concat-LP'],_,_=fit('Concat-LP',cx,cq,cfg['Concat']['alpha'])
 penalty=np.r_[np.full(x.shape[1],cfg['Block']['feature_alpha']),np.full(9,cfg['Block']['logit_alpha'])];scale=np.sqrt(penalty)
 predictions['Block-Concat'],_,_=fit('Block-Concat',cx/scale,cq/scale,1.)
 for name,anchor in [('No-target-scaling',raw),('No-source-scaling',sd[:,None]*np.mean([h['noscale'] for h in heads],0))]:
  predictions[name],_,_=fit(name,x,xq,cfg['SASM']['alpha'],anchor,cfg['SASM']['lam'])
 if bb=='ECGFounder':
  tab=pd.read_csv(B/'source/source_labels.csv');tasks=[n.strip().lower() for n in (B/'source/tasks.txt').read_text(encoding='utf-8-sig').splitlines() if n.strip()];native=np.zeros_like(raw);covered=np.zeros(9,bool)
  oldsd=np.mean([np.load(B/f'source/source_heads/ep{e:02d}_bce_signed_model.npz')['scaler_sd'] for e in range(10)],0);nativepaired=np.zeros_like(raw)
  for j,c in enumerate(cols):
   if c<0:continue
   name=str(tab.iloc[c].ecgfounder_label).strip().lower()
   if name in tasks:
    ind=tasks.index(name);native[:,j]=dw[ind];nativepaired[:,j]=dw[ind]/oldsd[ind];covered[j]=True
  for name,an in [('Native-raw',native),('Native-source-scaled',nativepaired)]:
   mask=np.where(covered,10.,0.);anchor=sd[:,None]*an;t=time.monotonic();w,b,d=certified_fit(x,y,.1,W_anchor=anchor,lam=10.,lam_vec=mask);predictions[name]=xq@w+b;models[name+'__w']=w;models[name+'__b']=b;diags[name]=d;timings[name]=time.monotonic()-t
  zs=fs@dw.T+db;zq=fq@dw.T+db;zm=zs.mean(0);zd=zs.std(0)+1e-6;predictions['LP-150logits'],_,_=fit('LP-150logits',(zs-zm)/zd,(zq-zm)/zd,cfg['LP']['alpha'])
 # Numerical invariants for the unsupported class: no source prior is present.
 assert np.allclose(predictions['Direct'][:,-1],lp[:,-1],rtol=0,atol=0)
 assert all(v.shape==(len(qi),9) and np.isfinite(v).all() for v in predictions.values())
 p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p.with_suffix('.npz'),**predictions,support_indices=si,query_indices=qi);np.savez_compressed(p.with_suffix('.weights.npz'),**models,scaler_mean=mu,scaler_sd=sd,source_raw=raw,source_offset=off,concat_mean=ms,concat_sd=ds,block_scale=scale)
 # Replay the central fitted models independently from saved coefficients.
 wz=np.load(p.with_suffix('.weights.npz'));replay=xq@np.mean([wz[f'SASM-head{i:02d}__w'] for i in range(10)],0)+np.mean([wz[f'SASM-head{i:02d}__b'] for i in range(10)],0)
 assert abs(replay-predictions['SASM-Ens']).max()<1e-9
 m=dict(backbone=bb,shot=k,seed=s,normal_mapping=normal,seconds=time.monotonic()-start,timings=timings,diagnostics=diags,methods=list(predictions),support_n=len(si),query_n=len(qi),predictions_sha256=sha(p.with_suffix('.npz')),weights_sha256=sha(p.with_suffix('.weights.npz')),protocol_sha256=sha(O/'protocol.lock.json'),query_labels_used_for_training_or_selection=False)
 dump(p,m);print('LINEAR_DONE',bb,normal,k,s,round(m['seconds'],2),flush=True);return m
def main(pilot=False):
 assert (O/'FEATURES_COMPLETE.json').exists();proto=seal()
 jobs=[(bb,k,s,norm) for bb in ['ECGFounder','MERL'] for norm in ['NORM','SR'] for k in [5,10] for s in range(20)]
 if pilot:jobs=[('ECGFounder',5,0,'NORM'),('MERL',5,0,'NORM')]
 start=time.monotonic();done=0
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(run_episode,*j) for j in jobs]
  for f in concurrent.futures.as_completed(futures):
   f.result();done+=1;dump(O/'linear_progress.json',dict(done=done,total=len(jobs),seconds=time.monotonic()-start,pilot=pilot))
 dump(O/('LINEAR_PILOT_COMPLETE.json' if pilot else 'LINEAR_COMPLETE.json'),dict(episodes=len(jobs),seconds=time.monotonic()-start))
if __name__=='__main__':
 try:main('--pilot' in sys.argv)
 except Exception:dump(O/'LINEAR_ERROR.json',dict(error=traceback.format_exc()));raise
