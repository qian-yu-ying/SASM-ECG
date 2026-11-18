from common import *
from concurrent.futures import ProcessPoolExecutor,as_completed
from scipy.optimize import minimize_scalar
def core(job):
 bb,ds,k,s,swap=job;bind([10,11]);tag='swap' if swap else 'main';p=E/'base'/tag/bb/ds/f'shot{k}_seed{s:02d}.npz'
 if p.exists():assert sha(p)==read(p.with_suffix('.json'))['sha256'];return
 si,qi,y,mu,sd,x,xq,raw,off=arrays(bb,ds,k,s,swap);pred={};refs={};diags={}
 for m in ['LP','SASM-Ens']:
  pred[m],path=hist(bb,ds,k,s,m,swap);refs[m]=dict(path=str(path),sha256=sha(path))
 a=sd[:,None]*raw.mean(0);pb=mu@raw.mean(0)+off.mean(0);lv=np.r_[np.full(8,10.),0.] if ds=='cpsc2018' else None;w,b,d=fit(x,y,.1,a,10.,lv,pb);pred['MeanAnchor']=xq@w+b;diags['MeanAnchor']=d
 direct=(xq*sd+mu)@raw.mean(0)+off.mean(0)
 if ds=='cpsc2018':direct[:,-1]=pred['LP'][:,-1]
 pred['Direct']=direct
 if not swap:
  mix=.3 if bb=='ECGFounder' else .45;pred['Direct-LP']=(1-mix)*pred['LP']+mix*direct
  # Restore support-only fixed LP coefficients, then refit only blend intercepts.
  if ds=='cpsc2018':z=np.load(O/'linear'/bb/'NORM'/f'shot{k}_seed{s:02d}.weights.npz');lp=x@z['LP__w']+z['LP__b']
  else:
   lpfile=Path(refs['LP']['path']).with_suffix('.weights.npz');z=np.load(lpfile);lp=x@z['weight'].mean(0)+z['bias'].mean(0)
  direct_s=(x*sd+mu)@raw.mean(0)+off.mean(0)
  if ds=='cpsc2018':direct_s[:,-1]=lp[:,-1]
  sup=(1-mix)*lp+mix*direct_s;delta=np.zeros(y.shape[1]);active=(y.sum(0)>0)&(y.sum(0)<len(y))
  for _ in range(100):
   prob=expit(sup+delta);g=(prob-y).mean(0);h=(prob*(1-prob)).mean(0)
   if abs(g[active]).max()<1e-11:break
   delta[active]-=np.clip(g[active]/np.maximum(h[active],1e-12),-5,5)
  assert abs((expit(sup+delta)-y).mean(0)[active]).max()<1e-9
  pred['Direct-LP-Refit-b']=pred['Direct-LP']+delta;diags['intercept_gradient']=float(abs((expit(sup+delta)-y).mean(0)[active]).max());diags['constant_intercept_columns']=np.flatnonzero(~active).tolist()
 p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**pred,support_indices=si,query_indices=qi);dump(p.with_suffix('.json'),dict(sha256=sha(p),references=refs,diagnostics=diags,protocol_sha256=sha(E/'protocol.lock.json')))
def basepath(bb,ds,k,s,swap=False):return E/'base'/('swap' if swap else 'main')/bb/ds/f'shot{k}_seed{s:02d}.npz'
def calibration_select():
 selected={}
 for bb in ['ECGFounder','MERL']:
  qi,y=labels('chapman');data=[np.load(basepath(bb,'chapman',k,s)) for k in [5,10] for s in range(10)];selected[bb]={}
  for m in ['LP','SASM-Ens','MeanAnchor','Direct-LP','Direct-LP-Refit-b']:
   zz=np.stack([z[m] for z in data]);assert all(np.array_equal(z['query_indices'],qi) for z in data)
   def objective(logT):
    z=zz/np.exp(logT);return float((np.logaddexp(0,z)-y[None]*z).mean())
   r=minimize_scalar(objective,bounds=(np.log(.05),np.log(20.)),method='bounded',options={'xatol':1e-10});assert r.success
   selected[bb][m]=dict(T=float(np.exp(r.x)),development_nll=float(r.fun),raw_nll=objective(0.),bound_hit=bool(abs(r.x-np.log(.05))<1e-4 or abs(r.x-np.log(20))<1e-4))
 dump(E/'temperature.lock.json',dict(selected=selected,selection='Chapman equal-episode BCE, no external labels loaded by this function',protocol_sha256=sha(E/'protocol.lock.json')))
 emit('TEMPERATURE_LOCKED',selected=selected)
def calibration_score():
 lock=read(E/'temperature.lock.json');rows=[];reliability=[];checks=[]
 for bb in ['ECGFounder','MERL']:
  for ds in ['chapman','ningbo','cpsc2018','sph']:
   qi,y=labels(ds)
   for k in [5,10]:
    for s in range(10 if ds=='chapman' else 20):
     z=np.load(basepath(bb,ds,k,s));assert np.array_equal(qi,z['query_indices'])
     for m,v in lock['selected'][bb].items():
      scores=z[m];T=v['T'];rawmetric=metric(y,scores);scaled=metric(y,scores/T);assert abs(rawmetric['ap']-scaled['ap'])<1e-12 and abs(rawmetric['auc']-scaled['auc'])<1e-12
      for mode,zz in [('Raw',scores),('DevTemperature',scores/T)]:
       rows.append(dict(backbone=bb,cohort=ds,shot=k,seed=s,method=m,calibration=mode,T=T if mode!='Raw' else 1.,**rawmetric,**calibration(y,zz)))
       # Macro reliability: each label contributes equal bin weight, no pooled micro-ECE substitution.
       pp=expit(zz)
       for c in range(y.shape[1]):
        bi=np.minimum((pp[:,c]*10).astype(int),9)
        for a in range(10):
         ix=bi==a
         reliability.append(dict(backbone=bb,cohort=ds,shot=k,seed=s,method=m,calibration=mode,label=c,bin=a,n=int(ix.sum()),positives=float(y[ix,c].sum()),sum_probability=float(pp[ix,c].sum())))
    emit('CALIBRATION_SCORE',backbone=bb,cohort=ds,shot=k)
 pd.DataFrame(rows).to_csv(E/'calibration_episodes.csv',index=False);pd.DataFrame(reliability).to_csv(E/'reliability_bins.csv',index=False);dump(E/'CALIBRATION_COMPLETE.json',dict(rows=len(rows),sha256=sha(E/'calibration_episodes.csv'),temperature_sha256=sha(E/'temperature.lock.json')))
def block_dev(job):
 bb,k,s=job;bind([10,11]);p=E/'swap_dev_block'/bb/f'shot{k}_seed{s:02d}.csv'
 if p.exists():return
 si,qi,y,mu,sd,x,xq,raw,off=arrays(bb,'sph',k,s,True);ss=(x*sd+mu)@raw.mean(0)+off.mean(0);sq=(xq*sd+mu)@raw.mean(0)+off.mean(0);sm=ss.mean(0);ssig=ss.std(0)+1e-6;cx=np.c_[x,(ss-sm)/ssig];cq=np.c_[xq,(sq-sm)/ssig];ix,yy=labels('sph',True);assert np.array_equal(ix,qi);rows=[]
 for a in [.01,.1,1.,10.,100.,1000.]:
  for b in [.01,.1,1.,10.,100.,1000.]:
   scale=np.sqrt(np.r_[np.full(x.shape[1],a),np.full(y.shape[1],b)]);w,bi,d=fit(cx/scale,y,1.);pr=cq/scale@w+bi;rows.append(dict(backbone=bb,shot=k,seed=s,feature_alpha=a,logit_alpha=b,**metric(yy,pr),max_gradient=d['max_gradient']))
 p.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(p,index=False)
def swap_select():
 rows=[];selected={}
 for bb in ['ECGFounder','MERL']:
  ix,y=labels('sph',True);values=[]
  for mix in np.linspace(0,1,21):
   vals=[]
   for k in [5,10]:
    for s in range(10):
     z=np.load(basepath(bb,'sph',k,s,True));vals.append(metric(y,(1-mix)*z['LP']+mix*z['Direct'])['ap'])
   values.append((float(np.mean(vals)),float(mix)));rows.append(dict(backbone=bb,method='Direct-LP',mix=mix,ap=np.mean(vals)))
  val,mix=max(values,key=lambda v:(v[0],-v[1]));f=pd.concat([pd.read_csv(p) for p in sorted((E/'swap_dev_block'/bb).glob('*.csv'))]);assert len(f)==720;g=f.groupby(['feature_alpha','logit_alpha'],as_index=False).ap.mean().sort_values(['ap','feature_alpha','logit_alpha'],ascending=[False,True,True]);best=g.iloc[0];selected[bb]=dict(mix=mix,development_mix_ap=val,feature_alpha=float(best.feature_alpha),logit_alpha=float(best.logit_alpha),development_block_ap=float(best.ap));g['backbone']=bb;g.to_csv(E/f'swap_block_candidates_{bb}.csv',index=False)
 dump(E/'swap_selection.lock.json',dict(selected=selected,development='SPH only',protocol_sha256=sha(E/'protocol.lock.json')));pd.DataFrame(rows).to_csv(E/'swap_mix_candidates.csv',index=False);emit('SWAP_SELECTION_LOCKED',selected=selected)
def swap_target(job):
 bb,ds,k,s=job;bind([10,11]);p=E/'swap'/bb/ds/f'shot{k}_seed{s:02d}.npz'
 if p.exists():assert sha(p)==read(p.with_suffix('.json'))['sha256'];return
 cfg=read(E/'swap_selection.lock.json')['selected'][bb];si,qi,y,mu,sd,x,xq,raw,off=arrays(bb,ds,k,s,True);z=np.load(basepath(bb,ds,k,s,True));pred={m:z[m] for m in ['LP','Direct','SASM-Ens','MeanAnchor']};pred['Direct-LP']=(1-cfg['mix'])*z['LP']+cfg['mix']*z['Direct'];ss=(x*sd+mu)@raw.mean(0)+off.mean(0);sq=(xq*sd+mu)@raw.mean(0)+off.mean(0);sm=ss.mean(0);ssig=ss.std(0)+1e-6;cx=np.c_[x,(ss-sm)/ssig];cq=np.c_[xq,(sq-sm)/ssig];scale=np.sqrt(np.r_[np.full(x.shape[1],cfg['feature_alpha']),np.full(y.shape[1],cfg['logit_alpha'])]);w,b,d=fit(cx/scale,y,1.);pred['Block-Concat']=cq/scale@w+b
 p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**pred,query_indices=qi);dump(p.with_suffix('.json'),dict(sha256=sha(p),config=cfg,diagnostics=d,selection_sha256=sha(E/'swap_selection.lock.json')))
def pool(fn,jobs,stage):
 with ProcessPoolExecutor(2) as ex:
  for i,f in enumerate(as_completed([ex.submit(fn,j) for j in jobs])):
   f.result()
   if (i+1)%10==0 or i+1==len(jobs):emit(stage,done=i+1,total=len(jobs))
def main():
 mainjobs=[(bb,ds,k,s,False) for bb in ['ECGFounder','MERL'] for ds in ['chapman','ningbo','cpsc2018','sph'] for k in [5,10] for s in range(10 if ds=='chapman' else 20)]
 pool(core,mainjobs,'MAIN_ASSETS');calibration_select();calibration_score()
 swapjobs=[(bb,ds,k,s,True) for bb in ['ECGFounder','MERL'] for ds in ['sph','chapman','ningbo','cpsc2018'] for k in [5,10] for s in range(10 if ds=='sph' else 20)]
 pool(core,swapjobs,'SWAP_ASSETS');pool(block_dev,[(bb,k,s) for bb in ['ECGFounder','MERL'] for k in [5,10] for s in range(10)],'SWAP_BLOCK_DEV');swap_select()
 jobs=[(bb,ds,k,s) for bb in ['ECGFounder','MERL'] for ds in ['chapman','ningbo','cpsc2018'] for k in [5,10] for s in range(20)];pool(swap_target,jobs,'SWAP_TARGET');rows=[]
 for bb,ds,k,s in jobs:
  ix,y=labels(ds,True);z=np.load(E/'swap'/bb/ds/f'shot{k}_seed{s:02d}.npz');assert np.array_equal(ix,z['query_indices'])
  for m in z.files:
   if m=='query_indices':continue
   rows.append(dict(backbone=bb,cohort=ds,shot=k,seed=s,method=m,**metric(y,z[m])))
 pd.DataFrame(rows).to_csv(E/'swap_episodes.csv',index=False);dump(E/'SWAP_COMPLETE.json',dict(rows=len(rows),sha256=sha(E/'swap_episodes.csv')))
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback;dump(E/'SUPPLEMENTS_ERROR.json',dict(traceback=traceback.format_exc()));raise
