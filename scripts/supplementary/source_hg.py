from common import *
from concurrent.futures import ProcessPoolExecutor,as_completed
ALPHAS=[.01,.1,1.,10.,100.,1000.,10000.]
def source(job):
 interface,e=job;bind([8] if interface=='h' else [9]);p=E/'source'/interface/f'ep{e:02d}.npz'
 if p.exists():assert sha(p)==read(p.with_suffix('.json'))['sha256'];return
 t=time.monotonic();sp=R/f'res/b3_sign_loss_diagnostic_20261014_v1/ep{e:02d}_support_and_cv.npz';z=np.load(sp);rows=z['support_rows'];y=z['support_y'].astype(float);mask=z['observation_mask'].astype(bool);fold=z['patient_cv_fold'];cols=z['cv_label_indices']
 h=np.load(R/'res/features_1024/ptbxl_feat1024_filtered.npz')['feat'][rows].astype(float);d=np.load(B/'assets/dense.npz');x=h if interface=='h' else h@d['Dw'].T+d['Db']
 db=pd.read_csv(R/'data/PTBXL_v1.0.3/ptbxl_database.csv');assert (db.strat_fold.to_numpy()[rows]<=8).all()
 pts=db.patient_id.to_numpy()[rows];assert not set(pts[fold==0])&set(pts[fold==1]);assert mask.all()
 solver=module('source_solver_supp',R/'code_takeover_20261020/src_v2/frozen_b3.py');oof={a:np.zeros(y.shape) for a in ALPHAS};diags=[]
 for held in [0,1]:
  tr=fold!=held;va=~tr;mu,sd=solver.scaler_fit(x[tr]);xt=(x[tr]-mu)/sd;xv=(x[va]-mu)/sd;warm=None
  for a in sorted(ALPHAS,reverse=True):
   w,b,dd=solver.solve_cell(xt,y[tr],mask[tr],a,'bce_signed',6000,5e-5,warm);assert dd['converged'];warm=(w,b);oof[a][va]=xv@w+b;diags.append(dict(dd,fold=held,alpha=a));emit('SOURCE_'+interface,episode=e,fold=held,alpha=a,fits=len(diags),seconds=time.monotonic()-t)
 scores={a:solver.oof_ap(y,oof[a],mask,cols) for a in ALPHAS};a=max(ALPHAS,key=lambda a:(scores[a],-a));mu,sd=solver.scaler_fit(x);w,b,dd=solver.solve_cell((x-mu)/sd,y,mask,a,'bce_signed',6000,5e-5);assert dd['converged']
 raw=w/sd[:,None] if interface=='h' else d['Dw'].T@(w/sd[:,None]);off=b-(mu/sd)@w if interface=='h' else b+((d['Db']-mu)/sd)@w
 err=float(abs((x-mu)/sd@w+b-(h@raw+off)).max());assert err<1e-7
 p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,weight=w,bias=b,scaler_mean=mu,scaler_sd=sd,raw=raw,off=off,source_rows=rows,source_oof=oof[a]);dump(p.with_suffix('.json'),dict(interface=interface,episode=e,alpha=a,scores=scores,diagnostics=diags,refit=dd,coordinate_error=err,sha256=sha(p),support_sha256=sha(sp),protocol_sha256=sha(E/'protocol.lock.json'),seconds=time.monotonic()-t));emit('SOURCE_DONE',interface=interface,episode=e,seconds=time.monotonic()-t)
def target(job):
 ds,k,s=job;bind([8,9]);p=E/'hg'/ds/f'shot{k}_seed{s:02d}.npz'
 if p.exists():assert sha(p)==read(p.with_suffix('.json'))['sha256'];return
 si,qi,y,mu,sd,x,xq,_,_=arrays('ECGFounder',ds,k,s);cols=context('ECGFounder',ds)[3];idx=[keep75().index(c) if c>=0 else -1 for c in cols];mask=np.array([c>=0 for c in cols]);pred={};diags={};weights={}
 for interface in ['h','g']:
  ww=[];bs=[];raws=[];offs=[]
  for e in range(10):
   z=np.load(E/'source'/interface/f'ep{e:02d}.npz');raw=np.zeros((x.shape[1],len(cols)));off=np.zeros(len(cols));raw[:,mask]=z['raw'][:,np.array(idx)[mask]];off[mask]=z['off'][np.array(idx)[mask]];a=sd[:,None]*raw;pb=mu@raw+off;w,b,d=fit(x,y,.1,a,10.,10.*mask,pb);ww.append(w);bs.append(b);raws.append(raw);offs.append(off);diags[f'{interface}{e}']=d
  pred[f'{interface}-SASM-Ens']=xq@np.mean(ww,0)+np.mean(bs,0);raw=np.mean(raws,0);off=np.mean(offs,0);w,b,d=fit(x,y,.1,sd[:,None]*raw,10.,10.*mask,mu@raw+off);pred[f'{interface}-MeanAnchor']=xq@w+b;diags[f'{interface}-MeanAnchor']=d;direct=context('ECGFounder',ds)[0][qi].astype(float)@raw+off
  if (~mask).any():lp,_=hist('ECGFounder',ds,k,s,'LP');direct[:,~mask]=lp[:,~mask]
  pred[f'{interface}-Direct']=direct;weights[f'{interface}-Ens-w']=np.mean(ww,0);weights[f'{interface}-Ens-b']=np.mean(bs,0);weights[f'{interface}-Mean-w']=w;weights[f'{interface}-Mean-b']=b
 p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**pred,query_indices=qi,support_indices=si);np.savez_compressed(p.with_suffix('.weights.npz'),**weights,scaler_mean=mu,scaler_sd=sd);dump(p.with_suffix('.json'),dict(sha256=sha(p),diags=diags,protocol_sha256=sha(E/'protocol.lock.json')))
def main():
 with ProcessPoolExecutor(2) as ex:
  for f in as_completed([ex.submit(source,(it,e)) for e in range(10) for it in ['h','g']]):f.result()
 emit('SOURCE_ALL_DONE')
 jobs=[(ds,k,s) for ds in ['chapman','ningbo','cpsc2018','sph'] for k in [5,10] for s in range(10 if ds=='chapman' else 20)]
 with ProcessPoolExecutor(2) as ex:
  for i,f in enumerate(as_completed([ex.submit(target,j) for j in jobs])):f.result();emit('HG_TARGET',done=i+1,total=len(jobs))
 rows=[]
 for ds,k,s in jobs:
  qi,y=labels(ds);z=np.load(E/'hg'/ds/f'shot{k}_seed{s:02d}.npz');assert np.array_equal(qi,z['query_indices'])
  for m in z.files:
   if m.endswith('indices'):continue
   rows.append(dict(cohort=ds,shot=k,seed=s,method=m,**metric(y,z[m])))
  for m in ['LP','SASM-Ens']:
   pr,_=hist('ECGFounder',ds,k,s,m);rows.append(dict(cohort=ds,shot=k,seed=s,method='historical-'+m,**metric(y,pr)))
 pd.DataFrame(rows).to_csv(E/'hg_episodes.csv',index=False);dump(E/'HG_COMPLETE.json',dict(source_heads=20,target_episodes=len(jobs),sha256=sha(E/'hg_episodes.csv')))
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback;dump(E/'HG_ERROR.json',dict(traceback=traceback.format_exc()));raise
