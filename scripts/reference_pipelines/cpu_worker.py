from core import *
import traceback,fcntl,pandas as pd
from target_contract import certified_fit

def context(ds):
 d=R/'data'/ds;meta=read(d/'metadata.json');q=np.load(d/'Q.npy');dense=np.load(R/'assets/dense.npz');dw=dense['Dw'];db=dense['Db'];heads=[];native=[];source_scales=[]
 tab=pd.read_csv(S/'source_labels.csv');tasks=[s.strip().lower() for s in (S/'tasks.txt').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
 keep=meta['source_label_keep'];native_map=np.zeros((150,len(meta['cols'])))
 for j,c in enumerate(meta['cols']):
  label=str(tab.iloc[c].ecgfounder_label).strip().lower()
  if label in tasks:native_map[tasks.index(label),j]=1
 for i in range(10):
  z=np.load(S/f'source_heads/ep{i:02d}_bce_signed_model.npz');w=z['weight'];bias=z['bias'];sd=z['scaler_sd'];mu=z['scaler_mean']
  assert w.shape==(150,71) and np.isfinite(w).all() and float(z['selected_alpha'])==10.
  wt=w@q;bt=bias@q;raw=dw.T@(wt/sd[:,None]);offset=bt+((db-mu)/sd)@wt
  heads.append((raw,offset))
  source_scales.append(sd.astype(float))
 mean_sd=np.mean(source_scales,axis=0)
 raw_native=dw.T@(native_map/mean_sd[:,None])
 native=[(raw_native,np.zeros(native_map.shape[1]))]
 return np.load(d/'features.npy',mmap_mode='r'),heads,native

def native_anchor(ds,target_sd):
 # Preserve the exact arithmetic order of the historical Native implementation.
 meta=read(R/'data'/ds/'metadata.json');dw=np.load(R/'assets/dense.npz')['Dw']
 tab=pd.read_csv(S/'source_labels.csv');tasks=[x.strip().lower() for x in (S/'tasks.txt').read_text(encoding='utf-8-sig').splitlines() if x.strip()]
 mean_sd=np.mean([np.load(S/'source_heads'/f'ep{i:02d}_bce_signed_model.npz')['scaler_sd'].astype(float) for i in range(10)],0)
 a=np.zeros((len(target_sd),len(meta['cols'])))
 for j,c in enumerate(meta['cols']):
  label=str(tab.iloc[c].ecgfounder_label).strip().lower()
  if label in tasks:
   k=tasks.index(label);a[:,j]=target_sd*dw[k]/mean_sd[k]
 return a

def fit_with_fallback(xs,ys,alpha,lam,anchor,bprior):
 w,b,diag=certified_fit(xs,ys,alpha,W_anchor=anchor,lam=lam)
 const=np.flatnonzero((ys.sum(0)==0)|(ys.sum(0)==len(ys)))
 if len(const):
  w[:,const]=anchor[:,const] if anchor is not None else 0.
  b[const]=bprior[const] if bprior is not None else np.log((ys[:,const].sum(0)+.5)/(len(ys)-ys[:,const].sum(0)+.5))
 assert diag['max_gradient']<=5e-5 and np.isfinite(w).all() and np.isfinite(b).all()
 return w,b,diag
def run(j,worker,ctx):
 t=time.time();ep=read(ep_path(j));si=np.array(ep['support']);qi=np.array(ep['query']);ys=np.asarray(ep['support_y'],float);F,heads,native=ctx
 fs=F[si].astype(float);mu=fs.mean(0);sd=fs.std(0)+1e-6;xs=(fs-mu)/sd;xq=(F[qi].astype(float)-mu)/sd
 cfg=j['config'];method=j['method'];diags=[];weights=[];biases=[]
 if method=='SASM-Ens':
  for raw,offset in heads:
   a=sd[:,None]*raw;bp=mu@raw+offset;w,b,diag=fit_with_fallback(xs,ys,cfg['alpha'],cfg['lam'],a,bp);diags.append(diag);weights.append(w);biases.append(b)
  pred=xq@np.mean(weights,axis=0)+np.mean(biases,axis=0)
 elif method=='Native-anchor':
  assert len(native)==1
  a=native_anchor(j['dataset'],sd)
  # Historical Native: one fit, free intercept; constant columns remain w=b=0.
  w,b,diag=certified_fit(xs,ys,cfg['alpha'],W_anchor=a,lam=cfg['lam'])
  diags=[diag];weights=[w];biases=[b];pred=xq@w+b
 elif method in ['LP','Concat-LP']:
  if method=='Concat-LP':
   raw=np.mean([h[0] for h in heads],axis=0);offset=np.mean([h[1] for h in heads],axis=0)
   fss=np.concatenate([fs,fs@raw+offset],axis=1);fqq=np.concatenate([F[qi].astype(float),F[qi].astype(float)@raw+offset],axis=1)
   mu=fss.mean(0);sd=fss.std(0)+1e-6;xs=(fss-mu)/sd;xq=(fqq-mu)/sd
  w,b,diag=fit_with_fallback(xs,ys,cfg['alpha'],0.,None,None);diags.append(diag);weights=[w];biases=[b];pred=xq@w+b
 else:raise ValueError(method)
 assert np.isfinite(pred).all()
 p=outputs(j);p.parent.mkdir(parents=True,exist_ok=True);wp=p.with_suffix('.weights.npz');np.savez_compressed(wp,weight=np.asarray(weights),bias=np.asarray(biases),scaler_mean=mu,scaler_sd=sd)
 zp=p.with_suffix('.npz');np.savez_compressed(zp,logits=pred,query_indices=qi,support_indices=si)
 dump(p,dict(job=j,method=method,seconds=time.time()-t,fits=diags,prediction_sha256=sha(zp),model_sha256=sha(wp),episode_sha256=sha(ep_path(j)),protocol_sha256=sha(R/'protocol.lock.json'),query_labels_loaded=False,worker=worker))
 if j['phase']=='dev':score_dev(j)
 print('LINEAR_DONE',key(j),round(time.time()-t,1),flush=True)
def score_dev(j):
 p=outputs(j);m=read(p);z=np.load(p.with_suffix('.npz'));v=np.load(R/'data'/j['dataset']/'development_labels.npz');assert np.array_equal(z['query_indices'],v['indices'])
 scores=metric(v['y'],z['logits']);assert scores['ap'] is not None
 dump(p.with_suffix('.score.json'),dict(**scores,job=j,prediction_sha256=m['prediction_sha256']))
def main(worker):
 lock=(R/'workers'/f'{worker}.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);contexts={}
 while not (R/'COMPLETE.json').exists():
  did=False;phases=['dev'] if not (R/'release_01.json').exists() else ['audit','external']
  for phase in phases:
   for j in job_list(phase):
    if j['method'] in ['FullFT','LoRA']:continue
    ds=j['dataset']
    if not (R/'data'/ds/'FEATURES_READY.json').exists():continue
    if done(j):
     if phase=='dev' and not outputs(j).with_suffix('.score.json').exists():score_dev(j)
     continue
    claim=R/'claims'/key(j)
    try:claim.mkdir()
    except FileExistsError:continue
    dump(claim/'owner.json',dict(pid=os.getpid(),worker=worker,job=j,time=time.time()))
    state(worker,stage='linear_fit',job=j)
    if ds not in contexts:contexts[ds]=context(ds)
    run(j,worker,contexts[ds]);did=True;break
   if did:break
  if not did:state(worker,stage='waiting');time.sleep(15)
 state(worker,stage='complete')
if __name__=='__main__':
 try:main(sys.argv[1])
 except Exception:dump(R/'errors'/f'{sys.argv[1]}_{os.getpid()}.json',dict(error=traceback.format_exc(),time=time.time()));raise
