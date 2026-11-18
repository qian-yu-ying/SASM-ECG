"""Frozen MERL pipeline, new locked Ningbo-only development and paired episodes."""
from core import *
import traceback,fcntl,importlib.util
import pandas as pd
OLD=R;BASE=R.parent
MR=R/'merl';CHECKPOINT=BASE/'source/merl_checkpoint.pth';ARCH=BASE/'source/merl_arch.py'
MM=['SASM-Ens','Concat-LP','LP']
_verified={}
def mp(j):return MR/'results'/j['phase']/j['dataset']/j['method']/f"c{j['candidate']:02d}_shot{j['shot']}_seed{j['seed']:02d}.json"
def mdone(j):
 p=mp(j)
 if not p.exists():return False
 signature=tuple((q.stat().st_size,q.stat().st_mtime_ns) for q in [p,p.with_suffix('.npz'),p.with_suffix('.weights.npz')])
 if _verified.get(str(p))==signature:return True
 m=read(p)
 for suffix,field in [('.npz','prediction_sha256'),('.weights.npz','model_sha256')]:assert sha(p.with_suffix(suffix))==m[field]
 _verified[str(p)]=signature
 return True
def mjobs():
 for method in MM:
  for c,cfg in enumerate(GRIDS[method]):
   for k in [5,10]:
    for seed in range(10):yield dict(phase='dev',dataset='chapman',method=method,candidate=c,config=cfg,shot=k,seed=seed)
 if (MR/'release.json').exists():
  rel=read(MR/'release.json')
  for phase,dss in [('external',EXTERNAL_DATASETS)]:
   for ds in dss:
    for method in MM:
     sel=rel['selected'][method]
     for k in [5,10]:
      for seed in range(20):yield dict(phase=phase,dataset=ds,method=method,candidate=sel['candidate'],config=sel['config'],shot=k,seed=seed)

def norm(x):
 assert x.ndim==2 and x.shape[0]==12 and np.isfinite(x).all()
 x=x[:,:5000]
 if x.shape[1]<5000:x=np.pad(x,((0,0),(0,5000-x.shape[1])))
 x=(x-x.min())/(x.max()-x.min()+1e-8);x[[4,5]]=x[[5,4]]
 return x.astype(np.float32)
def wave(row):
 p=Path(row['waveform'])
 if p.suffix=='.h5':
  import h5py
  with h5py.File(p) as f:x=np.asarray(f['ecg'])
  return norm(x)
 import wfdb
 x,m=wfdb.rdsamp(str(p.with_suffix('')));assert float(m['fs'])==500.
 assert [n.lower() for n in m['sig_name']]==['i','ii','iii','avr','avl','avf','v1','v2','v3','v4','v5','v6']
 return norm(x.T)

def extract():
 import torch,ast
 from concurrent.futures import ThreadPoolExecutor
 lock=(MR/'extract.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.manual_seed(0);np.random.seed(0);torch.set_num_threads(2)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 sp=importlib.util.spec_from_file_location('merl_architecture',ARCH);arch=importlib.util.module_from_spec(sp);sp.loader.exec_module(arch)
 ck=torch.load(CHECKPOINT,map_location='cpu',weights_only=True);model=arch.ResNet18(num_classes=ck['linear.weight'].shape[0]);model.load_state_dict(ck,strict=True);model.linear=torch.nn.Identity();model.cuda().eval()
 for p in model.parameters():p.requires_grad_(False)
 # Execute only the original standalone reader function to witness adapter equivalence.
 tree=ast.parse((BASE/'merl_source/run.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='read_wave');ns={'np':np,'sha':sha};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<original_merl_read_wave>','exec'),ns)
 for ds in DATASETS:
  dest=MR/'data'/ds;dest.mkdir(parents=True,exist_ok=True)
  if (dest/'FEATURES_READY.json').exists():continue
  manifest=R/'data'/ds/'manifest.csv';tab=pd.read_csv(manifest);assert tab['index'].tolist()==list(range(len(tab)))
  p=dest/'features.npy'
  if ds=='ningbo':
   src=BASE/'merl_ningbo';oldrecords=read(src/'records.json');idx={v:i for i,v in enumerate(oldrecords)};assert len(idx)==len(oldrecords)
   meta=read(src/'features.complete.json');assert sha(src/'features.npy')==meta['sha256']
   a=np.load(src/'features.npy',mmap_mode='r');order=[idx[x] for x in tab.record_id];assert len(set(order))==len(order)
   np.save(p,np.asarray(a[order]));reuse=dict(source=str(src/'features.npy'),source_sha256=meta['sha256'],ordered_record_mapping_verified=True)
  else:
   reuse=None;arr=np.lib.format.open_memmap(p.with_suffix('.tmp.npy'),mode='w+',dtype=np.float32,shape=(len(tab),512));items=tab.to_dict('records')
   with torch.inference_mode(),ThreadPoolExecutor(max_workers=2) as pool:
    first=wave(items[0])
    if ds!='sph':
     original=ns['read_wave']((Path(items[0]['waveform']),items[0]['record_id']))[0];assert np.array_equal(first,original)
    t=torch.from_numpy(first[None]).cuda();a=model(t);b=model(t);assert torch.equal(a,b) and torch.isfinite(a).all()
    dump(dest/'kernel_witness.json',dict(shape=list(a.shape),repeat_max_diff=float((a-b).abs().max()),feature_norm=float(a.norm()),first_raw_file_sha256=sha(items[0]['waveform']),checkpoint_sha256=sha(CHECKPOINT),adapter_original_equal=ds!='sph',sph_published_standard_lead_order=ds=='sph'))
    for start in range(0,len(items),32):
     batch=list(pool.map(wave,items[start:start+32]));values=model(torch.from_numpy(np.stack(batch)).cuda()).cpu().numpy();assert np.isfinite(values).all();arr[start:start+len(batch)]=values
     if start%512==0:arr.flush();dump(MR/'extraction_progress.json',dict(pid=os.getpid(),dataset=ds,completed=start+len(batch),total=len(tab),time=time.time()))
   arr.flush();assert np.isfinite(arr).all();del arr;p.with_suffix('.tmp.npy').replace(p)
  # Recompute cached/new feature rows from raw signals and current checkpoint.
  F=np.load(p,mmap_mode='r');sample=np.unique(np.linspace(0,len(tab)-1,8,dtype=int));errors=[]
  with torch.inference_mode():
   for i in sample:
    v=model(torch.from_numpy(wave(tab.iloc[int(i)].to_dict())[None]).cuda()).cpu().numpy()[0];errors.append(float(abs(v-F[i]).max()))
  assert max(errors)<1e-4,(ds,errors)
  dump(dest/'FEATURES_READY.json',dict(sha256=sha(p),manifest_sha256=sha(manifest),checkpoint_sha256=sha(CHECKPOINT),architecture_sha256=sha(ARCH),records=len(tab),shape=list(F.shape),reuse=reuse,raw_recompute_rows=sample.tolist(),raw_recompute_errors=errors,preprocess='first10s pad global minmax avl/avf swap',protocol_sha256=sha(MR/'protocol.lock.json')))
 dump(MR/'EXTRACTION_COMPLETE.json',dict(time=time.time(),datasets=DATASETS))

def context(ds):
 meta=read(R/'data'/ds/'metadata.json');cfg=read(BASE/'merl_source/protocol.lock.json');inv={v:i for i,v in enumerate(cfg['source_label_keep75'])};cols=[inv[c] for c in meta['cols']];heads=[]
 for e in range(10):
  p=BASE/f'merl_source/source_heads/ep{e:02d}.npz';m=read(p.with_suffix('.json'));assert sha(p)==m['model_sha256'];z=np.load(p);w=z['weight'][:,cols];mu=z['scaler_mean'];sd=z['scaler_sd'];assert w.shape[0]==512 and (sd>0).all()
  raw=w/sd[:,None];off=z['bias'][cols]-(mu/sd)@w;heads.append((raw,off))
 return np.load(MR/'data'/ds/'features.npy',mmap_mode='r'),heads,None

def score(j):
 p=mp(j);m=read(p);v=np.load(R/'data/chapman/development_labels.npz');z=np.load(p.with_suffix('.npz'));assert np.array_equal(v['indices'],z['query_indices']);sc=metric(v['y'],z['logits']);dump(p.with_suffix('.score.json'),dict(**sc,job=j,prediction_sha256=m['prediction_sha256']))

def worker(name):
 import cpu_worker as C
 C.outputs=mp;C.score_dev=score
 original_sha=C.sha
 C.sha=lambda p:original_sha(MR/'protocol.lock.json') if Path(p)==R/'protocol.lock.json' else original_sha(p)
 lock=(MR/f'{name}.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ctx={}
 while not (MR/'COMPLETE.json').exists():
  did=False
  for j in mjobs():
   if not (MR/'data'/j['dataset']/'FEATURES_READY.json').exists() or mdone(j):continue
   claim=R/'claims'/('merl_'+key(j))
   try:claim.mkdir()
   except FileExistsError:continue
   dump(claim/'owner.json',dict(pid=os.getpid(),worker=name,job=j,time=time.time()));ds=j['dataset']
   if ds not in ctx:ctx[ds]=context(ds)
   state(name,stage='merl_target_fit',job=j);C.run(j,name,ctx[ds]);did=True;break
  if not did:state(name,stage='waiting');time.sleep(15)

def select():
 if (MR/'release.json').exists():return
 js=list(mjobs())
 if not all(mdone(j) and mp(j).with_suffix('.score.json').exists() for j in js):return
 rows=[];selected={}
 for method in MM:
  candidates=[]
  for c,cfg in enumerate(GRIDS[method]):
   jobs=[j for j in js if j['method']==method and j['candidate']==c];vals=[]
   for j in jobs:
    sc=read(mp(j).with_suffix('.score.json'));assert sc['job']==j and sc['prediction_sha256']==sha(mp(j).with_suffix('.npz'));vals.append(sc['ap'])
   assert len(vals)==20
   row=dict(method=method,candidate=c,config=cfg,ap=float(np.mean(vals)));rows.append(row);candidates.append(row)
  best=max(x['ap'] for x in candidates);near=[x for x in candidates if best-x['ap']<=1e-4]
  selected[method]=min(near,key=lambda x:(x['config'].get('lam',0),x['config']['alpha']))
 dump(MR/'development_candidates.json',rows);dump(MR/'release.json',dict(selected=selected,protocol_sha256=sha(MR/'protocol.lock.json'),time=time.time(),query_used_for_selection=False))

def finalize():
 from target_contract import objective
 from supplement_finalize import save_tables
 rows=[];checks=[];contexts={};per=[]
 protocol=read(MR/'protocol.lock.json');assert read(MR/'release.json')['protocol_sha256']==sha(MR/'protocol.lock.json')
 for n,h in protocol['code_hashes'].items():assert sha(R/n)==h
 for asset in [protocol['checkpoint'],protocol['architecture'],protocol['source_protocol'],*protocol['source_heads']]:assert sha(asset['path'])==asset['sha256']
 for j in mjobs():
  if j['phase']=='dev':continue
  assert mdone(j);ds=j['dataset'];p=mp(j);m=read(p);ep=read(ep_path(j));assert sha(ep_path(j))==m['episode_sha256'] and j==m['job']
  z=np.load(p.with_suffix('.npz'));wz=np.load(p.with_suffix('.weights.npz'));si=z['support_indices'];qi=z['query_indices'];assert si.tolist()==ep['support'] and qi.tolist()==ep['query']
  if ds not in contexts:contexts[ds]=context(ds)
  F,heads,_=contexts[ds];fs=F[si].astype(float);fq=F[qi].astype(float);mu=wz['scaler_mean'];sd=wz['scaler_sd'];ys=np.asarray(ep['support_y'],float);weights=wz['weight'];bias=wz['bias']
  if j['method']=='Concat-LP':
   raw=np.mean([h[0] for h in heads],axis=0);off=np.mean([h[1] for h in heads],axis=0);fs=np.concatenate([fs,fs@raw+off],axis=1);fq=np.concatenate([fq,fq@raw+off],axis=1)
  assert np.allclose(mu,fs.mean(0),rtol=0,atol=1e-12) and np.allclose(sd,fs.std(0)+1e-6,rtol=0,atol=1e-12)
  xs=(fs-mu)/sd;pred=(fq-mu)/sd@weights.mean(0)+bias.mean(0);err=float(abs(pred-z['logits']).max());assert err<1e-9
  active=np.flatnonzero((ys.sum(0)>0)&(ys.sum(0)<len(ys)));grad=0.
  for h in range(len(weights)):
   a=sd[:,None]*heads[h][0] if j['method']=='SASM-Ens' else None;cfg=j['config'];fun=objective(xs,ys[:,active],cfg['alpha'],None if a is None else a[:,active],cfg.get('lam',0.));_,g=fun(np.r_[weights[h][:,active].ravel(),bias[h][active]]);grad=max(grad,float(abs(g).max()))
  assert grad<=5e-5
  v=np.load(OLD/'vault'/ds/f"{j['phase']}.npz");assert np.array_equal(qi,v['indices']);sc=metric(v['y'],z['logits']);assert sc['ap'] is not None
  rows.append(dict(backbone='MERL',dataset=ds,shot=j['shot'],seed=j['seed'],method=j['method'],ap=sc['ap'],auc=sc['auc'],result_path=str(p),result_sha256=sha(p)))
  for i,label in enumerate(read(R/'data'/ds/'metadata.json')['labels']):per.append(dict(dataset=ds,shot=j['shot'],seed=j['seed'],method=j['method'],label=label,ap=sc['ap_per_label'][i],auc=sc['auc_per_label'][i]))
  checks.append(dict(job=j,max_prediction_error=err,max_gradient=grad,prediction_sha256=m['prediction_sha256'],model_sha256=m['model_sha256']))
 pd.DataFrame(per).to_csv(R/'merl_per_label_metrics.csv',index=False)
 save_tables(rows,'merl');dump(MR/'verification.json',dict(checks=checks,release_sha256=sha(MR/'release.json')));dump(MR/'COMPLETE.json',dict(time=time.time(),verified_final_episodes=len(checks),summary_sha256=sha(R/'merl_summary.csv'),verification_sha256=sha(MR/'verification.json')))

def supervise():
 lock=(MR/'supervisor.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while True:
  select();js=list(mjobs());count=sum(mdone(j) for j in js)
  dump(MR/'status.json',dict(pid=os.getpid(),time=time.time(),completed=count,total=len(js),release=(MR/'release.json').exists()))
  if (MR/'release.json').exists() and count==len(js):finalize();return
  time.sleep(25)

if __name__=='__main__':
 try:
  if sys.argv[1]=='extract':extract()
  elif sys.argv[1]=='supervise':supervise()
  else:worker(sys.argv[1])
 except Exception:
  dump(R/'errors'/f'merl_{sys.argv[1]}_{os.getpid()}.json',dict(pid=os.getpid(),time=time.time(),error=traceback.format_exc()));raise
