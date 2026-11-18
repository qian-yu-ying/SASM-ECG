import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import json,time,hashlib,sys,socket
from pathlib import Path
import numpy as np
R=Path(os.environ.get('SASM_RUN_DIR', Path(__file__).resolve().parent))
B=Path(os.environ.get('SASM_ECGFOUNDER_DIR', './external/ECGFounder'))
S=R/'source';sys.path.insert(0,str(S));sys.path.insert(0,str(B))
METHODS=['SASM-Ens','Concat-LP','LP','Native-anchor','FullFT','LoRA']
DATASETS=['chapman','ningbo','cpsc2018','sph']
DEV_DATASETS=['chapman']
EXTERNAL_DATASETS=['ningbo','cpsc2018','sph']
PREVIOUS=Path(os.environ.get('SASM_PREVIOUS_DIR', './external/previous_run'))
GRIDS={
 'SASM-Ens':[dict(alpha=a,lam=l) for a in [.1,1.,10.] for l in [.1,1.,10.]],
 'Native-anchor':[dict(alpha=a,lam=l) for a in [.1,1.,10.] for l in [.1,1.,10.]],
 'LP':[dict(alpha=a) for a in [.01,.1,1.,10.,100.,1000.]],
 'Concat-LP':[dict(alpha=a) for a in [.01,.1,1.,10.,100.,1000.]],
 'FullFT':[dict(lr_backbone=lr,epochs=e) for lr in [1e-5,3e-5,1e-4] for e in [10,20,30,40,50]],
 'LoRA':[dict(lr_backbone=lr,epochs=e) for lr in [3e-5,1e-4,3e-4] for e in [10,20]],
}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 q=p.with_name(p.name+f'.{os.getpid()}.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');q.replace(p)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def state(name,**kw):dump(R/'workers'/f'{name}.json',dict(pid=os.getpid(),time=time.time(),**kw))
def key(j):return f"{j['phase']}__{j['dataset']}__{j['method']}__c{j['candidate']:02d}__k{j['shot']}__s{j['seed']:02d}"
def outputs(j):return R/'results'/j['phase']/j['dataset']/j['method']/f"c{j['candidate']:02d}_shot{j['shot']}_seed{j['seed']:02d}.json"
def done(j):
 p=outputs(j)
 if not p.exists():return False
 m=read(p)
 for file,field in [(p.with_suffix('.npz'),'prediction_sha256'),(p.with_suffix('.weights.npz') if j['method'] not in ['FullFT','LoRA'] else p.with_suffix('.pth'),'model_sha256')]:
  assert file.exists() and sha(file)==m[field],f'corrupt completed artifact {file}'
 return True
def metric(y,s):
 from sklearn.metrics import average_precision_score,roc_auc_score
 assert y.shape==s.shape and np.isfinite(s).all()
 valid=[i for i in range(y.shape[1]) if 0<y[:,i].sum()<len(y)]
 if len(valid)!=y.shape[1]:return dict(ap=None,auc=None,valid_labels=valid,ap_per_label=[None]*y.shape[1],auc_per_label=[None]*y.shape[1])
 ap=[float(average_precision_score(y[:,i],s[:,i])) for i in valid]
 auc=[float(roc_auc_score(y[:,i],s[:,i])) for i in valid]
 return dict(ap=float(np.mean(ap)),auc=float(np.mean(auc)),valid_labels=valid,ap_per_label=ap,auc_per_label=auc)
def ep_path(j):return R/'data'/j['dataset']/'episodes'/f"{j['phase']}_shot{j['shot']}_seed{j['seed']:02d}.json"
def job_list(phase):
 if phase=='audit':return []
 if phase=='dev':return [dict(phase='dev',dataset=ds,method=m,candidate=c,config=cfg,shot=k,seed=s) for ds in DEV_DATASETS for m in METHODS for c,cfg in enumerate(GRIDS[m]) for k in [5,10] for s in range(10)]
 release=read(R/'release_01.json')
 return [dict(phase=phase,dataset=ds,method=m,candidate=release['selected'][m]['candidate'],config=release['selected'][m]['config'],shot=k,seed=s) for ds in (DEV_DATASETS if phase=='audit' else EXTERNAL_DATASETS) for m in METHODS for k in [5,10] for s in range(20)]
