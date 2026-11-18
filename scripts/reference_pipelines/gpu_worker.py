from core import *
import importlib.util,traceback,fcntl
import torch
from torch import nn
sp=importlib.util.spec_from_file_location('FT',S/'locked_ft_runner.py');FT=importlib.util.module_from_spec(sp);sp.loader.exec_module(FT)
FT.ROOT=B
torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
DEVICE=torch.device('cuda:0')
CHECKPOINT=B/'checkpoint/12_lead_ECGFounder.pth'

class Waves:
 def __init__(self,ds):
  self.ds=ds;self.meta=read(R/'data'/ds/'metadata.json');self.kind=self.meta['wave']['kind'];self.cached={}
  if self.kind=='npy':self.array=np.load(self.meta['wave']['path'],mmap_mode='r')
 def __getitem__(self,key):
  i,sel=key;i=int(i)
  if self.kind=='npy':return self.array[i,sel]
  if i in self.cached:return self.cached[i][sel]
  import h5py
  with h5py.File(self.meta['wave']['paths'][i],'r') as f:a=np.asarray(f['ecg'],np.float32)
  chunks=[]
  for k in range(0,a.shape[1],5000):
   x=a[:,k:k+5000];x=np.pad(x,((0,0),(0,max(0,5000-x.shape[1]))));x=(x-x.mean())/(x.std()+1e-6);chunks.append(x)
  return np.asarray(chunks,np.float32)[sel]

class LoRAConv(nn.Module):
 def __init__(self,base):
  super().__init__();self.base=base
  self.lora_A=nn.Conv1d(base.in_channels,8*base.groups,1,groups=base.groups,bias=False)
  self.lora_B=nn.Conv1d(8*base.groups,base.out_channels,base.kernel_size,stride=base.stride,groups=base.groups,bias=False)
  nn.init.kaiming_uniform_(self.lora_A.weight,a=np.sqrt(5));nn.init.zeros_(self.lora_B.weight)
 def forward(self,x):return self.base(x)+2*self.lora_B(self.lora_A(x))
def inject(model,prefix=''):
 names=[]
 for n,ch in list(model.named_children()):
  full=prefix+'.'+n if prefix else n
  if isinstance(ch,nn.Conv1d) and ('conv1' in prefix or 'conv3' in prefix):
   assert ch.padding==(0,) and ch.dilation==(1,)
   setattr(model,n,LoRAConv(ch));names.append(full)
  else:names+=inject(ch,full)
 return names
def model_for(n,seed,method,backbone):
 m=FT.make_ft_model(backbone,n,seed,DEVICE)
 if method=='LoRA':
  for p in m.parameters():p.requires_grad=False
  names=inject(m)
  assert len(names)==40,('unexpected LoRA injection inventory',len(names))
  for name,p in m.named_parameters():
   if 'lora_' in name or name.startswith('dense.'):p.requires_grad=True
  m.to(DEVICE)
 return m
def predict(model,waves,lengths,indices):
 model.eval();parts=[]
 for st in range(0,len(indices),32):
  arr,own,cnt=FT.flatten_windows(waves,lengths,indices[st:st+32]);parts.append(FT.mean_pool_logits(model,arr,own,cnt,16,DEVICE).float().cpu().numpy())
 return np.concatenate(parts)
def feature_job(ds,worker,backbone):
 d=R/'data'/ds;p=d/'FEATURES_READY.json'
 if p.exists():return False
 claim=d/'features.claim'
 try:claim.mkdir()
 except FileExistsError:return False
 dump(claim/'owner.json',dict(pid=os.getpid(),worker=worker,host=socket.gethostname()))
 t=time.time();waves=Waves(ds);lengths=np.load(d/'lengths.npy');meta=read(d/'metadata.json');n=meta['records']
 model=FT.make_ft_model(backbone,150,730000,DEVICE,return_features=True)
 # Exact checkpoint backbone loaded; dense irrelevant for feature extraction.
 model.eval();F=np.lib.format.open_memmap(d/'features.npy.tmp',mode='w+',dtype=np.float32,shape=(n,1024))
 for st in range(0,n,64):
  ix=np.arange(st,min(st+64,n));F[ix]=FT.extract_pooled_features(model,waves,lengths,ix,DEVICE)
  if st%1024==0:state(worker,stage='extract_features',dataset=ds,completed=st,total=n)
 assert np.isfinite(F).all();F.flush();del F;Path(d/'features.npy.tmp').replace(d/'features.npy')
 del model;torch.cuda.empty_cache()
 dump(p,dict(sha256=sha(d/'features.npy'),seconds=time.time()-t,checkpoint_sha256=read(R/'protocol.lock.json')['checkpoint_sha256'],metadata_sha256=sha(d/'metadata.json'),producer='live exact locked fine-tuning waveform pipeline'))
 print('FEATURES_READY',ds,round(time.time()-t,1),flush=True);return True
def train_job(j,worker,backbone,cache):
 t=time.time();ep=read(ep_path(j));ds=j['dataset'];meta=read(R/'data'/ds/'metadata.json')
 waves,lengths=cache.setdefault(ds,(Waves(ds),np.load(R/'data'/ds/'lengths.npy')))
 si=np.array(ep['support']);qi=np.array(ep['query']);ys=np.asarray(ep['support_y'],np.float32)
 assert not set(si)&set(qi)
 seed=920000+j['seed']*100+99;model=model_for(len(meta['labels']),seed,j['method'],backbone)
 train_names=[n for n,p in model.named_parameters() if p.requires_grad]
 if j['method']=='LoRA':
  assert all('lora_' in n or n.startswith('dense.') for n in train_names)
  frozen=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters() if not p.requires_grad)).hexdigest()
  adapter_before={n:p.detach().cpu().clone() for n,p in model.named_parameters() if 'lora_B' in n}
 state(worker,stage='training',job=j,start=t)
 cfg=j['config'];out=FT._run_epochs(model,waves,lengths,si,ys,DEVICE,epochs=cfg['epochs'],batch_size=16,lr_backbone=cfg['lr_backbone'],lr_head=1e-3,weight_decay=1e-4,seed=seed)
 assert out['steps']>0 and np.isfinite(out['last_loss'])
 if j['method']=='LoRA':
  assert frozen==hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters() if not p.requires_grad)).hexdigest()
  delta=max(float((p.detach().cpu()-adapter_before[n]).abs().max()) for n,p in model.named_parameters() if n in adapter_before);assert delta>0
 else:delta=None
 state(worker,stage='query_inference',job=j)
 pred=predict(model,waves,lengths,qi);assert np.isfinite(pred).all()
 p=outputs(j);p.parent.mkdir(parents=True,exist_ok=True);mp=p.with_suffix('.pth');torch.save({n:v.detach().cpu() for n,v in model.state_dict().items() if j['method']=='FullFT' or 'lora_' in n or n.startswith('dense.')},mp)
 z=p.with_suffix('.npz');np.savez_compressed(z,logits=pred,query_indices=qi,support_indices=si)
 # Query labels are never loaded by this worker. Selection/scoring lives elsewhere.
 dump(p,dict(job=j,method=j['method'],seconds=time.time()-t,training=out,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),adapter_max_change=delta,frozen_backbone_unchanged=True if j['method']=='LoRA' else None,model_sha256=sha(mp),prediction_sha256=sha(z),episode_sha256=sha(ep_path(j)),protocol_sha256=sha(R/'protocol.lock.json'),feature_source='same waveforms as linear models',torch=torch.__version__,gpu=torch.cuda.get_device_name(),worker=worker,query_labels_loaded=False))
 if j['phase']=='dev':score_dev(j)
 del model;torch.cuda.empty_cache();print('TRAIN_DONE',key(j),round(time.time()-t,1),flush=True)
def score_dev(j):
 p=outputs(j);m=read(p);ep=read(ep_path(j));val=np.load(R/'data'/j['dataset']/'development_labels.npz');z=np.load(p.with_suffix('.npz'));assert np.array_equal(z['query_indices'],val['indices'])
 score=metric(val['y'],z['logits']);assert score['ap'] is not None
 dump(p.with_suffix('.score.json'),dict(**score,prediction_sha256=m['prediction_sha256'],job=j))
def main(worker):
 lock=(R/'workers'/f'{worker}.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 ck=torch.load(CHECKPOINT,map_location='cpu',weights_only=False);ck=ck.get('state_dict',ck);backbone={k:v for k,v in ck.items() if not k.startswith('dense.')};del ck
 cache={};state(worker,stage='started',gpu=torch.cuda.get_device_name(),torch=torch.__version__)
 # Real waveform repeat witness: ensured by the replay equality gate in every training step;
 # feature extraction is verified separately against saved model state during final audit.
 while not (R/'COMPLETE.json').exists():
  did=False
  for ds in DATASETS:
   if (R/'data'/ds/'DATA_READY.json').exists():
    if feature_job(ds,worker,backbone):did=True;break
  if did:continue
  phases=['dev'] if not (R/'release_01.json').exists() else ['audit','external']
  for phase in phases:
   for j in job_list(phase):
    if j['method'] not in ['FullFT','LoRA']:continue
    if not (R/'data'/j['dataset']/'FEATURES_READY.json').exists():continue
    if done(j):
     if phase=='dev' and not outputs(j).with_suffix('.score.json').exists():score_dev(j)
     continue
    claim=R/'claims'/key(j)
    try:claim.mkdir()
    except FileExistsError:continue
    dump(claim/'owner.json',dict(pid=os.getpid(),worker=worker,job=j,time=time.time()))
    train_job(j,worker,backbone,cache);did=True;break
   if did:break
  if not did:state(worker,stage='waiting');time.sleep(15)
 state(worker,stage='complete')
if __name__=='__main__':
 try:main(sys.argv[1])
 except Exception:dump(R/'errors'/f'{sys.argv[1]}_{os.getpid()}.json',dict(error=traceback.format_exc(),time=time.time()));raise
