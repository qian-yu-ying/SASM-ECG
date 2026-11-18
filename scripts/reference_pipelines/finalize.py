"""Independent scoring and artifact verification, never model selection."""
from core import *
import pandas as pd,traceback
from scipy.stats import wilcoxon

def holm(values):
 p=np.asarray(values);order=np.argsort(p);out=np.empty_like(p);out[order]=np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]).clip(0,1);return out.tolist()
def verify_linear(j,z,m):
 from cpu_worker import context,native_anchor
 from target_contract import objective
 ds=j['dataset'];F,heads,native=context(ds);ep=read(ep_path(j));si=np.array(ep['support']);qi=np.array(ep['query']);ys=np.array(ep['support_y'],float)
 p=outputs(j);wz=np.load(p.with_suffix('.weights.npz'));mu=wz['scaler_mean'];sd=wz['scaler_sd'];weights=wz['weight'];bias=wz['bias'];fq=F[qi].astype(float);fs=F[si].astype(float)
 if j['method']=='Concat-LP':
  raw=np.mean([x[0] for x in heads],axis=0);off=np.mean([x[1] for x in heads],axis=0)
  fq=np.concatenate([fq,fq@raw+off],axis=1);fs=np.concatenate([fs,fs@raw+off],axis=1)
 assert np.allclose(mu,fs.mean(0),rtol=0,atol=1e-12) and np.allclose(sd,fs.std(0)+1e-6,rtol=0,atol=1e-12)
 pred=((fq-mu)/sd)@weights.mean(0)+bias.mean(0);error=float(abs(pred-z['logits']).max());assert error<1e-9,error
 if j['method']=='Native-anchor':
  assert len(weights)==len(bias)==1
  const=np.flatnonzero((ys.sum(0)==0)|(ys.sum(0)==len(ys)))
  assert np.all(weights[0][:,const]==0) and np.all(bias[0][const]==0)
 xs=(fs-mu)/sd;active=np.flatnonzero((ys.sum(0)>0)&(ys.sum(0)<len(ys)));maxgrad=0.
 for h in range(len(weights)):
  cfg=j['config'];anchor=None
  if j['method']=='SASM-Ens':anchor=sd[:,None]*heads[h][0]
  elif j['method']=='Native-anchor':anchor=native_anchor(ds,sd)
  fun=objective(xs,ys[:,active],cfg['alpha'],None if anchor is None else anchor[:,active],cfg.get('lam',0.))
  _,g=fun(np.r_[weights[h][:,active].ravel(),bias[h][active]]);maxgrad=max(maxgrad,float(abs(g).max()))
 assert maxgrad<=5e-5
 return dict(full_prediction_recomputed=True,max_logit_error=error,max_gradient_recomputed=maxgrad,support_only_scaler_verified=True)
def verify_gpu(j,z,backbone):
 import torch
 from gpu_worker import model_for,Waves,predict,DEVICE
 model=model_for(z['logits'].shape[1],920000+j['seed']*100+99,j['method'],backbone)
 st=torch.load(outputs(j).with_suffix('.pth'),map_location='cpu',weights_only=False)
 inc=model.load_state_dict(st,strict=j['method']=='FullFT')
 if j['method']=='LoRA':assert not inc.unexpected_keys and all('lora_' not in k and not k.startswith('dense.') for k in inc.missing_keys)
 sample=np.arange(min(8,len(z['query_indices'])));pred=predict(model,Waves(j['dataset']),np.load(R/'data'/j['dataset']/'lengths.npy'),z['query_indices'][sample]);err=float(abs(pred-z['logits'][sample]).max());assert err<1e-4,err
 del model;torch.cuda.empty_cache()
 return dict(saved_model_reloaded=True,query_records_recomputed=len(sample),sample_max_logit_error=err,full_query_metrics_recomputed=True)
def main():
 assert (R/'release_01.json').exists();release=read(R/'release_01.json');assert release['protocol_sha256']==sha(R/'protocol.lock.json')
 for name,h in release['code_hashes'].items():assert sha(R/name)==h,f'code changed after lock: {name}'
 for name,h in read(R/'protocol.lock.json')['source_hashes'].items():assert sha(S/'source_heads'/name)==h
 for name,h in read(R/'protocol.lock.json')['asset_hashes'].items():assert sha(S/name)==h
 assert sha(B/'checkpoint/12_lead_ECGFounder.pth')==read(R/'protocol.lock.json')['checkpoint_sha256']
 import torch
 ck=torch.load(B/'checkpoint/12_lead_ECGFounder.pth',map_location='cpu',weights_only=False);ck=ck.get('state_dict',ck);backbone={k:v for k,v in ck.items() if not k.startswith('dense.')};del ck
 verification=[];rows=[];per=[]
 for phase in ['audit','external']:
  for j in job_list(phase):
   assert done(j),key(j);p=outputs(j);m=read(p);ep=read(ep_path(j));assert m['episode_sha256']==sha(ep_path(j))
   assert m['protocol_sha256'] in {sha(R/'protocol.lock.json'),*read(R/'protocol.lock.json').get('accepted_unaffected_result_protocol_hashes',[])},('unrecorded result protocol',j)
   z=np.load(p.with_suffix('.npz'));v=np.load(R/'vault'/j['dataset']/f'{phase}.npz');assert np.array_equal(v['indices'],z['query_indices']) and ep['support']==z['support_indices'].tolist()
   metrics=metric(v['y'],z['logits']);assert metrics['ap'] is not None,('undefined prelocked endpoints',j)
   proof=verify_gpu(j,z,backbone) if j['method'] in ['FullFT','LoRA'] else verify_linear(j,z,m)
   verification.append(dict(job=j,metadata_sha256=sha(p),prediction_sha256=sha(p.with_suffix('.npz')),model_sha256=m['model_sha256'],**proof))
   rows.append(dict(phase=phase,dataset=j['dataset'],method=j['method'],shot=j['shot'],seed=j['seed'],ap=metrics['ap'],auc=metrics['auc'],seconds=m['seconds']))
   labels=read(R/'data'/j['dataset']/'metadata.json')['labels']
   for i,label in enumerate(labels):per.append(dict(phase=phase,dataset=j['dataset'],method=j['method'],shot=j['shot'],seed=j['seed'],label=label,ap=metrics['ap_per_label'][i],auc=metrics['auc_per_label'][i]))
   if len(rows)%20==0:print('VERIFIED',len(rows),960,flush=True)
 df=pd.DataFrame(rows);df.to_csv(R/'episode_metrics.csv',index=False);pd.DataFrame(per).to_csv(R/'per_label_metrics.csv',index=False)
 summary=df.groupby(['phase','dataset','shot','method'])[['ap','auc']].agg(['mean','std','count']);summary.to_csv(R/'summary.csv')
 differences=[];effects=[]
 for (phase,ds,k),g in df.groupby(['phase','dataset','shot']):
  pivot=g.pivot(index='seed',columns='method',values=['ap','auc'])
  for method in METHODS:
   if method=='SASM-Ens':continue
   for met in ['ap','auc']:
    delta=(pivot[(met,'SASM-Ens')]-pivot[(met,method)]).to_numpy();assert len(delta)==20
    rng=np.random.default_rng(20261104);boot=delta[rng.integers(0,20,(10000,20))].mean(1);lo,hi=np.quantile(boot,[.025,.975]);p=1. if np.all(delta==0) else float(wilcoxon(delta,zero_method='wilcox',alternative='two-sided').pvalue)
    family=f'external_{met}_30' if phase=='external' else f'audit_{met}'
    effects.append(dict(phase=phase,dataset=ds,shot=k,baseline=method,metric=met,mean_difference_pp=float(delta.mean()*100),seed_CI_low_pp=float(lo*100),seed_CI_high_pp=float(hi*100),wins=int((delta>0).sum()),p=p,family=family))
    for seed,x in enumerate(delta):differences.append(dict(phase=phase,dataset=ds,shot=k,seed=seed,baseline=method,metric=met,difference_pp=float(x*100)))
 eff=pd.DataFrame(effects);eff['p_holm']=0.
 for family,g in eff.groupby('family'):eff.loc[g.index,'p_holm']=holm(g.p.tolist())
 eff.to_csv(R/'paired_effects_CI.csv',index=False);pd.DataFrame(differences).to_csv(R/'paired_differences.csv',index=False)
 dump(R/'verification_manifest.json',dict(artifacts=verification,source_hashes=read(R/'protocol.lock.json')['source_hashes'],protocol_sha256=sha(R/'protocol.lock.json'),release_sha256=sha(R/'release_01.json'),GPU_model_check_scope='reload and first 8 query records for every GPU final model; metrics recomputed on full predictions',CPU_model_check_scope='all final query logits reconstructed from saved weights and support scalers; gradients recomputed',time=time.time()))
 # Patient bootstrap primary comparisons run on CPU in independent processes.
 import subprocess
 subprocess.run([sys.executable,str(R/'patient_bootstrap.py')],cwd=R,check=True)
 lines=['# lambda不超过10、Native独立选参：全量重跑结果','',f'外部Ningbo/CPSC/SPH720组已完成并核验；Chapman仅开发，不留审计。共{len(rows)}组。','', '所有方法（包括Native）均只由Chapman开发验证集独立选参；SASM/Native的alpha和lambda网格均为0.1、1、10。Native保留历史单锚点单拟合构造，见release_01.json；外部训练未读取查询标签。','', '| 队列 | shot | 方法 | Macro-AP (%) | Macro-AUROC (%) |','|---|---|---|---|---|']
 for (phase,ds,k,method),g in df.groupby(['phase','dataset','shot','method']):lines.append(f'| {ds} ({phase}) | {k} | {method} | {g.ap.mean()*100:.2f} ± {g.ap.std()*100:.2f} | {g.auc.mean()*100:.2f} ± {g.auc.std()*100:.2f} |')
 lines+=['','此次方案在已查看前轮目标结果后修订；属于修订后的回顾性评价，不是此前从未访问的外部验证。Chapman是唯一开发队列。Native本轮独立Chapman选参；旧固定参数及含lambda100的旧结果单独保留，不混入本表。','','源信息：SASM及Concat-LP使用同一PTB-XL源监督；LP/FullFT/LoRA为目标支持监督。Native使用原生输出及源分布统计。','身份限制：SPH具有真实患者键；Chapman/Chapman依赖公开一记录一患者设计；CPSC仅验证记录和相同波形分组，无法独立核验不同记录的患者身份。','ECGFounder预训练患者重叠仍未知；本轮结果不证明预训练零重叠。','公开三库沿用原固定可映射子队列端点，不声称完整临床人群。','支持种子区间和患者区间单独报告，不能解释为机构层面独立重复。']
 (R/'RESULTS.md').write_text('\n'.join(lines),encoding='utf8')
 (R/'Claude_实验结果交接.md').write_text('先读RESULTS.md、release_01.json、paired_effects_CI.csv、patient_bootstrap_CI.csv和verification_manifest.json。新旧实验必须分开。此次为查看过旧轮结果后修订协议的回顾性评估，不能声称CPSC/SPH从未被查看。Chapman是唯一开发队列，不是外部验证。不得使用旧alpha/lambda成绩替代本轮；不得将CPSC记录隔离写成已完成独立患者身份核验；不得声称排除预训练患者重叠。GPU参数仅抽样重载复算8条query，CPU模型全量复算，指标均全量预测重算。所有正负结果如实报告。',encoding='utf8')
 files=['RESULTS.md','summary.csv','episode_metrics.csv','paired_effects_CI.csv','patient_bootstrap_CI.csv','verification_manifest.json','Claude_实验结果交接.md']
 dump(R/'COMPLETE.json',dict(time=time.time(),verified_final_method_episodes=len(rows),external=720,audit=0,files={x:sha(R/x) for x in files}))
 print('COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:dump(R/'errors'/f'finalize_{os.getpid()}.json',dict(error=traceback.format_exc(),time=time.time()));raise
