"""Retrospective DCA and strictly separated threshold locking. CPU only."""
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS']:
 os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys
from pathlib import Path
PARENT=Path('/home/wy/ECGFounder/supplement_queue_20261101_remote/supplement_core_20261118')
sys.path.insert(0,str(PARENT))
from common import *
from concurrent.futures import ProcessPoolExecutor,as_completed
from scipy.special import logit
RUN=Path(__file__).resolve().parent
METHODS=['LP','Direct-LP','Direct-LP-Refit-b','SASM-Ens','MeanAnchor']
PTS=[.02,.05,.10,.15,.20]

def notice(stage,**kw):
 v=dict(stage=stage,time=time.time(),pid=os.getpid(),**kw);print(json.dumps(v),flush=True);dump(RUN/'status.json',v)

def basepath(bb,ds,k,s):return PARENT/'base/main'/bb/ds/f'shot{k}_seed{s:02d}.npz'

def recover(job):
 bb,ds,k,s=job;bind([8,9]);path=RUN/'support'/bb/ds/f'shot{k}_seed{s:02d}.npz'
 if path.exists():assert sha(path)==read(path.with_suffix('.json'))['sha256'];return
 si,qi,y,mu,sd,x,xq,raw,off=arrays(bb,ds,k,s);bp=basepath(bb,ds,k,s);parent=np.load(bp);assert sha(bp)==read(bp.with_suffix('.json'))['sha256'];refs={str(bp):sha(bp)}
 assert np.array_equal(parent['support_indices'],si) and np.array_equal(parent['query_indices'],qi)
 assert not np.intersect1d(si,qi).size
 pred={};replay={};ma_reconstructed=False
 for m in ['LP','SASM-Ens','MeanAnchor']:
  if ds=='cpsc2018':
   wp=O/'linear'/bb/'NORM'/f'shot{k}_seed{s:02d}.weights.npz';z=np.load(wp)
   assert np.max(abs(z['scaler_mean']-mu))<1e-10 and np.max(abs(z['scaler_sd']-sd))<1e-10
   if m=='SASM-Ens':w=np.mean([z[f'SASM-head{i:02d}__w'] for i in range(10)],0);b=np.mean([z[f'SASM-head{i:02d}__b'] for i in range(10)],0)
   else:w,b=z[m+'__w'],z[m+'__b']
   refs[str(wp)]=sha(wp)
  elif m!='MeanAnchor':
   _,hp=hist(bb,ds,k,s,m);wp=hp.with_suffix('.weights.npz');z=np.load(wp);w=z['weight'].mean(0);b=z['bias'].mean(0);refs[str(wp)]=sha(wp)
   assert np.max(abs(z['scaler_mean']-mu))<1e-10 and np.max(abs(z['scaler_sd']-sd))<1e-10
  else:
   a=sd[:,None]*raw.mean(0);pb=mu@raw.mean(0)+off.mean(0);w,b,d=fit(x,y,.1,a,10.,None,pb);ma_reconstructed=True
  pred[m]=x@w+b;replay[m]=float(abs(xq@w+b-parent[m]).max());assert replay[m]<1e-7,(bb,ds,k,s,m,replay[m])
 direct=(x*sd+mu)@raw.mean(0)+off.mean(0)
 if ds=='cpsc2018':direct[:,-1]=pred['LP'][:,-1]
 mix=.3 if bb=='ECGFounder' else .45;pred['Direct-LP']=(1-mix)*pred['LP']+mix*direct
 shift=parent['Direct-LP-Refit-b'][0]-parent['Direct-LP'][0]
 assert np.max(abs(parent['Direct-LP-Refit-b']-parent['Direct-LP']-shift))<1e-10
 pred['Direct-LP-Refit-b']=pred['Direct-LP']+shift
 active=(y.sum(0)>0)&(y.sum(0)<len(y));gradient=float(abs((expit(pred['Direct-LP-Refit-b'])-y).mean(0)[active]).max());assert gradient<1e-9
 # All local support thresholds are locked before external query labels are read.
 thresholds={};negative_counts=(y==0).sum(0)
 for m in METHODS:
  thresholds[m]=np.array([np.quantile(pred[m][y[:,c]==0,c],.95,method='higher') if negative_counts[c] else np.inf for c in range(y.shape[1])])
 path.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(path,**pred,**{m+'__threshold':v for m,v in thresholds.items()},support_indices=si,query_indices=qi,support_y=y,negative_counts=negative_counts)
 dump(path.with_suffix('.json'),dict(sha256=sha(path),query_score_replay=replay,refit_gradient=gradient,MeanAnchor_reconstructed=ma_reconstructed,references=refs,no_external_query_labels_read=True))

def select_global():
 qi,y=labels('chapman');out={};devrows=[]
 T=read(PARENT/'temperature.lock.json')
 for bb in ['ECGFounder','MERL']:
  out[bb]={}
  for m in METHODS:
   values=[];weights=[]
   for k in [5,10]:
    for s in range(10):
     z=np.load(basepath(bb,'chapman',k,s));assert np.array_equal(z['query_indices'],qi)
     for c in range(y.shape[1]):
      v=z[m][y[:,c]==0,c];assert len(v);values.append(v);weights.append(np.full(len(v),1/(20*y.shape[1]*len(v))))
   v=np.concatenate(values);wt=np.concatenate(weights);ix=np.argsort(v,kind='stable');q=float(v[ix[np.searchsorted(np.cumsum(wt[ix]),.95,side='left')]])
   dev_spec=float(wt[v<=q].sum()/wt.sum());assert dev_spec>=.95-1e-12
   temp=T['selected'][bb][m]['T'];out[bb][m]=dict(raw_logit_threshold=q,raw_probability_threshold=float(expit(q)),temperature=temp,temperature_logit_threshold=q/temp,temperature_probability_threshold=float(expit(q/temp)),development_macro_specificity=dev_spec)
 dump(RUN/'global_thresholds.lock.json',dict(selected=out,source='Chapman development only, 20 equally weighted episodes and 16 equally weighted negative class CDFs',parent_temperature_sha256=sha(PARENT/'temperature.lock.json'),protocol_sha256=sha(RUN/'protocol.lock.json'),time=time.time()))

def confusion(y,decision):
 tp=(decision*y).sum(0);fp=(decision*(1-y)).sum(0);pos=y.sum(0);neg=len(y)-pos;tn=neg-fp;fn=pos-tp
 return dict(TP=tp,FP=fp,TN=tn,FN=fn,specificity=tn/neg,sensitivity=tp/pos,PPV=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=tp+fp>0),f1=np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp),where=2*tp+fp+fn>0))

def score(job):
 bb,ds,k,s=job;bind([8,9]);out=RUN/'episodes'/bb/ds/f'shot{k}_seed{s:02d}.json'
 if out.exists():return
 # Stage barrier: both global and ALL support thresholds are complete and immutable.
 assert (RUN/'ALL_THRESHOLDS_LOCKED.json').exists()
 qi,y=labels(ds);z=np.load(basepath(bb,ds,k,s));u=np.load(RUN/'support'/bb/ds/f'shot{k}_seed{s:02d}.npz');assert np.array_equal(qi,z['query_indices']);assert np.array_equal(qi,u['query_indices'])
 assert ((y.sum(0)>0)&(y.sum(0)<len(y))).all()
 cfg=read(RUN/'global_thresholds.lock.json')['selected'][bb];ts=read(PARENT/'temperature.lock.json')['selected'][bb]
 names=read(B/'data'/ds/'metadata.json')['labels'] if ds!='cpsc2018' else np.load(O/'labels.npz')['names'].astype(str).tolist()
 counts=dict(n=len(y),classes=y.shape[1]);prev=y.mean(0);dr=[];de=[];lr=[];le=[];max_decision_diff=0
 for m in METHODS:
  for mode,T in [('Raw',1.),('DevTemperature',ts[m]['T'])]:
   scores=z[m]/T
   for pt in PTS:
    cc=confusion(y,scores>=logit(pt));nb=cc['TP']/len(y)-cc['FP']/len(y)*pt/(1-pt);sn=nb/prev;allnb=prev-(1-prev)*pt/(1-pt)
    common=dict(backbone=bb,cohort=ds,shot=k,seed=s,method=m,calibration=mode,pt=pt)
    for c,name in enumerate(names):dr.append(dict(**common,label=c,label_name=name,n=len(y),positives=int(y[:,c].sum()),prevalence=prev[c],TP=int(cc['TP'][c]),FP=int(cc['FP'][c]),NB=nb[c],standardized_NB=sn[c],treat_all_NB=allnb[c],treat_all_standardized_NB=allnb[c]/prev[c],treat_none_NB=0.))
    de.append(dict(**common,**counts,NB=float(nb.mean()),standardized_NB=float(sn.mean()),treat_all_NB=float(allnb.mean()),treat_all_standardized_NB=float((allnb/prev).mean()),treat_none_NB=0.,NB_minus_treat_all=float((nb-allnb).mean())))
   for rule in ['A_ChapmanGlobal','B_SupportNegative']:
    rawcut=np.full(y.shape[1],cfg[m]['raw_logit_threshold']) if rule.startswith('A') else u[m+'__threshold']
    decision=scores>rawcut/T;rawdecision=z[m]>rawcut;assert np.array_equal(decision,rawdecision)
    cc=confusion(y,decision);common=dict(backbone=bb,cohort=ds,shot=k,seed=s,method=m,calibration=mode,rule=rule)
    for c,name in enumerate(names):lr.append(dict(**common,label=c,label_name=name,n=len(y),positives=int(y[:,c].sum()),prevalence=prev[c],support_negative_n=int(u['negative_counts'][c]),no_support_negative=bool(u['negative_counts'][c]==0),threshold_logit=rawcut[c]/T,threshold_probability=float(expit(rawcut[c]/T)),**{key:float(val[c]) for key,val in cc.items()}))
    le.append(dict(**common,**counts,**{key:float(cc[key].mean()) for key in ['specificity','sensitivity','PPV','f1']},zero_predicted_positive_classes=int(((cc['TP']+cc['FP'])==0).sum()),no_support_negative_classes=int((u['negative_counts']==0).sum()),min_support_negative=int(u['negative_counts'].min())))
 out.parent.mkdir(parents=True,exist_ok=True)
 for suffix,rows in [('dca_class',dr),('dca_episode',de),('threshold_class',lr),('threshold_episode',le)]:pd.DataFrame(rows).to_csv(out.with_suffix('.'+suffix+'.csv'),index=False)
 dump(out,dict(query_labels_first_used_after_threshold_lock=True,query_npz_sha256=sha(basepath(bb,ds,k,s)),support_npz_sha256=sha(RUN/'support'/bb/ds/f'shot{k}_seed{s:02d}.npz'),global_threshold_sha256=sha(RUN/'global_thresholds.lock.json'),temperature_threshold_decision_difference=max_decision_diff))

def witness():
 y=np.array([[1,0],[0,1],[1,0],[0,0]],float);d=np.array([[1,1],[1,1],[0,0],[0,0]],bool);c=confusion(y,d);assert np.array_equal(c['TP'],[1,1]) and np.array_equal(c['FP'],[1,1])
 nb=c['TP']/4-c['FP']/4*.1/.9;assert np.max(abs(nb-np.array([2/9,2/9])))<1e-12
 neg=np.array([1.,2.,2.,3.]);cut=np.quantile(neg,.95,method='higher');assert np.mean(neg>cut)<=.05
 dump(RUN/'WITNESS.json',dict(manual_confusion_and_net_benefit=True,tie_conservative_threshold=True))

def pool(fn,jobs,stage):
 with ProcessPoolExecutor(2) as ex:
  for i,f in enumerate(as_completed([ex.submit(fn,j) for j in jobs])):
   f.result()
   if (i+1)%20==0 or i+1==len(jobs):notice(stage,done=i+1,total=len(jobs))

def main():
 witness();dev=[(bb,'chapman',k,s) for bb in ['ECGFounder','MERL'] for k in [5,10] for s in range(10)]
 ext=[(bb,ds,k,s) for bb in ['ECGFounder','MERL'] for ds in ['ningbo','cpsc2018','sph'] for k in [5,10] for s in range(20)]
 select_global();notice('GLOBAL_THRESHOLDS_LOCKED');pool(recover,dev+ext,'SUPPORT_RECOVERY')
 files=sorted((RUN/'support').glob('*/*/shot*.npz'));assert len(files)==280
 refs={};maxerr=0.;reconstructed=0
 for p in files:
  v=read(p.with_suffix('.json'));assert sha(p)==v['sha256'];refs.update(v['references']);maxerr=max(maxerr,max(v['query_score_replay'].values()));reconstructed+=int(v['MeanAnchor_reconstructed'])
 refs[str(PARENT/'temperature.lock.json')]=sha(PARENT/'temperature.lock.json')
 dump(RUN/'ALL_THRESHOLDS_LOCKED.json',dict(time=time.time(),support_files={str(p.relative_to(RUN)):sha(p) for p in files},global_lock_sha256=sha(RUN/'global_thresholds.lock.json'),no_external_query_labels_read_by_recovery_or_selection=True,max_query_coefficient_replay_error=maxerr,MeanAnchor_identical_fit_reconstructions=reconstructed,references=refs))
 notice('ALL_THRESHOLDS_LOCKED',replay_max_error=maxerr)
 pool(score,ext,'EXTERNAL_SCORE')
 for suffix,name in [('dca_class','dca_classes.csv'),('dca_episode','dca_episodes.csv'),('threshold_class','locked_threshold_classes.csv'),('threshold_episode','locked_threshold_episodes.csv')]:
  files=sorted((RUN/'episodes').glob('*/*/shot*.'+suffix+'.csv'));assert len(files)==240;pd.concat([pd.read_csv(p) for p in files],ignore_index=True).to_csv(RUN/name,index=False)
 dump(RUN/'SCORING_COMPLETE.json',dict(episodes=240,time=time.time(),protocol_sha256=sha(RUN/'protocol.lock.json')));notice('SCORING_COMPLETE')
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback;dump(RUN/'ERROR.json',dict(traceback=traceback.format_exc()));raise
