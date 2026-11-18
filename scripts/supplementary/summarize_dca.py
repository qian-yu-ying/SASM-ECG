from run_dca import *
from scipy.special import logit

KEY=['backbone','cohort','shot']
def mean_intervals(f,group,metrics,rng):
 rows=[];ix=rng.integers(0,20,(10000,20))
 for keys,g in f.groupby(group,sort=True):
  assert len(g)==20 and g.seed.nunique()==20
  vals=g.sort_values('seed')[metrics].to_numpy();boots=vals[ix].mean(1);lo,hi=np.quantile(boots,[.025,.975],axis=0);rec=dict(zip(group,keys if isinstance(keys,tuple) else [keys]))
  for j,m in enumerate(metrics):rec.update({m+'_mean':float(vals[:,j].mean()),m+'_sd':float(vals[:,j].std(ddof=1)),m+'_CI_low':float(lo[j]),m+'_CI_high':float(hi[j])})
  rec['seed_n']=20;rows.append(rec)
 return pd.DataFrame(rows)

def comparisons(f,group,metrics,rng):
 rows=[];ix=rng.integers(0,20,(10000,20))
 for keys,g in f.groupby(group,sort=True):
  for met in metrics:
   tab=g.pivot(index='seed',columns='method',values=met).sort_index();assert len(tab)==20
   for a in ['MeanAnchor','SASM-Ens']:
    for b in ['LP','Direct-LP']:
     v=(tab[a]-tab[b]).to_numpy();lo,hi=np.quantile(v[ix].mean(1),[.025,.975]);rows.append(dict(zip(group,keys if isinstance(keys,tuple) else [keys]),comparison=a+' minus '+b,metric=met,difference=float(v.mean()),CI_low=float(lo),CI_high=float(hi),seed_n=20,constant_seed_difference=bool(np.ptp(v)<1e-12),CI_unit='paired_support_seed_fixed_query'))
 return pd.DataFrame(rows)

def add_overall(f,metrics,extra):
 # Seed index is resampled jointly over all fixed conditions; conditions are not independent samples.
 over=f.groupby(['seed','method','calibration']+extra,as_index=False)[metrics].mean();over['backbone']='Overall';over['cohort']='Equal12Conditions';over['shot']=0
 return pd.concat([f,over],ignore_index=True)

def main():
 bind([8]);assert (RUN/'SCORING_COMPLETE.json').exists();rng=np.random.default_rng(2026111806)
 d=pd.read_csv(RUN/'dca_episodes.csv');l=pd.read_csv(RUN/'locked_threshold_episodes.csv');assert len(d)==12000 and len(l)==4800
 assert d.groupby(KEY+['seed','method','calibration']).size().eq(5).all()
 assert l.groupby(KEY+['seed','method','calibration']).size().eq(2).all()
 dm=['NB','standardized_NB','treat_all_NB','treat_all_standardized_NB','NB_minus_treat_all']
 lm=['specificity','sensitivity','PPV','f1']
 dd=add_overall(d,dm,['pt']);ll=add_overall(l,lm,['rule'])
 mean_intervals(dd,KEY+['method','calibration','pt'],dm,rng).to_csv(RUN/'dca_summary.csv',index=False)
 comparisons(dd,KEY+['calibration','pt'],['NB','standardized_NB'],rng).to_csv(RUN/'dca_paired_effects.csv',index=False)
 mean_intervals(ll,KEY+['method','calibration','rule'],lm,rng).to_csv(RUN/'locked_threshold_summary.csv',index=False)
 comparisons(ll,KEY+['calibration','rule'],lm,rng).to_csv(RUN/'locked_threshold_paired_effects.csv',index=False)
 for raw,out,extra,metrics in [('dca_classes.csv','dca_class_summary.csv',['pt'],['NB','standardized_NB','prevalence','treat_all_NB','treat_all_standardized_NB']),('locked_threshold_classes.csv','locked_threshold_class_summary.csv',['rule'],['specificity','sensitivity','PPV','f1','support_negative_n','threshold_probability','prevalence'])]:
  f=pd.read_csv(RUN/raw);f.groupby(KEY+['method','calibration']+extra+['label','label_name'])[metrics].agg(['mean','std']).to_csv(RUN/out)
 # All decision metrics exactly invariant under jointly transformed scores and threshold.
 piv=l.pivot(index=KEY+['seed','method','rule'],columns='calibration',values=lm);errs={m:float(abs(piv[m]['Raw']-piv[m]['DevTemperature']).max()) for m in lm};assert max(errs.values())==0
 # Manual independent witness against parent logits and actual query labels, no optimizer reuse.
 bb,ds,k,s='ECGFounder','cpsc2018',5,0;qi,y=labels(ds);z=np.load(basepath(bb,ds,k,s));sc=z['MeanAnchor'];decision=sc>=np.log(.05/.95);tp=np.sum(decision&(y==1),axis=0);fp=np.sum(decision&(y==0),axis=0);nb=tp/len(y)-fp/len(y)*.05/.95
 saved=pd.read_csv(RUN/'dca_classes.csv');sel=saved[(saved.backbone==bb)&(saved.cohort==ds)&(saved.shot==k)&(saved.seed==s)&(saved.method=='MeanAnchor')&(saved.calibration=='Raw')&(saved.pt==.05)].sort_values('label');err=float(abs(sel.NB.to_numpy()-nb).max());assert err<1e-12
 # Class and episode outputs must have identical macro averaging, including rare labels.
 joined=saved.groupby(KEY+['seed','method','calibration','pt'])[['NB','standardized_NB']].mean();orig=d.set_index(KEY+['seed','method','calibration','pt']);macroerr=float(abs(joined-orig[['NB','standardized_NB']]).to_numpy().max());assert macroerr<1e-10
 thresholdlock=read(RUN/'ALL_THRESHOLDS_LOCKED.json');drift=[]
 for path,h in thresholdlock['references'].items():
  if sha(path)!=h:drift.append(path)
 for path,h in thresholdlock['support_files'].items():
  if sha(RUN/path)!=h:drift.append(path)
 assert not drift
 assert sha(RUN/'global_thresholds.lock.json')==thresholdlock['global_lock_sha256']
 allrows=pd.read_csv(RUN/'locked_threshold_classes.csv');neg=allrows[allrows.calibration=='Raw'].support_negative_n
 dump(RUN/'NUMERICAL_CHECKS.json',dict(episodes=240,dca_class_rows=len(saved),dca_episode_rows=len(d),threshold_class_rows=len(allrows),threshold_episode_rows=len(l),threshold_temperature_decision_invariance=errs,independent_DCA_error=err,class_to_macro_error=macroerr,historical_input_drift=drift,support_coefficient_max_replay_error=thresholdlock['max_query_coefficient_replay_error'],MeanAnchor_identical_fit_reconstructions=thresholdlock['MeanAnchor_identical_fit_reconstructions'],min_support_negatives=int(neg.min()),no_support_negative_rows=int(allrows.no_support_negative.sum()),primary_STE_lambda0_unchanged=True,GPU_used=False,clinical_deployment_validated=False,query_bootstrap_performed=False,CI_unit='support-seed conditional on fixed query, not patients',source_label_selection_exposure='prior Chapman development and prior external results already observed; retrospective analysis'))
 dump(RUN/'CODE_MANIFEST.json',dict(files={p.name:sha(p) for p in RUN.glob('*.py')},protocol_sha256=sha(RUN/'protocol.lock.json')))
 files=[p for pat in ['*.csv','*.json','*.py','*.md'] for p in RUN.glob(pat) if p.name not in ['FINAL_MANIFEST.json','ALL_COMPLETE.json']]
 dump(RUN/'FINAL_MANIFEST.json',dict(files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in files},time=time.time()))
 dump(RUN/'ALL_COMPLETE.json',dict(time=time.time(),checks_sha256=sha(RUN/'NUMERICAL_CHECKS.json'),manifest_sha256=sha(RUN/'FINAL_MANIFEST.json')));notice('ALL_COMPLETE')

if __name__=='__main__':
 try:main()
 except Exception:
  import traceback;dump(RUN/'SUMMARY_ERROR.json',dict(traceback=traceback.format_exc()));raise
