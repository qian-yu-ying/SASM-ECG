from common import *
from scipy.stats import wilcoxon
def paired(f,group,contrasts,endpoint='ap'):
 rows=[];rng=np.random.default_rng(2026111803)
 for keys,g in f.groupby(group):
  keys=keys if isinstance(keys,tuple) else (keys,);tab=g.pivot(index='seed',columns='method',values=endpoint)
  for a,b in contrasts:
   if a not in tab or b not in tab:continue
   v=(tab[a]-tab[b]).dropna().to_numpy();assert len(v)==20;idx=rng.integers(0,20,(10000,20));lo,hi=np.quantile(v[idx].mean(1),[.025,.975]);constant=np.ptp(v)<1e-12
   p=None if constant else float(wilcoxon(v,method='auto').pvalue)
   rows.append(dict(zip(group,keys),comparison=a+' minus '+b,metric=endpoint,mean_difference=float(v.mean()),CI_low=float(lo),CI_high=float(hi),seed_sd=float(v.std(ddof=1)),p=p,constant_seed_difference=constant))
 r=pd.DataFrame(rows)
 # One comparison/endpoint family across all shown external conditions, with missing deterministic tests excluded.
 for _,idx in r.groupby(['comparison','metric']).groups.items():
  valid=r.loc[idx].dropna(subset=['p']).sort_values('p');previous=0
  for rank,(i,row) in enumerate(valid.iterrows()):
   v=min(1.,max(previous,(len(valid)-rank)*row.p));r.loc[i,'p_holm']=v;previous=v
 return r
def main():
 assert (E/'MODEL_ASSET_VERIFICATION.json').exists()
 for n in ['HG','CALIBRATION','SWAP','TIMING','SUBGROUP']:
  assert (E/(n+'_COMPLETE.json')).exists(),n
 labels_by_cohort={ds:read(B/'data'/ds/'metadata.json')['labels'] for ds in ['chapman','ningbo','sph']}
 labels_by_cohort['cpsc2018']=np.load(O/'labels.npz')['names'].astype(str).tolist()
 pd.DataFrame([dict(cohort=ds,label=i,label_name=name) for ds,names in labels_by_cohort.items() for i,name in enumerate(names)]).to_csv(E/'label_dictionary.csv',index=False)
 sub=pd.read_csv(E/'subgroup_summary.csv');assert len(sub)==240
 assert sub.groupby(['backbone','cohort','shot','subgroup']).method.nunique().eq(5).all()
 assert np.isfinite(sub[['ap_mean','auc_mean']].to_numpy()).all()
 bc=pd.read_csv(E/'subgroup_CI.csv');assert (bc.valid_replicates<=bc.total_replicates).all()
 assert (bc.CI_low<=bc.CI_high).all()
 tr=pd.read_csv(E/'timing_raw.csv');assert len(tr)==360
 assert tr.groupby(['backbone','cohort','shot','method']).size().eq(10).all()
 assert tr.seconds.gt(0).all()
 assert len(list((E/'source').glob('*/*.npz')))==20
 tables={}
 f=pd.read_csv(E/'hg_episodes.csv');f.groupby(['cohort','shot','method'])[['ap','auc']].agg(['mean','std']).to_csv(E/'hg_summary.csv');paired(f[f.cohort!='chapman'],['cohort','shot'],[('h-SASM-Ens','g-SASM-Ens'),('h-MeanAnchor','g-MeanAnchor'),('h-SASM-Ens','historical-LP'),('g-SASM-Ens','historical-SASM-Ens')]).to_csv(E/'hg_paired_effects.csv',index=False)
 f=pd.read_csv(E/'swap_episodes.csv');f.groupby(['backbone','cohort','shot','method'])[['ap','auc']].agg(['mean','std']).to_csv(E/'swap_summary.csv');paired(f,['backbone','cohort','shot'],[('MeanAnchor','LP'),('MeanAnchor','SASM-Ens'),('MeanAnchor','Direct-LP'),('Block-Concat','MeanAnchor')]).to_csv(E/'swap_paired_effects.csv',index=False)
 f=pd.read_csv(E/'calibration_episodes.csv');f.groupby(['backbone','cohort','shot','method','calibration'])[['ap','auc','ece','brier','f1','nll']].agg(['mean','std']).to_csv(E/'calibration_summary.csv');a=f.pivot(index=['backbone','cohort','shot','seed','method'],columns='calibration',values=['ap','auc','f1']);invariants={m:float(abs(a[m]['Raw']-a[m]['DevTemperature']).max()) for m in ['ap','auc','f1']};assert max(invariants.values())<1e-12
 h=f[f.cohort!='chapman'].copy();h['method']=h.method+' / '+h.calibration
 allstats=[]
 for endpoint in ['ece','brier','nll']:
  allstats.append(paired(h,['backbone','cohort','shot'],[('Direct-LP / DevTemperature','Direct-LP / Raw'),('Direct-LP-Refit-b / DevTemperature','Direct-LP-Refit-b / Raw'),('MeanAnchor / Raw','Direct-LP-Refit-b / DevTemperature')],endpoint))
 pd.concat(allstats).to_csv(E/'calibration_paired_effects.csv',index=False)
 mf=read(E/'INPUT_MANIFEST.json');drift=[]
 for p,v in mf['files'].items():
  if sha(p)!=v['sha256']:drift.append(p)
 assert not drift
 dump(E/'ALL_NUMERICAL_CHECKS.json',dict(positive_temperature_invariants=invariants,historical_input_drift=drift,source_head_count=len(list((E/'source').glob('*/*.npz'))),hg_episodes=len(list((E/'hg').glob('*/shot*.npz')))-len(list((E/'hg').glob('*/shot*.weights.npz'))),subgroup_validity='fixed common class denominator and invalid bootstrap replicate counts exported',papers_modified=False,public_release_modified=False))
 dump(E/'EXECUTION_CODE_MANIFEST.json',dict(time=time.time(),files={p.name:sha(p) for p in E.glob('*.py')},revision_note='Original protocol/code locks retained. Implementation fixes: diagnostic duplicate alpha serialization; historical import bridge; numpy string concatenation. Subgroup implementation added metadata hashes, SPH patient matching, AUC seed intervals and exclusive instance lock. No external-result-based hyperparameter or primary protocol replacement. Summarizer additionally requires model replay verification.'))
 files=[p for p in E.glob('*.csv')]+[p for p in E.glob('*.json')]+list((E/'source').glob('*/*.json'));files=[p for p in files if p.name not in ['FINAL_MANIFEST.json','ALL_COMPLETE.json']];dump(E/'FINAL_MANIFEST.json',dict(time=time.time(),files={str(p.relative_to(E)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files}));dump(E/'ALL_COMPLETE.json',dict(time=time.time(),protocol_sha256=sha(E/'protocol.lock.json'),checks_sha256=sha(E/'ALL_NUMERICAL_CHECKS.json')));emit('ALL_COMPLETE')
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback;dump(E/'SUMMARY_ERROR.json',dict(traceback=traceback.format_exc()));raise
