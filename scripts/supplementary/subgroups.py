from common import *
from concurrent.futures import ProcessPoolExecutor,as_completed
import re
METHODS=['LP','Direct','Direct-LP','SASM-Ens','MeanAnchor']
def demographics():
 if (E/'DEMOGRAPHICS_COMPLETE.json').exists():return
 cover=[]
 for ds in ['ningbo','cpsc2018','sph','chapman']:
  out=E/'demographics'/f'{ds}.csv'
  if out.exists():continue
  tab=pd.read_csv((O if ds=='cpsc2018' else B/'data'/ds)/'manifest.csv');ages=[];sexes=[];header_hashes=[]
  if ds=='sph':
   src=R/'data/ECG_新增数据集/SPH/raw/metadata.csv';md=pd.read_csv(src);assert not md.ECG_ID.duplicated().any();md=md.set_index('ECG_ID');patient_by_record=tab.set_index('record_id').patient_id.to_dict()
   for rid in tab.record_id:
    row=md.loc[rid];assert 'SPH:'+str(row.Patient_ID)==str(patient_by_record[rid]);ages.append(row.Age);sexes.append(row.Sex)
  else:
   src=None
   for path in tab.waveform:
    raw=Path(path).read_bytes();header_hashes.append(hashlib.sha256(raw).hexdigest());text=raw.decode();a=re.search(r'^#\s*Age\s*:\s*(.*)$',text,re.M|re.I);s=re.search(r'^#\s*Sex\s*:\s*(.*)$',text,re.M|re.I);ages.append(a.group(1).strip() if a else None);sexes.append(s.group(1).strip() if s else None)
  age=pd.to_numeric(pd.Series(ages),errors='coerce');valid=age.between(0,120);sex=pd.Series(sexes).astype(str).str.strip().str.lower().map({'male':'Male','m':'Male','female':'Female','f':'Female'}).fillna('Unknown');age=age.where(valid)
  out.parent.mkdir(parents=True,exist_ok=True);res=pd.DataFrame(dict(index=np.arange(len(tab)),record_id=tab.record_id,age=age,sex=sex,group_id=tab.patient_id if ds=='sph' else tab.normalized_wave_sha256));assert not res.record_id.duplicated().any();res['metadata_header_sha256']=header_hashes if header_hashes else sha(src);res.to_csv(out,index=False);cover.append(dict(cohort=ds,n=len(tab),age_known=int(valid.sum()),sex_known=int((sex!='Unknown').sum()),sex_distribution=sex.value_counts().to_dict(),patient_id_match_verified=ds=='sph',sex_discordant_groups=int((res[res.sex!='Unknown'].groupby('group_id').sex.nunique()>1).sum()),manifest_sha256=sha((O if ds=='cpsc2018' else B/'data'/ds)/'manifest.csv'),metadata_sha256=sha(src) if src else 'original WFDB headers, manifest-linked'))
 dump(E/'DEMOGRAPHICS_COMPLETE.json',dict(coverage=cover,warning='Subject grouping only verified in SPH. Other query uncertainty clustered by full-waveform hash, not patient.'))
def ci(a):return tuple(float(v) for v in np.quantile(a,[.025,.975]))
def weighted_ap_batch(y,score,weights):
 order=np.argsort(-score,kind='stable');s=score[order];yy=y[order];ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1];ww=weights[order];tp=np.cumsum(ww*yy[:,None],axis=0)[ends];nn=np.cumsum(ww,axis=0)[ends];den=tp[-1];inc=np.diff(np.r_[np.zeros((1,ww.shape[1])),tp],axis=0)
 return np.divide((np.divide(tp,nn,out=np.zeros_like(tp),where=nn>0)*inc).sum(0),den,out=np.full_like(den,np.nan),where=den>0)
def one(job):
 bb,ds,k=job;bind([10,11]);out=E/'subgroups'/bb/ds/f'shot{k}.json'
 if out.exists():return
 qi,y=labels(ds);md=pd.read_csv(E/'demographics'/f'{ds}.csv').iloc[qi].reset_index(drop=True);assert len(md)==len(y);base=[np.load(E/'base/main'/bb/ds/f'shot{k}_seed{s:02d}.npz') for s in range(20)]
 assert all(np.array_equal(z['query_indices'],qi) for z in base)
 pred={m:[z[m] for z in base] for m in METHODS};rows=[];counts=[];summary=[];boot=[];groupmasks={'Male':md.sex.to_numpy()=='Male','Female':md.sex.to_numpy()=='Female','Age_lt60':md.age.to_numpy()<60,'Age_ge60':md.age.to_numpy()>=60};rng=np.random.default_rng(2026111802);seedidx=rng.integers(0,20,(10000,20))
 for family,gnames in [('sex',['Male','Female']),('age',['Age_lt60','Age_ge60'])]:
  common=np.ones(y.shape[1],bool)
  for name in gnames:
   yy=y[groupmasks[name]];common&=(yy.sum(0)>0)&(yy.sum(0)<len(yy))
  cols=np.flatnonzero(common);assert len(cols)
  for name in gnames:
   mask=groupmasks[name];yy=y[mask];n=len(yy);assert n>0
   for c in range(y.shape[1]):counts.append(dict(backbone=bb,cohort=ds,shot=k,subgroup=name,family=family,label=c,n=n,positives=int(yy[:,c].sum()),prevalence=float(yy[:,c].mean()),included_in_common_macro=bool(common[c])))
   vals={}
   for m in METHODS:
    ap=[];auc=[]
    for s in range(20):
     sc=pred[m][s][mask];aa=[];uu=[]
     for c in range(y.shape[1]):
      a=float(average_precision_score(yy[:,c],sc[:,c])) if 0<yy[:,c].sum()<n else None;u=float(roc_auc_score(yy[:,c],sc[:,c])) if a is not None else None
      rows.append(dict(backbone=bb,cohort=ds,shot=k,seed=s,subgroup=name,method=m,label=c,n=n,positives=int(yy[:,c].sum()),ap=a,auc=u,included_in_common_macro=bool(common[c])))
      if common[c]:aa.append(a);uu.append(u)
     ap.append(np.mean(aa));auc.append(np.mean(uu))
    vals[m]=np.array(ap);lo,hi=ci(vals[m][seedidx].mean(1));summary.append(dict(backbone=bb,cohort=ds,shot=k,subgroup=name,method=m,n=n,common_classes=len(cols),ap_mean=float(np.mean(ap)),ap_sd=float(np.std(ap,ddof=1)),ap_seed_CI_low=lo,ap_seed_CI_high=hi,auc_mean=float(np.mean(auc)),auc_seed_CI_low=ci(np.asarray(auc)[seedidx].mean(1))[0],auc_seed_CI_high=ci(np.asarray(auc)[seedidx].mean(1))[1],omitted_labels=','.join(map(str,np.flatnonzero(~common)))))
   for comp in ['LP','Direct-LP','SASM-Ens']:
    diff=vals['MeanAnchor']-vals[comp];lo,hi=ci(diff[seedidx].mean(1));boot.append(dict(backbone=bb,cohort=ds,shot=k,subgroup=name,comparison='MeanAnchor-minus-'+comp,unit='support_seed_fixed_query',difference=float(diff.mean()),CI_low=lo,CI_high=hi,valid_replicates=10000,total_replicates=10000))
   # Identical query-cluster draws for all support episodes and methods. Metric first, then average across seeds.
   groups,inverse=np.unique(md.loc[mask,'group_id'].astype(str),return_inverse=True);G=len(groups);W=rng.multinomial(G,np.full(G,1/G),size=500).T.astype(np.float64);weights=W[inverse];apboot={m:np.zeros(500) for m in ['LP','Direct-LP','MeanAnchor']}
   # Fixed common label denominator. Undefined bootstrap replicates remain NaN, never silently drop rare labels.
   valid=np.ones(500,bool)
   for c in cols:
    pos=yy[:,c]@weights;neg=(1-yy[:,c])@weights;valid&=(pos>0)&(neg>0)
   for m in apboot:
    for s in range(20):
     scores=pred[m][s][mask]
     for c in cols:
      value=weighted_ap_batch(yy[:,c],scores[:,c],weights);apboot[m]+=value/(20*len(cols))
   for comp in ['LP','Direct-LP']:
    v=apboot['MeanAnchor']-apboot[comp];ok=valid&np.isfinite(v);assert ok.sum()>=250;lo,hi=ci(v[ok]);boot.append(dict(backbone=bb,cohort=ds,shot=k,subgroup=name,comparison='MeanAnchor-minus-'+comp,unit='SPH_patient_cluster' if ds=='sph' else 'waveform_hash_cluster',difference=float((vals['MeanAnchor']-vals[comp]).mean()),CI_low=lo,CI_high=hi,valid_replicates=int(ok.sum()),total_replicates=500))
   emit('SUBGROUP_DETAIL',backbone=bb,cohort=ds,shot=k,subgroup=name,n=n,common_classes=len(cols),bootstrap_valid=int(valid.sum()))
 out.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out.with_suffix('.labels.csv'),index=False);pd.DataFrame(counts).to_csv(out.with_suffix('.counts.csv'),index=False);pd.DataFrame(summary).to_csv(out.with_suffix('.summary.csv'),index=False);pd.DataFrame(boot).to_csv(out.with_suffix('.CI.csv'),index=False);dump(out,dict(backbone=bb,cohort=ds,shot=k,summary_sha256=sha(out.with_suffix('.summary.csv'))))
def witness():
 rng=np.random.default_rng(91);y=rng.integers(0,2,50);s=rng.integers(0,8,50);w=rng.integers(0,3,(50,9)).astype(float);actual=weighted_ap_batch(y,s,w);ref=np.array([average_precision_score(y,s,sample_weight=w[:,j]) for j in range(9)]);assert abs(actual-ref).max()<1e-12;dump(E/'SUBGROUP_WITNESS.json',dict(weighted_tie_AP_error=float(abs(actual-ref).max())))
def main():
 import fcntl
 guard=(E/'subgroups.lock').open('a+')
 fcntl.flock(guard,fcntl.LOCK_EX)
 if (E/'SUBGROUP_COMPLETE.json').exists():return
 witness();demographics();jobs=[(bb,ds,k) for bb in ['ECGFounder','MERL'] for ds in ['ningbo','cpsc2018','sph'] for k in [5,10]]
 with ProcessPoolExecutor(2) as ex:
  for i,f in enumerate(as_completed([ex.submit(one,j) for j in jobs])):f.result();emit('SUBGROUP_PROGRESS',done=i+1,total=len(jobs))
 for suf in ['labels','counts','summary','CI']:
  files=sorted((E/'subgroups').glob(f'*/*/shot*.{suf}.csv'));assert len(files)==12;pd.concat([pd.read_csv(p) for p in files],ignore_index=True).to_csv(E/f'subgroup_{suf}.csv',index=False)
 dump(E/'SUBGROUP_COMPLETE.json',dict(conditions=12,sha256=sha(E/'subgroup_summary.csv')))
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback;dump(E/'SUBGROUP_ERROR.json',dict(traceback=traceback.format_exc()));raise
