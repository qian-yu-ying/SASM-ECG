"""Rebuild CPSC labels from raw headers, without loading prediction results."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,hashlib,time,sys,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import wfdb
O=Path(__file__).resolve().parent
R=Path('/home/wy/ECGFounder'); Q=R/'supplement_queue_20261101_remote'
B=Q/'chapman_dev_native_same_20261103'
RAW=R/'data/ECG_新增数据集/CPSC2018/kaggle_extracted/Training_WFDB'
NAMES=['Normal','AF','I-AVB','LBBB','RBBB','PAC','PVC','STD','STE']
CODES=['426783006','164889003','270492004','164909002','59118001','284470004','164884008','429622005','164931005']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(4194304),b''):h.update(block)
 return h.hexdigest()
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False));t.replace(p)
def one(p):
 text=p.read_text();dx=next(l.split(':',1)[1].strip() for l in text.splitlines() if l.startswith('#Dx:'))
 codes=dx.replace(' ','').split(',');assert set(codes)<=set(CODES),(p,codes)
 x,m=wfdb.rdsamp(str(p.with_suffix('')))
 assert m['fs']==500 and x.shape[1]==12 and np.isfinite(x).all(),str(p)
 assert [n.lower() for n in m['sig_name']]==['i','ii','iii','avr','avl','avf','v1','v2','v3','v4','v5','v6']
 x=x.T.astype(np.float32);h=hashlib.sha256();nw=0
 for st in range(0,x.shape[1],5000):
  z=x[:,st:st+5000];z=np.pad(z,((0,0),(0,5000-z.shape[1])));z=(z-z.mean())/(z.std()+1e-6);h.update(np.asarray(z,np.float32).tobytes());nw+=1
 files={p}|{p.parent/l.split()[0] for l in text.splitlines()[1:13]};fh=hashlib.sha256()
 for f in sorted(files):fh.update(f.name.encode());fh.update(bytes.fromhex(sha(f)))
 return dict(record_id=p.name,waveform=str(p),codes=','.join(codes),header_sha256=sha(p),raw_sha256=fh.hexdigest(),normalized_wave_sha256=h.hexdigest(),samples=x.shape[1],windows=nw,**{n:int(c in codes) for n,c in zip(NAMES,CODES)})
def main():
 O.mkdir(parents=True,exist_ok=True);paths=sorted(RAW.glob('*.hea'));assert len(paths)==6877,len(paths)
 old=pd.read_csv(B/'data/cpsc2018/manifest.csv');known=old.set_index('record_id');rows=[];start=time.monotonic()
 with ThreadPoolExecutor(max_workers=2) as pool:
  for i,row in enumerate(pool.map(one,paths),1):
   row['historical_included']=row['record_id'] in known.index
   if row['historical_included']:
    oldrow=known.loc[row['record_id']];assert row['normalized_wave_sha256']==oldrow.normalized_wave_sha256,('historical waveform mismatch',row['record_id'])
   rows.append(row)
   if i%512==0:dump(O/'audit_progress.json',dict(completed=i,total=len(paths),seconds=time.monotonic()-start));print('AUDIT',i,flush=True)
 tab=pd.DataFrame(rows);Y=tab[NAMES].to_numpy(np.uint8);assert (Y.sum(1)>0).all()
 seven=[1,2,3,4,5,7,0];expected=Y[:,seven].any(1);assert np.array_equal(expected,tab.historical_included), 'historical eligibility differs from only-PVC/STE explanation'
 # Exact normalized waveform groups are indivisible. Full patient linkage is unavailable.
 tab['split_group']=tab.normalized_wave_sha256
 other={}
 for ds in ['chapman','ningbo','sph']:
  f=pd.read_csv(B/'data'/ds/'manifest.csv');other[ds]=set(f.normalized_wave_sha256)
 overlap={ds:tab[tab.normalized_wave_sha256.isin(h)].record_id.tolist() for ds,h in other.items()}
 # Preserve all existing assignments; assign restored groups by deterministic hash.
 group_role={}
 for row in old.itertuples():
  h=row.normalized_wave_sha256
  assert h not in group_role or group_role[h]==row.role
  group_role[h]=row.role
 for h in sorted(set(tab.split_group)-set(group_role)):
  u=int(hashlib.sha256(('cpsc9_restore_v1|'+h).encode()).hexdigest()[:16],16)/2**64
  group_role[h]='support_pool' if u<.4 else 'query'
 tab['role']=[group_role[h] for h in tab.split_group]
 contaminated=set(sum(overlap.values(),[]));badgroups=set(tab.loc[tab.record_id.isin(contaminated),'split_group'])
 tab.loc[tab.split_group.isin(badgroups),'role']='excluded_cross_cohort_duplicate'
 pool=np.flatnonzero(tab.role=='support_pool');qi=np.flatnonzero(tab.role=='query');assert not set(tab.iloc[pool].split_group)&set(tab.iloc[qi].split_group)
 # One representative per identical-waveform group in support sampling.
 reps=np.array([min(g.index) for _,g in tab.iloc[pool].groupby('split_group')]);assert (Y[reps].sum(0)>=10).all()
 tab.insert(0,'index',np.arange(len(tab)));tab.to_csv(O/'manifest.csv',index=False)
 np.savez_compressed(O/'labels.npz',y=Y,query_indices=qi,support_pool=pool,records=tab.record_id.to_numpy(str),names=np.array(NAMES))
 epstats=[]
 for k in [5,10]:
  for s in range(20):
   rng=np.random.default_rng(2026111700+s);selected=[]
   for c in range(9):selected.extend(rng.choice(reps[Y[reps,c]==1],size=k,replace=False).tolist())
   si=np.array(sorted(set(selected)));assert not set(si)&set(qi)
   ep=dict(shot=k,seed=s,support=si.tolist(),query=qi.tolist(),support_y=Y[si].tolist(),support_positive_counts=Y[si].sum(0).tolist(),support_groups_distinct=len(set(tab.iloc[si].split_group))==len(si))
   assert ep['support_groups_distinct'] and (Y[si].sum(0)>=k).all() and (Y[si].sum(0)<len(si)).all()
   dump(O/'episodes'/f'shot{k}_seed{s:02d}.json',ep);epstats.append(dict(shot=k,seed=s,n=len(si),**dict(zip(NAMES,Y[si].sum(0).astype(int).tolist()))))
 pd.DataFrame(epstats).to_csv(O/'episode_counts.csv',index=False)
 mapping=[dict(target=n,snomed=c,source_row75=j,source_code=sc,policy=pol) for n,c,j,sc,pol in zip(NAMES,CODES,[13,63,19,9,16,49,44,57,-1],['NORM','AFIB','1AVB','CLBBB','CRBBB','PAC','PVC','STD_',None],['normal ECG semantic mapping; SR row70 predefined sensitivity']+['matched']*7+['unmatched; no anchor penalty; fit from support labels'])]
 dump(O/'mapping.json',mapping)
 report=dict(total=len(tab),historical=len(old),restored=int((~tab.historical_included).sum()),class_positive_counts=dict(zip(NAMES,Y.sum(0).astype(int).tolist())),restored_codes=tab[~tab.historical_included].codes.value_counts().to_dict(),roles=tab.role.value_counts().to_dict(),cross_cohort_overlap=overlap,duplicate_records=int(tab.split_group.duplicated().sum()),distinct_waveforms=tab.split_group.nunique(),header_manifest_sha256=sha(O/'manifest.csv'),labels_sha256=sha(O/'labels.npz'),seconds=time.monotonic()-start,patient_identity='record keys only; duplicate waveform grouping does not prove absence of repeated patients',retrospective=True,prior_result_exposure=True,STE_policy='unmatched by design; broad target STE is not asserted equivalent to non-specific source STE',source_mapping_sha256=sha(B/'source/source_labels.csv'))
 dump(O/'AUDIT_COMPLETE.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:dump(O/'AUDIT_ERROR.json',dict(error=traceback.format_exc()));raise
