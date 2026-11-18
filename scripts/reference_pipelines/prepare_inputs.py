from core import *
import shutil,pandas as pd
PRE=R.parent/'lambda10_native_tuned_full_rerun_20261103'
def main():
 assert not (R/'INPUTS_READY.json').exists()
 for n in ['source']:(R/n).symlink_to((PRE/n).resolve(),target_is_directory=True)
 (R/'assets').mkdir();shutil.copy2(PRE/'assets/dense.npz',R/'assets/dense.npz')
 shutil.copy2(PRE/'CROSS_DATASET_WAVEFORM_AUDIT.json',R/'CROSS_DATASET_WAVEFORM_AUDIT.json')
 evidence={}
 for ds in DATASETS:
  old=PRE/'data'/ds;d=R/'data'/ds;d.mkdir(parents=True);(d/'episodes').mkdir();(R/'vault'/ds).mkdir(parents=True)
  for n in ['features.npy','Q.npy','lengths.npy','dedup_report.json']:
   if (old/n).exists():(d/n).symlink_to((old/n).resolve())
  tab=pd.read_csv(old/'manifest.csv');oldtab=tab.copy();meta=read(old/'metadata.json')
  if ds=='chapman':tab.loc[tab.role=='query','role']='dev_val';origin_phase='external';new_phase='dev';seeds=range(10)
  elif ds=='ningbo':
   tab.loc[tab.role=='dev_val','role']='unused_previous_development';tab.loc[tab.role=='audit','role']='query';origin_phase='audit';new_phase='external';seeds=range(20)
  else:origin_phase=new_phase='external';seeds=range(20)
  tab.to_csv(d/'manifest.csv',index=False)
  assert tab.record_id.tolist()==oldtab.record_id.tolist() and tab.patient_id.tolist()==oldtab.patient_id.tolist() and tab.split_group.tolist()==oldtab.split_group.tolist()
  pool=set(tab[tab.role=='support_pool']['index']);qrole='dev_val' if ds=='chapman' else 'query';query=tab[tab.role==qrole]['index'].to_numpy()
  assert not set(tab[tab.role=='support_pool'].split_group)&set(tab[tab.role==qrole].split_group)
  for k in [5,10]:
   for seed in seeds:
    ep=read(old/'episodes'/f'{origin_phase}_shot{k}_seed{seed:02d}.json');ep['phase']=new_phase
    assert set(ep['support'])<=pool and ep['query']==query.tolist() and not set(ep['support'])&set(ep['query'])
    dump(d/'episodes'/f'{new_phase}_shot{k}_seed{seed:02d}.json',ep)
  vp=PRE/'vault'/ds/f'{origin_phase}.npz';v=np.load(vp,allow_pickle=True);assert np.array_equal(v['indices'],query)
  if ds=='chapman':np.savez_compressed(d/'development_labels.npz',indices=v['indices'],y=v['y'])
  else:(R/'vault'/ds/'external.npz').symlink_to(vp.resolve())
  meta.update(development_dataset=ds=='chapman',role_counts={str(k):int(v) for k,v in tab.role.value_counts().items()},manifest_sha256=sha(d/'manifest.csv'),cohort_role_revision='Chapman dev/no audit; Ningbo previous audit becomes external query; existing support/query identities retained',prior_result_exposure=True)
  dump(d/'metadata.json',meta);dump(d/'DATA_READY.json',dict(time=time.time(),manifest_sha256=sha(d/'manifest.csv'),metadata_sha256=sha(d/'metadata.json'),predecessor_manifest_sha256=sha(old/'manifest.csv'),record_order_unchanged=True))
  ef=read(old/'FEATURES_READY.json');ef.update(metadata_sha256=sha(d/'metadata.json'),manifest_sha256=sha(d/'manifest.csv'),predecessor_metadata_sha256=sha(old/'metadata.json'),role_only_revision=True);dump(d/'FEATURES_READY.json',ef)
  md=R/'merl/data'/ds;md.mkdir(parents=True)
  (md/'features.npy').symlink_to((PRE/'merl/data'/ds/'features.npy').resolve())
  mf=read(PRE/'merl/data'/ds/'FEATURES_READY.json');mf.update(manifest_sha256=sha(d/'manifest.csv'),role_only_revision=True);dump(md/'FEATURES_READY.json',mf)
  assert sha(d/'features.npy')==read(d/'FEATURES_READY.json')['sha256'];assert sha(md/'features.npy')==mf['sha256']
  evidence[ds]=dict(manifest_sha256=sha(d/'manifest.csv'),metadata_sha256=sha(d/'metadata.json'),ECG_feature_sha256=sha(d/'features.npy'),MERL_feature_sha256=sha(md/'features.npy'),Q_sha256=sha(d/'Q.npy'),episodes={p.name:sha(p) for p in sorted((d/'episodes').glob('*.json'))},vault={p.name:sha(p) for p in (R/'vault'/ds).glob('*.npz')},role_counts=meta['role_counts'])
 dump(R/'INPUTS_READY.json',dict(time=time.time(),datasets=evidence,predecessor_protocol_sha256=sha(PRE/'protocol.lock.json')))
if __name__=='__main__':main()
