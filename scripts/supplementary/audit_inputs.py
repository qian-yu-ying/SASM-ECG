from common import *
def main():
 files=set();checks=[];p=R/'res/features_1024/ptbxl_feat1024_filtered.npz';files.add(p);z=np.load(p);db=pd.read_csv(R/'data/PTBXL_v1.0.3/ptbxl_database.csv');assert np.array_equal(z['records'].astype(str),np.char.add(db.filename_hr.to_numpy(str),'.hea'));dense=np.load(B/'assets/dense.npz');rows=np.unique(np.concatenate([np.load(R/f'res/b3_sign_loss_diagnostic_20261014_v1/ep{e:02d}_support_and_cv.npz')['support_rows'] for e in range(10)]));assert len(rows)==9871 and z['valid'][rows].all();h=z['feat'][rows].astype(float);g=h@dense['Dw'].T+dense['Db'];aff=float(abs(g-z['logits_mean'][rows]).max());assert aff<1e-4
 old=pd.read_csv(R/'res/eval/all_pred.csv').to_numpy(np.float32)[rows];old=np.log(np.clip(old,1e-6,1-1e-6)/np.clip(1-old,1e-6,1)).astype(np.float32);checks.append(dict(check='source representation',rows=len(rows),all_record_ids_aligned=True,all_source_features_valid=True,affine_cached_logits_error=aff,historical_source_g_vs_same_h_affine_max_error=float(abs(g-old).max()),historical_source_g_vs_same_h_affine_mean_error=float(abs(g-old).mean()),interpretation='Historical source clinical scores and current filtered h affine outputs differ; controlled h/g both reconstructed from identical h, historical method reported separately.'))
 files|={R/'data/PTBXL_v1.0.3/ptbxl_database.csv',R/'res/eval/all_pred.csv',B/'assets/dense.npz',B/'source/source_labels.csv',B/'source/tasks.txt',B/'source/target_contract.py'}
 for e in range(10):files|={R/f'res/b3_sign_loss_diagnostic_20261014_v1/ep{e:02d}_support_and_cv.npz',B/f'source/source_heads/ep{e:02d}_bce_signed_model.npz',Q/f'merl_source/source_heads/ep{e:02d}.npz'}
 for bb in ['ECGFounder','MERL']:
  for ds in ['chapman','ningbo','sph']:
   d=B/('data' if bb=='ECGFounder' else 'merl/data')/ds;p=d/'features.npy';a=np.load(p,mmap_mode='r');tab=pd.read_csv(B/'data'/ds/'manifest.csv');assert a.shape==(len(tab),1024 if bb=='ECGFounder' else 512);v=read(d/'FEATURES_READY.json');digest=sha(p);expected=v.get('sha256',v.get('feature_sha256'));assert digest==expected,(str(p),v.keys());files|={p,d/'FEATURES_READY.json',B/'data'/ds/'manifest.csv',B/'data'/ds/'metadata.json'};checks.append(dict(check='target feature cache',backbone=bb,cohort=ds,shape=list(a.shape),matches_historical_feature_receipt=True))
   files.update((B/'data'/ds/'episodes').glob('*.json'));files.update((SW/'data'/ds/'episodes').glob('*.json'))
  files.add(O/f'{bb}_features.npy')
 files.update((O/'episodes').glob('*.json'));files|={O/'FEATURES_COMPLETE.json',O/'manifest.csv',O/'labels.npz',B/'data/chapman/development_labels.npz',SW/'data/sph/development_labels.npz',R/'data/ECG_新增数据集/SPH/raw/metadata.csv'}
 for base in [B,SW]:files.update((base/'vault').glob('*/external.npz'))
 dump(E/'INPUT_MANIFEST.json',dict(time=time.time(),snapshot='Inputs verified during execution against historical receipts; not represented as a new prospective registration.',files={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(files)},checks=checks));emit('INPUT_AUDIT_COMPLETE',files=len(files))
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback;dump(E/'INPUT_AUDIT_ERROR.json',dict(traceback=traceback.format_exc()));raise
