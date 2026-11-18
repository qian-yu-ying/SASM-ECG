from common import *
def main():
 bind([8]);models=[];worst=0.;maxsource=0.;maxtarget=0.;files={}
 for it in ['h','g']:
  for p in sorted((E/'source'/it).glob('ep*.npz')):
   m=read(p.with_suffix('.json'));assert sha(p)==m['sha256'];assert m['refit']['converged'];assert all(d['converged'] for d in m['diagnostics']);g=max([d['kkt_max_abs_mean_per_record'] for d in m['diagnostics']]+[m['refit']['kkt_max_abs_mean_per_record']]);assert g<=5e-5;maxsource=max(maxsource,g);files[str(p.relative_to(E))]=sha(p)
 for ds in ['chapman','ningbo','cpsc2018','sph']:
  F,*_=context('ECGFounder',ds)
  for p in sorted((E/'hg'/ds).glob('shot*.npz')):
   if '.weights.' in p.name:continue
   m=read(p.with_suffix('.json'));assert sha(p)==m['sha256'];z=np.load(p);wp=p.with_suffix('.weights.npz');w=np.load(wp);qi=z['query_indices'];xq=(F[qi].astype(float)-w['scaler_mean'])/w['scaler_sd']
   for it in ['h','g']:
    for method,key in [('SASM-Ens','Ens'),('MeanAnchor','Mean')]:
     replay=xq@w[f'{it}-{key}-w']+w[f'{it}-{key}-b'];err=float(abs(replay-z[f'{it}-{method}']).max());assert err<1e-8;worst=max(worst,err)
   maxtarget=max(maxtarget,max(d['max_gradient'] for d in m['diags'].values()));files[str(p.relative_to(E))]=sha(p);files[str(wp.relative_to(E))]=sha(wp);models.append(str(p.relative_to(E)))
 for directory in ['base','swap']:
  for p in (E/directory).glob('**/shot*.npz'):
   m=read(p.with_suffix('.json'));assert sha(p)==m['sha256'];files[str(p.relative_to(E))]=m['sha256']
 assert len(models)==140
 dump(E/'MODEL_ASSET_VERIFICATION.json',dict(hg_replayed_episodes=len(models),hg_saved_coefficient_max_score_error=worst,source_max_KKT_residual=maxsource,target_max_gradient=maxtarget,checked_assets=len(files),files=files))
 emit('MODEL_ASSET_VERIFIED',episodes=len(models),max_score_error=worst,assets=len(files))
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback;dump(E/'MODEL_ASSET_ERROR.json',dict(traceback=traceback.format_exc()));raise
