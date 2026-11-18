from common import *
import platform,subprocess
def main():
 bind([12]);rows=[];rng=np.random.default_rng(2026111801)
 info=dict(cpuinfo=subprocess.check_output(['lscpu'],text=True),platform=platform.platform(),affinity=sorted(os.sched_getaffinity(0)),threadpools=__import__('threadpoolctl').threadpool_info(),protocol_sha256=sha(E/'protocol.lock.json'),timing_excludes=['encoder','disk IO','query scoring','source head training'])
 dump(E/'timing_environment.json',info)
 for bb in ['ECGFounder','MERL']:
  for ds in ['ningbo','cpsc2018','sph']:
   for k in [5,10]:
    F,raw,off,cols=context(bb,ds);ep,si,qi,y=episode(ds,k,0);fs=F[si].astype(float);lv=np.r_[np.full(8,10.),0.] if ds=='cpsc2018' else None;lpalpha=100. if bb=='ECGFounder' else 10.
    # All arrays reside in memory before timing. Include scaler and anchor transformation.
    for rep in range(-1,10):
     for name in rng.permutation(['LP','SASM-Ens','MeanAnchor']):
      t=time.perf_counter();mu=fs.mean(0);sd=fs.std(0)+1e-6;x=(fs-mu)/sd;diags=[]
      if name=='LP':w,b,d=fit(x,y,lpalpha);diags.append(d)
      elif name=='MeanAnchor':
       a=sd[:,None]*raw.mean(0);pb=mu@raw.mean(0)+off.mean(0);w,b,d=fit(x,y,.1,a,10.,lv,pb);diags.append(d)
      else:
       ww=[];bs=[]
       for r,o in zip(raw,off):
        a=sd[:,None]*r;pb=mu@r+o;w,b,d=fit(x,y,.1,a,10.,lv,pb);diags.append(d);ww.append(w);bs.append(b)
       w=np.mean(ww,0);b=np.mean(bs,0)
      seconds=time.perf_counter()-t
      assert np.isfinite(w).all() and np.isfinite(b).all()
      if rep>=0:rows.append(dict(backbone=bb,cohort=ds,shot=k,seed=0,method=name,repeat=rep,seconds=seconds,max_gradient=max(d['max_gradient'] for d in diags),support_records=len(y),CPU=12))
     pd.DataFrame(rows).to_csv(E/'timing_raw.csv',index=False)
    emit('TIMING',backbone=bb,cohort=ds,shot=k,done=len(rows),total=360)
 f=pd.DataFrame(rows);g=f.groupby(['backbone','cohort','shot','method']).seconds.agg(n='size',median='median',q25=lambda x:x.quantile(.25),q75=lambda x:x.quantile(.75),mean='mean',std='std').reset_index();g['IQR']=g.q75-g.q25;g.to_csv(E/'timing_summary.csv',index=False);dump(E/'TIMING_COMPLETE.json',dict(rows=len(rows),sha256=sha(E/'timing_raw.csv')))
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback;dump(E/'TIMING_ERROR.json',dict(traceback=traceback.format_exc()));raise
