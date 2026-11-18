from ablation_common import *
from threadpoolctl import threadpool_info
import platform,traceback

def support(ds,k,seed):
 if k in [5,10]:
  ep=read(episode(ds,k,seed));return np.array(ep['support']),np.array(ep['support_y'],float)
 tab=pd.read_csv(OLD/'data'/ds/'manifest.csv');meta=read(OLD/'data'/ds/'metadata.json')
 if ds=='sph':Y=pd.read_csv(OLD/'source/sph_manifest.csv')[meta['labels']].to_numpy(np.uint8)
 else:
  z=np.load(Path(os.environ.get('SASM_TARGET_FULL_DIR', './external/target_full_v3'))/f'{ds}_unfiltered/cohort.npz');assert z['records'].astype(str).tolist()==tab.record_id.tolist();Y=z['Y'][:,z['cols']]
 pool=tab[tab.role=='support_pool'];represent=np.array(sorted(int(g.sort_values('record_id').iloc[0]['index']) for _,g in pool.groupby('split_group')));rng=np.random.default_rng(seed);indices=set()
 for c in range(Y.shape[1]):indices.update(rng.choice(represent[Y[represent,c]==1],k,replace=False).tolist())
 si=np.array(sorted(indices));assert (Y[si].sum(0)>=k).all() and (tab.iloc[si].role=='support_pool').all()
 dump(ROOT/'timing_supports'/f'{ds}_shot{k}_seed{seed:02d}.json',dict(indices=si.tolist(),support_records=tab.iloc[si].record_id.tolist(),support_y=Y[si].tolist(),sampling='original support-pool representative sampler; no query evaluation'))
 return si,Y[si].astype(float)

def main():
 cfg=read(ROOT/'protocol.lock.json')['selected']['ECGFounder'];available=sorted(os.sched_getaffinity(0));cpu=available[-1];os.sched_setaffinity(0,{cpu});assert os.sched_getaffinity(0)=={cpu}
 LIMITER=threadpool_limits(limits=1);assert all(x['num_threads']==1 for x in threadpool_info())
 dump(ROOT/'timing_environment.json',dict(cpu_affinity=[cpu],hostname=platform.node(),platform=platform.platform(),processor=platform.processor(),cpuinfo=Path('/proc/cpuinfo').read_text(),threadpools=threadpool_info(),repetitions=3,warmups_per_dataset_shot_method=1,seeds=list(range(5)),includes='support scaler and target anchors/constant fallback plus sequential fitting',excludes='feature/source extraction, disk IO, query prediction, scoring',numpy=np.__version__))
 rows=[]
 for ds in ['ningbo','chapman','cpsc2018','sph']:
  F=context(ds);heads=source_context(ds);lp_cfg=read(OLD/'release_01.json')['selected']['LP']['config']
  for k in [5,10,15]:
   for seed in range(5):
    si,ys=support(ds,k,seed);fs=np.asarray(F[si],float)
    def fit(method):
     start=time.perf_counter();mu=fs.mean(0);sd=fs.std(0)+1e-6;xs=(fs-mu)/sd
     if method=='LP':C.fit_with_fallback(xs,ys,lp_cfg['alpha'],0.,None,None)
     else:
      for h in heads[:1 if method=='SASM-Single-ep00' else 10]:solve(xs,ys,cfg['alpha'],cfg['lam'],sd[:,None]*h['raw'],mu@h['raw']+h['offset'])
     return time.perf_counter()-start
    methods=['LP','SASM-Single-ep00','SASM-Ens']
    # Rotate execution order deterministically to reduce order effects without performance selection.
    methods=methods[seed%3:]+methods[:seed%3]
    for method in methods:
     if seed==0:fit(method)
     for repeat in range(3):rows.append(dict(dataset=ds,shot=k,seed=seed,method=method,repeat=repeat,seconds=fit(method),support_records=len(si),cpu=cpu))
    dump(ROOT/'timing_status.json',dict(pid=os.getpid(),time=time.time(),dataset=ds,shot=k,seed=seed,rows=len(rows)));pd.DataFrame(rows).to_csv(ROOT/'timing_raw.csv',index=False)
 df=pd.DataFrame(rows);s=df.groupby(['dataset','shot','method']).seconds.agg(n='size',median='median',q25=lambda x:x.quantile(.25),q75=lambda x:x.quantile(.75),minimum='min',maximum='max').reset_index();s.to_csv(ROOT/'timing_summary.csv',index=False);dump(ROOT/'TIMING_COMPLETE.json',dict(time=time.time(),rows=len(rows),raw_sha256=sha(ROOT/'timing_raw.csv'),summary_sha256=sha(ROOT/'timing_summary.csv')))
if __name__=='__main__':
 try:main()
 except Exception:dump(ROOT/'errors'/f'timing_{os.getpid()}.json',dict(error=traceback.format_exc(),time=time.time()));raise
