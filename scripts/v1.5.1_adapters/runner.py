"""Development-only grids followed by a barrier and locked external fits."""
from adapters import *
from concurrent.futures import ProcessPoolExecutor,as_completed
from scipy.optimize import minimize_scalar
import traceback

def model_tag(i):return f'm{i}'
def run_dir(mode):return RUN/'fits'/mode
def grid_path(mode,bb,k,s,mi):return RUN/'grid'/mode/bb/f'{model_tag(mi)}_shot{k}_seed{s:02d}.json'
def freeze_path(mode):return OUT/('dev_chapman_grid_search.json' if mode=='main' else 'dev_sph_roleswap_grid_search.json')
def configuration(mode,bb,mi):return read(freeze_path(mode))['selected'][bb][METHODS[mi]]['config']
def guards():
    assert read(OUT/'sanity_gate_v1.5.1_check.json')['all_passed']
    assert read(OUT/'lpplus_monotone_unit_test.json')['all_monotone']
    assert sha(RUN/'SASM_v1.5.1_BSPC_Adapter_Server_Spec_LOCKED.md')==SPEC_SHA

def grid_job(job):
    mode,bb,k,s,mi=job;pin();guards();p=grid_path(mode,bb,k,s,mi)
    if p.exists():
        old=read(p);assert old['spec_sha256']==SPEC_SHA and old['complete'];return str(p)
    ds='chapman' if mode=='main' else 'sph';swap=mode=='swap';ep=support(bb,ds,k,s,swap,True)
    qi,y=H.labels(ds,swap);assert np.array_equal(qi,ep['qi'])
    method=METHODS[mi];configs=grid(method);start=time.time();scores=[]
    if mi==0:
        for space in ['raw_l2','std_l2']:
            us=views(ep,space);uq=views(ep,space,True);similarity=uq@us.T;base=tip_base(ep,True)
            for beta in [.5,1.,2.5,5.,10.,20.]:
                affinity=np.exp(-beta*(1.-similarity));pos=affinity@ep['y']/(affinity.sum(1,keepdims=True)+1e-8)
                signed=(affinity@(2*ep['y']-1))/(affinity.sum(1,keepdims=True)+1e-8)
                for encoding,v in [('signed',signed),('positive',pos)]:
                    for alpha in [.1,.5,1.,2.,5.,10.]:
                        cfg=dict(space=space,encoding=encoding,beta=beta,alpha=alpha)
                        scores.append(dict(config=cfg,ap=macro_ap(y,base+alpha*v)))
    elif mi==1:
        inherited=configuration(mode,bb,0)
        for cfg in configs:
            full=dict(inherited,**cfg);model=fit(ep,method,full);scores.append(dict(config=full,ap=macro_ap(y,predict(ep,model,True))))
    elif mi==2:
        for epochs in [20,50,100]:
            cc=[v for v in configs if v['epochs']==epochs];models=fit_task_batch(ep,cc)
            for cfg,model in zip(cc,models):scores.append(dict(config=cfg,ap=macro_ap(y,predict(ep,model,True))))
    else:
        for space in ['std','raw_l2']:
            cc=[v for v in configs if v['space']==space];models=fit_lp_batch(ep,cc)
            for cfg,model in zip(cc,models):scores.append(dict(config=cfg,ap=macro_ap(y,predict(ep,model,True)),max_objective_rise=model['max_objective_rise']))
    assert len(scores)==[144,24,135,160][mi]
    assert all(np.isfinite(row['ap']) for row in scores)
    dump(p,dict(complete=True,mode=mode,backbone=bb,cohort=ds,shot=k,seed=s,method=method,development_only=True,
                spec_sha256=SPEC_SHA,candidates=scores,seconds=time.time()-start))
    return str(p)

def select(mode,mis):
    p=freeze_path(mode);out=read(p) if p.exists() else dict(spec_sha256=SPEC_SHA,development='chapman' if mode=='main' else 'sph',selected={},grid={},files={},external_query_labels_used=False)
    for bb in ['ECGFounder','MERL']:
        out['selected'].setdefault(bb,{});out['grid'].setdefault(bb,{})
        for mi in mis:
            files=[grid_path(mode,bb,k,s,mi) for k in [5,10] for s in range(10)]
            allrows=[]
            for f in files:
                v=read(f);assert v['complete'] and len(v['candidates'])==[144,24,135,160][mi]
                out['files'][str(f.relative_to(RUN))]=sha(f)
                for row in v['candidates']:allrows.append(dict(config_id=json.dumps(row['config'],sort_keys=True),ap=row['ap']))
            frame=pd.DataFrame(allrows);g=frame.groupby('config_id',sort=True).ap.agg(['mean','std','size']).reset_index()
            assert (g['size']==20).all()
            # Predeclared deterministic tie rule: descending exact mean AP, then canonical config JSON.
            g=g.sort_values(['mean','config_id'],ascending=[False,True]);best=g.iloc[0]
            out['selected'][bb][METHODS[mi]]=dict(config=json.loads(best.config_id),mean_dev_ap=float(best['mean']),episodes=20)
            out['grid'][bb][METHODS[mi]]=[dict(config=json.loads(r.config_id),mean_dev_ap=float(r['mean']),sd_dev_ap=float(r['std']),episodes=int(r['size'])) for _,r in g.iterrows()]
    out['all_four_methods_locked']=all(len(out['selected'].get(bb,{}))==4 for bb in ['ECGFounder','MERL'])
    dump(p,out);notice('SELECTION_'+mode,methods=[METHODS[i] for i in mis],all_locked=out['all_four_methods_locked'])

def fit_path(mode,bb,ds,k,s):return run_dir(mode)/bb/ds/f'shot{k}_seed{s:02d}.npz'
def fit_job(job):
    mode,bb,ds,k,s=job;pin();guards();lock=read(freeze_path(mode));assert lock['all_four_methods_locked']
    path=fit_path(mode,bb,ds,k,s)
    if path.exists():
        assert sha(path)==read(path.with_suffix('.json'))['sha256'];return str(path)
    swap=mode=='swap';ep=support(bb,ds,k,s,swap,True);save={'support_indices':ep['si'],'query_indices':ep['qi'],'support_y':ep['y']};metas={}
    for mi,method in enumerate(METHODS):
        cfg=lock['selected'][bb][method]['config'];start=time.perf_counter();model=fit(ep,method,cfg);fit_sec=time.perf_counter()-start
        ss=predict(ep,model);start=time.perf_counter();qq=predict(ep,model,True);query_sec=time.perf_counter()-start
        assert np.isfinite(ss).all() and np.isfinite(qq).all()
        save[f'm{mi}__support']=ss;save[f'm{mi}__query']=qq
        for key,value in model.items():
            if isinstance(value,np.ndarray):save[f'm{mi}__model_{key}']=value
        metas[method]=dict(config=cfg,kind=model['kind'],fit_time_s=fit_sec,query_infer_time_s=query_sec)
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**save);tmp.replace(path)
    dump(path.with_suffix('.json'),dict(sha256=sha(path),spec_sha256=SPEC_SHA,selection_sha256=sha(freeze_path(mode)),
      mode=mode,backbone=bb,cohort=ds,shot=k,seed=s,models=metas,query_labels_read_in_fit=False))
    return str(path)

def calibration_lock(mode):
    assert read(freeze_path(mode))['all_four_methods_locked'];ds='chapman' if mode=='main' else 'sph';swap=mode=='swap';out={}
    qi,y=H.labels(ds,swap)
    for bb in ['ECGFounder','MERL']:
        out[bb]={};files=[fit_path(mode,bb,ds,k,s) for k in [5,10] for s in range(10)]
        for mi,m in enumerate(METHODS):
            scores=[np.load(p)[f'm{mi}__query'] for p in files]
            def objective(log_t):
                temp=np.exp(log_t);return float(np.mean([np.mean(np.logaddexp(0,z/temp)-y*z/temp) for z in scores]))
            result=minimize_scalar(objective,bounds=(np.log(.05),np.log(20.)),method='bounded',options={'xatol':1e-10});assert result.success
            values=[];weights=[]
            for z in scores:
                for c in range(y.shape[1]):
                    v=z[y[:,c]==0,c];values.append(v);weights.append(np.full(len(v),1/(20*y.shape[1]*len(v))))
            v=np.concatenate(values);wt=np.concatenate(weights);ix=np.argsort(v,kind='stable');cut=float(v[ix[np.searchsorted(np.cumsum(wt[ix]),.95,side='left')]])
            spec=float(wt[v<=cut].sum()/wt.sum());assert spec>=.95-1e-12
            out[bb][m]=dict(T=float(np.exp(result.x)),development_nll=float(result.fun),raw_logit_threshold=cut,development_macro_specificity=spec)
    dump(OUT/f'calibration_thresholds_{mode}.lock.json',dict(selected=out,spec_sha256=SPEC_SHA,development=ds,
      external_labels_used=False,selection_sha256=sha(freeze_path(mode))))
    notice('CALIBRATION_LOCKED_'+mode)

def pool(fn,jobs,stage):
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(fn,j) for j in jobs]
        for i,f in enumerate(as_completed(futures),1):
            f.result()
            notice(stage,done=i,total=len(jobs))

def run_mode(mode):
    guards()
    jobs=[(mode,bb,k,s,mi) for mi in [0,2,3] for bb in ['ECGFounder','MERL'] for k in [5,10] for s in range(10)]
    pool(grid_job,jobs,'GRID_'+mode);select(mode,[0,2,3])
    pool(grid_job,[(mode,bb,k,s,1) for bb in ['ECGFounder','MERL'] for k in [5,10] for s in range(10)],'TIPF_GRID_'+mode);select(mode,[1])
    dev='chapman' if mode=='main' else 'sph'
    pool(fit_job,[(mode,bb,dev,k,s) for bb in ['ECGFounder','MERL'] for k in [5,10] for s in range(10)],'DEV_FITS_'+mode)
    calibration_lock(mode)
    targets=['ningbo','cpsc2018','sph'] if mode=='main' else ['chapman','cpsc2018','ningbo']
    pool(fit_job,[(mode,bb,ds,k,s) for bb in ['ECGFounder','MERL'] for ds in targets for k in [5,10] for s in range(20)],'EXTERNAL_FITS_'+mode)
    dump(RUN/f'{mode.upper()}_FITS_COMPLETE.json',dict(spec_sha256=SPEC_SHA,selection_sha256=sha(freeze_path(mode)),development_fits=40,external_fits=240,methods=METHODS))

if __name__=='__main__':
    try:
        mode=sys.argv[1] if len(sys.argv)>1 else 'main';assert mode in ['main','swap'];run_mode(mode)
    except Exception:
        dump(RUN/'status'/'RUNNER_ERROR.json',dict(traceback=traceback.format_exc(),argv=sys.argv));raise
