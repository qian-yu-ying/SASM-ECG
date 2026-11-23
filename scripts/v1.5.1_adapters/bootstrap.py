"""Paired query-unit resampling, averaging metrics over fixed support episodes."""
from core import *
from evaluate import load_predictions,rocpoint
from concurrent.futures import ProcessPoolExecutor,as_completed
BOOT_METHODS=METHODS+['MeanAnchor','SASM-Ens']

def weighted_rank(y,score,counts,inv,chunk=64,ap_only=False):
    order=np.argsort(-score,kind='stable');yy=y[order];ss=score[order]
    ends=np.r_[np.flatnonzero(ss[1:]!=ss[:-1]),len(ss)-1]
    result=np.full((len(counts),3),np.nan)
    for start in range(0,len(counts),chunk):
        w=np.ascontiguousarray(counts[start:start+chunk,inv[order]],dtype=float)
        tp=np.cumsum(w*yy,axis=1)[:,ends];total=np.cumsum(w,axis=1)[:,ends];fp=total-tp
        pp=tp[:,-1];nn=fp[:,-1];dt=np.diff(np.c_[np.zeros(len(w)),tp],axis=1)
        precision=np.divide(tp,total,out=np.zeros_like(tp),where=total>0)
        ap=np.divide((precision*dt).sum(1),pp,out=np.full(len(w),np.nan),where=pp>0)
        result[start:start+len(w),0]=ap
        if ap_only:continue
        dfp=np.diff(np.c_[np.zeros(len(w)),fp],axis=1)
        auc=np.divide(((2*tp-dt)*.5*dfp).sum(1),pp*nn,out=np.full(len(w),np.nan),where=(pp*nn)>0)
        # sklearn first removes zero-weight observations, then drops points
        # whose second differences in both TP and FP are zero. Apply this
        # exact operation to adjacent surviving score groups, not to padded
        # zero-weight groups. This matters at a discrete specificity cutoff.
        sen=np.full(len(w),np.nan)
        for r in range(len(w)):
            if pp[r]<=0 or nn[r]<=0:continue
            active=np.flatnonzero(np.diff(np.r_[0.,total[r]])>0)
            j=int(np.searchsorted(fp[r,active],nn[r]*(.05+1e-12),side='right'))-1
            # Only inspect the last eligible point and any dropped predecessors.
            # Other retained points cannot determine the constrained maximum.
            while 0<j<len(active)-1:
                a,b,c=active[j-1:j+2]
                if fp[r,c]-2*fp[r,b]+fp[r,a]!=0 or tp[r,c]-2*tp[r,b]+tp[r,a]!=0:break
                j-=1
            sen[r]=tp[r,active[j]]/pp[r] if j>=0 else 0.
        result[start:start+len(w)]=np.c_[ap,auc,sen]
    return result

def get_manifest(ds):return pd.read_csv(H.O/'manifest.csv' if ds=='cpsc2018' else H.B/'data'/ds/'manifest.csv')
def drawfile(ds,subgroup=None):return RUN/'draws'/f'{ds}{"_"+subgroup if subgroup else ""}.npz'

def prepare_draws():
    provenance=[]
    for ds in ['ningbo','cpsc2018','sph']:
        qi,y=H.labels(ds);tab=get_manifest(ds).iloc[qi].reset_index(drop=True)
        md=pd.read_csv(OLD/'demographics'/f'{ds}.csv').iloc[qi].reset_index(drop=True)
        groups=tab.patient_id.astype(str).to_numpy() if ds=='sph' else tab.record_id.astype(str).to_numpy() if ds=='ningbo' else tab.split_group.astype(str).to_numpy()
        unit='patient_id' if ds=='sph' else 'subject_record_not_verified_patient' if ds=='ningbo' else 'normalized_full_waveform_hash'
        unique,inv=np.unique(groups,return_inverse=True);ng=len(unique)
        if ds=='cpsc2018':assert ng==3966
        masks={'Male':md.sex.eq('Male').to_numpy(),'Female':md.sex.eq('Female').to_numpy(),'Age_lt60':md.age.to_numpy()<60,'Age_ge60':md.age.to_numpy()>=60}
        for name,mask,draws in [(None,np.ones(len(y),bool),2000)]+[(name,mask,500) for name,mask in masks.items()]:
            p=drawfile(ds,name);p.parent.mkdir(parents=True,exist_ok=True)
            # Main Ningbo uses released subject-record units; demographic slices
            # retain waveform grouping outside SPH. Neither implies patient IDs.
            subgroups=groups if name is None or ds=='sph' else md.group_id.astype(str).to_numpy()
            uniq,iv=np.unique(subgroups[mask],return_inverse=True);g=len(uniq)
            if not p.exists():
                rng=np.random.default_rng(20261122);counts=rng.multinomial(g,np.full(g,1/g),size=draws).astype(np.int16)
                np.savez_compressed(p,counts=counts,inverse=iv,query_indices=qi[mask],group_ids=uniq)
            z=np.load(p);assert np.array_equal(z['query_indices'],qi[mask]) and np.array_equal(z['inverse'],iv)
            provenance.append(dict(cohort=ds,subgroup=name,records=int(mask.sum()),groups=g,draws=draws,seed=20261122,
              unit=unit if name is None or ds=='sph' else 'normalized_full_waveform_hash_not_patient',file=str(p.relative_to(RUN)),sha256=sha(p)))
    dump(OUT/'bootstrap_draws_manifest.json',dict(provenance=provenance,new_sampling=True,
      reason='No archived cross-cohort shared index array matching this locked protocol was located; deterministic new draws as specified.',conditional_on_support_episodes=True,
      invalid_draw_policy='fixed label denominator; remove undefined entire replicate and report count; no imputation'))

def validate():
    rng=np.random.default_rng(151);n=30;inv=np.repeat(np.arange(10),3);counts=rng.multinomial(10,np.full(10,.1),size=25);y=rng.integers(0,2,n).astype(float);s=np.round(rng.normal(size=n),1)
    got=weighted_rank(y,s,counts,inv,chunk=7);errors=[]
    for i,row in enumerate(counts):
        w=row[inv];sen,ppv=rocpoint_weighted(y,s,w)
        ref=np.array([average_precision_score(y,s,sample_weight=w),roc_auc_score(y,s,sample_weight=w),sen]);errors.append(float(abs(got[i]-ref).max()))
    assert max(errors)<=1e-12
    dump(OUT/'bootstrap_implementation_witness.json',dict(passed=True,sklearn_max_error=max(errors),replicates=25,metrics=['AP','AUROC','Sensitivity95'],tied_scores=True,zero_weight_records=True))
def rocpoint_weighted(y,z,w):
    fpr,tpr,thr=roc_curve(y,z,sample_weight=w,drop_intermediate=True);i=np.flatnonzero(1-fpr>=.95-1e-12)[-1];pred=z>=thr[i];tp=float((w*y*pred).sum());fp=float((w*(1-y)*pred).sum())
    return float(tpr[i]),tp/(tp+fp) if tp+fp else 0.

def valid_draws(y,counts,inv,cols):
    ng=counts.shape[1];valid=np.ones(len(counts),bool)
    for c in cols:
        pos=counts@np.bincount(inv,weights=y[:,c],minlength=ng);neg=counts@np.bincount(inv,weights=1-y[:,c],minlength=ng)
        valid&=(pos>0)&(neg>0)
    return valid
def interval(values,valid):
    v=values[valid&np.isfinite(values)]
    if not len(v):return dict(ci_low=None,ci_high=None,valid_draws=0,total_draws=len(values),undefined_replicates=len(values),undefined_reason='no valid fixed-label replicate')
    return dict(ci_low=float(np.quantile(v,.025)),ci_high=float(np.quantile(v,.975)),valid_draws=len(v),total_draws=len(values),undefined_replicates=len(values)-len(v),undefined_reason=None)

def main_condition(job):
    bb,ds,k=job;pin();path=RUN/'query_ci'/bb/ds/f'shot{k}.json'
    if path.exists():return
    qi,y=H.labels(ds);z=np.load(drawfile(ds));counts=z['counts'];inv=z['inverse'];assert np.array_equal(qi,z['query_indices'])
    valid=valid_draws(y,counts,inv,range(y.shape[1]));B=len(counts);acc={m:np.zeros((B,3)) for m in BOOT_METHODS};point={m:np.zeros(3) for m in BOOT_METHODS}
    start=time.time()
    for s in range(20):
        pred,_,_,order=load_predictions('main',bb,ds,k,s);assert np.array_equal(qi,order)
        for m in BOOT_METHODS:
            for c in range(y.shape[1]):
                acc[m]+=weighted_rank(y[:,c],pred[m][:,c],counts,inv)/(20*y.shape[1])
                point[m]+=np.array([average_precision_score(y[:,c],pred[m][:,c]),roc_auc_score(y[:,c],pred[m][:,c]),rocpoint(y[:,c],pred[m][:,c])[0]])/(20*y.shape[1])
        notice('QUERY_CI_PROGRESS',backbone=bb,cohort=ds,shot=k,seeds=s+1,total=20,seconds=time.time()-start)
    rows=[]
    for ref in ['MeanAnchor','SASM-Ens']:
        for m in METHODS:
            for j,metric in enumerate(['ap','auc','sensitivity95']):
                delta=acc[ref][:,j]-acc[m][:,j]
                rows.append(dict(backbone=bb,cohort=ds,shot=k,reference=ref,comparator=m,metric=metric,difference=float(point[ref][j]-point[m][j]),
                  conditional_support_seeds=20,query_groups=counts.shape[1],**interval(delta,valid)))
    path.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(path.with_suffix('.csv'),index=False)
    np.savez_compressed(path.with_suffix('.npz'),**acc,valid=valid)
    dump(path,dict(complete=True,backbone=bb,cohort=ds,shot=k,seconds=time.time()-start,draws_sha256=sha(drawfile(ds)),bootstrap_sha256=sha(path.with_suffix('.npz')),rows=len(rows)))

def subgroup_condition(job):
    bb,ds,k=job;pin();path=RUN/'subgroup_ci'/bb/ds/f'shot{k}.json'
    if path.exists():return
    qi,y=H.labels(ds);md=pd.read_csv(OLD/'demographics'/f'{ds}.csv').iloc[qi].reset_index(drop=True)
    preds=[load_predictions('main',bb,ds,k,s)[0] for s in range(20)]
    methods=METHODS+CONTROLS;rows=[];counts_rows=[];seedrows=[]
    masks={'Male':md.sex.eq('Male').to_numpy(),'Female':md.sex.eq('Female').to_numpy(),'Age_lt60':md.age.to_numpy()<60,'Age_ge60':md.age.to_numpy()>=60}
    for family,names in [('sex',['Male','Female']),('age',['Age_lt60','Age_ge60'])]:
        common=np.ones(y.shape[1],bool)
        for name in names:
            yy=y[masks[name]];common&=(yy.sum(0)>0)&(yy.sum(0)<len(yy))
        cols=np.flatnonzero(common);assert len(cols)>0
        for name in names:
            mask=masks[name];yy=y[mask];d=np.load(drawfile(ds,name));co=d['counts'];iv=d['inverse'];assert np.array_equal(qi[mask],d['query_indices'])
            valid=valid_draws(yy,co,iv,cols);acc={m:np.zeros(len(co)) for m in methods};points={m:[] for m in methods}
            for c in range(y.shape[1]):counts_rows.append(dict(backbone=bb,cohort=ds,shot=k,subgroup=name,label=c,n=len(yy),positives=int(yy[:,c].sum()),included_in_common_macro=bool(common[c])))
            for s in range(20):
                for m in methods:
                    sc=preds[s][m][mask];ap=float(np.mean([average_precision_score(yy[:,c],sc[:,c]) for c in cols]));points[m].append(ap)
                    seedrows.append(dict(backbone=bb,cohort=ds,shot=k,subgroup=name,seed=s,method=m,ap=ap,common_classes=len(cols)))
                    for c in cols:acc[m]+=weighted_rank(yy[:,c],sc[:,c],co,iv,ap_only=True)[:,0]/(20*len(cols))
            for m in methods:rows.append(dict(backbone=bb,cohort=ds,shot=k,subgroup=name,family=family,method=m,n=len(yy),common_classes=len(cols),ap_mean=float(np.mean(points[m])),ap_sd=float(np.std(points[m],ddof=1)),**interval(acc[m],valid)))
            notice('SUBGROUP_CI_PROGRESS',backbone=bb,cohort=ds,shot=k,subgroup=name,valid=int(valid.sum()),draws=len(valid))
    path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(path.with_suffix('.csv'),index=False);pd.DataFrame(counts_rows).to_csv(path.with_suffix('.counts.csv'),index=False);pd.DataFrame(seedrows).to_csv(path.with_suffix('.seeds.csv'),index=False)
    dump(path,dict(complete=True,rows=len(rows),sha256=sha(path.with_suffix('.csv'))))

def pool(fn,jobs,stage):
    with ProcessPoolExecutor(max_workers=4) as ex:
        for i,f in enumerate(as_completed([ex.submit(fn,j) for j in jobs]),1):f.result();notice(stage,done=i,total=len(jobs))
def main():
    assert (RUN/'MAIN_FITS_COMPLETE.json').exists();pin(10);validate();prepare_draws()
    jobs=[(bb,ds,k) for bb in ['ECGFounder','MERL'] for ds in ['ningbo','cpsc2018','sph'] for k in [5,10]]
    pool(main_condition,jobs,'QUERY_CI_CONDITIONS')
    pd.concat([pd.read_csv(RUN/'query_ci'/bb/ds/f'shot{k}.csv') for bb,ds,k in jobs],ignore_index=True).to_csv(OUT/'main_query_cluster_ci.csv',index=False)
    pool(subgroup_condition,jobs,'SUBGROUP_CI_CONDITIONS')
    pd.concat([pd.read_csv(RUN/'subgroup_ci'/bb/ds/f'shot{k}.csv') for bb,ds,k in jobs],ignore_index=True).to_csv(OUT/'subgroup_48slices_summary.csv',index=False)
    pd.concat([pd.read_csv(RUN/'subgroup_ci'/bb/ds/f'shot{k}.counts.csv') for bb,ds,k in jobs],ignore_index=True).to_csv(OUT/'subgroup_counts.csv',index=False)
    dump(RUN/'BOOTSTRAPS_COMPLETE.json',dict(query_draws=2000,subgroup_draws=500,conditions=12,draws_manifest_sha256=sha(OUT/'bootstrap_draws_manifest.json')));notice('ALL_BOOTSTRAPS_COMPLETE')

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback;dump(RUN/'status'/'BOOTSTRAP_ERROR.json',dict(traceback=traceback.format_exc()));raise
