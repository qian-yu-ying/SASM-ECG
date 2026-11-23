"""Isolated single-core adaptation and deployable cached-feature inference."""
from runner import configuration,fit_path
from adapters import *
import gc

def fold(ep,w,b):return w/ep['sd'][:,None],b-(ep['mu']/ep['sd'])@w

def lp_coeff(ep):
    bb,ds,k,s,swap=(ep[v] for v in ['bb','ds','shot','seed','swap'])
    if ds=='cpsc2018':
        z=np.load(H.O/'linear'/bb/'NORM'/f'shot{k}_seed{s:02d}.weights.npz');return z['LP__w'],z['LP__b']
    base=H.SW if swap else H.B;phase='dev' if ds==('sph' if swap else 'chapman') else 'external'
    rel=read(base/('release_01.json' if bb=='ECGFounder' else 'merl/release.json'));ci=rel['selected']['LP']['candidate'];d=base if bb=='ECGFounder' else base/'merl'
    z=np.load(d/'results'/phase/ds/'LP'/f'c{ci:02d}_shot{k}_seed{s:02d}.weights.npz')
    return z['weight'].mean(0),z['bias'].mean(0)

def compile_predictor(ep,model,lp=None):
    """No precomputed query logits; count all normalization/prior matmuls."""
    if model['kind']=='tip':
        w=ep['raw'].copy();b=ep['off'].copy()
        if not ep['matched'].all():
            lw,lb=lp_coeff(ep) if lp is None else lp;rw,rb=fold(ep,lw,lb)
            w[:,~ep['matched']]=rw[:,~ep['matched']];b[~ep['matched']]=rb[~ep['matched']]
        keys=unit_rows(model['keys']) if model['normalize_keys'] else model['keys']
        val=2*model['values']-1 if model['encoding']=='signed' else model['values']
        def predict_raw(h):
            u=unit_rows(h if model['space']=='raw_l2' else (h-ep['mu'])/ep['sd'])
            affinity=np.exp(-model['beta']*(1.-u@keys.T))
            return h@w+b+model['alpha']*(affinity@val)/(affinity.sum(1,keepdims=True)+1e-8)+model['delta_b']
        return predict_raw
    if model['kind']=='linear':
        w,b=fold(ep,model['weight'],model['bias']);return lambda h:h@w+b
    if model['space']=='std':
        w=model['weight']+ep['a']*model['mix'];b=model['bias']+ep['pb']*model['mix'];w,b=fold(ep,w,b)
        return lambda h:h@w+b
    w=model['weight'];b=model['bias']+ep['off']*model['mix'];rw=ep['raw']*model['mix']
    return lambda h:unit_rows(h)@w+h@rw+b

def baseline_fit(ep,method):
    alpha=100. if ep['bb']=='ECGFounder' else 10.
    x,y=ep['x'],ep['y'];lv=np.full(y.shape[1],10.);lv[~ep['matched']]=0.
    if method in ['LP','Direct-LP']:
        w,b,_=H.fit(x,y,alpha)
        if method=='Direct-LP':
            mix=.3 if ep['bb']=='ECGFounder' else .45;a=ep['a'].copy();pb=ep['pb'].copy()
            a[:,~ep['matched']]=w[:,~ep['matched']];pb[~ep['matched']]=b[~ep['matched']]
            w=(1-mix)*w+mix*a;b=(1-mix)*b+mix*pb
    elif method=='MeanAnchor':w,b,_=H.fit(x,y,.1,ep['a'],10.,lv,ep['pb'])
    else:
        _,raw,off,_=H.context(ep['bb'],ep['ds']);ws=[];bs=[]
        for a,ob in zip(raw,off):
            wa=ep['sd'][:,None]*a;pb=ep['mu']@a+ob;w,b,_=H.fit(x,y,.1,wa,10.,lv,pb);ws.append(w);bs.append(b)
        w=np.mean(ws,axis=0);b=np.mean(bs,axis=0)
    return dict(kind='linear',space='std',weight=w,bias=b)

def main():
    assert (RUN/'SWAP_FITS_COMPLETE.json').exists();pin(10)
    # Caller runs this stage serially, with no other experiment workers.
    cpu=next(v.split(':',1)[1].strip() for v in Path('/proc/cpuinfo').read_text().splitlines() if v.startswith('model name'))
    rows=[];rawrows=[];witness=[]
    for bb in ['ECGFounder','MERL']:
        for ds in ['ningbo','cpsc2018','sph']:
            for k in [5,10]:
                ep=support(bb,ds,k,0,False,True);lp=lp_coeff(ep) if ds=='cpsc2018' else None
                for m in METHODS+CONTROLS:
                    mi=METHODS.index(m) if m in METHODS else None;cfg=configuration('main',bb,mi) if mi is not None else None
                    def train():return fit(ep,m,cfg) if mi is not None else baseline_fit(ep,m)
                    model=train();fn=compile_predictor(ep,model,lp);ref=predict(ep,model,True);warm=fn(ep['hq'])
                    err=float(np.max(abs(warm-ref)));assert err<=1e-7,(bb,ds,k,m,err)
                    if mi is not None:
                        archive=np.load(fit_path('main',bb,ds,k,0));err2=float(abs(warm-archive[f'm{mi}__query']).max());assert err2<=1e-7
                    else:err2=None
                    witness.append(dict(backbone=bb,cohort=ds,shot=k,method=m,compiled_vs_native_max_logit_error=err,compiled_vs_archive_max_logit_error=err2))
                    times=[];infer=[]
                    for rep in range(10):
                        gc.collect();start=time.perf_counter();model=train();fn=compile_predictor(ep,model,lp);ft=time.perf_counter()-start
                        start=time.perf_counter();zz=fn(ep['hq']);it=time.perf_counter()-start;assert np.isfinite(zz).all()
                        times.append(ft);infer.append(it);rawrows.append(dict(backbone=bb,cohort=ds,shot=k,method=m,repeat=rep,fit_time_s=ft,query_infer_time_s=it))
                    row=dict(backbone=bb,cohort=ds,shot=k,method=m,CPU=cpu,CPU_affinity='10',BLAS_threads=1,torch_threads=1,warmup_runs=1,formal_runs=10,support_seed=0,support_n=len(ep['y']),query_n=len(ep['hq']),cached_frozen_features=True,
                        includes_encoder_forward=False,includes_hyperparameter_search=False,includes_query_label_access=False,
                        LP_fallback_coefficients='cached v1.4.1 LP; fallback refit excluded' if mi in [0,1] and ds=='cpsc2018' else 'not required',
                        fit_scope='cached support coordinates to deployable predictor; includes key normalization/weight folding; feature extraction and loading excluded',
                        inference_scope='raw cached h to full-query logits; prior projections and feature normalization included')
                    for name,v in [('fit_time_s',times),('query_infer_time_s',infer)]:row.update({name:float(np.median(v)),name+'_q25':float(np.quantile(v,.25)),name+'_q75':float(np.quantile(v,.75)),name+'_iqr':float(np.quantile(v,.75)-np.quantile(v,.25))})
                    rows.append(row);pd.DataFrame(rows).to_csv(OUT/'cpu_singlecore_latency.csv',index=False);pd.DataFrame(rawrows).to_csv(OUT/'cpu_singlecore_repeats.csv',index=False)
                    notice('CPU_TIMING',backbone=bb,cohort=ds,shot=k,method=m,fit=row['fit_time_s'],inference=row['query_infer_time_s'])
    dump(OUT/'cpu_deployable_predictor_witness.json',dict(all_passed=True,tests=witness,encoder_forward_excluded=True,CPSC_tip_LP_fallback_cached=True))
    dump(RUN/'TIMING_COMPLETE.json',dict(rows=len(rows),repeats=len(rawrows),CPU=cpu,single_core=True))

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback;dump(RUN/'status'/'TIMING_ERROR.json',dict(traceback=traceback.format_exc()));raise
