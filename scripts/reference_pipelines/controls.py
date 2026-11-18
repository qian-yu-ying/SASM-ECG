"""Remote-only, frozen-input source-transfer controls; no training on query labels."""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,time,hashlib,fcntl,traceback,concurrent.futures,multiprocessing
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from threadpoolctl import threadpool_limits,threadpool_info
OUT=Path(__file__).resolve().parent
BASE=OUT.parent/'chapman_dev_native_same_20261103'
sys.path.insert(0,str(BASE));sys.path.insert(0,str(BASE/'source'))
import core as C
import cpu_worker as CPU
import merl as MERL
from target_contract import certified_fit,objective
LIMIT=threadpool_limits(limits=1)
BB=['ECGFounder','MERL'];DS=['ningbo','cpsc2018','sph']
GRID=[dict(alpha=a,lam=l) for a in [.1,1.,10.] for l in [.1,1.,10.]]
DIRECT='Mapped-source-direct';TUNED='No-scale-tuned-Ens'
CTX={};SEALED=None
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+f'.{os.getpid()}.tmp')
    q.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');q.replace(p)
def sealmap():return read(OUT/'protocol.lock.json')['input_files']
def checkseal(p):
    global SEALED
    if SEALED is None:SEALED=sealmap()
    s=str(Path(p));assert s in SEALED and sha(p)==SEALED[s],f'input drift: {s}'
def epfile(ds,k,seed,phase):return BASE/'data'/ds/'episodes'/f'{phase}_shot{k}_seed{seed:02d}.json'
def path(j):return OUT/'models'/j['phase']/j['backbone']/j['dataset']/f"c{j['candidate']:02d}_k{j['shot']}_s{j['seed']:02d}.json"
def context(bb,ds):
    key=(bb,ds)
    if key in CTX:return CTX[key]
    fdir=BASE/('data' if bb=='ECGFounder' else 'merl/data')/ds
    F=np.load(fdir/'features.npy',mmap_mode='r');heads=[]
    if bb=='ECGFounder':
        q=np.load(BASE/'data'/ds/'Q.npy');dw=np.load(BASE/'assets/dense.npz')['Dw']
        _,converted,_=CPU.context(ds)
        for e,(raw,off) in enumerate(converted):
            z=np.load(BASE/'source/source_heads'/f'ep{e:02d}_bce_signed_model.npz')
            heads.append(dict(raw=raw,off=off,no_scale=dw.T@(z['weight']@q)))
    else:
        _,converted,_=MERL.context(ds)
        conf=read(BASE.parent/'merl_source/protocol.lock.json')
        inv={v:i for i,v in enumerate(conf['source_label_keep75'])}
        cols=[inv[c] for c in read(BASE/'data'/ds/'metadata.json')['cols']]
        for e,(raw,off) in enumerate(converted):
            z=np.load(BASE.parent/'merl_source/source_heads'/f'ep{e:02d}.npz')
            heads.append(dict(raw=raw,off=off,no_scale=z['weight'][:,cols].copy()))
    assert len(heads)==10 and all(h['raw'].shape==h['no_scale'].shape for h in heads)
    assert F.shape[1]==heads[0]['raw'].shape[0]
    CTX[key]=(F,heads);return F,heads
def arrays(j):
    p=epfile(j['dataset'],j['shot'],j['seed'],j['phase']);checkseal(p);e=read(p)
    si=np.asarray(e['support']);qi=np.asarray(e['query']);ys=np.asarray(e['support_y'],float)
    assert not set(si)&set(qi)
    F,heads=context(j['backbone'],j['dataset']);fs=F[si].astype(float)
    mu=fs.mean(0);sd=fs.std(0)+1e-6
    return e,si,qi,ys,mu,sd,(fs-mu)/sd,(F[qi].astype(float)-mu)/sd,heads
def gradients(xs,ys,weights,biases,anchors,cfg):
    active=np.flatnonzero((ys.sum(0)>0)&(ys.sum(0)<len(ys)))
    worst=0.
    for w,b,a in zip(weights,biases,anchors):
        if len(active):
            fn=objective(xs,ys[:,active],cfg['alpha'],a[:,active],cfg['lam'])
            _,g=fn(np.r_[w[:,active].ravel(),b[active]])
            worst=max(worst,float(abs(g).max()))
    assert worst<=5e-5,('gradient',worst)
    return worst
def predict_train(j,save=True):
    p=path(j)
    if p.exists():
        m=read(p);assert m['job']==j
        for suf,key in [('.weights.npz','weights_sha256'),('.npz','predictions_sha256')]:assert sha(p.with_suffix(suf))==m[key]
        return str(p)
    started=time.time();e,si,qi,ys,mu,sd,xs,xq,heads=arrays(j);cfg=j['config']
    weights=[];biases=[];diags=[];anchors=[]
    for h in heads:
        a=h['no_scale'];bp=mu@h['raw']+h['off']
        w,b,d=certified_fit(xs,ys,cfg['alpha'],W_anchor=a,lam=cfg['lam'])
        const=np.flatnonzero((ys.sum(0)==0)|(ys.sum(0)==len(ys)))
        w[:,const]=a[:,const];b[const]=bp[const]
        weights.append(w);biases.append(b);diags.append(d);anchors.append(a)
    w=np.asarray(weights);b=np.asarray(biases)
    pred=xq@w.mean(0)+b.mean(0)
    replay=sum((xq@ww+bb)/10 for ww,bb in zip(w,b))
    err=float(abs(pred-replay).max());assert err<1e-9 and np.isfinite(pred).all()
    grad=gradients(xs,ys,w,b,anchors,cfg)
    p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p.with_suffix('.weights.npz'),weight=w,bias=b,scaler_mean=mu,scaler_sd=sd)
    np.savez_compressed(p.with_suffix('.npz'),logits=pred,query_indices=qi,support_indices=si)
    m=dict(job=j,seconds=time.time()-started,episode_sha256=sha(epfile(j['dataset'],j['shot'],j['seed'],j['phase'])),
           protocol_sha256=sha(OUT/'protocol.lock.json'),weights_sha256=sha(p.with_suffix('.weights.npz')),
           predictions_sha256=sha(p.with_suffix('.npz')),max_gradient=grad,head_replay_error=err,diagnostics=diags,
           fits=10,worker_pid=os.getpid(),query_labels_loaded=False)
    if j['phase']=='dev':
        v=np.load(BASE/'data/chapman/development_labels.npz');assert np.array_equal(qi,v['indices'])
        m['development_score']=C.metric(v['y'],pred);assert m['development_score']['ap'] is not None
        m['query_labels_loaded']='Chapman development only after fitting'
    dump(p,m);return str(p)
def init_worker(cpus):
    global LIMIT
    ident=multiprocessing.current_process()._identity
    cpu=cpus[((ident[-1] if ident else 1)-1)%len(cpus)]
    os.sched_setaffinity(0,{cpu});os.nice(10);LIMIT=threadpool_limits(limits=1)
def pool_jobs(jobs,stage,workers=24):
    cpus=sorted(os.sched_getaffinity(0));workers=min(workers,max(1,len(cpus)//2))
    done=[];total=len(jobs)
    with concurrent.futures.ProcessPoolExecutor(workers,initializer=init_worker,initargs=(cpus,)) as ex:
        fs={ex.submit(predict_train,j):j for j in jobs}
        for future in concurrent.futures.as_completed(fs):
            done.append(future.result())
            if len(done)%12==0 or len(done)==total:
                dump(OUT/'status.json',dict(stage=stage,completed=len(done),total=total,time=time.time(),pid=os.getpid()))
                print(stage,len(done),total,flush=True)
    return done
def make_job(bb,ds,k,seed,candidate,phase,cfg):
    return dict(backbone=bb,dataset=ds,shot=k,seed=seed,candidate=candidate,phase=phase,config=cfg)
def reference_file(bb,ds,k,seed):
    rel=read(BASE/('release_01.json' if bb=='ECGFounder' else 'merl/release.json'))
    candidate=rel['selected']['SASM-Ens']['candidate'];root=BASE if bb=='ECGFounder' else BASE/'merl'
    return root/'results/external'/ds/'SASM-Ens'/f'c{candidate:02d}_shot{k}_seed{seed:02d}.json'
def verify_reference(bb,ds,k,seed):
    p=reference_file(bb,ds,k,seed);m=read(p)
    assert sha(p.with_suffix('.npz'))==m['prediction_sha256'] and sha(p.with_suffix('.weights.npz'))==m['model_sha256']
    j=make_job(bb,ds,k,seed,0,'external',m['job']['config'])
    _,si,qi,ys,mu,sd,xs,xq,heads=arrays(j)
    z=np.load(p.with_suffix('.npz'));wz=np.load(p.with_suffix('.weights.npz'))
    assert np.array_equal(si,z['support_indices']) and np.array_equal(qi,z['query_indices'])
    assert np.allclose(mu,wz['scaler_mean'],rtol=0,atol=1e-12) and np.allclose(sd,wz['scaler_sd'],rtol=0,atol=1e-12)
    w=wz['weight'];b=wz['bias'];assert len(w)==10
    pred=xq@w.mean(0)+b.mean(0);err=float(abs(pred-z['logits']).max());assert err<1e-9
    grad=gradients(xs,ys,w,b,[sd[:,None]*h['raw'] for h in heads],j['config'])
    return dict(backbone=bb,dataset=ds,shot=k,seed=seed,model_metadata_sha256=sha(p),prediction_error=err,max_gradient=grad),pred,qi
def preflight():
    seals={}
    def add(p,expected=None):
        p=Path(p);h=sha(p)
        if expected is not None:assert h==expected,('hash mismatch',str(p),expected,h)
        seals[str(p)]=h
    complete=read(BASE/'ALL_COMPLETE.json')
    for n,h in complete['files'].items():add(BASE/n,h)
    add(BASE/'ALL_COMPLETE.json');add(BASE/'INPUTS_READY.json')
    inp=read(BASE/'INPUTS_READY.json');proto=read(BASE/'protocol.lock.json');mp=read(BASE/'merl/protocol.lock.json')
    for n,h in proto['source_hashes'].items():add(BASE/'source/source_heads'/n,h)
    for n,h in proto['asset_hashes'].items():add(BASE/'source'/n,h)
    for asset in mp['source_heads']+[mp['source_protocol']]:add(asset['path'],asset['sha256'])
    for n,h in mp['code_hashes'].items():add(BASE/n,h)
    add(BASE/'assets/dense.npz')
    for ds,meta in inp['datasets'].items():
        for name,field in [('manifest.csv','manifest_sha256'),('metadata.json','metadata_sha256'),('Q.npy','Q_sha256')]:add(BASE/'data'/ds/name,meta[field])
        for sub,field in [('data','ECG_feature_sha256'),('merl/data','MERL_feature_sha256')]:
            f=BASE/sub/ds/'features.npy';add(f,meta[field]);add(f.parent/'FEATURES_READY.json')
        for n,h in meta['episodes'].items():add(BASE/'data'/ds/'episodes'/n,h)
        for n,h in meta['vault'].items():add(BASE/'vault'/ds/n,h)
    add(BASE/'data/chapman/development_labels.npz')
    code=sha(__file__)
    proto_new=dict(created=time.time(),base=str(BASE),backbones=BB,external=DS,shots=[5,10],dev_seeds=10,eval_seeds=20,
        grid=GRID,methods=[DIRECT,TUNED],selection='Chapman mean AP over 5/10shot x ten seeds; tolerance1e-4, smaller lambda then alpha',
        loss=proto['loss'],direct='mean ten mapped affine source logits; no support fitting or target statistics',
        no_scale={'ECGFounder':'Dw.T @ (Ws @ Q)','MERL':'mapped standardized source weights'},
        prior_bias='full affine conversion retained for constant-label fallback; active intercept free',
        comparisons='SASM minus each new control; Holm 12 external contrasts per backbone/metric; paired-seed CI10000; patient AP CI2000',
        retrospective=True,pretraining_overlap='unknown',CPSC_patient_identity='record-level limitation',input_files=seals,
        code_sha256=code,source_head_count=10,workers=24,cpu_only=True)
    p=OUT/'protocol.lock.json'
    if p.exists():assert read(p)['input_files']==seals and read(p)['code_sha256']==code
    else:dump(p,proto_new)
    smoke=[]
    for bb in BB:
        for ds in DS:
            check,_,_=verify_reference(bb,ds,5,0);smoke.append(check)
    # Numerically reproduce the existing exact `none` ablation, without importing its results as new fits.
    j=make_job('ECGFounder','cpsc2018',5,0,8,'dev',GRID[8])
    j.update(phase='external',candidate=6,config=dict(alpha=.1,lam=10.))
    _,si,qi,ys,mu,sd,xs,xq,heads=arrays(j)
    cfg=j['config'];pred=np.zeros((len(qi),ys.shape[1]))
    for h in heads:
        a=h['no_scale'];w,b,_=certified_fit(xs,ys,cfg['alpha'],W_anchor=a,lam=cfg['lam'])
        c=np.flatnonzero((ys.sum(0)==0)|(ys.sum(0)==len(ys)));w[:,c]=a[:,c];b[c]=(mu@h['raw']+h['off'])[c]
        pred+=(xq@w+b)/10
    old=BASE/'ablation/results/ECGFounder/none/cpsc2018/shot5_seed00.npz'
    z=np.load(old);err=float(abs(pred-z['logits']).max());assert err<1e-9,('none regression',err)
    smoke.append(dict(test='exact historical none regression',max_prediction_error=err))
    dump(OUT/'PREFLIGHT_PASS.json',dict(time=time.time(),input_count=len(seals),code_sha256=code,checks=smoke,
         environment=C.read(BASE/'protocol.lock.json')['environment'],threadpools=threadpool_info(),cpu_affinity=sorted(os.sched_getaffinity(0))))
    print('PREFLIGHT_PASS',len(seals),flush=True)
def direct_predictions():
    checks=[]
    for bb in BB:
        for ds in DS:
            F,heads=context(bb,ds);ep=read(epfile(ds,5,0,'external'));qi=np.array(ep['query'])
            raw=np.mean([h['raw'] for h in heads],0);off=np.mean([h['off'] for h in heads],0)
            fq=F[qi].astype(float);pred=fq@raw+off
            replay=sum((fq@h['raw']+h['off'])/10 for h in heads)
            err=float(abs(pred-replay).max());assert err<1e-9 and np.isfinite(pred).all()
            for k in [5,10]:
                for s in range(20):assert np.array_equal(qi,read(epfile(ds,k,s,'external'))['query'])
            p=OUT/'direct'/bb/f'{ds}.npz';p.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(p,logits=pred,query_indices=qi)
            checks.append(dict(backbone=bb,dataset=ds,unique_prediction_file=str(p),sha256=sha(p),affine_replay_error=err,
                               uses_target_support_labels=False,uses_target_support_statistics=False,head_count=10))
    dump(OUT/'direct_verification.json',dict(checks=checks))
def select():
    rows=[];selected={}
    for bb in BB:
        group=[]
        for c,cfg in enumerate(GRID):
            scores=[]
            for k in [5,10]:
                for seed in range(10):
                    j=make_job(bb,'chapman',k,seed,c,'dev',cfg);m=read(path(j));assert m['job']==j
                    assert sha(path(j).with_suffix('.npz'))==m['predictions_sha256']
                    scores.append(m['development_score']['ap'])
            row=dict(backbone=bb,candidate=c,config=cfg,ap_mean=float(np.mean(scores)),episodes=20)
            rows.append(row);group.append(row)
        best=max(r['ap_mean'] for r in group);near=[r for r in group if best-r['ap_mean']<=1e-4]
        selected[bb]=min(near,key=lambda r:(r['config']['lam'],r['config']['alpha']))
    dump(OUT/'development_candidates.json',rows)
    pd.DataFrame([dict(backbone=r['backbone'],candidate=r['candidate'],alpha=r['config']['alpha'],lam=r['config']['lam'],ap_mean=r['ap_mean']) for r in rows]).to_csv(OUT/'development_candidates.csv',index=False)
    release=dict(selected=selected,protocol_sha256=sha(OUT/'protocol.lock.json'),time=time.time(),external_labels_used=False)
    if (OUT/'release.json').exists():assert read(OUT/'release.json')['selected']==selected
    else:dump(OUT/'release.json',release)
    print('RELEASE',json.dumps(selected),flush=True);return selected
def verify_model(j):
    p=path(j);m=read(p);assert m['job']==j and m['protocol_sha256']==sha(OUT/'protocol.lock.json')
    assert sha(p.with_suffix('.npz'))==m['predictions_sha256'] and sha(p.with_suffix('.weights.npz'))==m['weights_sha256']
    _,si,qi,ys,mu,sd,xs,xq,heads=arrays(j)
    z=np.load(p.with_suffix('.npz'));wz=np.load(p.with_suffix('.weights.npz'))
    assert np.array_equal(qi,z['query_indices']) and np.array_equal(si,z['support_indices'])
    assert np.allclose(mu,wz['scaler_mean'],rtol=0,atol=1e-12) and np.allclose(sd,wz['scaler_sd'],rtol=0,atol=1e-12)
    pred=xq@wz['weight'].mean(0)+wz['bias'].mean(0);err=float(abs(pred-z['logits']).max());assert err<1e-9
    g=gradients(xs,ys,wz['weight'],wz['bias'],[h['no_scale'] for h in heads],j['config'])
    const=np.flatnonzero((ys.sum(0)==0)|(ys.sum(0)==len(ys)))
    for h,w,b in zip(heads,wz['weight'],wz['bias']):
        assert np.array_equal(w[:,const],h['no_scale'][:,const])
        assert np.array_equal(b[const],(mu@h['raw']+h['off'])[const])
    return dict(backbone=j['backbone'],dataset=j['dataset'],shot=j['shot'],seed=j['seed'],prediction_error=err,max_gradient=g,
                metadata_sha256=sha(p),weights_sha256=m['weights_sha256'],predictions_sha256=m['predictions_sha256']),pred,qi
def finish_metrics(selected):
    rows=[];per=[];checks=[]
    for bb in BB:
        cfg=selected[bb]['config'];c=selected[bb]['candidate']
        for ds in DS:
            v=np.load(BASE/'vault'/ds/'external.npz',allow_pickle=True);y=v['y']
            direct=np.load(OUT/'direct'/bb/f'{ds}.npz');assert np.array_equal(v['indices'],direct['query_indices'])
            dscore=C.metric(y,direct['logits'])
            for k in [5,10]:
                for s in range(20):
                    j=make_job(bb,ds,k,s,c,'external',cfg)
                    check,pred,qi=verify_model(j);assert np.array_equal(qi,v['indices']);checks.append(check)
                    rcheck,rpred,rqi=verify_reference(bb,ds,k,s);assert np.array_equal(rqi,qi);checks.append(rcheck)
                    scores=[(TUNED,C.metric(y,pred)),('SASM-Ens',C.metric(y,rpred)),(DIRECT,dscore)]
                    for method,score in scores:
                        assert score['ap'] is not None and score['auc'] is not None
                        rows.append(dict(backbone=bb,dataset=ds,shot=k,seed=s,method=method,ap=score['ap'],auc=score['auc']))
                        for i,label in enumerate(read(BASE/'data'/ds/'metadata.json')['labels']):
                            per.append(dict(backbone=bb,dataset=ds,shot=k,seed=s,method=method,label=label,
                                            ap=score['ap_per_label'][i],auc=score['auc_per_label'][i]))
            print('VERIFIED_METRICS',bb,ds,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'episode_metrics.csv',index=False)
    pd.DataFrame(per).to_csv(OUT/'per_label_metrics.csv',index=False)
    summary=frame.groupby(['backbone','dataset','shot','method']).agg(n=('ap','size'),ap_mean=('ap','mean'),ap_sd=('ap','std'),auc_mean=('auc','mean'),auc_sd=('auc','std')).reset_index()
    summary['seed_unit']=np.where(summary.method==DIRECT,'same deterministic prediction repeated for pairing; not twenty independent direct runs','twenty paired target support seeds')
    summary.to_csv(OUT/'summary.csv',index=False)
    pairs=[];rng=np.random.default_rng(20261105)
    for (bb,ds,k),g in frame.groupby(['backbone','dataset','shot']):
        a=g[g.method=='SASM-Ens'].sort_values('seed')
        for method in [DIRECT,TUNED]:
            b=g[g.method==method].sort_values('seed');assert a.seed.tolist()==b.seed.tolist()==list(range(20))
            for metric in ['ap','auc']:
                delta=a[metric].to_numpy()-b[metric].to_numpy();boots=delta[rng.integers(0,20,(10000,20))].mean(1)
                pairs.append(dict(backbone=bb,dataset=ds,shot=int(k),comparator=method,metric=metric,
                    mean_difference_pp=float(delta.mean()*100),ci_low_pp=float(np.quantile(boots,.025)*100),ci_high_pp=float(np.quantile(boots,.975)*100),
                    wins=int((delta>0).sum()),ties=int((delta==0).sum()),p=float(wilcoxon(delta).pvalue) if np.any(delta) else 1.))
    t=pd.DataFrame(pairs)
    for _,g in t.groupby(['backbone','metric']):
        ix=g.sort_values('p').index;p=t.loc[ix,'p'].to_numpy();assert len(p)==12
        t.loc[ix,'p_holm']=np.minimum(1,np.maximum.accumulate(p*(len(p)-np.arange(len(p)))))
    t.to_csv(OUT/'paired_effects_CI.csv',index=False)
    dump(OUT/'verification_manifest.json',dict(checks=checks,new_models=240,reference_replays=240,zero_shot_unique_predictions=6,
             release_sha256=sha(OUT/'release.json'),protocol_sha256=sha(OUT/'protocol.lock.json')))
    dump(OUT/'METRICS_COMPLETE.json',dict(time=time.time(),rows=len(rows),new_models=240,reference_replays=240))
def ap_boot(y,score,counts,inv,batch=16):
    reps=len(counts);value=np.zeros(reps)
    for col in range(y.shape[1]):
        order=np.argsort(-score[:,col],kind='stable');yy=y[order,col];ss=score[order,col];ix=inv[order]
        ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1]
        for start in range(0,reps,batch):
            w=counts[start:start+batch,ix].astype(float)
            tp=np.cumsum(w*yy[None,:],axis=1);den=np.cumsum(w,axis=1)
            t=tp[:,ends];d=den[:,ends];inc=np.diff(np.c_[np.zeros(len(w)),t],axis=1)
            prec=np.divide(t,d,out=np.zeros_like(t),where=d>0)
            result=np.divide((inc*prec).sum(1),tp[:,-1],out=np.full(len(w),np.nan),where=tp[:,-1]>0)
            value[start:start+batch]+=result/y.shape[1]
    return value
def patient_one(pair):
    bb,ds,k=pair;p=OUT/'patient'/f'{bb}_{ds}_{k}.json'
    if p.exists():return read(p)['rows']
    rel=read(OUT/'release.json')['selected'][bb];v=np.load(BASE/'vault'/ds/'external.npz',allow_pickle=True)
    y=v['y'];uniq,inv=np.unique(v['patients'].astype(str),return_inverse=True);n=len(uniq)
    rng=np.random.default_rng(20261105);counts=rng.multinomial(n,np.full(n,1/n),size=2000).astype(np.int16)
    valid=np.ones(2000,bool)
    for col in range(y.shape[1]):
        pos=counts@np.bincount(inv,weights=y[:,col],minlength=n)
        neg=counts@np.bincount(inv,weights=1-y[:,col],minlength=n);valid&=(pos>0)&(neg>0)
    # Witness tied/untied weighted AP against sklearn before bootstrapping.
    from sklearn.metrics import average_precision_score
    direct=np.load(OUT/'direct'/bb/f'{ds}.npz')['logits']
    witness=ap_boot(y,direct,np.ones((1,n),np.int16),inv)[0]
    expected=np.mean([average_precision_score(y[:,c],direct[:,c]) for c in range(y.shape[1])])
    assert abs(witness-expected)<1e-12,(witness,expected)
    distributions={DIRECT:ap_boot(y,direct,counts,inv),'SASM-Ens':np.zeros(2000),TUNED:np.zeros(2000)}
    for s in range(20):
        j=make_job(bb,ds,k,s,rel['candidate'],'external',rel['config'])
        for method,zp in [('SASM-Ens',reference_file(bb,ds,k,s).with_suffix('.npz')),(TUNED,path(j).with_suffix('.npz'))]:
            z=np.load(zp);assert np.array_equal(z['query_indices'],v['indices'])
            distributions[method]+=ap_boot(y,z['logits'],counts,inv)/20
        if s%5==4:print('PATIENT_PROGRESS',bb,ds,k,s+1,flush=True)
    rows=[]
    for method in [DIRECT,TUNED]:
        delta=distributions['SASM-Ens']-distributions[method];good=valid&np.isfinite(delta)
        assert good.mean()>=.95
        lo,hi=np.quantile(delta[good],[.025,.975])*100
        rows.append(dict(backbone=bb,dataset=ds,shot=k,baseline=method,metric='ap',replications=2000,valid_replications=int(good.sum()),
            patient_CI_low_pp=float(lo),patient_CI_high_pp=float(hi),scope='query patient-cluster sampling conditional on twenty fixed support sets; CPSC record-level linkage limitation'))
    p.parent.mkdir(exist_ok=True)
    np.savez_compressed(p.with_suffix('.npz'),valid=valid,**{m:a for m,a in distributions.items()})
    dump(p,dict(rows=rows,weighted_AP_unit_weight_error=float(abs(witness-expected)),distribution_sha256=sha(p.with_suffix('.npz'))))
    return rows
def patient_all():
    jobs=[(bb,ds,k) for bb in BB for ds in DS for k in [5,10]];rows=[];cpus=sorted(os.sched_getaffinity(0))
    with concurrent.futures.ProcessPoolExecutor(12,initializer=init_worker,initargs=(cpus,)) as ex:
        fs={ex.submit(patient_one,j):j for j in jobs}
        for f in concurrent.futures.as_completed(fs):
            rows.extend(f.result());dump(OUT/'status.json',dict(stage='patient_bootstrap',completed_groups=len(rows)//2,total_groups=12,pid=os.getpid(),time=time.time()))
            print('PATIENT_COMPLETE',fs[f],flush=True)
    pd.DataFrame(rows).to_csv(OUT/'patient_bootstrap_CI.csv',index=False)
def final_report():
    s=pd.read_csv(OUT/'summary.csv');p=pd.read_csv(OUT/'paired_effects_CI.csv');patients=pd.read_csv(OUT/'patient_bootstrap_CI.csv')
    lines=['# 两项源迁移补实验：最终结果','',
      '冻结输入来自2026-11-03 Chapman开发重跑。补实验在观察既有外部结果后于2026-11-05设计，属于回顾性补充评价。',
      'Chapman仅用于无尺度对照独立选参；Ningbo、CPSC2018、SPH各5/10-shot、20个相同支持集种子。两个骨干使用各自原十个源头。',
      '', '## 定义', '',
      '- Mapped-source-direct：十个映射源分类器logits平均，无目标标签拟合、无目标支持集统计。相同查询集下预测不随shot/seed改变；表内重复仅用于配对，零SD不是20次独立零样本试验。',
      '- No-scale-tuned-Ens：保留原目标标准化及十头集成，锚点按原none定义移除双端尺度换算；ECG为Dw.T@(Ws@Q)，MERL为映射后的源标准化权重。常量列沿用完整换算偏置，活跃列自由截距。',
      '- 该对照不是所有L2-SP实现的复现；仅检验这个明确定义、获得独立选参机会的无尺度方案。Native及原训练结果未修改。',
      '', '## Chapman独立选参','']
    for bb,r in read(OUT/'release.json')['selected'].items():lines.append(f"- {bb}: alpha={r['config']['alpha']}, lambda={r['config']['lam']}; 开发平均Macro-AP={r['ap_mean']*100:.4f}%。")
    lines+=['','## 外部主结果','', '| 骨干 | 队列 | shot | 方法 | Macro-AP均值±SD (%) | Macro-AUROC均值±SD (%) |','|---|---|---|---|---|---|']
    for r in s.itertuples():lines.append(f'| {r.backbone} | {r.dataset} | {r.shot} | {r.method} | {100*r.ap_mean:.3f} ± {100*r.ap_sd:.3f} | {100*r.auc_mean:.3f} ± {100*r.auc_sd:.3f} |')
    lines+=['','## 完整SASM减补充对照：Macro-AP配对差值','',
      '种子CI为10000次配对支持种子bootstrap；Holm每骨干/指标包含12个新外部比较。患者区间另列，不能混用。', '',
      '| 骨干 | 队列 | shot | 对照 | AP差值(pp) | 种子95%CI(pp) | 胜出/20 | Holm p | 患者95%CI(pp) |','|---|---|---|---|---|---|---|---|---|']
    for r in p[p.metric=='ap'].itertuples():
        ci=patients[(patients.backbone==r.backbone)&(patients.dataset==r.dataset)&(patients.shot==r.shot)&(patients.baseline==r.comparator)].iloc[0]
        lines.append(f'| {r.backbone} | {r.dataset} | {r.shot} | {r.comparator} | {r.mean_difference_pp:+.3f} | [{r.ci_low_pp:+.3f}, {r.ci_high_pp:+.3f}] | {r.wins} | {r.p_holm:.6g} | [{ci.patient_CI_low_pp:+.3f}, {ci.patient_CI_high_pp:+.3f}] |')
    lines+=['','## 解释范围','',
       '直接源头是比较起点而非性能上限。支持种子重复不能当作独立患者队列。患者bootstrap为固定二十支持集条件下的查询患者整群重采样，2000次；无效缺单类重采样剔除且报告有效次数。',
       'CPSC2018没有独立跨记录患者身份核验；预训练患者重叠未知。本轮为回顾性补实验，全部正负零结果均保留。',
       '所有新适配模型复算完整预测和梯度，逐头与平均预测误差<1e-9，梯度<=5e-5；原SASM参考同样完整重放。原输入与结果封印在运行前后逐文件核验。',
       '新增统计比较族与历史报告分开；若在论文中合并为同一预先定义假设族，需要对合并后的p值重新实施多重校正。',
       '', '## 文件', '', 'summary.csv、episode_metrics.csv、per_label_metrics.csv、development_candidates.csv/json、release.json、paired_effects_CI.csv、patient_bootstrap_CI.csv、verification_manifest.json、direct_verification.json、protocol.lock.json。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    unchanged={}
    for name,h in sealmap().items():assert sha(name)==h,('input changed during run',name)
    dump(OUT/'INPUTS_UNCHANGED.json',dict(time=time.time(),verified_files=len(sealmap()),all_unchanged=True))
    names=['RESULTS.md','AUTHORIZED_PLAN.md','controls.py','protocol.lock.json','PREFLIGHT_PASS.json','direct_verification.json',
           'development_candidates.json','development_candidates.csv','release.json','summary.csv','episode_metrics.csv','per_label_metrics.csv',
           'paired_effects_CI.csv','patient_bootstrap_CI.csv','verification_manifest.json','METRICS_COMPLETE.json','INPUTS_UNCHANGED.json']
    dump(OUT/'COMPLETE.json',dict(time=time.time(),dev_tasks=360,new_final_models=240,unique_direct_predictions=6,reference_replays=240,
              files={n:sha(OUT/n) for n in names}))
    dump(OUT/'status.json',dict(stage='complete',pid=os.getpid(),time=time.time()))
    print('ALL_COMPLETE',str(OUT/'RESULTS.md'),flush=True)
def main():
    lock=(OUT/'pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'COMPLETE.json').exists():print('ALREADY_COMPLETE');return
    dump(OUT/'status.json',dict(stage='preflight',pid=os.getpid(),time=time.time()))
    preflight();direct_predictions()
    dev=[make_job(bb,'chapman',k,s,c,'dev',cfg) for bb in BB for c,cfg in enumerate(GRID) for k in [5,10] for s in range(10)]
    pool_jobs(dev,'Chapman_development');selected=select()
    external=[make_job(bb,ds,k,s,selected[bb]['candidate'],'external',selected[bb]['config']) for bb in BB for ds in DS for k in [5,10] for s in range(20)]
    pool_jobs(external,'external_fits');finish_metrics(selected)
    dump(OUT/'status.json',dict(stage='patient_bootstrap',completed_groups=0,total_groups=12,pid=os.getpid(),time=time.time()))
    patient_all();final_report()
if __name__=='__main__':
    try:main()
    except Exception:
        dump(OUT/'FAILURE.json',dict(time=time.time(),pid=os.getpid(),error=traceback.format_exc()))
        raise
