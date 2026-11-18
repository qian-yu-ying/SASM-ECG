"""Remote CPU-only mean-prior control; all historical experiment files read-only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import sys,json,time,fcntl,traceback,concurrent.futures,importlib.util,itertools
from pathlib import Path
OUT=Path(__file__).resolve().parent
OLD=OUT.parent/'chapman_source_controls_20261105'
spec=importlib.util.spec_from_file_location('frozen_controls',OLD/'controls.py')
L=importlib.util.module_from_spec(spec);sys.modules[spec.name]=L;spec.loader.exec_module(L)
L.OUT=OUT;L.SEALED=None
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
np=L.np;pd=L.pd;BASE=L.BASE;BB=L.BB;DS=L.DS;GRID=L.GRID
sha=L.sha;read=L.read;dump=L.dump
FIXED='MeanAnchor-fixed';TUNED='MeanAnchor-tuned';ENS='SASM-Ens'
LIMIT=L.threadpool_limits(limits=1)
def state(stage,**kw):dump(OUT/'status.json',dict(stage=stage,pid=os.getpid(),time=time.time(),**kw))
def sas_config(bb):
    p=BASE/('release_01.json' if bb=='ECGFounder' else 'merl/release.json')
    return read(p)['selected'][ENS]['config']
def path(j):
    return OUT/'models'/j['phase']/j['backbone']/j['method']/j['dataset']/f"c{j['candidate']:02d}_k{j['shot']}_s{j['seed']:02d}.json"
def job(bb,ds,k,s,c,phase,cfg,method=TUNED):
    return dict(backbone=bb,dataset=ds,shot=k,seed=s,candidate=c,phase=phase,config=cfg,method=method)
def aggregate(heads,mu,sd):
    raw=np.mean([h['raw'] for h in heads],axis=0)
    off=np.mean([h['off'] for h in heads],axis=0)
    a=sd[:,None]*raw;bp=mu@raw+off
    other=np.mean([sd[:,None]*h['raw'] for h in heads],axis=0)
    ob=np.mean([mu@h['raw']+h['off'] for h in heads],axis=0)
    assert np.allclose(a,other,rtol=1e-12,atol=1e-12)
    assert np.allclose(bp,ob,rtol=1e-12,atol=1e-12)
    return a,bp
def train(j):
    p=path(j)
    if p.exists():
        m=read(p);assert m['job']==j and m['protocol_sha256']==sha(OUT/'protocol.lock.json')
        assert sha(p.with_suffix('.npz'))==m['predictions_sha256']
        assert sha(p.with_suffix('.weights.npz'))==m['weights_sha256']
        return str(p)
    started=time.perf_counter();_,si,qi,ys,mu,sd,xs,xq,heads=L.arrays(j)
    a,bp=aggregate(heads,mu,sd);cfg=j['config']
    w,b,d=L.certified_fit(xs,ys,cfg['alpha'],W_anchor=a,lam=cfg['lam'])
    const=np.flatnonzero((ys.sum(0)==0)|(ys.sum(0)==len(ys)))
    w[:,const]=a[:,const];b[const]=bp[const]
    g=L.gradients(xs,ys,[w],[b],[a],cfg);pred=xq@w+b
    assert np.isfinite(pred).all()
    p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p.with_suffix('.weights.npz'),weight=w[None],bias=b[None],anchor=a,prior_bias=bp,scaler_mean=mu,scaler_sd=sd)
    np.savez_compressed(p.with_suffix('.npz'),logits=pred,query_indices=qi,support_indices=si)
    m=dict(job=j,seconds=time.perf_counter()-started,fits=1,worker_pid=os.getpid(),max_gradient=g,diagnostics=d,
        protocol_sha256=sha(OUT/'protocol.lock.json'),episode_sha256=sha(L.epfile(j['dataset'],j['shot'],j['seed'],j['phase'])),
        weights_sha256=sha(p.with_suffix('.weights.npz')),predictions_sha256=sha(p.with_suffix('.npz')),query_labels_loaded=False)
    if j['phase']=='dev':
        z=np.load(BASE/'data/chapman/development_labels.npz');assert np.array_equal(qi,z['indices'])
        m['development_score']=L.C.metric(z['y'],pred);assert m['development_score']['ap'] is not None
        m['query_labels_loaded']='Chapman development scorer only after fit'
    dump(p,m);return str(p)
def verify(j):
    p=path(j);m=read(p);assert m['job']==j and m['protocol_sha256']==sha(OUT/'protocol.lock.json')
    assert sha(p.with_suffix('.npz'))==m['predictions_sha256'] and sha(p.with_suffix('.weights.npz'))==m['weights_sha256']
    _,si,qi,ys,mu,sd,xs,xq,heads=L.arrays(j);a,bp=aggregate(heads,mu,sd)
    z=np.load(p.with_suffix('.npz'));wz=np.load(p.with_suffix('.weights.npz'));w=wz['weight'][0];b=wz['bias'][0]
    assert np.array_equal(qi,z['query_indices']) and np.array_equal(si,z['support_indices'])
    assert np.array_equal(mu,wz['scaler_mean']) and np.array_equal(sd,wz['scaler_sd'])
    assert np.array_equal(a,wz['anchor']) and np.array_equal(bp,wz['prior_bias'])
    const=np.flatnonzero((ys.sum(0)==0)|(ys.sum(0)==len(ys)))
    assert np.array_equal(w[:,const],a[:,const]) and np.array_equal(b[const],bp[const])
    err=float(abs(xq@w+b-z['logits']).max());assert err<1e-9
    g=L.gradients(xs,ys,[w],[b],[a],j['config'])
    return dict(job=j,prediction_error=err,max_gradient=g,metadata_sha256=sha(p)),z['logits'],qi
def pool(items,fn,stage,workers=32):
    cpus=sorted(os.sched_getaffinity(0))[:workers]
    out=[]
    with concurrent.futures.ProcessPoolExecutor(min(workers,len(items)),initializer=L.init_worker,initargs=(cpus,)) as ex:
        fs={ex.submit(fn,j):j for j in items}
        for f in concurrent.futures.as_completed(fs):
            out.append(f.result());state(stage,completed=len(out),total=len(items))
            if len(out)%12==0 or len(out)==len(items):print(stage,len(out),len(items),flush=True)
    return out
def preflight():
    old=read(OLD/'protocol.lock.json');seals=old['input_files'].copy()
    seals[str(OLD/'controls.py')]=old['code_sha256']
    seals[str(OLD/'protocol.lock.json')]=sha(OLD/'protocol.lock.json')
    for n,h in seals.items():assert sha(n)==h,('input drift',n)
    spec=dict(base=str(BASE),historical_control_root=str(OLD),backbones=BB,development='chapman',external=DS,shots=[5,10],
        dev_seeds=10,eval_seeds=20,grid=GRID,methods=[FIXED,TUNED,ENS],fixed_configs={bb:sas_config(bb) for bb in BB},
        selection='equal mean Chapman5/10 AP over10 seeds each; AP tolerance1e-4 then smaller lambda then alpha',
        anchor='mean ten fully converted source anchors in common target-standardized coordinates; single certified_fit',
        intercept='active free; constant w=mean anchor and b=mean full affine prior intercept',loss=read(BASE/'protocol.lock.json')['loss'],
        input_files=seals,code_sha256=sha(__file__),plan_sha256=sha(OUT/'AUTHORIZED_PLAN.md'),
        original_environment=read(BASE/'protocol.lock.json')['environment'],cpu_only=True,workers=32,
        statistics='SASM-minus each of two controls; Holm12 per backbone and metric; paired seed bootstrap10000; patient AP2000',
        timing='same fixed CPU,BLAS1,warmup,5 seeds x3 reps; support scaler+all conversions+fit; no IO/features/query; fixed control vs SASM',
        retrospective=True,pretraining_overlap='unknown',CPSC_patient_identity='record-level limitation')
    p=OUT/'protocol.lock.json'
    if p.exists():assert read(p)==spec
    else:dump(p,spec)
    checks=[]
    for bb in BB:
        for ds in DS:
            c,_,_=L.verify_reference(bb,ds,5,0);checks.append(c)
        j=job(bb,'chapman',5,0,2,'dev',sas_config(bb),FIXED)
        train(j);c,_,_=verify(j);checks.append(c)
    dump(OUT/'PREFLIGHT_PASS.json',dict(time=time.time(),input_files=len(seals),checks=checks,threadpools=L.threadpool_info(),python=sys.executable))
    print('PREFLIGHT_PASS',len(seals),flush=True)
def select():
    rows=[];selected={}
    for bb in BB:
        group=[]
        for c,cfg in enumerate(GRID):
            scores=[]
            for k in [5,10]:
                for s in range(10):
                    j=job(bb,'chapman',k,s,c,'dev',cfg);m=read(path(j));assert m['job']==j
                    assert sha(path(j).with_suffix('.npz'))==m['predictions_sha256']
                    scores.append(m['development_score']['ap'])
            r=dict(backbone=bb,candidate=c,config=cfg,ap_mean=float(np.mean(scores)),episodes=20);rows.append(r);group.append(r)
        best=max(r['ap_mean'] for r in group);near=[r for r in group if best-r['ap_mean']<=1e-4]
        selected[bb]=min(near,key=lambda r:(r['config']['lam'],r['config']['alpha']))
    dump(OUT/'development_candidates.json',rows)
    pd.DataFrame([dict(backbone=r['backbone'],candidate=r['candidate'],alpha=r['config']['alpha'],lam=r['config']['lam'],ap_mean=r['ap_mean']) for r in rows]).to_csv(OUT/'development_candidates.csv',index=False)
    release=dict(selected=selected,fixed_configs={bb:sas_config(bb) for bb in BB},protocol_sha256=sha(OUT/'protocol.lock.json'),external_labels_used=False)
    if (OUT/'release.json').exists():assert read(OUT/'release.json')==release
    else:dump(OUT/'release.json',release)
    print('RELEASE',json.dumps(release),flush=True);return selected
def extjob(bb,ds,k,s,method):
    if method==FIXED:return job(bb,ds,k,s,2,'external',sas_config(bb),FIXED)
    r=read(OUT/'release.json')['selected'][bb]
    return job(bb,ds,k,s,r['candidate'],'external',r['config'],TUNED)
def metric_group(arg):
    bb,ds,k=arg;v=np.load(BASE/'vault'/ds/'external.npz',allow_pickle=True);y=v['y'];rows=[];per=[];checks=[]
    labels=read(BASE/'data'/ds/'metadata.json')['labels']
    for s in range(20):
        check,pred,qi=L.verify_reference(bb,ds,k,s);checks.append(check);assert np.array_equal(qi,v['indices'])
        preds={ENS:pred}
        for method in [FIXED,TUNED]:
            check,pred,qi=verify(extjob(bb,ds,k,s,method));checks.append(check);assert np.array_equal(qi,v['indices']);preds[method]=pred
        for method,pred in preds.items():
            score=L.C.metric(y,pred);assert score['ap'] is not None and score['auc'] is not None
            rows.append(dict(backbone=bb,dataset=ds,shot=k,seed=s,method=method,ap=score['ap'],auc=score['auc']))
            for c,label in enumerate(labels):per.append(dict(backbone=bb,dataset=ds,shot=k,seed=s,method=method,label=label,ap=score['ap_per_label'][c],auc=score['auc_per_label'][c]))
    return rows,per,checks
def finish_metrics():
    results=pool(list(itertools.product(BB,DS,[5,10])),metric_group,'metrics_verification',12)
    rows=[];per=[];checks=[]
    for a,b,c in results:rows+=a;per+=b;checks+=c
    df=pd.DataFrame(rows);df.to_csv(OUT/'episode_metrics.csv',index=False);pd.DataFrame(per).to_csv(OUT/'per_label_metrics.csv',index=False)
    summary=df.groupby(['backbone','dataset','shot','method']).agg(n=('ap','size'),ap_mean=('ap','mean'),ap_sd=('ap','std'),auc_mean=('auc','mean'),auc_sd=('auc','std')).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False);pairs=[];rng=np.random.default_rng(20261105)
    for (bb,ds,k),g in df.groupby(['backbone','dataset','shot']):
        a=g[g.method==ENS].sort_values('seed')
        for method in [FIXED,TUNED]:
            b=g[g.method==method].sort_values('seed');assert a.seed.tolist()==b.seed.tolist()==list(range(20))
            for metric in ['ap','auc']:
                d=a[metric].to_numpy()-b[metric].to_numpy();boots=d[rng.integers(0,20,(10000,20))].mean(1)
                pairs.append(dict(backbone=bb,dataset=ds,shot=int(k),comparator=method,metric=metric,mean_difference_pp=float(d.mean()*100),
                    ci_low_pp=float(np.quantile(boots,.025)*100),ci_high_pp=float(np.quantile(boots,.975)*100),wins=int((d>0).sum()),ties=int((d==0).sum()),
                    p=float(L.wilcoxon(d).pvalue) if np.any(d) else 1.))
    t=pd.DataFrame(pairs)
    for _,g in t.groupby(['backbone','metric']):
        ix=g.sort_values('p').index;ps=t.loc[ix,'p'].to_numpy();assert len(ps)==12
        t.loc[ix,'p_holm']=np.minimum(1,np.maximum.accumulate(ps*(len(ps)-np.arange(len(ps)))))
    t.to_csv(OUT/'paired_effects_CI.csv',index=False)
    dump(OUT/'verification_manifest.json',dict(checks=checks,new_external_models=480,reference_replays=240))
    dump(OUT/'METRICS_COMPLETE.json',dict(time=time.time(),rows=len(rows),new_external_models=480,reference_replays=240))
def patient(arg):
    bb,ds,k=arg;p=OUT/'patient'/f'{bb}_{ds}_{k}.json'
    if p.exists():
        m=read(p);assert sha(p.with_suffix('.npz'))==m['distribution_sha256'];return m['rows']
    v=np.load(BASE/'vault'/ds/'external.npz',allow_pickle=True);y=v['y'];uniq,inv=np.unique(v['patients'].astype(str),return_inverse=True);n=len(uniq)
    rng=np.random.default_rng(20261105);counts=rng.multinomial(n,np.full(n,1/n),size=2000).astype(np.int16);valid=np.ones(2000,bool)
    for c in range(y.shape[1]):
        pos=counts@np.bincount(inv,weights=y[:,c],minlength=n);neg=counts@np.bincount(inv,weights=1-y[:,c],minlength=n);valid&=(pos>0)&(neg>0)
    distributions={m:np.zeros(2000) for m in [ENS,FIXED,TUNED]};witness=0.
    for s in range(20):
        paths={ENS:L.reference_file(bb,ds,k,s).with_suffix('.npz'),**{m:path(extjob(bb,ds,k,s,m)).with_suffix('.npz') for m in [FIXED,TUNED]}}
        for method,zp in paths.items():
            z=np.load(zp);assert np.array_equal(z['query_indices'],v['indices'])
            if s==0:
                from sklearn.metrics import average_precision_score
                actual=L.ap_boot(y,z['logits'],np.ones((1,n),np.int16),inv)[0]
                expected=np.mean([average_precision_score(y[:,c],z['logits'][:,c]) for c in range(y.shape[1])])
                witness=max(witness,float(abs(actual-expected)));assert witness<1e-12
            distributions[method]+=L.ap_boot(y,z['logits'],counts,inv)/20
        if s%5==4:print('PATIENT_PROGRESS',bb,ds,k,s+1,flush=True)
    rows=[]
    for method in [FIXED,TUNED]:
        delta=distributions[ENS]-distributions[method];good=valid&np.isfinite(delta);assert good.mean()>=.95
        lo,hi=np.quantile(delta[good],[.025,.975])*100
        rows.append(dict(backbone=bb,dataset=ds,shot=k,baseline=method,metric='ap',replications=2000,valid_replications=int(good.sum()),
            patient_CI_low_pp=float(lo),patient_CI_high_pp=float(hi),scope='query patient-cluster sampling conditional on20 fixed support sets; CPSC record-level linkage'))
    p.parent.mkdir(exist_ok=True);np.savez_compressed(p.with_suffix('.npz'),valid=valid,**distributions)
    dump(p,dict(rows=rows,weighted_AP_unit_weight_error=witness,distribution_sha256=sha(p.with_suffix('.npz'))));return rows
def fit_support(bb,ds,k,s,method):
    e=read(L.epfile(ds,k,s,'external'));F,heads=L.context(bb,ds)
    # IO and support-row gathering intentionally happen outside timed region.
    fs=F[np.asarray(e['support'])].astype(float);ys=np.asarray(e['support_y'],float);cfg=sas_config(bb)
    start=time.perf_counter();mu=fs.mean(0);sd=fs.std(0)+1e-6;xs=(fs-mu)/sd
    anchors=[sd[:,None]*h['raw'] for h in heads];biases=[mu@h['raw']+h['off'] for h in heads]
    if method==FIXED:anchors=[np.mean(anchors,axis=0)];biases=[np.mean(biases,axis=0)]
    const=np.flatnonzero((ys.sum(0)==0)|(ys.sum(0)==len(ys)));weights=[];bs=[]
    for a,bp in zip(anchors,biases):
        w,b,_=L.certified_fit(xs,ys,cfg['alpha'],W_anchor=a,lam=cfg['lam']);w[:,const]=a[:,const];b[const]=bp[const];weights.append(w);bs.append(b)
    # Include construction of the final effective linear predictor, but no query data.
    w=np.mean(weights,axis=0);b=np.mean(bs,axis=0)
    elapsed=time.perf_counter()-start;assert np.isfinite(w).all() and np.isfinite(b).all();return elapsed
def timing():
    original=os.sched_getaffinity(0);cpu=max(original);os.sched_setaffinity(0,{cpu});limit=L.threadpool_limits(limits=1);rows=[]
    try:
        for bb,ds,k in itertools.product(BB,DS,[5,10]):
            for method in [FIXED,ENS]:fit_support(bb,ds,k,0,method)
            for s in range(5):
                for rep in range(3):
                    order=[FIXED,ENS] if (s+rep)%2==0 else [ENS,FIXED]
                    for method in order:rows.append(dict(backbone=bb,dataset=ds,shot=k,seed=s,rep=rep,method=method,seconds=fit_support(bb,ds,k,s,method),cpu=cpu))
            pd.DataFrame(rows).to_csv(OUT/'timing_raw.csv',index=False);state('single_core_timing',completed=len(rows),total=360,cpu=cpu)
            print('TIMING_GROUP',bb,ds,k,len(rows),flush=True)
        df=pd.DataFrame(rows);assert len(df)==360
        summary=df.groupby(['backbone','dataset','shot','method']).seconds.agg(n='size',median='median',q25=lambda x:x.quantile(.25),q75=lambda x:x.quantile(.75),minimum='min',maximum='max').reset_index()
        summary.to_csv(OUT/'timing_summary.csv',index=False)
        dump(OUT/'TIMING_COMPLETE.json',dict(time=time.time(),measurements=len(df),cpu=cpu,threadpools=L.threadpool_info(),scope='support scaler+ten transported anchors+one or ten fits+effective predictor; excludes IO/features/query/scoring'))
    finally:os.sched_setaffinity(0,original)
def report(final=False):
    summary=pd.read_csv(OUT/'summary.csv');pairs=pd.read_csv(OUT/'paired_effects_CI.csv')
    lines=['# 平均源锚点后单次适配：Chapman开发补实验','',
        'ECGFounder与MERL；Chapman唯一开发，Ningbo/CPSC2018/SPH外部；5/10-shot，每格20配对支持种子。',
        'MeanAnchor-fixed固定完整SASM参数；MeanAnchor-tuned在Chapman原9格独立选择。所有尺度、映射、截距、常量回退一致；新单锚点模型重新拟合。SASM参考为历史冻结结果，经完整hash、预测和梯度重放，不称为新训练。',
        '这是已见旧外部结果后的回顾性补实验。预训练重叠未知；CPSC患者仅记录级；bootstrap不能证明无泄漏；此方法不是Native。','',
        '## 独立选参','']
    for bb,r in read(OUT/'release.json')['selected'].items():lines.append(f"- {bb}: {r['config']}; Chapman开发AP={100*r['ap_mean']:.4f}%; fixed={sas_config(bb)}")
    lines+=['','## 外部结果','', '| 骨干 | 数据集 | Shot | 方法 | AP mean±SD (%) | AUROC mean±SD (%) |','|---|---|---|---|---|---|']
    for r in summary.itertuples():lines.append(f'| {r.backbone} | {r.dataset} | {r.shot} | {r.method} | {100*r.ap_mean:.3f}±{100*r.ap_sd:.3f} | {100*r.auc_mean:.3f}±{100*r.auc_sd:.3f} |')
    lines+=['','## SASM减单锚点：配对种子统计','', '95%CI为10000次配对支持种子bootstrap，未作同时区间校正；Holm每骨干/指标12个外部比较。新比较族与历史分别报告。','',
        '| 骨干 | 数据集 | Shot | 对照 | 指标 | 差值pp | 95%CI pp | 胜出/20 | Holm p |','|---|---|---|---|---|---|---|---|---|']
    for r in pairs.itertuples():lines.append(f'| {r.backbone} | {r.dataset} | {r.shot} | {r.comparator} | {r.metric} | {r.mean_difference_pp:+.4f} | [{r.ci_low_pp:+.4f}, {r.ci_high_pp:+.4f}] | {r.wins} | {r.p_holm:.6g} |')
    for fn,title in [('patient_bootstrap_CI.csv','患者AP Bootstrap：2000次，固定20支持集'),('timing_summary.csv','单核耗时：中位数与IQR，秒')]:
        if (OUT/fn).exists():
            df=pd.read_csv(OUT/fn);lines+=['','## '+title,'','```csv',df.to_csv(index=False).strip(),'```']
    lines+=['','## 完成状态','', '全部完成并核验。' if final else '主结果已核验；患者bootstrap与独立单核计时尚待完成。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
def finish():
    for n,h in L.sealmap().items():assert sha(n)==h,('input drift after run',n)
    dump(OUT/'INPUTS_UNCHANGED.json',dict(time=time.time(),all_unchanged=True,files=len(L.sealmap())))
    report(True)
    names=['mean_anchor.py','AUTHORIZED_PLAN.md','protocol.lock.json','PREFLIGHT_PASS.json','development_candidates.json','development_candidates.csv','release.json',
        'episode_metrics.csv','per_label_metrics.csv','summary.csv','paired_effects_CI.csv','verification_manifest.json','METRICS_COMPLETE.json','patient_bootstrap_CI.csv',
        'timing_raw.csv','timing_summary.csv','TIMING_COMPLETE.json','INPUTS_UNCHANGED.json','RESULTS.md']
    dump(OUT/'COMPLETE.json',dict(time=time.time(),dev_tasks=360,preflight_extra_fits=2,new_final_models=480,reference_replays=240,timing_measurements=360,files={n:sha(OUT/n) for n in names}))
    state('complete');print('ALL_COMPLETE',str(OUT/'RESULTS.md'),flush=True)
def main():
    lock=(OUT/'pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'COMPLETE.json').exists():print('ALREADY_COMPLETE');return
    state('preflight');preflight()
    dev=[job(bb,'chapman',k,s,c,'dev',cfg) for bb in BB for c,cfg in enumerate(GRID) for k in [5,10] for s in range(10)]
    pool(dev,train,'Chapman_development');select()
    external=[extjob(bb,ds,k,s,m) for bb in BB for ds in DS for k in [5,10] for s in range(20) for m in [FIXED,TUNED]]
    pool(external,train,'external_fits');finish_metrics();report()
    rows=pool(list(itertools.product(BB,DS,[5,10])),patient,'patient_bootstrap',12)
    pd.DataFrame([r for group in rows for r in group]).to_csv(OUT/'patient_bootstrap_CI.csv',index=False)
    state('single_core_timing',completed=0,total=360);timing();finish()
if __name__=='__main__':
    try:main()
    except Exception:
        dump(OUT/'FAILURE.json',dict(time=time.time(),pid=os.getpid(),error=traceback.format_exc()));raise
