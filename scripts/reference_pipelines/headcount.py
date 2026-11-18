"""Frozen independent-head aggregation sensitivity; no new fits or GPU work."""
import os,sys,json,time,hashlib,fcntl,traceback,itertools,concurrent.futures,importlib.util
from pathlib import Path
OUT=Path(__file__).resolve().parent
GATE=OUT.parent/'chapman_mean_anchor_20261105'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.{os.getpid()}.tmp')
    q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');q.replace(p)
def state(stage,**kw):dump(OUT/'status.json',dict(stage=stage,time=time.time(),pid=os.getpid(),**kw))
if __name__=='__main__':
    state('waiting_for_mean_anchor_timing')
    while not (GATE/'COMPLETE.json').exists():
        if (GATE/'FAILURE.json').exists():raise RuntimeError((GATE/'FAILURE.json').read_text())
        launch=read(GATE/'launch.json');p=Path('/proc')/str(launch['pid'])/'cmdline'
        if not p.exists():raise RuntimeError('Gate pipeline exited without completion; inspect rather than restart')
        time.sleep(15)
    # Wait for its actual process to leave the final completion path as well.
    oldpid=read(GATE/'launch.json')['pid']
    while (Path('/proc')/str(oldpid)/'cmdline').exists():
        data=(Path('/proc')/str(oldpid)/'cmdline').read_bytes()
        if str(GATE/'mean_anchor.py').encode() not in data:break
        time.sleep(1)
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
OLD=OUT.parent/'chapman_source_controls_20261105'
spec=importlib.util.spec_from_file_location('frozen_headcount_helpers',OLD/'controls.py')
L=importlib.util.module_from_spec(spec);sys.modules[spec.name]=L;spec.loader.exec_module(L)
L.OUT=OUT;L.SEALED=None
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
np=L.np;pd=L.pd;sha=L.sha;BASE=L.BASE;BB=L.BB;DS=L.DS
LIMIT=L.threadpool_limits(limits=1)
def config(bb):return read(BASE/('release_01.json' if bb=='ECGFounder' else 'merl/release.json'))['selected']['SASM-Ens']
def src(bb,ds,k,s):
    root=BASE if bb=='ECGFounder' else BASE/'merl';c=config(bb)['candidate'];phase='dev' if ds=='chapman' else 'external'
    return root/'results'/phase/ds/'SASM-Ens'/f'c{c:02d}_shot{k}_seed{s:02d}.json'
def result(arg):
    bb,ds,k,s=arg;return OUT/'episodes'/bb/ds/f'shot{k}_seed{s:02d}.json'
def schedule():
    rng=np.random.default_rng(2026110511);p0=rng.permutation(10).tolist()
    def windows(p,m):return {tuple(sorted((p+p)[i:i+m])) for i in range(10)}
    for attempt in range(100000):
        p1=rng.permutation(10).tolist()
        if all(not windows(p0,m)&windows(p1,m) for m in range(2,9)):break
    else:raise RuntimeError('Could not construct disjoint balanced schedules')
    groups={1:[(i,) for i in range(10)],9:list(itertools.combinations(range(10),9)),10:[tuple(range(10))]}
    for m in range(2,9):groups[m]=sorted(windows(p0,m)|windows(p1,m))
    rows=[]
    for m in range(1,11):
        ss=groups[m];assert len(ss)==(10 if m in [1,9] else 1 if m==10 else 20)
        assert len(set(ss))==len(ss)
        counts=[sum(i in t for t in ss) for i in range(10)]
        assert len(set(counts))==1
        for ids in ss:assert len(ids)==m and len(set(ids))==m
        for n,ids in enumerate(ss):rows.append(dict(head_count=m,subset_index=n,heads=list(ids)))
    assert len(rows)==161
    return dict(seed=2026110511,circular_permutations=[p0,p1],second_permutation_attempts=attempt+1,subsets=rows,
        interpretation='balanced sampled average for M2..8; exhaustive for M1/9/10; same schedule all episodes')
def preflight():
    for n,h in read(GATE/'COMPLETE.json')['files'].items():assert sha(GATE/n)==h
    seals=read(GATE/'protocol.lock.json')['input_files'].copy()
    for n,h in seals.items():assert sha(n)==h,('input drift',n)
    seals[str(GATE/'COMPLETE.json')]=sha(GATE/'COMPLETE.json')
    for bb in BB:
        for ds in ['chapman',*DS]:
            for k in [5,10]:
                for s in range(10 if ds=='chapman' else 20):
                    p=src(bb,ds,k,s);m=read(p)
                    assert m['job']['config']==config(bb)['config']
                    for pp,h in [(p,sha(p)),(p.with_suffix('.weights.npz'),m['model_sha256']),(p.with_suffix('.npz'),m['prediction_sha256'])]:
                        assert sha(pp)==h;seals[str(pp)]=h
                    if ds!='chapman':
                        ab=BASE/'ablation/results'/bb/'ensemble'/ds/f'shot{k}_seed{s:02d}.json'
                        assert read(ab)['checks']['source_result_sha256']==sha(p);seals[str(ab)]=sha(ab)
    subsets=schedule();p=OUT/'subsets.json'
    if p.exists():assert read(p)==subsets
    else:dump(p,subsets)
    lock=dict(base=str(BASE),backbones=BB,development='chapman',external=DS,shots=[5,10],dev_seeds=10,eval_seeds=20,
        source_heads=10,counts=list(range(1,11)),target_fits='one per retained source head; total=M; frozen fits reused exactly',
        fixed_configs={bb:config(bb)['config'] for bb in BB},selection='per-backbone Chapman mean AP, equal shots/support seeds; numerical tie1e-8 then smaller M',
        parsimonious_secondary='smallest M within1e-4 AP of maximum; descriptive, not equivalence/noninferiority',
        statistics='average subset metrics within support seed; paired seed bootstrap10000; selected-vs1/10, omit self, Holm across both backbones per metric',
        subset_manifest_sha256=sha(p),input_files=seals,code_sha256=sha(__file__),plan_sha256=sha(OUT/'AUTHORIZED_PLAN.md'),
        review_sha256=sha(OUT/'DESIGN_REVIEW.md'),cpu_only=True,workers=32,retrospective=True,pretraining_overlap='unknown',CPSC_patient_identity='record-level')
    p=OUT/'protocol.lock.json'
    if p.exists():assert read(p)==lock
    else:dump(p,lock)
    checks=[evaluate((bb,'chapman',5,0)) for bb in BB]
    dump(OUT/'PREFLIGHT_PASS.json',dict(time=time.time(),sealed_files=len(seals),subsets_per_episode=161,checks=checks,threadpools=L.threadpool_info()))
    print('PREFLIGHT_PASS',len(seals),flush=True)
def evaluate(arg):
    p=result(arg)
    if p.exists():
        m=read(p);assert m['protocol_sha256']==sha(OUT/'protocol.lock.json') and sha(p.with_suffix('.npz'))==m['scores_sha256'];return str(p)
    bb,ds,k,s=arg;phase='dev' if ds=='chapman' else 'external';j=L.make_job(bb,ds,k,s,config(bb)['candidate'],phase,config(bb)['config'])
    mp=src(bb,ds,k,s);L.checkseal(mp);meta=read(mp)
    assert sha(mp.with_suffix('.weights.npz'))==meta['model_sha256'] and sha(mp.with_suffix('.npz'))==meta['prediction_sha256']
    _,si,qi,ys,mu,sd,xs,xq,heads=L.arrays(j)
    wz=np.load(mp.with_suffix('.weights.npz'));z=np.load(mp.with_suffix('.npz'));w=wz['weight'];b=wz['bias']
    assert len(w)==len(b)==10
    assert np.allclose(mu,wz['scaler_mean'],rtol=0,atol=1e-12) and np.allclose(sd,wz['scaler_sd'],rtol=0,atol=1e-12)
    assert np.array_equal(si,z['support_indices']) and np.array_equal(qi,z['query_indices'])
    g=L.gradients(xs,ys,w,b,[sd[:,None]*h['raw'] for h in heads],config(bb)['config'])
    const=np.flatnonzero((ys.sum(0)==0)|(ys.sum(0)==len(ys)))
    for h,ww,bi in zip(heads,w,b):
        assert np.allclose(ww[:,const],(sd[:,None]*h['raw'])[:,const],rtol=0,atol=1e-12)
        assert np.allclose(bi[const],(mu@h['raw']+h['off'])[const],rtol=0,atol=1e-12)
    all_logits=np.asarray([xq@ww+bi for ww,bi in zip(w,b)])
    err=float(abs(all_logits.mean(0)-z['logits']).max());assert err<1e-9
    if ds=='chapman':v=np.load(BASE/'data/chapman/development_labels.npz')
    else:
        assert (OUT/'release.json').exists(),'External metrics prohibited before count release'
        v=np.load(BASE/'vault'/ds/'external.npz',allow_pickle=True)
    assert np.array_equal(qi,v['indices']);y=v['y'];subs=read(OUT/'subsets.json')['subsets'];scores=[]
    for sub in subs:
        pred=z['logits'] if sub['head_count']==10 else all_logits[sub['heads']].mean(0)
        sc=L.C.metric(y,pred);assert sc['ap'] is not None and sc['auc'] is not None
        scores.append([sc['ap'],sc['auc']])
    scores=np.asarray(scores);counts=np.asarray([x['head_count'] for x in subs]);rows=[]
    for m in range(1,11):
        a=scores[counts==m];assert np.isfinite(a).all()
        rows.append(dict(backbone=bb,dataset=ds,shot=k,seed=s,head_count=m,target_fits=m,subsets=len(a),ap=float(a[:,0].mean()),auc=float(a[:,1].mean()),
            subset_ap_sd=float(a[:,0].std(ddof=1)) if len(a)>1 else 0.,subset_auc_sd=float(a[:,1].std(ddof=1)) if len(a)>1 else 0.))
    full=L.C.metric(y,z['logits']);assert abs(rows[-1]['ap']-full['ap'])<1e-12 and abs(rows[-1]['auc']-full['auc'])<1e-12
    endpoint_error=0.
    if ds!='chapman':
        ab=BASE/'ablation/results'/bb/'ensemble'/ds/f'shot{k}_seed{s:02d}.json';L.checkseal(ab);old=read(ab)['metrics']['Mean-single-head']
        endpoint_error=max(abs(rows[0]['ap']-old['ap']),abs(rows[0]['auc']-old['auc']));assert endpoint_error<1e-12
    p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p.with_suffix('.npz'),scores=scores,head_count=counts)
    dump(p,dict(rows=rows,source_model_metadata_sha256=sha(mp),scores_sha256=sha(p.with_suffix('.npz')),protocol_sha256=sha(OUT/'protocol.lock.json'),
        ensemble_replay_error=err,single_head_mean_replay_error=endpoint_error,max_gradient=g,worker_pid=os.getpid(),new_fits=0))
    return str(p)
def pool(args,stage,workers=32):
    out=[];cpus=sorted(os.sched_getaffinity(0))[:workers]
    with concurrent.futures.ProcessPoolExecutor(min(workers,len(args)),initializer=L.init_worker,initargs=(cpus,)) as ex:
        futures=[ex.submit(evaluate,a) for a in args]
        for f in concurrent.futures.as_completed(futures):
            out.append(f.result());state(stage,completed=len(out),total=len(args))
            if len(out)%4==0 or len(out)==len(args):print(stage,len(out),len(args),flush=True)
    return out
def frame(paths):return pd.DataFrame([row for p in paths for row in read(p)['rows']])
def choose(df):
    dev=df.groupby(['backbone','head_count']).ap.mean().reset_index();selected={}
    for bb,g in dev.groupby('backbone'):
        best=float(g.ap.max());m=int(g[g.ap>=best-1e-8].head_count.min());small=int(g[g.ap>=best-1e-4].head_count.min())
        selected[bb]=dict(head_count=m,ap_mean=best,within_1e4_smallest=small)
    shared=dev.groupby('head_count').ap.mean();best=float(shared.max());m=int(shared[shared>=best-1e-8].index.min())
    release=dict(selected=selected,shared_descriptive=dict(head_count=m,ap_mean=best),external_labels_used=False,
        protocol_sha256=sha(OUT/'protocol.lock.json'),subset_manifest_sha256=sha(OUT/'subsets.json'))
    if (OUT/'release.json').exists():assert read(OUT/'release.json')==release
    else:dump(OUT/'release.json',release)
    dev.to_csv(OUT/'development_candidates.csv',index=False);df.to_csv(OUT/'development_episode_metrics.csv',index=False)
    print('RELEASE',json.dumps(release),flush=True);return release
def summarize(dev,ext):
    ext.to_csv(OUT/'external_episode_metrics.csv',index=False);allrows=pd.concat([dev,ext],ignore_index=True);summ=[];rng=np.random.default_rng(2026110513)
    for (bb,ds,k,m),g in allrows.groupby(['backbone','dataset','shot','head_count']):
        assert sorted(g.seed.tolist())==list(range(10 if ds=='chapman' else 20));r=dict(backbone=bb,dataset=ds,shot=int(k),head_count=int(m),target_fits=int(m),n=len(g))
        for metric in ['ap','auc']:
            a=g.sort_values('seed')[metric].to_numpy();boots=a[rng.integers(0,len(a),(10000,len(a)))].mean(1)
            r[metric+'_mean']=float(a.mean());r[metric+'_sd']=float(a.std(ddof=1));r[metric+'_ci_low']=float(np.quantile(boots,.025));r[metric+'_ci_high']=float(np.quantile(boots,.975))
        summ.append(r)
    summary=pd.DataFrame(summ);summary.to_csv(OUT/'summary.csv',index=False);pairs=[];sel=read(OUT/'release.json')['selected']
    for (bb,ds,k),g in ext.groupby(['backbone','dataset','shot']):
        m=sel[bb]['head_count'];a=g[g.head_count==m].sort_values('seed')
        for base in [1,10]:
            if base==m:continue
            b=g[g.head_count==base].sort_values('seed')
            for metric in ['ap','auc']:
                d=a[metric].to_numpy()-b[metric].to_numpy();boots=d[rng.integers(0,20,(10000,20))].mean(1)
                pairs.append(dict(backbone=bb,dataset=ds,shot=int(k),selected_count=m,baseline_count=base,metric=metric,mean_difference_pp=float(d.mean()*100),
                    ci_low_pp=float(np.quantile(boots,.025)*100),ci_high_pp=float(np.quantile(boots,.975)*100),wins=int((d>0).sum()),ties=int((d==0).sum()),
                    p=float(L.wilcoxon(d).pvalue) if np.any(d) else 1.))
    pairs=pd.DataFrame(pairs)
    for _,g in pairs.groupby('metric'):
        ix=g.sort_values('p').index;ps=pairs.loc[ix,'p'].to_numpy();pairs.loc[ix,'p_holm']=np.minimum(1,np.maximum.accumulate(ps*(len(ps)-np.arange(len(ps)))));pairs.loc[ix,'family_size']=len(ps)
    pairs.to_csv(OUT/'paired_effects_CI.csv',index=False)
def report():
    df=pd.read_csv(OUT/'summary.csv');release=read(OUT/'release.json');stats=pd.read_csv(OUT/'paired_effects_CI.csv')
    lines=['# SASM逐头适配：1–10个源头敏感性','',
        '每个保留源头各拟合一次，所以算法总目标拟合次数=M。本次复用已验证的独立适配权重，新增拟合0次。不是分组平均，不是同一头重复训练，也不是优化迭代轮数扫描。',
        'Chapman开发选择头数；Ningbo/CPSC2018/SPH仅外部描述和锁定比较；固定原SASM超参数。M1/9/10穷举，M2–8每个20个平衡预定子集。所有结果先在支持seed内平均子集指标，再统计seed，不把子集当额外seed。',
        '这是回顾性敏感性分析；最佳仅指本轮固定超参数/源头池/预定子集下开发均值最高。不能证明普适最优，也不能证明多次拟合优于平均锚点后单次拟合。','', '## Chapman选择','']
    for bb,r in release['selected'].items():lines.append(f"- {bb}: 最高开发AP头数={r['head_count']}; AP={100*r['ap_mean']:.4f}%; 距最高AP不超过0.01pp的最小头数={r['within_1e4_smallest']}（仅描述，不称等效）。")
    lines.append(f"- 跨骨干共享头数（附加描述）：{release['shared_descriptive']}")
    for bb in BB:
        for metric in ['ap','auc']:
            lines+=['',f'## {bb} {metric.upper()}均值 (%)','', '| 头数/目标拟合次数 | Chapman5开发 | Chapman10开发 | Ningbo5 | Ningbo10 | CPSC5 | CPSC10 | SPH5 | SPH10 |','|---|---|---|---|---|---|---|---|---|']
            for m in range(1,11):
                vals=[]
                for ds,k in [('chapman',5),('chapman',10),('ningbo',5),('ningbo',10),('cpsc2018',5),('cpsc2018',10),('sph',5),('sph',10)]:
                    r=df[(df.backbone==bb)&(df.dataset==ds)&(df.shot==k)&(df.head_count==m)].iloc[0];vals.append(f"{100*r[metric+'_mean']:.4f}")
                lines.append('| '+str(m)+' | '+' | '.join(vals)+' |')
    lines+=['','## 开发选中头数与端点的配对统计','',
        '支持seed是推断单位；CI为固定查询队列、源头池和子集表条件下10000次配对seed bootstrap；不是患者总体CI；点区间未作同时校正。Holm跨两个骨干全部锁定比较，按AP/AUROC分别计算，自比较剔除。','',
        '```csv',stats.to_csv(index=False).strip(),'```','',
        '## 解释限制','',
        '中间头数仅为20平衡子集的平均表现，不是全部组合穷举，也没有选择最强子集。外部曲线最高点仅描述，不能用于改选头数。源头数与逐头拟合次数在本扫描绑定；用户另问的独立拟合次数尚需定义。',
        '预训练重叠未知，CPSC记录级患者身份限制。没有修改Native、原模型或现有平均锚点对照。没有用外部结果调参，所有正负零保留。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
def finish():
    files=list((OUT/'episodes').glob('*/*/*.json'));assert len(files)==280
    checks=[];subset_rows=[];subs=read(OUT/'subsets.json')['subsets']
    for p in files:
        m=read(p);assert sha(p.with_suffix('.npz'))==m['scores_sha256'];a=np.load(p.with_suffix('.npz'))['scores']
        checks.append(dict(path=str(p),metadata_sha256=sha(p),scores_sha256=m['scores_sha256'],ensemble_replay_error=m['ensemble_replay_error'],single_head_mean_replay_error=m['single_head_mean_replay_error'],max_gradient=m['max_gradient']))
        r=m['rows'][0]
        for sub,sc in zip(subs,a):subset_rows.append(dict(backbone=r['backbone'],dataset=r['dataset'],shot=r['shot'],seed=r['seed'],head_count=sub['head_count'],subset_index=sub['subset_index'],ap=float(sc[0]),auc=float(sc[1])))
    assert len(subset_rows)==45080;pd.DataFrame(subset_rows).to_csv(OUT/'subset_episode_metrics.csv',index=False)
    dump(OUT/'verification_manifest.json',dict(checks=checks,episodes=280,subset_scores=45080,new_fits=0,subsets_sha256=sha(OUT/'subsets.json')))
    for n,h in L.sealmap().items():assert sha(n)==h,('input drift after scan',n)
    dump(OUT/'INPUTS_UNCHANGED.json',dict(time=time.time(),all_unchanged=True,files=len(L.sealmap())))
    report()
    names=['headcount.py','AUTHORIZED_PLAN.md','DESIGN_REVIEW.md','protocol.lock.json','subsets.json','PREFLIGHT_PASS.json','release.json','development_candidates.csv',
        'development_episode_metrics.csv','external_episode_metrics.csv','summary.csv','paired_effects_CI.csv','subset_episode_metrics.csv','verification_manifest.json','INPUTS_UNCHANGED.json','RESULTS.md']
    dump(OUT/'COMPLETE.json',dict(time=time.time(),episodes=280,development_episodes=40,external_episodes=240,subset_scores=45080,new_fits=0,files={n:sha(OUT/n) for n in names}))
    state('complete');print('ALL_COMPLETE',str(OUT/'RESULTS.md'),flush=True)
def main():
    lock=(OUT/'pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'COMPLETE.json').exists():print('ALREADY_COMPLETE');return
    state('preflight');preflight()
    devargs=list(itertools.product(BB,['chapman'],[5,10],range(10)));dev=frame(pool(devargs,'Chapman_headcount_selection',16));choose(dev)
    extargs=list(itertools.product(BB,DS,[5,10],range(20)));ext=frame(pool(extargs,'external_curves',32));summarize(dev,ext);finish()
if __name__=='__main__':
    try:main()
    except Exception:
        dump(OUT/'FAILURE.json',dict(time=time.time(),pid=os.getpid(),error=traceback.format_exc()));raise
