"""Development-only expanded grids, frozen solver attribution and role sensitivity."""
from solver import *
from concurrent.futures import ProcessPoolExecutor,as_completed
from scipy.stats import wilcoxon
import itertools
ALPHAS=[0.,.001,.01,.1,1.,10.];LAMS=[.1,1.,3.,10.,30.,40.,100.,300.]
ARMS=['PureProx-Fixed40','PureProx-Fixed10','PureProx-DevSelected','DualReg-DevSelected']
POLICIES=['STE-matched-L2','STE-LP-fallback']

def grid_path(dev,bb,k,s):return RUN/'grid'/dev/bb/f'shot{k}_seed{s:02d}.json'
def grid_job(job):
    dev,bb,k,s=job;pin();guard();ep=episode(bb,dev,k,s,dev=='sph');qi,y=H.labels(dev,dev=='sph');assert np.array_equal(qi,ep['qi']);rows=[]
    for alpha,lam in itertools.product(ALPHAS,LAMS):
        w,b,d=solve(ep,alpha,lam);z=ep['xq']@w+b
        rows.append(dict(alpha=alpha,lambda_=lam,ap=macro_ap(y,z),auc=float(np.mean([roc_auc_score(y[:,c],z[:,c]) for c in range(y.shape[1])])),ece=probability_metrics(y,z)['ece'],gradient=d['max_gradient']))
    dump(grid_path(dev,bb,k,s),dict(complete=True,development_only=True,dev_cohort=dev,backbone=bb,shot=k,seed=s,candidates=rows))

def pool(fn,jobs,stage):
    with ProcessPoolExecutor(max_workers=4) as ex:
        for i,f in enumerate(as_completed([ex.submit(fn,j) for j in jobs]),1):f.result();notice(stage,done=i,total=len(jobs))

def select():
    rows=[];selected={};plateau={}
    for dev in ['chapman','sph']:
        selected[dev]={};plateau[dev]={}
        for bb in BB:
            raw=[]
            for k,s in itertools.product(SHOTS,range(10)):
                for row in read(grid_path(dev,bb,k,s))['candidates']:raw.append(dict(shot=k,seed=s,**row))
            g=pd.DataFrame(raw).groupby(['alpha','lambda_']).agg(mean_dev_ap=('ap','mean'),sd_dev_ap=('ap','std'),mean_dev_auc=('auc','mean'),mean_dev_ece=('ece','mean'),episodes=('ap','size')).reset_index()
            assert len(g)==48 and g.episodes.eq(20).all();g=g.sort_values(['mean_dev_ap','alpha','lambda_'],ascending=[False,True,True]);best=g.iloc[0];old=g[(g.alpha==.1)&(g.lambda_==10.)].iloc[0]
            g['rank_overall']=np.arange(1,49);g['delta_vs_opt_pp']=(g.mean_dev_ap-best.mean_dev_ap)*100;g['delta_vs_01_10_pp']=(g.mean_dev_ap-old.mean_dev_ap)*100
            g['dev_cohort']=dev;g['backbone']=bb;g['reg_family']=np.where(g.alpha==0,'pure_proximity','dual_regularization')
            rows.extend(g.rename(columns={'lambda_':'lambda'}).to_dict('records'))
            def cfg(part):
                v=part.iloc[0];return dict(alpha=float(v.alpha),lam=float(v.lambda_),mean_dev_ap=float(v.mean_dev_ap))
            selected[dev][bb]=dict(pure=cfg(g[g.alpha==0]),dual=cfg(g[g.alpha>0]),global48=cfg(g))
            bands={}
            for width in [.10,.25,.50]:
                bands[str(width)]={}
                for family,part in [('all48',g),('dual40',g[g.alpha>0])]:
                    b=part[part.delta_vs_opt_pp>=-width-1e-12]
                    bands[str(width)][family]=dict(count=len(b),alpha_range=[float(b.alpha.min()),float(b.alpha.max())] if len(b) else None,lambda_range=[float(b.lambda_.min()),float(b.lambda_.max())] if len(b) else None)
            plateau[dev][bb]=dict(best=cfg(g),is_boundary=bool(best.alpha in [0.,10.] or best.lambda_ in [.1,300.]),old_rank=int(g[(g.alpha==.1)&(g.lambda_==10.)].rank_overall.iloc[0]),old_minus_opt_pp=float((old.mean_dev_ap-best.mean_dev_ap)*100),old_within_025pp=bool((best.mean_dev_ap-old.mean_dev_ap)*100<=.25+1e-12),bands=bands)
    pd.DataFrame(rows).to_csv(OUT/'module2_expanded_dev_grid.csv',index=False);dump(OUT/'module2_dev_plateau_summary.json',plateau)
    dump(OUT/'module12_selection.lock.json',dict(selected=selected,spec_sha256=SPEC_SHA,external_labels_used=False,tie_rule='descending exact mean dev AP; ascending alpha then lambda',interpretation='development-cohort sensitivity, not new untouched external validation'))

def configs(bb,dev='chapman'):
    d=read(OUT/'module12_selection.lock.json')['selected'][dev][bb]
    return {'PureProx-Fixed40':dict(alpha=0.,lam=40.),'PureProx-Fixed10':dict(alpha=0.,lam=10.),'PureProx-DevSelected':d['pure'],'DualReg-DevSelected':d['dual'],'MeanAnchor(0.1,10)':dict(alpha=.1,lam=10.)}
def fit_path(phase,bb,ds,k,s):return RUN/'fits'/phase/bb/ds/f'shot{k}_seed{s:02d}.npz'
def fit_job(job):
    phase,bb,ds,k,s=job;pin();guard();swap=phase=='swap';dev='sph' if swap else 'chapman';ep=episode(bb,ds,k,s,swap)
    cc=configs(bb,dev)
    if swap:
        d=read(OUT/'module12_selection.lock.json')['selected']['sph'][bb]
        cc={'PureProx-SPHOpt':d['pure'],'DualReg-SPHOpt':d['dual'],'Global48-SPHOpt':d['global48'],'MeanAnchor(0.1,10)':dict(alpha=.1,lam=10.)}
    pred=dict(query_indices=ep['qi'],support_indices=ep['si']);meta={};cache={}
    policies=['STE-LP-fallback'] if swap or ds=='chapman' else POLICIES
    for mode in policies:
        for method,c in cc.items():
            key=(float(c['alpha']),float(c['lam']),mode if ds=='cpsc2018' else 'matched')
            if key not in cache:
                w,b,diag=solve(ep,c['alpha'],c['lam'],mode);cache[key]=(ep['xq']@w+b,diag)
            z,diag=cache[key];tag=mode+'::'+method;pred[tag]=z;meta[tag]=dict(config=c,solver=diag)
    p=fit_path(phase,bb,ds,k,s);p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**pred);dump(p.with_suffix('.json'),dict(sha256=sha(p),models=meta,query_labels_used=False,selection_sha256=sha(OUT/'module12_selection.lock.json')))

def lock_temperature():
    qi,y=H.labels('chapman');rows={}
    for bb in BB:
        rows[bb]={}
        for method in ARMS+['MeanAnchor(0.1,10)']:
            zz=[np.load(fit_path('main',bb,'chapman',k,s))['STE-LP-fallback::'+method] for k in SHOTS for s in range(10)]
            def obj(logt):
                t=np.exp(logt);return float(np.mean([np.mean(np.logaddexp(0,z/t)-y*z/t) for z in zz]))
            opt=minimize_scalar(obj,bounds=(np.log(.05),np.log(20.)),method='bounded',options={'xatol':1e-10});assert opt.success
            for mode in POLICIES:rows[bb].setdefault(mode,{})[method]=dict(T=float(np.exp(opt.x)),development_nll=float(opt.fun),development='chapman',episodes=20)
    dump(OUT/'module1_temperature.lock.json',dict(selected=rows,external_labels_used=False))

def paired_stats(f):
    rng=np.random.default_rng(20261122);draws=rng.integers(0,20,size=(10000,20));rows=[]
    contrasts=[('PureProx-Fixed40','TaskRes-ECG/BCE'),('MeanAnchor(0.1,10)','PureProx-Fixed10'),('DualReg-DevSelected','PureProx-DevSelected'),('DualReg-DevSelected','MeanAnchor(0.1,10)')]
    group=['backbone','cohort','shot','ste_policy','calibration','evaluation_scope']
    for keys,g in f.groupby(group):
        prefix=dict(zip(group,keys))
        for metric in ['ap','auc','ece','brier','nll','sensitivity95','ppv95']:
            piv=g.pivot(index='seed',columns='method',values=metric);assert np.array_equal(piv.index,np.arange(20))
            for ref,cmp in contrasts:
                d=(piv[ref]-piv[cmp]).to_numpy();boot=d[draws].mean(1);constant=bool(np.ptp(d)<1e-14)
                p=None if constant else float(wilcoxon(d,alternative='two-sided',method='auto').pvalue)
                rows.append(dict(**prefix,reference=ref,comparator=cmp,metric=metric,difference=float(d.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),p_raw=p,constant_difference=constant,bootstrap_draws=10000,query_conditional=True))
    out=pd.DataFrame(rows);out['p_holm']=np.nan;fam=['ste_policy','calibration','evaluation_scope','reference','comparator','metric']
    for _,ix in out.groupby(fam).groups.items():
        g=out.loc[ix];assert len(g)==12;pp=g.p_raw.fillna(1.).to_numpy();order=np.argsort(pp,kind='stable');adj=np.minimum(1.,np.maximum.accumulate(pp[order]*(12-np.arange(12))))
        for ind,p in zip(g.index.to_numpy()[order],adj):
            if pd.notna(out.at[ind,'p_raw']):out.at[ind,'p_holm']=p
    out.to_csv(OUT/'module1_paired_stats_holm.csv',index=False)

def score_main():
    rows=[];T=read(OUT/'module1_temperature.lock.json')['selected']
    for bb,ds,k in itertools.product(BB,DS,SHOTS):
        qi,y=H.labels(ds)
        for s in range(20):
            p=fit_path('main',bb,ds,k,s);assert sha(p)==read(p.with_suffix('.json'))['sha256'];a=np.load(p);assert np.array_equal(qi,a['query_indices']);task=frozen_predictions(bb,ds,k,s)['TaskRes-ECG/BCE']
            cache={}
            for mode in POLICIES:
                for m in ARMS+['MeanAnchor(0.1,10)','TaskRes-ECG/BCE']:
                    z=task if m=='TaskRes-ECG/BCE' else a[(mode if ds!='chapman' else 'STE-LP-fallback')+'::'+m]
                    t=temperature(bb,m) if m=='TaskRes-ECG/BCE' else T[bb][mode][m]['T']
                    for scope in ['CPSC_all9_other_all','CPSC_matched8_other_all']:
                        sl=slice(None,-1) if ds=='cpsc2018' and scope.startswith('CPSC_matched8') else slice(None)
                        yy,zz=y[:,sl],z[:,sl]
                        key=(m,mode if ds=='cpsc2018' else 'shared',scope if ds=='cpsc2018' else 'all')
                        if key not in cache:cache[key]=dict(rank=rank_metrics(yy,zz),Raw=probability_metrics(yy,zz),DevTemperature=probability_metrics(yy,zz/t))
                        rank=cache[key]['rank']
                        for cal,logits in [('Raw',zz),('DevTemperature',zz/t)]:
                            rows.append(dict(backbone=bb,cohort=ds,shot=k,seed=s,method=m,ste_policy=mode,calibration=cal,evaluation_scope=scope,**rank,**cache[key][cal]))
        notice('MODULE1_SCORE',backbone=bb,cohort=ds,shot=k)
    f=pd.DataFrame(rows);f.to_csv(OUT/'module1_episode_metrics.csv',index=False);vals=['ap','auc','ece','brier','nll','sensitivity95','ppv95'];groups=['backbone','cohort','shot','method','ste_policy','calibration','evaluation_scope']
    g=f.groupby(groups)[vals].agg(['mean','std','size']).reset_index();g.columns=['_'.join(v for v in c if v) if isinstance(c,tuple) else c for c in g.columns]
    lookup=g[g.evaluation_scope=='CPSC_matched8_other_all'].set_index([c for c in groups if c!='evaluation_scope']).ap_mean
    g['cpsc2018_matched8_ap']=[float(lookup.loc[tuple(v[c] for c in groups if c!='evaluation_scope')]) if v.cohort=='cpsc2018' else np.nan for _,v in g.iterrows()]
    g['matched8_missing_reason']=np.where(g.cohort=='cpsc2018','not_missing','not_applicable_non_CPSC')
    g.to_csv(OUT/'module1_solver_disentanglement_12cond.csv',index=False)
    scope=[]
    for (m,mode,cal,ev),part in f.groupby(['method','ste_policy','calibration','evaluation_scope']):
        for sc,bb in [('ECGFounder',['ECGFounder']),('MERL',['MERL']),('Overall',BB)]:
            p=part[part.backbone.isin(bb)].groupby(['backbone','cohort','shot'])[vals].mean()
            scope.append(dict(method=m,ste_policy=mode,calibration=cal,evaluation_scope=ev,scope=sc,conditions=len(p),**p.mean().to_dict()))
    pd.DataFrame(scope).to_csv(OUT/'module1_scope_summary.csv',index=False);paired_stats(f)

def score_swap():
    rows=[]
    for bb,ds,k in itertools.product(BB,['chapman','ningbo','cpsc2018'],SHOTS):
        qi,y=H.labels(ds,True)
        for s in range(20):
            p=fit_path('swap',bb,ds,k,s);assert sha(p)==read(p.with_suffix('.json'))['sha256'];a=np.load(p);assert np.array_equal(qi,a['query_indices'])
            taskpath=V151/'fits/swap'/bb/ds/f'shot{k}_seed{s:02d}.npz';assert sha(taskpath)==read(taskpath.with_suffix('.json'))['sha256'];task=np.load(taskpath)['m2__query']
            pred={m:a['STE-LP-fallback::'+m] for m in ['PureProx-SPHOpt','DualReg-SPHOpt','Global48-SPHOpt','MeanAnchor(0.1,10)']};pred['TaskRes-ECG/BCE']=task
            for m,z in pred.items():
                for ev,sl in [('CPSC_all9_other_all',slice(None)),('CPSC_matched8_other_all',slice(None,-1) if ds=='cpsc2018' else slice(None))]:
                    rows.append(dict(backbone=bb,cohort=ds,shot=k,seed=s,method=m,ste_policy='STE-LP-fallback',evaluation_scope=ev,analysis_type='development_cohort_sensitivity_not_untouched_external',ap=macro_ap(y[:,sl],z[:,sl]),auc=float(np.mean([roc_auc_score(y[:,c],z[:,c]) for c in range(y[:,sl].shape[1])]))))
    f=pd.DataFrame(rows);f.to_csv(OUT/'module2_roleswap_episode_metrics.csv',index=False)
    g=f.groupby(['backbone','cohort','shot','method','ste_policy','evaluation_scope','analysis_type'])[['ap','auc']].agg(['mean','std','size']).reset_index();g.columns=['_'.join(v for v in c if v) if isinstance(c,tuple) else c for c in g.columns];g.to_csv(OUT/'module2_roleswap_expanded_check.csv',index=False)

def main():
    guard();pin(10)
    pool(grid_job,[(dev,bb,k,s) for dev in ['chapman','sph'] for bb in BB for k in SHOTS for s in range(10)],'EXPANDED_DEV_GRID');select()
    pool(fit_job,[('main',bb,'chapman',k,s) for bb in BB for k in SHOTS for s in range(10)],'MODULE1_DEV_FITS');lock_temperature()
    pool(fit_job,[('main',bb,ds,k,s) for bb in BB for ds in DS for k in SHOTS for s in range(20)],'MODULE1_EXTERNAL_FITS')
    pool(fit_job,[('swap',bb,ds,k,s) for bb in BB for ds in ['chapman','ningbo','cpsc2018'] for k in SHOTS for s in range(20)],'MODULE2_SWAP_FITS')
    dump(RUN/'MODULE12_FITS_COMPLETE.json',dict(all_passed=True));score_main();score_swap()
    dump(RUN/'MODULE12_COMPLETE.json',dict(all_passed=True));notice('MODULE12_COMPLETE')

if __name__=='__main__':
    try:main()
    except Exception:dump(RUN/'status/MODULE12_ERROR.json',dict(error=traceback.format_exc()));raise
