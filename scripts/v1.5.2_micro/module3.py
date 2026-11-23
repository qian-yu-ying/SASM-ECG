"""Frozen-prediction calibration resampling, pathology DCA and figure inputs."""
from micro_common import *
from concurrent.futures import ProcessPoolExecutor,as_completed
import ast,itertools

def weighted_metrics(y,z,counts,inverse,chunk=200):
    """Exact classwise ECE before seed averaging; no mean-logit shortcut."""
    p=expit(z);n,c=y.shape;ng=counts.shape[1];bins=np.minimum((p*10).astype(int),9)
    mat=np.zeros((ng,c*10+2));col=np.arange(c)[None,:]*10+bins
    np.add.at(mat,(np.repeat(inverse,c),col.ravel()),(p-y).ravel())
    mat[:,-2]=np.bincount(inverse,weights=((p-y)**2).mean(1),minlength=ng)
    mat[:,-1]=np.bincount(inverse,weights=(np.logaddexp(0,z)-y*z).mean(1),minlength=ng)
    sizes=np.bincount(inverse,minlength=ng);out=np.empty((len(counts),3))
    for start in range(0,len(counts),chunk):
        co=np.ascontiguousarray(counts[start:start+chunk],dtype=float);den=co@sizes;num=co@mat
        out[start:start+len(co),0]=abs(num[:,:c*10]).sum(1)/(c*den)
        out[start:start+len(co),1:]=num[:,-2:]/den[:,None]
    return out

def unit_test():
    guard();pin(10);p=V151/'scripts/v1.5.1_adapters/evaluate.py';tree=ast.parse(p.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='probability_metrics')
    ns=dict(np=np,expit=expit);exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns)
    errors=[];checks=[]
    for bb in BB:
        for ds in DS:
            qi,y=H.labels(ds);pred=frozen_predictions(bb,ds,5,0)
            for m,z in pred.items():
                got=weighted_metrics(y,z,np.ones((1,len(y)),dtype=np.int16),np.arange(len(y)))[0];ref=ns['probability_metrics'](y,z)
                err=max(abs(got[i]-ref[k]) for i,k in enumerate(['ece','brier','nll']));errors.append(err);checks.append(dict(backbone=bb,cohort=ds,method=m,max_abs_diff=err))
    out=dict(passed=max(errors)<=1e-13,max_abs_diff=float(max(errors)),source_sha256=sha(p),w_equals_one=True,checks=checks)
    dump(OUT/'calib_weighted_unit_test.json',out)
    if not out['passed']:raise RuntimeError('Weighted calibration gate failed; no later module may run')
    notice('CALIB_WEIGHTED_TEST_PASSED',max_abs_diff=out['max_abs_diff'])

def valid_draws(y,co,iv):
    valid=np.ones(len(co),bool);g=co.shape[1]
    for c in range(y.shape[1]):
        p=co@np.bincount(iv,weights=y[:,c],minlength=g);n=co@np.bincount(iv,weights=1-y[:,c],minlength=g);valid&=(p>0)&(n>0)
    return valid
def interval(x,valid):
    ok=valid&np.isfinite(x);v=x[ok]
    return dict(ci_low=float(np.quantile(v,.025)) if len(v) else None,ci_high=float(np.quantile(v,.975)) if len(v) else None,valid_draws=len(v),total_draws=len(x),undefined_replicates=int((~ok).sum()),undefined_reason='one or more fixed labels absent from resampled positives/negatives' if not ok.all() else 'none')

def endpoints(ds):
    names=label_names(ds);norm=lambda s:''.join(c for c in s.upper() if c.isalnum())
    normal=[i for i,n in enumerate(names) if norm(n) in ['NORMAL','NORM','NORMALECG']]
    sinus=[i for i,n in enumerate(names) if norm(n) in ['SR','SINUSRHYTHM','NORMALSINUSRHYTHM']]
    exact={'AF':'AFIB/AFL','AFIB':'AFIB/AFL','ATRIALFIBRILLATION':'AFIB/AFL','AFL':'AFIB/AFL','ATRIALFLUTTER':'AFIB/AFL',
      '1AVB':'1AVB/I-AVB','IAVB':'1AVB/I-AVB','WITH1STDEGREEAVBLOCK':'1AVB/I-AVB',
      'LBBB':'LBBB/CLBBB','CLBBB':'LBBB/CLBBB','LEFTBUNDLEBRANCHBLOCK':'LBBB/CLBBB',
      'RBBB':'RBBB/CRBBB','CRBBB':'RBBB/CRBBB','RIGHTBUNDLEBRANCHBLOCK':'RBBB/CRBBB',
      'PVC':'PVC','PREMATUREVENTRICULARCOMPLEXES':'PVC','PAC':'PAC','ATRIALPREMATURECOMPLEX':'PAC'}
    selected={i:exact[norm(n)] for i,n in enumerate(names) if norm(n) in exact}
    masks={'All_44':list(range(len(names))),'Pathology_Excl_Normal':[i for i in range(len(names)) if i not in normal],'Pathology_Excl_Normal_And_SR':[i for i in range(len(names)) if i not in normal+sinus]}
    return dict(label_names=names,normal_indices=normal,sinus_rhythm_indices=sinus,subsets=masks,selected_endpoints=[dict(index=i,label_name=names[i],clinical_family=f) for i,f in selected.items()],normal_endpoint_present=bool(normal),note='Ningbo selected 22 endpoints may contain SR without a separate Normal endpoint; no artificial label is introduced. Incomplete RBBB is not relabeled complete RBBB.')

def condition(job):
    bb,ds,k=job;pin();guard();assert read(OUT/'calib_weighted_unit_test.json')['passed']
    qi,y=H.labels(ds);a=np.load(V151/'draws'/f'{ds}.npz');assert np.array_equal(qi,a['query_indices']);co=a['counts'];iv=a['inverse'];valid=valid_draws(y,co,iv);B=len(co)
    methods=METHODS;acc={(m,cal):np.zeros((B,3)) for m in methods for cal in ['Raw','DevTemperature']};point={key:np.zeros(3) for key in acc}
    classes=[];dca=[];selected=[];reliability=[];meta=endpoints(ds);names=meta['label_names'];prev=y.mean(0);start=time.time()
    for s in range(20):
        pred=frozen_predictions(bb,ds,k,s)
        for m,z in pred.items():
            # Tip-F is required for classwise/DCA, not for the CI core set.
            for cal,zz in [('Raw',z),('DevTemperature',z/temperature(bb,m))]:
                if m in methods:
                    acc[(m,cal)]+=weighted_metrics(y,zz,co,iv)/20
                    q=probability_metrics(y,zz);point[(m,cal)]+=np.array([q[n] for n in ['ece','brier','nll']])/20
                prob=expit(zz)
                for c,n in enumerate(names):classes.append(dict(backbone=bb,cohort=ds,shot=k,seed=s,method=m,calibration=cal,label=c,label_name=n,**probability_metrics(y[:,c:c+1],zz[:,c:c+1])))
                for pt in [.02,.05,.10,.15,.20]:
                    decision=zz>=logit(pt);tp=(decision*y).sum(0);fp=(decision*(1-y)).sum(0);nb=tp/len(y)-fp/len(y)*pt/(1-pt);allnb=prev-(1-prev)*pt/(1-pt)
                    base=dict(backbone=bb,cohort=ds,shot=k,seed=s,method=m,calibration=cal,pt=pt)
                    for subset,cols in meta['subsets'].items():
                        dca.append(dict(**base,subset=subset,endpoints=len(cols),NB=float(nb[cols].mean()),standardized_NB=float((nb[cols]/prev[cols]).mean()),treat_all_NB=float(allnb[cols].mean()),treat_none_NB=0.,aggregation='equal endpoints within cohort; scope averages conditions equally'))
                    if m in ['MeanAnchor','TaskRes-ECG/BCE','Direct-LP','LP']:
                        for item in meta['selected_endpoints']:
                            c=item['index'];selected.append(dict(**base,label=c,label_name=names[c],clinical_family=item['clinical_family'],query_n=len(y),positives=int(y[:,c].sum()),NB=float(nb[c]),standardized_NB=float(nb[c]/prev[c]),treat_all_NB=float(allnb[c]),treat_none_NB=0.))
                if cal=='Raw' and m in ['LP','Direct-LP','Direct-LP-Refit-b','TaskRes-ECG/BCE','LP++-ECG/BCE','MeanAnchor']:
                    ix=np.minimum((prob*10).astype(int),9).ravel();counts=np.bincount(ix,minlength=10);ps=np.bincount(ix,weights=prob.ravel(),minlength=10);ys=np.bincount(ix,weights=y.ravel(),minlength=10)
                    for b in range(10):reliability.append(dict(backbone=bb,cohort=ds,shot=k,seed=s,method=m,bin=b,count=int(counts[b]),probability_sum=float(ps[b]),positive_sum=float(ys[b])))
        notice('MODULE3_CALIB_EPISODES',backbone=bb,cohort=ds,shot=k,done=s+1,total=20,seconds=time.time()-start)
    path=RUN/'module3_conditions'/bb/ds/f'shot{k}';path.parent.mkdir(parents=True,exist_ok=True);arr={f'{m}::{cal}':v for (m,cal),v in acc.items()};arr['valid']=valid
    np.savez_compressed(path.with_suffix('.npz'),**arr);rows=[]
    for m in methods:
        if m=='MeanAnchor':continue
        for cal in ['Raw','DevTemperature']:
            for j,n in enumerate(['ece','brier','nll']):rows.append(dict(backbone=bb,cohort=ds,shot=k,reference='MeanAnchor',comparator=m,calibration=cal,metric=n,difference=float(point[('MeanAnchor',cal)][j]-point[(m,cal)][j]),conditional_support_seeds=20,**interval(acc[('MeanAnchor',cal)][:,j]-acc[(m,cal)][:,j],valid)))
    for suffix,data in [('ci',rows),('class',classes),('dca',dca),('selected',selected),('reliability',reliability)]:pd.DataFrame(data).to_csv(path.with_suffix('.'+suffix+'.csv'),index=False)
    dump(path.with_suffix('.json'),dict(complete=True,seconds=time.time()-start,arrays_sha256=sha(path.with_suffix('.npz')),valid_draws=int(valid.sum()),total_draws=B))

def aggregate():
    jobs=list(itertools.product(BB,DS,SHOTS))
    def join(suffix):return pd.concat([pd.read_csv((RUN/'module3_conditions'/bb/ds/f'shot{k}').with_suffix('.'+suffix+'.csv')) for bb,ds,k in jobs],ignore_index=True)
    join('ci').to_csv(OUT/'module3_calib_query_cluster_ci_12cond.csv',index=False)
    c=join('class');c.groupby(['backbone','cohort','shot','method','calibration','label','label_name'])[['ece','brier','nll']].mean().reset_index().to_csv(OUT/'module3_classwise_calibration_44endpoints.csv',index=False)
    d=join('dca');d.to_csv(OUT/'module3_pathology_dca_episodes.csv',index=False)
    keys=['backbone','cohort','shot','method','calibration','pt','subset','endpoints'];s=d.groupby(keys)[['NB','standardized_NB','treat_all_NB','treat_none_NB']].mean().reset_index();s.to_csv(OUT/'module3_pathology_dca_summary.csv',index=False)
    wins=[]
    for (subset,cal),g in s.groupby(['subset','calibration']):
        piv=g.pivot(index=['backbone','cohort','shot','pt'],columns='method',values='NB')
        for m in ['TaskRes-ECG/BCE','LP++-ECG/BCE','Direct-LP','LP','Tip-Adapter-F-ECG/BCE']:
            delta=piv['MeanAnchor']-piv[m];wins.append(dict(subset=subset,calibration=cal,reference='MeanAnchor',comparator=m,pairs=len(delta),wins=int((delta>0).sum()),ties=int((delta==0).sum()),losses=int((delta<0).sum()),significance_claim=False))
    pd.DataFrame(wins).to_csv(OUT/'module3_pathology_dca_wins.csv',index=False)
    v=join('selected');v.groupby(['backbone','cohort','shot','method','calibration','pt','label','label_name','clinical_family','query_n','positives'])[['NB','standardized_NB','treat_all_NB','treat_none_NB']].mean().reset_index().to_csv(OUT/'module3_selected_endpoints_dca.csv',index=False)
    rb=join('reliability').groupby(['backbone','method','bin'])[['count','probability_sum','positive_sum']].sum().reset_index()
    rb['mean_predicted_probability']=np.divide(rb.probability_sum,rb['count'],out=np.full(len(rb),np.nan),where=rb['count']>0);rb['empirical_positive_rate']=np.divide(rb.positive_sum,rb['count'],out=np.full(len(rb),np.nan),where=rb['count']>0)
    rb['undefined_reason']=np.where(rb['count']>0,'none','empty pooled reliability bin');rb['aggregation']='pooled record-label pairs over fixed queries and 20 supports; figure legend ECE is equal-condition macro ECE'
    rb.to_csv(OUT/'module3_reliability_bins_ecgf_merl.csv',index=False)
    dump(OUT/'module3_pathology_dca_meta.json',dict(cohorts={ds:endpoints(ds) for ds in DS},aggregation='cohort/class endpoints first; equal 12 conditions at scope level; All_44 indicates coverage, not equal weight of all 44 labels globally',exploratory=True))
    rng=np.random.default_rng(20261123);perms={'ningbo':np.arange(2000),'cpsc2018':rng.permutation(2000),'sph':rng.permutation(2000)}
    dump(OUT/'module3_crosscohort_permutations.json',dict(seed=20261123,permutations={k:v.tolist() for k,v in perms.items()},shared_across_backbones_and_shots=True))
    scopes=[];cond=join('ci')
    for scope,bbs in [('ECGFounder',['ECGFounder']),('MERL',['MERL']),('Overall',BB)]:
        selected_jobs=[j for j in jobs if j[0] in bbs];valid=np.ones(2000,bool);acc={}
        for bb,ds,k in selected_jobs:
            a=np.load((RUN/'module3_conditions'/bb/ds/f'shot{k}').with_suffix('.npz'));ix=perms[ds];valid&=a['valid'][ix]
            for m in METHODS:
                for cal in ['Raw','DevTemperature']:
                    key=m+'::'+cal;acc.setdefault(key,np.zeros((2000,3)));acc[key]+=a[key][ix]/len(selected_jobs)
        for m in METHODS:
            if m=='MeanAnchor':continue
            for cal in ['Raw','DevTemperature']:
                for j,n in enumerate(['ece','brier','nll']):
                    part=cond[(cond.backbone.isin(bbs))&(cond.comparator==m)&(cond.calibration==cal)&(cond.metric==n)]
                    scopes.append(dict(scope=scope,conditions=len(selected_jobs),reference='MeanAnchor',comparator=m,calibration=cal,metric=n,difference=float(part.difference.mean()),permutation_seed=20261123,conditional_support_seeds=20,**interval(acc['MeanAnchor::'+cal][:,j]-acc[m+'::'+cal][:,j],valid)))
    pd.DataFrame(scopes).to_csv(OUT/'module3_calib_query_cluster_ci_scope.csv',index=False)

def main():
    pin(10);guard();assert (RUN/'MODULE12_COMPLETE.json').exists();unit_test()
    jobs=list(itertools.product(BB,DS,SHOTS))
    with ProcessPoolExecutor(max_workers=4) as ex:
        for i,f in enumerate(as_completed([ex.submit(condition,j) for j in jobs]),1):f.result();notice('MODULE3_CONDITIONS',done=i,total=len(jobs))
    aggregate();dump(RUN/'MODULE3_COMPLETE.json',dict(all_passed=True));notice('MODULE3_COMPLETE')

if __name__=='__main__':
    try:main()
    except Exception:dump(RUN/'status/MODULE3_ERROR.json',dict(error=traceback.format_exc()));raise
