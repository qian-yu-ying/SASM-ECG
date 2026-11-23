"""Scoring occurs only after all parameters and thresholds are locked."""
from runner import fit_path,freeze_path
from core import *
from scipy.stats import wilcoxon
import gzip

def rocpoint(y,z,spec=.95):
    fpr,tpr,t=roc_curve(y,z,drop_intermediate=True);i=np.flatnonzero(1-fpr>=spec-1e-12)[-1]
    pred=z>=t[i];tp=float((y*pred).sum());fp=float(((1-y)*pred).sum())
    return float(tpr[i]),tp/(tp+fp) if tp+fp else 0.
def probability_metrics(y,z):
    p=expit(z);ec=[];f=[];precision=[]
    for c in range(y.shape[1]):
        ix=np.minimum((p[:,c]*10).astype(int),9);ec.append(abs(np.bincount(ix,weights=p[:,c]-y[:,c],minlength=10)).sum()/len(y))
        pred=p[:,c]>=.5;tp=float((pred*y[:,c]).sum());den=float(pred.sum()+y[:,c].sum())
        f.append(2*tp/den if den else 0.);precision.append(tp/float(pred.sum()) if pred.any() else 0.)
    return dict(ece=float(np.mean(ec)),brier=float(((p-y)**2).mean()),nll=float((np.logaddexp(0,z)-y*z).mean()),f1=float(np.mean(f)),ppv50=float(np.mean(precision)))
def label_names(ds):
    if ds=='cpsc2018':return np.load(H.O/'labels.npz')['names'].astype(str).tolist()
    return read(H.B/'data'/ds/'metadata.json')['labels']

def load_predictions(mode,bb,ds,k,s,include_support=False):
    p=fit_path(mode,bb,ds,k,s);meta=read(p.with_suffix('.json'));assert sha(p)==meta['sha256']
    z=np.load(p);pred={m:z[f'm{i}__query'] for i,m in enumerate(METHODS)}
    sup={m:z[f'm{i}__support'] for i,m in enumerate(METHODS)}
    baseline,bp=archived(bb,ds,k,s,mode=='swap')
    for m in CONTROLS:pred[m]=baseline[m]
    if include_support:
        us=np.load(DCA/'support'/bb/ds/f'shot{k}_seed{s:02d}.npz')
        for m in CONTROLS:sup[m]=us[m]
    return pred,sup,z['support_y'],z['query_indices']

def score_main():
    assert (RUN/'MAIN_FITS_COMPLETE.json').exists()
    new=read(OUT/'calibration_thresholds_main.lock.json')['selected'];temp=read(OLD/'temperature.lock.json')['selected'];oldcut=read(DCA/'global_thresholds.lock.json')['selected']
    metrics_rows=[];labels_rows=[];dca_rows=[];rule_rows=[]
    for bb in ['ECGFounder','MERL']:
        for ds in ['ningbo','cpsc2018','sph']:
            qi,y=H.labels(ds);names=label_names(ds)
            for k in [5,10]:
                for s in range(20):
                    pred,sup,sy,order=load_predictions('main',bb,ds,k,s,True);assert np.array_equal(qi,order)
                    adjustment=logit(y.mean(0))-logit((sy.sum(0)+.5)/(len(sy)+1))
                    for m,z in pred.items():
                        ap=np.array([average_precision_score(y[:,c],z[:,c]) for c in range(y.shape[1])]);auc=np.array([roc_auc_score(y[:,c],z[:,c]) for c in range(y.shape[1])])
                        roc95=[rocpoint(y[:,c],z[:,c],.95) for c in range(y.shape[1])];roc90=[rocpoint(y[:,c],z[:,c],.90) for c in range(y.shape[1])]
                        T=new[bb][m]['T'] if m in METHODS else temp[bb][m]['T']
                        prefix=dict(backbone=bb,cohort=ds,shot=k,seed=s,method=m)
                        for mode,zz in [('Raw',z),('DevTemperature',z/T),('Adj',z+adjustment)]:
                            metrics_rows.append(dict(**prefix,calibration=mode,ap=float(ap.mean()),auc=float(auc.mean()),
                              sensitivity95=float(np.mean([v[0] for v in roc95])),ppv95=float(np.mean([v[1] for v in roc95])),
                              sensitivity90=float(np.mean([v[0] for v in roc90])),ppv90=float(np.mean([v[1] for v in roc90])),**probability_metrics(y,zz)))
                        if k==5:
                            for c,name in enumerate(names):labels_rows.append(dict(**prefix,label=c,label_name=name,ap=float(ap[c]),auc=float(auc[c]),query_positives=int(y[:,c].sum()),support_positives=int(sy[:,c].sum())))
                        for cal,zz in [('Raw',z),('DevTemperature',z/T)]:
                            for pt in PTS:dca_rows.append(dict(**prefix,calibration=cal,pt=pt,**dca_values(y,zz,pt)))
                        cut=new[bb][m]['raw_logit_threshold'] if m in METHODS else oldcut[bb][m]['raw_logit_threshold']
                        support_threshold=np.array([np.quantile(sup[m][sy[:,c]==0,c],.95,method='higher') if np.any(sy[:,c]==0) else np.inf for c in range(y.shape[1])])
                        for rule,threshold in [('A_ChapmanGlobal',cut),('B_SupportNegative',support_threshold)]:
                            decision=z>threshold;assert np.array_equal(decision,(z/T)>(threshold/T))
                            co=confusion(y,decision);v={a:float(co[a].mean()) for a in ['specificity','sensitivity','PPV','f1']}
                            for cal in ['Raw','DevTemperature']:rule_rows.append(dict(**prefix,calibration=cal,rule=rule,temperature_decision_invariant=True,**v))
                notice('MAIN_SCORE',backbone=bb,cohort=ds,shot=k)
    f=pd.DataFrame(metrics_rows);lf=pd.DataFrame(labels_rows);df=pd.DataFrame(dca_rows);rf=pd.DataFrame(rule_rows)
    for frame,name in [(f,'metric_episodes.csv'),(lf,'classwise_5shot_episodes.csv'),(df,'dca_episodes.csv'),(rf,'locked_rule_episodes.csv')]:frame.to_csv(OUT/name,index=False)
    summaries(f,['backbone','cohort','shot','method','calibration'],['ap','auc','ece','brier','nll','f1','ppv50','sensitivity95','ppv95','sensitivity90','ppv90']).to_csv(OUT/'translational_calibration_summary.csv',index=False)
    rank=f[f.calibration=='Raw'];summaries(rank,['backbone','cohort','shot','method'],['ap','auc']).to_csv(OUT/'main_12cond_summary.csv',index=False)
    scopes=[]
    for (method,cal),g in f.groupby(['method','calibration']):
        for scope,part in [('ECGFounder',g[g.backbone=='ECGFounder']),('MERL',g[g.backbone=='MERL']),('Overall',g)]:
            condition=part.groupby(['backbone','cohort','shot'])[['ap','auc','ece','brier','nll','f1','ppv50','sensitivity95','ppv95']].mean()
            scopes.append(dict(method=method,calibration=cal,scope=scope,conditions=len(condition),**condition.mean().to_dict()))
    pd.DataFrame(scopes).to_csv(OUT/'scope_summary.csv',index=False)
    cs=summaries(lf,['backbone','cohort','method','label','label_name'],['ap','auc','support_positives']);cs['low_shot']=cs['support_positives_mean']<=10
    cs.to_csv(OUT/'classwise_5shot_ap_44endpoints.csv',index=False)
    assert cs.groupby(['backbone','method']).size().eq(44).all()
    low=cs.groupby(['backbone','method','low_shot'],as_index=False).agg(endpoints=('label','size'),mean_ap=('ap_mean','mean'))
    low.to_csv(OUT/'classwise_lowshot_strata.csv',index=False)
    for name,frame,groups,vals in [('translational_dca_summary.csv',df,['backbone','cohort','shot','method','calibration','pt'],['NB','standardized_NB']),('translational_locked_rules_summary.csv',rf,['backbone','cohort','shot','method','calibration','rule'],['specificity','sensitivity','PPV','f1'])]:summaries(frame,groups,vals).to_csv(OUT/name,index=False)
    wins=[]
    for cal in ['Raw','DevTemperature']:
        for m in METHODS:
            g=df[df.calibration==cal].groupby(['backbone','cohort','shot','pt','method']).NB.mean().unstack('method')
            delta=g['MeanAnchor']-g[m]
            wins.append(dict(calibration=cal,contrast='MeanAnchor minus '+m,pairs=len(delta),wins=int((delta>0).sum()),ties=int((delta==0).sum()),losses=int((delta<0).sum())))
    pd.DataFrame(wins).to_csv(OUT/'dca_pair_wins.csv',index=False)
    with gzip.open(OUT/'per_seed_raw_records.json.gz','wt',encoding='utf-8') as stream:
        json.dump(dict(spec_sha256=SPEC_SHA,metric_units='fractions',metric_episodes=metrics_rows,dca_episodes=dca_rows,locked_rules=rule_rows,classwise_5shot=labels_rows),stream,allow_nan=False)
    dump(RUN/'MAIN_SCORING_COMPLETE.json',dict(rows=len(f),methods=sorted(f.method.unique()),spec_sha256=SPEC_SHA))

def summaries(frame,groups,values):
    g=frame.groupby(groups,dropna=False)[values].agg(['mean','std','size']).reset_index()
    g.columns=['_'.join(str(v) for v in c if v) if isinstance(c,tuple) else c for c in g.columns]
    return g

def paired(frame,group_fields,values):
    rows=[];rng=np.random.default_rng(20261122);draws=rng.integers(0,20,size=(10000,20))
    for key,g in frame.groupby(group_fields,dropna=False):
        key=key if isinstance(key,tuple) else (key,);prefix=dict(zip(group_fields,key))
        for metric in values:
            pivot=g.pivot(index='seed',columns='method',values=metric);assert np.array_equal(pivot.index,np.arange(20))
            for ref in ['MeanAnchor','SASM-Ens']:
                for m in METHODS:
                    d=(pivot[ref]-pivot[m]).to_numpy();boot=d[draws].mean(1);constant=bool(np.ptp(d)<1e-14)
                    p=None if constant else float(wilcoxon(d,alternative='two-sided',method='auto').pvalue)
                    rows.append(dict(**prefix,metric=metric,reference=ref,comparator=m,difference=float(d.mean()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),p_raw=p,constant_difference=constant,support_seeds=20,bootstrap_draws=10000,query_conditional=True))
    result=pd.DataFrame(rows);result['p_holm']=None
    family=[c for c in group_fields if c not in ['backbone','cohort','shot']]+['metric','reference','comparator']
    for _,indices in result.groupby(family,dropna=False).groups.items():
        group=result.loc[indices];assert len(group)==12
        pp=group.p_raw.fillna(1.).to_numpy();order=np.argsort(pp,kind='stable');adj=np.minimum(1,np.maximum.accumulate(pp[order]*(len(pp)-np.arange(len(pp)))))
        for ix,v in zip(group.index.to_numpy()[order],adj):
            if result.at[ix,'p_raw'] is not None and pd.notna(result.at[ix,'p_raw']):result.at[ix,'p_holm']=float(v)
    return result

def score_swap():
    assert (RUN/'SWAP_FITS_COMPLETE.json').exists();rows=[]
    for bb in ['ECGFounder','MERL']:
        for ds in ['chapman','ningbo','cpsc2018']:
            qi,y=H.labels(ds,True)
            for k in [5,10]:
                for s in range(20):
                    pred,_,sy,order=load_predictions('swap',bb,ds,k,s);assert np.array_equal(qi,order)
                    for m,z in pred.items():rows.append(dict(backbone=bb,cohort=ds,shot=k,seed=s,method=m,ap=macro_ap(y,z),auc=float(np.mean([roc_auc_score(y[:,c],z[:,c]) for c in range(y.shape[1])]))))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'roleswap_episodes.csv',index=False)
    summaries(frame,['backbone','cohort','shot','method'],['ap','auc']).to_csv(OUT/'roleswap_sph_12cond_summary.csv',index=False)
    paired(frame,['backbone','cohort','shot'],['ap','auc']).to_csv(OUT/'roleswap_paired_stats_holm.csv',index=False)
    dump(RUN/'SWAP_SCORING_COMPLETE.json',dict(rows=len(frame)))

def main():
    pin(10);score_main();f=pd.read_csv(OUT/'metric_episodes.csv')
    paired(f,['backbone','cohort','shot','calibration'],['ap','auc','ece','brier','nll','f1','sensitivity95','ppv95']).to_csv(OUT/'main_paired_stats_holm.csv',index=False)
    paired(pd.read_csv(OUT/'dca_episodes.csv'),['backbone','cohort','shot','calibration','pt'],['NB','standardized_NB']).to_csv(OUT/'dca_paired_stats_holm.csv',index=False)
    paired(pd.read_csv(OUT/'locked_rule_episodes.csv'),['backbone','cohort','shot','calibration','rule'],['specificity','sensitivity','PPV','f1']).to_csv(OUT/'locked_rule_paired_stats_holm.csv',index=False)
    score_swap();notice('SCORING_AND_SEED_STATS_COMPLETE')

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback;dump(RUN/'status'/'EVALUATION_ERROR.json',dict(traceback=traceback.format_exc()));raise
