from core import *
import pandas as pd
from scipy.stats import wilcoxon

def paired_table(rows):
 df=pd.DataFrame(rows);out=[];rng=np.random.default_rng(20261103)
 for (backbone,ds,k),g in df.groupby(['backbone','dataset','shot']):
  a=g[g.method=='SASM-Ens'].sort_values('seed')
  for method in sorted(set(g.method)-{'SASM-Ens'}):
   b=g[g.method==method].sort_values('seed');assert a.seed.tolist()==b.seed.tolist() and len(a)==20
   for metric_name in ['ap','auc']:
    delta=a[metric_name].to_numpy()-b[metric_name].to_numpy();boots=delta[rng.integers(0,20,(10000,20))].mean(1)
    out.append(dict(backbone=backbone,dataset=ds,shot=int(k),comparator=method,metric=metric_name,mean_difference=float(delta.mean()),ci_low=float(np.quantile(boots,.025)),ci_high=float(np.quantile(boots,.975)),wins=int((delta>0).sum()),p=float(wilcoxon(delta).pvalue) if np.any(delta) else 1.))
 if out:
  # One explicitly defined family per backbone/metric across all external comparisons.
  t=pd.DataFrame(out)
  for _,g in t.groupby(['backbone','metric']):
   ix=g.sort_values('p').index;p=t.loc[ix,'p'].to_numpy();t.loc[ix,'p_holm']=np.minimum(1,np.maximum.accumulate(p*(len(p)-np.arange(len(p)))))
  return t
 return pd.DataFrame()

def save_tables(rows,prefix):
 df=pd.DataFrame(rows);df.to_csv(R/f'{prefix}_episode_metrics.csv',index=False)
 summary=df.groupby(['backbone','dataset','shot','method']).agg(n=('ap','size'),ap_mean=('ap','mean'),ap_sd=('ap','std'),auc_mean=('auc','mean'),auc_sd=('auc','std')).reset_index();summary.to_csv(R/f'{prefix}_summary.csv',index=False)
 paired_table([r for r in rows if r['dataset'] in EXTERNAL_DATASETS]).to_csv(R/f'{prefix}_paired_effects.csv',index=False)
 return summary

