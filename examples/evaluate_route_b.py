"""Re-run Route B from released frozen features; assert archived per-seed parity."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import argparse,csv,json,sys,time
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sasm.cache import load_cohort
from sasm.route_b import load_heads,selected_config,fit_route_b,BACKBONES,COHORTS,SUITES

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=ROOT/'data')
    p.add_argument('--suite',choices=SUITES+('all',),default='fullpool')
    p.add_argument('--backbone',choices=BACKBONES+('all',),default='ecgfounder')
    p.add_argument('--cohort',choices=COHORTS+('all',),default='ningbo')
    p.add_argument('--shot',choices=['5','10','all'],default='5')
    p.add_argument('--seeds',default='0',help='0..19 comma-separated, or all')
    p.add_argument('--config',choices=['selected','fixed'],default='selected')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/route_b.json')
    a=p.parse_args()
    seeds=list(range(20)) if a.seeds=='all' else list(map(int,a.seeds.split(',')))
    if not seeds or len(set(seeds))!=len(seeds) or any(s not in range(20) for s in seeds):p.error('invalid seeds')
    suites=SUITES if a.suite=='all' else [a.suite]
    bbs=BACKBONES if a.backbone=='all' else [a.backbone]
    cohorts=[c for c in COHORTS if c != "cpsc2018"] if a.cohort=='all' else [a.cohort]
    if "cpsc2018" in cohorts:
        p.error("CPSC2018 uses the nine-class protocol: run examples/evaluate_cpsc9.py instead.")
    shots=[5,10] if a.shot=='all' else [int(a.shot)]
    with (ROOT/'results/route_b/episode_metrics.csv').open(newline='',encoding='utf-8-sig') as f:
        ref={(r['suite'],r['backbone'].lower(),r['dataset'],int(r['shot']),int(r['seed']),r['method']):r for r in csv.DictReader(f)}
    rows=[];atol=5e-5 # fractions, i.e. 0.005 percentage points; no retuning on external data
    with threadpool_limits(limits=1):
        for bb in bbs:
            for ds in cohorts:
                F,_,_,qi,qy,_=load_cohort(a.data_root,bb,ds)
                fq=F[qi].astype(float)
                for suite in suites:
                    if bb=='merl' and suite!='fullpool':
                        if a.suite!='all':p.error('native suites require ECGFounder')
                        continue
                    rw,rb,covered=load_heads(a.data_root,bb,ds,suite)
                    cfg=selected_config(a.data_root,bb,suite,a.config)
                    for shot in shots:
                        for seed in seeds:
                            ep=json.loads((a.data_root/f'cohorts/{ds}/episodes/shot{shot}_seed{seed:02d}.json').read_text())
                            si=np.asarray(ep['support'],int);ys=np.asarray(ep['support_y'],float)
                            if np.intersect1d(si,qi).size:raise ValueError('support/query overlap')
                            start=time.perf_counter()
                            w,b,diags=fit_route_b(F[si],ys,rw,rb,cfg,suite)
                            pred=fq@w+b
                            methods={a.config:pred}
                            if suite=='fullpool':methods['direct']=fq@rw[0]+rb[0]
                            for method,score in methods.items():
                                metrics={'ap':float(average_precision_score(qy[:,covered],score[:,covered],average='macro')),
                                         'auc':float(roc_auc_score(qy[:,covered],score[:,covered],average='macro'))}
                                expected=ref[(suite,bb,ds,shot,seed,method)]
                                errors={k:abs(metrics[k]-float(expected[k])) for k in ['ap','auc']}
                                row=dict(suite=suite,backbone=bb,dataset=ds,shot=shot,seed=seed,method=method,
                                         covered_labels=len(covered),**metrics,reference_abs_error=errors,
                                         reference_pass=max(errors.values())<=atol,
                                         max_gradient=max(d['max_gradient'] for d in diags),
                                         fit_and_prediction_seconds=time.perf_counter()-start)
                                rows.append(row);print(json.dumps(row),flush=True)
                                if not row['reference_pass']:
                                    a.output.parent.mkdir(parents=True,exist_ok=True)
                                    a.output.write_text(json.dumps({'rows':rows,'passed':False},indent=2))
                                    raise RuntimeError('archived per-seed reference mismatch; output retained')
    result={'rows':rows,'passed':True,'metric_units':'fractions','reference_atol_fraction':atol,
            'selection':'archived Chapman-only config; no search in this command'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'passed':True,'rows':len(rows),'output':str(a.output)}))

if __name__=='__main__':main()
