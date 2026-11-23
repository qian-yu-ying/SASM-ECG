"""Replay archived predictions with independently recomputed metrics; never fit adapters."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,hashlib,traceback,importlib.util
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path('/home/wy/ECGFounder/supplement_queue_20261101_remote')
OLD=ROOT/'supplement_core_20261118'
RUN=ROOT/'sasm_v151_adapters_20261122'
OUT=RUN/'results/v1.5.1_adapters'
OUT.mkdir(parents=True,exist_ok=True)
METHODS=['LP','Direct','SASM-Ens','MeanAnchor','Direct-LP']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def independent(y,z):
    assert y.shape==z.shape and np.isfinite(z).all()
    pp=expit(z);ec=[]
    for c in range(y.shape[1]):
        ix=np.minimum(np.floor(pp[:,c]*10).astype(int),9)
        cnt=np.bincount(ix,minlength=10)
        diff=np.bincount(ix,weights=pp[:,c]-y[:,c],minlength=10)
        ec.append(np.abs(diff).sum()/len(y))
    return {'ap':float(np.mean([average_precision_score(y[:,c],z[:,c]) for c in range(y.shape[1])])),
            'auc':float(np.mean([roc_auc_score(y[:,c],z[:,c]) for c in range(y.shape[1])])),
            'ece':float(np.mean(ec))}

def main():
    start=time.time();os.sched_setaffinity(0,{10});os.nice(8)
    sys.path.insert(0,str(OLD));import common as C
    tab=pd.read_csv(OLD/'calibration_episodes.csv')
    temperatures=json.loads((OLD/'temperature.lock.json').read_text())['selected']
    rows=[];errors=[];manifest=[];details=[]
    def check(name,actual,expected,tol=.01,kind='authoritative'):
        error=abs(float(actual)-float(expected))
        row=dict(name=name,actual_pp=float(actual),reference_pp=float(expected),abs_diff_pp=error,tolerance_pp=tol,passed=error<=tol,kind=kind)
        details.append(row)
        if not row['passed'] and kind=='authoritative':errors.append(name)
    for bb in ['ECGFounder','MERL']:
        for ds,nq,nc,ns in [('chapman',6132,16,10),('ningbo',6878,22,20),('cpsc2018',4122,9,20),('sph',15473,13,20)]:
            qi,y=C.labels(ds);assert y.shape==(nq,nc)
            for k in [5,10]:
                for s in range(ns):
                    p=OLD/'base/main'/bb/ds/f'shot{k}_seed{s:02d}.npz'
                    z=np.load(p);assert np.array_equal(qi,z['query_indices'])
                    ep,si,eqi,sy=C.episode(ds,k,s)
                    assert np.array_equal(eqi,qi) and not np.intersect1d(si,qi).size
                    metadata=json.loads(p.with_suffix('.json').read_text())
                    digest=sha(p);assert digest==metadata['sha256']
                    manifest.append(dict(file=str(p),sha256=digest))
                    for m in METHODS:
                        a=z[m];me=independent(y,a)
                        r=dict(backbone=bb,cohort=ds,shot=k,seed=s,method=m,**me)
                        rows.append(r)
                        ref=tab[(tab.backbone==bb)&(tab.cohort==ds)&(tab.shot==k)&(tab.seed==s)&(tab.method==m)&(tab.calibration=='Raw')]
                        if m!='Direct':
                            assert len(ref)==1
                            for metric in ['ap','auc','ece']:check(f'episode/{bb}/{ds}/{k}/{s}/{m}/{metric}',100*me[metric],100*ref.iloc[0][metric],tol=1e-9)
                    if s==0:
                        # Independent raw/standardized affine score equivalence, no target labels involved.
                        si2,qi2,sy,mu,sd,x,xq,raw,off=C.arrays(bb,ds,k,s)
                        aa=sd[:,None]*raw.mean(0);bi=mu@raw.mean(0)+off.mean(0)
                        raw_score=(xq*sd+mu)@raw.mean(0)+off.mean(0)
                        std_score=xq@aa+bi
                        assert np.max(np.abs(raw_score-std_score))<1e-8
                        use=slice(None,-1) if ds=='cpsc2018' else slice(None)
                        assert np.max(np.abs(raw_score[:,use]-z['Direct'][:,use]))<1e-8
                        if ds=='cpsc2018':assert np.array_equal(z['Direct'][:,-1],z['LP'][:,-1])
                print(json.dumps(dict(stage='SANITY_REPLAY',backbone=bb,cohort=ds,shot=k,episodes=ns)),flush=True)
    f=pd.DataFrame(rows);f.to_csv(OUT/'sanity_gate_episode_metrics.csv',index=False)
    # Printed paper values are checked only to their displayed accuracy. The two
    # erroneous MeanAnchor development references remain explicitly failed.
    expected_dev={('ECGFounder','LP'):54.53,('ECGFounder','Direct-LP'):58.61,('ECGFounder','SASM-Ens'):58.89,('ECGFounder','MeanAnchor'):58.88,
                  ('MERL','LP'):34.35,('MERL','Direct-LP'):39.49,('MERL','SASM-Ens'):40.36,('MERL','MeanAnchor'):40.37}
    for (bb,m),expected in expected_dev.items():
        actual=f[(f.backbone==bb)&(f.cohort=='chapman')&(f.method==m)].ap.mean()*100
        check(f'original_spec/dev/{bb}/{m}',actual,expected,kind='original_spec')
    refs={
      'ECGFounder':{
       'LP':[(53.84,91.15),(58.08,92.76),(62.73,89.45),(70.82,92.62),(67.05,93.69),(72.20,95.46)],
       'Direct':[(51.94,91.90),(51.94,91.90),(61.47,90.23),(62.65,90.63),(72.54,96.09),(72.54,96.09)],
       'SASM-Ens':[(59.50,94.83),(60.71,95.08),(69.53,92.76),(73.44,93.92),(74.10,96.47),(75.15,96.80)],
       'MeanAnchor':[(59.51,94.84),(60.71,95.09),(69.50,92.76),(73.42,93.91),(74.11,96.47),(75.14,96.80)]},
      'MERL':{
       'LP':[(32.52,81.71),(37.40,85.05),(47.11,81.96),(54.57,85.22),(39.36,83.58),(45.20,86.53)],
       'Direct':[(37.18,85.55),(37.18,85.55),(53.98,84.90),(54.31,85.08),(55.10,89.90),(55.10,89.90)],
       'SASM-Ens':[(40.30,87.00),(42.07,88.03),(57.07,86.35),(59.86,87.38),(52.46,88.88),(53.91,89.48)],
       'MeanAnchor':[(40.32,87.06),(42.06,88.07),(57.03,86.33),(59.80,87.36),(52.54,88.91),(53.95,89.50)]}}
    cond=[(ds,k) for ds in ['ningbo','cpsc2018','sph'] for k in [5,10]]
    for bb,mm in refs.items():
        for m,vv in mm.items():
            for (ds,k),(ap,auc) in zip(cond,vv):
                group=f[(f.backbone==bb)&(f.cohort==ds)&(f.shot==k)&(f.method==m)]
                assert len(group)==20
                check(f'paper/{bb}/{ds}/{k}/{m}/ap',group.ap.mean()*100,ap)
                check(f'paper/{bb}/{ds}/{k}/{m}/auc',group.auc.mean()*100,auc)
    original_failures=[r for r in details if r['kind']=='original_spec' and not r['passed']]
    for bb,m in [('ECGFounder','MeanAnchor'),('MERL','MeanAnchor')]:
        g=f[(f.backbone==bb)&(f.cohort=='chapman')&(f.method==m)]
        reference=tab[(tab.backbone==bb)&(tab.cohort=='chapman')&(tab.method==m)&(tab.calibration=='Raw')]
        check(f'authoritative/dev/{bb}/{m}',100*g.ap.mean(),100*reference.ap.mean())
    # Keep strict original gate status false on a specification error. This script
    # does not silently turn a failed user-supplied gate into a pass.
    report=dict(all_passed=not errors and not original_failures,archived_asset_replay_passed=not errors,
      original_specification_passed=not original_failures,max_abs_diff_pp=max(r['abs_diff_pp'] for r in details),
      max_authoritative_abs_diff_pp=max((r['abs_diff_pp'] for r in details if r['kind']=='authoritative'),default=0),
      failures=errors,original_specification_failures=original_failures,checks=details,
      source_reference=dict(calibration_csv=str(OLD/'calibration_episodes.csv'),sha256=sha(OLD/'calibration_episodes.csv')),
      input_manifest=manifest,episodes=len(manifest),metric_rows=len(rows),seconds=time.time()-start,
      adapter_fits_started=0,GPU_used=False,
      state='STOPPED_REFERENCE_SPECIFICATION_MISMATCH' if original_failures else 'PASSED' if not errors else 'STOPPED_REPLAY_MISMATCH')
    dump(OUT/'gate_prediction_replay.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['checks','input_manifest']},ensure_ascii=False),flush=True)
    return 2 if not report['all_passed'] else 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception:
        dump(OUT/'SANITY_ERROR.json',dict(traceback=traceback.format_exc(),adapter_fits_started=0));raise
