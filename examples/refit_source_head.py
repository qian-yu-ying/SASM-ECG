"""Refit a fullpool source head, optionally re-running patient-grouped source CV.

This command uses source arrays only. It does not extract ECG waveforms or
access target data. It can take substantially longer than the target demos.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(k,'1')
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sasm.source_solver import scaler_fit,solve_cell,oof_ap

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=ROOT/'data')
    p.add_argument('--backbone',choices=['ecgfounder','merl'],required=True)
    p.add_argument('--run-cv',action='store_true',help='re-run all 7 alphas x 2 preserved folds before refitting')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.output.with_suffix('.json').exists():p.error('choose a new output path')
    d=a.data_root/'route_b';z=np.load(d/'source_training.npz',allow_pickle=False)
    x=np.load(d/f'{a.backbone}_source_features.npy',allow_pickle=False).astype(float)
    y=z['y'].astype(float);folds=z['folds'];mask=np.ones_like(y,bool)
    if x.shape[0]!=17418 or y.shape!=(17418,71):raise ValueError('unexpected fullpool shape')
    if set(z['patient_groups'][folds==0])&set(z['patient_groups'][folds==1]):raise ValueError('CV patient overlap')
    alpha=10.;cv_scores=None;started=time.perf_counter()
    with threadpool_limits(limits=1):
        if a.run_cv:
            alphas=[.01,.1,1.,10.,100.,1000.,10000.]
            oof={v:np.full(y.shape,np.nan) for v in alphas}
            for held in [0,1]:
                tr=folds!=held;va=~tr;mu,sd=scaler_fit(x[tr]);warm=None
                for v in sorted(alphas,reverse=True):
                    w,b,diag=solve_cell((x[tr]-mu)/sd,y[tr],mask[tr],v,'bce_signed',6000,5e-5,warm)
                    if not diag['converged']:raise RuntimeError('source CV stationarity failed')
                    warm=(w,b);oof[v][va]=(x[va]-mu)/sd@w+b
                    print(json.dumps({'stage':'cv','fold':held,'alpha':v}),flush=True)
            cv_scores={v:oof_ap(y,oof[v],mask,z['cv_labels']) for v in alphas}
            alpha=max(alphas,key=lambda v:(cv_scores[v],-v))
            if alpha!=10.:raise RuntimeError('source selection differs from archived alpha=10; do not publish silently')
        mu,sd=scaler_fit(x)
        w,b,diag=solve_cell((x-mu)/sd,y,mask,alpha,'bce_signed',6000,5e-5)
    if not diag['converged']:raise RuntimeError('source refit stationarity failed')
    old=np.load(d/f'{a.backbone}_fullpool_head.npz',allow_pickle=False)
    # Numerical optimizers may differ across BLAS/SciPy versions. Report weight
    # and prediction drift without presenting bitwise identity as a requirement.
    drift=float(np.max(abs(((x-mu)/sd)@(w-old['weight'])+(b-old['bias']))))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,weight=w,bias=b,scaler_mean=mu,scaler_sd=sd,selected_alpha=alpha)
    meta={'backbone':a.backbone,'source_records':len(y),'selected_alpha':alpha,'cv_rerun':a.run_cv,
          'cv_scores':cv_scores,'diagnostics':diag,'max_source_logit_difference_from_archive':drift,
          'seconds':time.perf_counter()-started,'target_data_accessed':False}
    a.output.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print(json.dumps(meta))

if __name__=='__main__':main()
