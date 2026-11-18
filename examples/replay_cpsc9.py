"""Score archived NORM target coefficients on the released nine-class features.

This verifies fitted-model/feature/result consistency; it is not fresh fitting.
"""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[k]='1'
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sasm.cache import load_cohort

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=ROOT/'data')
    p.add_argument('--backbone',choices=['ecgfounder','merl','all'],default='ecgfounder')
    p.add_argument('--shot',choices=['5','10','all'],default='5');p.add_argument('--seeds',default='0')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/cpsc9_replay.json');a=p.parse_args()
    seeds=list(range(20)) if a.seeds=='all' else list(map(int,a.seeds.split(',')))
    if not seeds or len(set(seeds))!=len(seeds) or any(s not in range(20) for s in seeds):p.error('invalid seeds')
    with (ROOT/'results/cpsc9_primary_episodes.csv').open(encoding='utf-8-sig',newline='') as f:
        refs={(r['backbone'],int(r['shot']),int(r['seed']),r['method'],r['scope']):r for r in csv.DictReader(f) if r['normal_mapping']=='NORM'}
    rows=[];maxerr=0.
    for bb in (['ecgfounder','merl'] if a.backbone=='all' else [a.backbone]):
        long='ECGFounder' if bb=='ecgfounder' else 'MERL'
        F,_,_,qi,y,_=load_cohort(a.data_root,bb,'cpsc2018');fq=F[qi].astype(float)
        for shot in ([5,10] if a.shot=='all' else [int(a.shot)]):
            for seed in seeds:
                path=a.data_root/f'cpsc9_fitted/{long}/NORM/shot{shot}_seed{seed:02d}.weights.npz'
                with np.load(path,allow_pickle=False) as z:
                    xq=(fq-z['scaler_mean'])/z['scaler_sd']
                    pred={n:xq@z[n+'__w']+z[n+'__b'] for n in ['LP','MeanAnchor','FullPool-SASM','No-target-scaling','No-source-scaling']}
                    pred['SASM-Ens']=xq@np.mean([z[f'SASM-head{e:02d}__w'] for e in range(10)],0)+np.mean([z[f'SASM-head{e:02d}__b'] for e in range(10)],0)
                    pred['SASM-Single']=xq@z['SASM-head00__w']+z['SASM-head00__b']
                    direct=fq@z['source_raw']+z['source_offset'];direct[:,-1]=pred['LP'][:,-1];pred['Direct']=direct
                    mix=.30 if bb=='ecgfounder' else .45;pred['Direct-LP']=(1-mix)*pred['LP']+mix*direct
                    for method,score in pred.items():
                        for scope,cols in [('all9',np.arange(9)),('matched8',np.arange(8)),('STE',np.array([8]))]:
                            ap=float(average_precision_score(y[:,cols],score[:,cols],average='macro'));auc=float(roc_auc_score(y[:,cols],score[:,cols],average='macro'))
                            ref=refs[(long,shot,seed,method,scope)];err=max(abs(ap-float(ref['ap'])),abs(auc-float(ref['auc'])));maxerr=max(maxerr,err)
                            if err>1e-10:raise RuntimeError('archived coefficient replay mismatch: '+str((long,shot,seed,method,scope,err)))
                            rows.append(dict(backbone=long,shot=shot,seed=seed,method=method,scope=scope,ap=ap,auc=auc,reference_error_fraction=err))
                print(json.dumps(dict(backbone=long,shot=shot,seed=seed,passed=True)),flush=True)
    result=dict(passed=True,mode='archived fitted coefficients; no training',normal_mapping='NORM',protocol='cpsc2018-nineclass-20261117',tolerance_fraction=1e-10,max_reference_error_fraction=maxerr,rows=rows)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
if __name__=='__main__':main()
