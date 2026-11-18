"""Independently check fullpool source stationarity and preserved CV disjointness."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,default=ROOT/'data')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/source_verification.json');a=p.parse_args()
    d=a.data_root/'route_b';z=np.load(d/'source_training.npz',allow_pickle=False)
    y=z['y'].astype(float);folds=z['folds'];groups=z['patient_groups'];rows=[]
    if y.shape!=(17418,71) or set(folds.tolist())!={0,1}:raise ValueError('source shape/folds')
    if set(groups[folds==0])&set(groups[folds==1]):raise ValueError('source CV patient overlap')
    with threadpool_limits(limits=1):
        for bb in ['ecgfounder','merl']:
            x=np.load(d/f'{bb}_source_features.npy',allow_pickle=False).astype(float)
            h=np.load(d/f'{bb}_fullpool_head.npz',allow_pickle=False)
            np.testing.assert_allclose(x.mean(0),h['scaler_mean'],atol=1e-10,rtol=0)
            np.testing.assert_allclose(x.std(0)+1e-6,h['scaler_sd'],atol=1e-10,rtol=0)
            xs=(x-h['scaler_mean'])/h['scaler_sd'];w=h['weight'];b=h['bias'];alpha=float(h['selected_alpha'])
            active=np.flatnonzero((y.sum(0)>0)&(y.sum(0)<len(y)))
            residual=expit(xs@w[:,active]+b[active])-y[:,active]
            gw=(xs.T@residual+2*alpha*w[:,active])/len(y);gb=residual.mean(0)
            grad=float(max(abs(gw).max(),abs(gb).max()))
            if not np.isfinite(grad) or grad>5e-5:raise RuntimeError('source stationarity failed')
            rows.append(dict(backbone=bb,records=len(y),labels=y.shape[1],alpha=alpha,max_gradient=grad,passed=True))
    result=dict(rows=rows,patient_disjoint_folds=True,passed=True)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
