"""Historical L-BFGS-B objective and operations, extended with per-column alpha."""
from micro_common import *

def fit(x,y,alpha,anchor,lam=10.,lam_vec=None,prior_bias=None,alpha_vec=None):
    x=np.asarray(x,float);y=np.asarray(y,float);n,p=x.shape;m=y.shape[1]
    if n<1 or not np.isfinite(x).all() or not np.isfinite(y).all() or not ((y==0)|(y==1)).all():raise ValueError('invalid fit input')
    if alpha<0 or lam<0:raise ValueError('negative regularization')
    active=np.flatnonzero((y.sum(0)>0)&(y.sum(0)<n));ma=len(active);const=np.setdiff1d(np.arange(m),active)
    wfull=np.zeros((p,m));bfull=np.zeros(m)
    if ma:
        ya=y[:,active];anc=anchor[:,active];la=None if lam_vec is None else np.asarray(lam_vec)[active]
        av=alpha if alpha_vec is None else np.asarray(alpha_vec)[active]
        w0=anc.copy();lower=np.full_like(w0,-np.inf);upper=np.full_like(w0,np.inf);w0=np.clip(w0,lower,upper)
        my=ya.mean(0);b0=np.log(np.clip(my,1e-8,1-1e-8)/np.clip(1-my,1e-8,1));t0=np.r_[w0.ravel(),b0]
        def fun(theta):
            w=theta[:p*ma].reshape(p,ma);b=theta[p*ma:];z=x@w+b
            # Preserve scalar-alpha arithmetic exactly for the historical gate.
            if alpha_vec is None:
                v=float(np.sum(np.logaddexp(0,z)-ya*z))/n+alpha*float(np.sum(w*w))/n
            else:v=float(np.sum(np.logaddexp(0,z)-ya*z))/n+float((av*(w*w)).sum())/n
            d=(expit(z)-ya)/n;gw=x.T@d+2*av*w/n;gb=d.sum(0)
            weight=lam if la is None else la;delta=w-anc;v+=(weight*(delta*delta)).sum()/n;gw+=2*weight*delta/n
            return float(v),np.r_[gw.ravel(),gb]
        def gradmax(t):
            val,g=fun(t);gw=g[:p*ma].reshape(p,ma).copy();w=t[:p*ma].reshape(p,ma)
            gw[(w<=lower+1e-12)&(gw>0)]=0;gw[(w>=upper-1e-12)&(gw<0)]=0
            return float(max(abs(gw).max(),abs(g[p*ma:]).max())),val
        res=minimize(fun,t0,jac=True,method='L-BFGS-B',bounds=None,options={'maxiter':3000,'ftol':1e-12,'gtol':1e-7});g,v=gradmax(res.x);retry=not res.success or g>5e-5
        if retry:
            res=minimize(fun,res.x,jac=True,method='L-BFGS-B',bounds=None,options={'maxiter':6000,'ftol':1e-14,'gtol':1e-8});g,v=gradmax(res.x)
        if not np.isfinite(res.x).all() or not np.isfinite(v) or g>5e-5:raise RuntimeError('stationarity gate failed: '+str(dict(gradient=g,message=res.message)))
        wfull[:,active]=res.x[:p*ma].reshape(p,ma);bfull[active]=res.x[p*ma:]
        diag=dict(retry=retry,nit=int(res.nit),max_gradient=g,objective=v,success_flag=bool(res.success),constant_columns=const.tolist())
    else:diag=dict(retry=False,nit=0,max_gradient=0.,constant_columns=const.tolist())
    if len(const):
        wfull[:,const]=anchor[:,const]
        if prior_bias is not None:bfull[const]=prior_bias[const]
    return wfull,bfull,diag

def solve(ep,alpha,lam,ste_mode='zero_unmatched'):
    a=ep['a'].copy();pb=ep['pb'].copy();lv=np.full(ep['y'].shape[1],lam);av=None
    if ep['ds']=='cpsc2018':
        a[:,-1]=0.
        if ste_mode in ['zero_unmatched','STE-LP-fallback']:
            lv[-1]=0.
            if alpha==0:
                av=np.full(ep['y'].shape[1],alpha);av[-1]=.1
        elif ste_mode=='STE-matched-L2':lv[-1]=lam
        else:raise ValueError(ste_mode)
    return fit(ep['x'],ep['y'],alpha,a,lam,lv,pb,av)
