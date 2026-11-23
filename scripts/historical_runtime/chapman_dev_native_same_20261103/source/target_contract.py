"""Fail-closed target gates; original BCE convention and optimizer retained."""
from pathlib import Path
import hashlib,json,platform,sys
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
import scipy
from threadpoolctl import threadpool_info

def runtime_fingerprint():
 return {'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'threadpools':threadpool_info()}

def header_digest(headers,root):
 h=hashlib.sha256()
 for p in headers:
  h.update(p.relative_to(root).as_posix().encode('utf8'));h.update(bytes([0]))
  h.update(p.read_bytes());h.update(bytes([0]))
 return h.hexdigest()

def check_cache_ids(records,headers,root):
 expected=[str(p.relative_to(root)).replace(chr(92),'/') for p in headers]
 actual=[str(s).replace(chr(92),'/') for s in records]
 if actual!=expected or len(set(actual))!=len(actual):raise ValueError('cache/header record ID order or uniqueness mismatch')

def check_dense_identity(feat,probs,logits,dw,db,use):
 worst=0.
 rows=np.flatnonzero(use)
 if not len(rows):raise ValueError('empty eligible cohort')
 for ix in np.array_split(rows,max(1,len(rows)//1024)):
  z=feat[ix].astype(float)@dw.T+db
  if not np.isfinite(z).all():raise ValueError('nonfinite dense reconstruction')
  if logits is None:
   err=float(abs(expit(z)-probs[ix]).max());bound=1e-4
  else:
   if not np.isfinite(logits[ix]).all():raise ValueError('nonfinite logits_mean')
   err=float(abs(z-logits[ix]).max());bound=1e-4*max(1.,abs(logits[ix]).max())
  if err>bound:raise ValueError('affine dense identity fails: '+str(err))
  worst=max(worst,err)
 return worst

def finite_record_mask(audit_path,dataset,headers,root):
 if audit_path=='historical_replay_only':return np.ones(len(headers),bool)
 audit=json.loads(Path(audit_path).read_text())
 if audit.get('state')!='complete':raise ValueError('waveform audit incomplete')
 entry=audit['datasets'][dataset]
 if len(headers)!=entry['records'] or not entry['cache_ID_order_verified']:raise ValueError('audit cohort mismatch')
 failed={str(Path(x['record']).relative_to(root)) for x in entry['waveform_failures']}
 return np.array([str(p.relative_to(root)) not in failed for p in headers])

def load_mapping(path):
 obj=json.loads(path.read_text())
 if not isinstance(obj,dict) or not all(isinstance(v,list) for v in obj.values()):raise ValueError('mapping schema')
 return obj

def verify_direct_assets(folder,source_folder):
 folder=Path(folder);source_folder=Path(source_folder)
 for ep in range(10):
  p=folder/f'direct_ep{ep:02d}.npz';meta=json.loads((folder/f'direct_ep{ep:02d}.json').read_text())
  if hashlib.sha256(p.read_bytes()).hexdigest()!=meta['model_sha256']:raise ValueError('Direct model hash changed')
  seal=hashlib.sha256((source_folder/f'ep{ep:02d}_support_and_cv.npz').read_bytes()).hexdigest()
  if seal!=meta['support_file_sha256']:raise ValueError('Direct/B3 source supports differ')
  if not meta['final_converged'] or len(meta['cv_fits'])!=14 or not all(v['converged'] for v in meta['cv_fits']):raise ValueError('Direct unconverged CV/model')
  with np.load(p,allow_pickle=False) as z:
   if z['weight'].shape!=(1024,75) or z['scaler_sd'].shape!=(1024,) or not (z['scaler_sd']>0).all():raise ValueError('Direct layout/scaler')
   if not all(np.isfinite(z[k]).all() for k in ['weight','bias','scaler_mean','scaler_sd']):raise ValueError('Direct nonfinite')

def checked_macro(ap,auc):
 if not len(ap) or not np.isfinite(ap).all() or not np.isfinite(auc).all():raise ValueError('fixed evaluation denominator contains undefined metric')
 return float(np.mean(ap)),float(np.mean(auc))

def objective(x,y,alpha,anchor=None,lam=0.,lam_vec=None,b_anchor=None,lam_b=0.):
 n,p=x.shape;m=y.shape[1]
 def fun(t):
  w=t[:p*m].reshape(p,m);b=t[p*m:];z=x@w+b
  v=float(np.sum(np.logaddexp(0,z)-y*z))/n+alpha*float(np.sum(w*w))/n
  d=(expit(z)-y)/n;gw=x.T@d+2*alpha*w/n;gb=d.sum(0)
  if anchor is not None:
   weight=lam if lam_vec is None else np.asarray(lam_vec)
   delta=w-anchor;v+=(weight*(delta*delta)).sum()/n;gw+=2*weight*delta/n
  if b_anchor is not None and lam_b>0:
   delta=b-b_anchor;v+=lam_b*(delta*delta).sum()/n;gb+=2*lam_b*delta/n
  return float(v),np.r_[gw.ravel(),gb]
 return fun

def certified_fit(x,y,alpha,W_anchor=None,lam=0.,lam_vec=None,ub=None,nonneg=False,b_anchor=None,lam_b=0.):
 x=np.asarray(x,float);y=np.asarray(y,float);n,p=x.shape;m=y.shape[1]
 if n<1 or not np.isfinite(x).all() or not np.isfinite(y).all() or not ((y==0)|(y==1)).all():raise ValueError('invalid fit input')
 if alpha<0 or lam<0 or lam_b<0:raise ValueError('negative regularization')
 active=np.flatnonzero((y.sum(0)>0)&(y.sum(0)<n));ma=len(active)
 wfull=np.zeros((p,m));bfull=np.zeros(m)
 # Historical constant-label convention is retained explicitly for reproducibility.
 if not ma:return wfull,bfull,{'retry':False,'nit':0,'max_gradient':0.,'constant_columns':list(range(m))}
 ya=y[:,active];anc=None if W_anchor is None else W_anchor[:,active]
 la=None if lam_vec is None else np.asarray(lam_vec)[active]
 ba=None if b_anchor is None else np.asarray(b_anchor)[active]
 w0=np.zeros((p,ma)) if anc is None else anc.copy()
 lower=np.zeros_like(w0) if nonneg else np.full_like(w0,-np.inf)
 upper=np.full_like(w0,np.inf) if ub is None else np.asarray(ub)[:,active]
 w0=np.clip(w0,lower,upper)
 my=ya.mean(0);b0=np.log(np.clip(my,1e-8,1-1e-8)/np.clip(1-my,1e-8,1))
 t0=np.r_[w0.ravel(),b0];fun=objective(x,ya,alpha,anc,lam,la,ba,lam_b)
 bounds=None if not nonneg and ub is None else list(zip(lower.ravel(),upper.ravel()))+[(None,None)]*ma
 def gradmax(t):
  val,g=fun(t);gw=g[:p*ma].reshape(p,ma).copy();w=t[:p*ma].reshape(p,ma)
  gw[(w<=lower+1e-12)&(gw>0)]=0;gw[(w>=upper-1e-12)&(gw<0)]=0
  return float(max(abs(gw).max(),abs(g[p*ma:]).max())),val
 res=minimize(fun,t0,jac=True,method='L-BFGS-B',bounds=bounds,options={'maxiter':3000,'ftol':1e-12,'gtol':1e-7})
 g,v=gradmax(res.x);retry=not res.success or g>5e-5
 if retry:
  res=minimize(fun,res.x,jac=True,method='L-BFGS-B',bounds=bounds,options={'maxiter':6000,'ftol':1e-14,'gtol':1e-8})
  g,v=gradmax(res.x)
 if not np.isfinite(res.x).all() or not np.isfinite(v) or g>5e-5:raise RuntimeError('target stationarity gate failed: '+str({'max_gradient':g,'message':str(res.message)}))
 wfull[:,active]=res.x[:p*ma].reshape(p,ma);bfull[active]=res.x[p*ma:]
 return wfull,bfull,{'retry':retry,'nit':int(res.nit),'max_gradient':g,'objective':v,'success_flag':bool(res.success),'constant_columns':np.setdiff1d(np.arange(m),active).tolist()}
