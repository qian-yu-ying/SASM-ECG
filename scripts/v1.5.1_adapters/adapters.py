"""SHA-locked multilabel ECG/BCE variants; no external labels used in fitting."""
from core import *
import torch
import torch.nn.functional as TF
from scipy.linalg import qr,eigh

def unit_rows(x):return x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-12)
def views(ep,space,query=False):
    if space=='raw_l2':return unit_rows(ep['hq'] if query else ep['hs'])
    if space=='std_l2':return unit_rows(ep['xq'] if query else ep['x'])
    if space=='std':return ep['xq'] if query else ep['x']
    raise KeyError(space)
def tip_base(ep,query=False):
    v=(ep['sq'] if query else ep['ss']).copy()
    if not ep['matched'].all():
        a,b=lp_scores(ep);v[:,~ep['matched']]=(b if query else a)[:,~ep['matched']]
    return v
def kernel_readout(u,keys,y,beta,encoding,normalize_keys=True):
    keys=unit_rows(keys) if normalize_keys else keys
    k=np.exp(-beta*(1.-u@keys.T));values=2*y-1 if encoding=='signed' else y
    return k@values/(k.sum(1,keepdims=True)+1e-8)

def fit_tip(ep,cfg):
    u=views(ep,cfg['space'])
    return dict(kind='tip',normalize_keys=False,keys=u.copy(),values=ep['y'].copy(),delta_b=np.zeros(ep['y'].shape[1]),**cfg)
def fit_tipf(ep,cfg):
    torch.manual_seed(ep['seed']);u=views(ep,cfg['space']);n,c=ep['y'].shape
    keys=torch.nn.Parameter(torch.tensor(u,dtype=torch.float64));b=torch.nn.Parameter(torch.zeros(c,dtype=torch.float64))
    x=torch.tensor(u,dtype=torch.float64);yy=torch.tensor(ep['y'],dtype=torch.float64)
    val=2*yy-1 if cfg['encoding']=='signed' else yy
    base=torch.tensor(tip_base(ep),dtype=torch.float64)
    opt=torch.optim.AdamW([{'params':[keys],'weight_decay':cfg['wd']},{'params':[b],'weight_decay':0.}],lr=cfg['lr'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['epochs'])
    for _ in range(cfg['epochs']):
        opt.zero_grad(set_to_none=True)
        kk=keys/(torch.linalg.vector_norm(keys,dim=1,keepdim=True)+1e-12)
        affinity=torch.exp(-cfg['beta']*(1.-x@kk.T));out=base+cfg['alpha']*(affinity@val)/(affinity.sum(1,keepdim=True)+1e-8)+b
        loss=TF.binary_cross_entropy_with_logits(out,yy,reduction='mean');assert torch.isfinite(loss)
        loss.backward();opt.step();scheduler.step()
    return dict(kind='tip',normalize_keys=True,keys=keys.detach().numpy().copy(),values=ep['y'].copy(),delta_b=b.detach().numpy().copy(),**cfg)

def fit_task_batch(ep,configs):
    """Exact independent Adam parameter groups; batching changes GEMM layout only."""
    torch.manual_seed(ep['seed']);n,d=ep['x'].shape;c=ep['y'].shape[1]
    epochs={v['epochs'] for v in configs};assert len(epochs)==1;epochs=epochs.pop()
    x=torch.tensor(ep['x'],dtype=torch.float64);y=torch.tensor(np.tile(ep['y'],(1,len(configs))),dtype=torch.float64)
    base=ep['ss'].copy();a=ep['a'].copy();pb=ep['pb'].copy()
    if not ep['matched'].all():
        a[:,~ep['matched']]=0.;p=(ep['y'][:,~ep['matched']].sum(0)+.5)/(n+1)
        pb[~ep['matched']]=logit(p);base=ep['x']@a+pb
    bb=torch.tensor(np.tile(base,(1,len(configs))),dtype=torch.float64)
    ga=torch.tensor(np.repeat([v['gamma'] for v in configs],c),dtype=torch.float64)
    la=torch.tensor(np.repeat([v['lambda_res'] for v in configs],c),dtype=torch.float64)
    # Each column block has its own Adam state and learning rate. This is the
    # specified torch.optim.Adam(weight_decay=0), not AdamW or a rotated solver.
    ws=[torch.nn.Parameter(torch.zeros((d,c),dtype=torch.float64)) for _ in configs]
    bs=[torch.nn.Parameter(torch.zeros(c,dtype=torch.float64)) for _ in configs]
    opt=torch.optim.Adam([{'params':[w,b],'lr':cfg['lr'],'weight_decay':0.} for w,b,cfg in zip(ws,bs,configs)],weight_decay=0.)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True);w=torch.cat(ws,dim=1);b=torch.cat(bs)
        scores=bb+(x@w)*ga+b
        loss=TF.binary_cross_entropy_with_logits(scores,y,reduction='sum')/n+(w.square()*la).sum()/n
        assert torch.isfinite(loss);loss.backward();opt.step();sched.step()
    return [dict(kind='linear',space='std',weight=a+cfg['gamma']*w.detach().numpy(),bias=pb+b.detach().numpy(),**cfg) for cfg,w,b in zip(configs,ws,bs)]

def row_basis(u):
    q,r,pivot=qr(u.T,mode='economic',pivoting=True,check_finite=True)
    diag=abs(np.diag(r));tol=np.finfo(float).eps*max(u.shape)*max(1.,diag.max(initial=0.))
    rank=int((diag>tol).sum());q=q[:,:rank]
    assert np.max(abs(u-(u@q)@q.T))<1e-8
    return q,u@q

def reduced_bound(b,source,lam,n,matched=True):
    r=b.shape[1];xx=np.c_[b,np.ones(n),source] if matched else np.c_[b,np.ones(n)]
    h=xx.T@xx/(4*n);h[:r,:r]+=np.eye(r)*(2*lam/n)
    top=float(eigh(h,subset_by_index=[len(h)-1,len(h)-1],eigvals_only=True,check_finite=False)[0])
    return max(2*lam/n,top)

def fit_lp_batch(ep,configs,checkpoints=None):
    space={v['space'] for v in configs};assert len(space)==1;space=space.pop()
    # Configurations may share exact trajectories across stopping steps.
    u=views(ep,space);q,b=row_basis(u);n,d=u.shape;c=ep['y'].shape[1];y=ep['y'];source=ep['ss'].copy()
    source[:,~ep['matched']]=0.;proto=unit_rows(u).T@y/np.maximum(y.sum(0),1.)[None,:]
    vb=q.T@proto;bias=logit((y.sum(0)+.5)/(n+1));out={}
    keys=[]
    for cfg in configs:
        key=(cfg['alpha0'],cfg['lambda_W'])
        if key not in keys:keys.append(key)
    count=len(keys);v=np.tile(vb,(1,count));bb=np.tile(bias,count);aa=np.repeat([k[0] for k in keys],c);aa[np.tile(~ep['matched'],count)]=0.
    prior=np.tile(source,(1,count));yy=np.tile(y,(1,count));lam=np.repeat([k[1] for k in keys],c)
    eta=[]
    for a,l in keys:
        eta.extend(1./max(reduced_bound(b,source[:,cc],l,n,bool(ep['matched'][cc])),1e-6) for cc in range(c))
    eta=np.array(eta);needed=sorted({cfg['steps'] for cfg in configs});initial=b@v+bb+prior*aa
    previous=(np.logaddexp(0,initial)-yy*initial).mean(0)+lam*(v*v).sum(0)/n;max_rise=0.
    for step in range(1,max(needed)+1):
        scores=b@v+bb+prior*aa;grad=(expit(scores)-yy)/n
        v-=eta[None,:]*(b.T@grad+2*lam[None,:]*v/n);bb-=eta*grad.sum(0);aa-=eta*(prior*grad).sum(0)
        aa[np.tile(~ep['matched'],count)]=0.
        new=b@v+bb+prior*aa
        objective=(np.logaddexp(0,new)-yy*new).mean(0)+lam*(v*v).sum(0)/n
        assert np.isfinite(objective).all()
        if previous is not None:
            rise=float((objective-previous).max());max_rise=max(max_rise,rise)
            assert rise<=1e-12,(space,step,rise)
        previous=objective
        if step in needed:
            for i,(alpha,l) in enumerate(keys):
                cfg=dict(space=space,alpha0=alpha,lambda_W=l,steps=step)
                ix=slice(i*c,(i+1)*c)
                out[json.dumps(cfg,sort_keys=True)]=dict(kind='lp',weight=q@v[:,ix],bias=bb[ix].copy(),mix=aa[ix].copy(),max_objective_rise=max_rise,**cfg)
    return [out[json.dumps(cfg,sort_keys=True)] for cfg in configs]

def fit(ep,method,cfg):
    if method==METHODS[0]:return fit_tip(ep,cfg)
    if method==METHODS[1]:return fit_tipf(ep,cfg)
    if method==METHODS[2]:return fit_task_batch(ep,[cfg])[0]
    if method==METHODS[3]:return fit_lp_batch(ep,[cfg])[0]
    raise KeyError(method)
def predict(ep,model,query=False):
    u=views(ep,model['space'],query)
    if model['kind']=='tip':return tip_base(ep,query)+model['alpha']*kernel_readout(u,model['keys'],model['values'],model['beta'],model['encoding'],model['normalize_keys'])+model['delta_b']
    if model['kind']=='linear':return u@model['weight']+model['bias']
    return u@model['weight']+model['bias']+(ep['sq'] if query else ep['ss'])*model['mix']

def lp_unit_test():
    pin(12);rng=np.random.default_rng(151);tests=[];maxdiff=0.
    for n,d,matched in [(2,1,True),(7,18,True),(8,14,False),(20,6,True)]:
        u=np.ones((2,1)) if n==2 else rng.normal(size=(n,d));source=np.ones(n) if n==2 else rng.normal(size=n)
        q,b=row_basis(u)
        for lam in [0.,.01,1.,10.]:
            x=np.c_[u,np.ones(n),source] if matched else np.c_[u,np.ones(n)]
            h=x.T@x/(4*n);h[:d,:d]+=np.eye(d)*2*lam/n
            full=float(np.linalg.eigvalsh(h)[-1]);red=reduced_bound(b,source,lam,n,matched);diff=abs(full-red);maxdiff=max(maxdiff,diff);assert diff<=1e-11
            y=np.array([0.,1.]) if n==2 else rng.integers(0,2,n).astype(float);theta=np.zeros(x.shape[1]);theta[0]=1.
            vals=[]
            def objective(t):
                s=x@t;return float((np.logaddexp(0,s)-y*s).mean()+lam*(t[:d]**2).sum()/n)
            vals.append(objective(theta))
            for _ in range(30):
                gradient=x.T@(expit(x@theta)-y)/n;gradient[:d]+=2*lam*theta[:d]/n;theta-=gradient/max(red,1e-6);vals.append(objective(theta))
            assert np.max(np.diff(vals))<=1e-12
            tests.append(dict(n=n,d=d,matched=matched,lambda_W=lam,full_L=full,reduced_L=red,abs_diff=diff,initial=vals[0],final=vals[-1],max_rise=float(np.max(np.diff(vals)))))
    witness=read(RUN/'lpplus_joint_step_witness.json');assert witness['loss_increased']
    unit=dict(all_monotone=True,full_reduced_eigenvalue_max_abs_diff=maxdiff,tests=tests,witness_sha256=sha(RUN/'lpplus_joint_step_witness.json'),GPU_used=False)
    w=[t for t in tests if t['n']==2 and t['lambda_W']==0. and t['matched']][0];assert abs(w['reduced_L']-.75)<1e-12
    unit.update(witness_L=w['reduced_L'],witness_eta=1/w['reduced_L'])
    dump(OUT/'lpplus_monotone_unit_test.json',unit);notice('LPPLUS_UNIT_PASSED',max_eigen_error=maxdiff)
    return unit

if __name__=='__main__':lp_unit_test()
