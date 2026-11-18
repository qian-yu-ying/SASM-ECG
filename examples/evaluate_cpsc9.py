"""Reproduce the locked nine-class CPSC2018 cache experiment on local CPU."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
import argparse,json,sys,time,csv
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sasm.cache import load_cohort
from sasm.optimizer import certified_fit

METHODS=['LP','SASM-Ens','MeanAnchor','Direct','Direct-LP','FullPool-SASM','FullPool-Direct',
         'SASM-Single','Concat-LP','Block-Concat','No-target-scaling','No-source-scaling',
         'Native-raw','Native-source-scaled','LP-150logits']

def mapped_heads(root,bb,normal,fullpool=False):
    with np.load(root/'mappings/cpsc2018.npz',allow_pickle=False) as z:
        q=z['concept_map'].copy();keep=z['source_rows'].tolist()
    if normal=='SR':q[:,0]=0;q[keep.index(70),0]=1
    projection=np.load(root/'source_heads/ecgfounder/native_affine_projection.npz',allow_pickle=False) if bb=='ecgfounder' else None
    paths=[root/f'route_b/{bb}_fullpool_head.npz'] if fullpool else [root/f'source_heads/{bb}/source_ep{e:02d}.npz' for e in range(10)]
    out=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as z:
            w=z['weight']@q;b=z['bias']@q;sd=z['scaler_sd'];mu=z['scaler_mean']
            if projection is not None:
                raw=projection['Dw'].T@(w/sd[:,None]);off=b+((projection['Db']-mu)/sd)@w;noscale=projection['Dw'].T@w
            else:raw=w/sd[:,None];off=b-(mu/sd)@w;noscale=w
            out.append((raw,off,noscale))
    return out

def run(root,bb,k,seed,normal,requested):
    F,_,_,qi,qy,meta=load_cohort(root,bb,'cpsc2018')
    ep=json.loads((root/f'cohorts/cpsc2018/episodes/shot{k}_seed{seed:02d}.json').read_text())
    si=np.asarray(ep['support'],int);y=np.asarray(ep['support_y'],float)
    if np.intersect1d(si,qi).size or y.shape!=(len(si),9):raise ValueError('episode mismatch')
    if np.any((y.sum(0)==0)|(y.sum(0)==len(y))):raise ValueError('inactive support class')
    lock=json.loads((ROOT/'provenance/cpsc9_protocol.lock.json').read_text())
    cfg=lock['configs']['ECGFounder' if bb=='ecgfounder' else 'MERL']
    fs=F[si].astype(float);fq=F[qi].astype(float);mu=fs.mean(0);sd=fs.std(0)+1e-6
    x=(fs-mu)/sd;xq=(fq-mu)/sd;heads=mapped_heads(root,bb,normal)
    raw=np.mean([h[0] for h in heads],0);off=np.mean([h[1] for h in heads],0)
    pred={};diags={};mask=np.r_[np.ones(8),0.]
    def fit(name,alpha,anchor=None,lam=0.,xx=x,xxq=xq,mask_=mask):
        w,b,d=certified_fit(xx,y,alpha,W_anchor=anchor,lam=lam,lam_vec=lam*mask_ if anchor is not None else None)
        diags[name]=d;return xxq@w+b,w,b
    lp,lw,lb=fit('LP',cfg['LP']['alpha']);pred['LP']=lp
    if set(requested)-{'LP'}:
        direct=fq@raw+off;direct[:,-1]=lp[:,-1];pred['Direct']=direct
        pred['Direct-LP']=(1-cfg['DirectLP']['mix'])*lp+cfg['DirectLP']['mix']*direct
    if set(requested)&{'SASM-Ens','SASM-Single'}:
        ww=[];bs=[]
        for e,h in enumerate(heads):
            a=sd[:,None]*h[0];beta=mu@h[0]+h[1]
            if np.max(np.abs(x@a+beta-(fs@h[0]+h[1])))>1e-7:raise ValueError('coordinate conversion failed')
            _,w,b=fit('head%02d'%e,cfg['SASM']['alpha'],a,cfg['SASM']['lam']);ww.append(w);bs.append(b)
        pred['SASM-Ens']=xq@np.mean(ww,0)+np.mean(bs,0);pred['SASM-Single']=xq@ww[0]+bs[0]
    if 'MeanAnchor' in requested:
        pred['MeanAnchor'],_,_=fit('MeanAnchor',cfg['SASM']['alpha'],sd[:,None]*raw,cfg['SASM']['lam'])
    if set(requested)&{'FullPool-SASM','FullPool-Direct'}:
        full=mapped_heads(root,bb,normal,True)[0]
        pred['FullPool-SASM'],_,_=fit('FullPool-SASM',cfg['FullPool']['alpha'],sd[:,None]*full[0],cfg['FullPool']['lam'])
        d=fq@full[0]+full[1];d[:,-1]=lp[:,-1];pred['FullPool-Direct']=d
    if set(requested)&{'Concat-LP','Block-Concat'}:
        ss=fs@raw+off;ms=ss.mean(0);ds=ss.std(0)+1e-6
        cx=np.c_[x,(ss-ms)/ds];cq=np.c_[xq,(fq@raw+off-ms)/ds]
        if 'Concat-LP' in requested:pred['Concat-LP'],_,_=fit('Concat-LP',cfg['Concat']['alpha'],xx=cx,xxq=cq)
        if 'Block-Concat' in requested:
            scale=np.sqrt(np.r_[np.full(x.shape[1],cfg['Block']['feature_alpha']),np.full(9,cfg['Block']['logit_alpha'])])
            pred['Block-Concat'],_,_=fit('Block-Concat',1.,xx=cx/scale,xxq=cq/scale)
    for name,a in [('No-target-scaling',raw),('No-source-scaling',sd[:,None]*np.mean([h[2] for h in heads],0))]:
        if name in requested:pred[name],_,_=fit(name,cfg['SASM']['alpha'],a,cfg['SASM']['lam'])
    if bb=='ecgfounder' and set(requested)&{'Native-raw','Native-source-scaled','LP-150logits'}:
        d=np.load(root/'source_heads/ecgfounder/native_affine_projection.npz',allow_pickle=False);dw=d['Dw'];db=d['Db']
        if 'LP-150logits' in requested:
            zs=fs@dw.T+db;zq=fq@dw.T+db;zm=zs.mean(0);zd=zs.std(0)+1e-6
            pred['LP-150logits'],_,_=fit('LP-150logits',cfg['LP']['alpha'],xx=(zs-zm)/zd,xxq=(zq-zm)/zd)
        nm=json.loads((root/'route_b/native_mapping.json').read_text())['cohorts']['cpsc2018']
        ix=np.asarray(nm['native_indices']).copy()
        if normal=='SR':ix[0]=3
        cov=ix>=0;native=np.zeros_like(raw);native[:,cov]=dw[ix[cov]].T
        oldsd=np.mean([np.load(root/f'source_heads/ecgfounder/source_ep{e:02d}.npz',allow_pickle=False)['scaler_sd'] for e in range(10)],0)
        paired=native.copy();paired[:,cov]/=oldsd[ix[cov]][None,:]
        for name,a in [('Native-raw',native),('Native-source-scaled',paired)]:
            if name in requested:pred[name],_,_=fit(name,.1,sd[:,None]*a,10.,mask_=cov.astype(float))
    result=[]
    for name in requested:
        score=pred[name]
        for scope,cols in [('all9',np.arange(9)),('matched8',np.arange(8)),('native_shared7',np.arange(7)),('STE',np.array([8]))]:
            result.append(dict(backbone='ECGFounder' if bb=='ecgfounder' else 'MERL',normal_mapping=normal,shot=k,seed=seed,method=name,scope=scope,
                ap=float(average_precision_score(qy[:,cols],score[:,cols],average='macro')),
                auc=float(roc_auc_score(qy[:,cols],score[:,cols],average='macro'))))
    return result,diags

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=ROOT/'data')
    p.add_argument('--backbone',choices=['ecgfounder','merl','all'],default='ecgfounder')
    p.add_argument('--shot',choices=['5','10','all'],default='5');p.add_argument('--seeds',default='0')
    p.add_argument('--normal',choices=['NORM','SR','all'],default='NORM')
    p.add_argument('--methods',default='LP,SASM-Ens,MeanAnchor');p.add_argument('--output',type=Path,default=ROOT/'outputs/cpsc9.json')
    a=p.parse_args();seeds=list(range(20)) if a.seeds=='all' else [int(s) for s in a.seeds.split(',')]
    if len(set(seeds))!=len(seeds) or not seeds or any(s not in range(20) for s in seeds):p.error('invalid seeds')
    methods=METHODS if a.methods=='all' else a.methods.split(',')
    if not methods or set(methods)-set(METHODS):p.error('unknown methods')
    with (ROOT/'results/cpsc9_primary_episodes.csv').open(encoding='utf-8-sig',newline='') as stream:
        refs={(r['backbone'],r['normal_mapping'],int(r['shot']),int(r['seed']),r['method'],r['scope']):r for r in csv.DictReader(stream)}
    rows=[];diagnostics=[];start=time.monotonic();worst=0.;gateworst=0.
    with threadpool_limits(limits=1):
        for bb in (['ecgfounder','merl'] if a.backbone=='all' else [a.backbone]):
            selected=[m for m in methods if bb=='ecgfounder' or m not in ['Native-raw','Native-source-scaled','LP-150logits']]
            for normal in (['NORM','SR'] if a.normal=='all' else [a.normal]):
                for shot in ([5,10] if a.shot=='all' else [int(a.shot)]):
                    for seed in seeds:
                        rr,dd=run(a.data_root,bb,shot,seed,normal,selected)
                        for r in rr:
                            key=(r['backbone'],normal,shot,seed,r['method'],r['scope'])
                            if key in refs:
                                err=max(abs(r[k]-float(refs[key][k])) for k in ['ap','auc']);worst=max(worst,err)
                                r['reference_abs_error_fraction']=err
                                r['reference_gate_applied']=r['scope']!='STE'
                                if r['reference_gate_applied']:gateworst=max(gateworst,err)
                                if r['scope']!='STE' and err>5e-5:raise RuntimeError('per-seed macro reference mismatch: '+str((key,err)))
                            rows.append(r)
                        diagnostics.append(dict(backbone=bb,normal=normal,shot=shot,seed=seed,solvers=dd))
                        print(json.dumps(dict(backbone=bb,normal=normal,shot=shot,seed=seed,passed=True,seconds=time.monotonic()-start)),flush=True)
    out=dict(release='v1.2.0',protocol='cpsc2018-nineclass-20261117',rows=rows,diagnostics=diagnostics,passed=True,max_reference_error_fraction=worst,max_gated_reference_error_fraction=gateworst,reference_tolerance_fraction=5e-5,reference_gate_scopes=['all9','matched8'],STE='reported diagnostic; sensitive to solver/BLAS ranking ties; use replay_cpsc9.py for archived coefficients',metric_units='fractions',seconds=time.monotonic()-start)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ['rows','diagnostics']}))
if __name__=='__main__':main()
