"""Portable Route B evaluation using frozen public-release arrays only."""
import json
from pathlib import Path
import numpy as np
from .model import fit_cached_features

BACKBONES = ('ecgfounder', 'merl')
COHORTS = ('ningbo', 'cpsc2018', 'sph')
SUITES = ('fullpool', 'native_raw', 'native_paired')

def load_heads(data_root, backbone, cohort, suite):
    """Return raw-coordinate heads and the fixed evaluation columns.

    Fullpool uses all target endpoints. Native suites use 19/6/13 shared
    endpoints for reporting, while fitting the original full target panel.
    """
    root=Path(data_root)
    if backbone not in BACKBONES or cohort not in COHORTS or suite not in SUITES:
        raise ValueError('unknown Route B configuration')
    mapping=np.load(root/f'mappings/{cohort}.npz',allow_pickle=False)
    if suite=='fullpool':
        z=np.load(root/f'route_b/{backbone}_fullpool_head.npz',allow_pickle=False)
        q=mapping['concept_map'];w=z['weight']@q;b=z['bias']@q
        if backbone=='ecgfounder':
            d=np.load(root/'source_heads/ecgfounder/native_affine_projection.npz',allow_pickle=False)
            raw=d['Dw'].T@(w/z['scaler_sd'][:,None])
            off=b+((d['Db']-z['scaler_mean'])/z['scaler_sd'])@w
        else:
            raw=w/z['scaler_sd'][:,None]
            off=b-(z['scaler_mean']/z['scaler_sd'])@w
        return raw[None],off[None],np.arange(q.shape[1])
    if backbone!='ecgfounder':
        raise ValueError('native controls are defined only for ECGFounder')
    native=json.loads((root/'route_b/native_mapping.json').read_text(encoding='utf-8'))['cohorts'][cohort]
    ix=np.asarray(native['native_indices'],int)
    covered=np.asarray(native['covered_target_columns'],int)
    d=np.load(root/'source_heads/ecgfounder/native_affine_projection.npz',allow_pickle=False)
    raw=np.zeros((d['Dw'].shape[1],len(ix)));off=np.zeros(len(ix))
    raw[:,covered]=d['Dw'][ix[covered]].T;off[covered]=d['Db'][ix[covered]]
    if suite=='native_raw': return raw[None],off[None],covered
    heads=[]
    for e in range(10):
        old=np.load(root/f'source_heads/ecgfounder/source_ep{e:02d}.npz',allow_pickle=False)
        h=raw.copy();h[:,covered]/=old['scaler_sd'][ix[covered]][None,:]
        heads.append(h)
    # Original per-source-SD control uses zero raw offsets. Active target
    # intercepts are fitted freely in both native variants.
    return np.asarray(heads),np.zeros((10,len(ix))),covered

def selected_config(data_root,backbone,suite,config='selected'):
    if config=='fixed':return dict(alpha=.1,lam=10.)
    if config!='selected':raise ValueError('config must be selected or fixed')
    selections=json.loads((Path(data_root)/'route_b/selections.json').read_text(encoding='utf-8'))
    return selections[suite][backbone]['selected']['config']

def fit_route_b(features,labels,raw_weights,raw_biases,config,suite):
    if suite.startswith('native'):
        y=np.asarray(labels)
        if np.any((y.sum(0)==0)|(y.sum(0)==len(y))):
            raise ValueError('Route B native convention requires active support labels')
    return fit_cached_features(features,labels,raw_weights,raw_biases,
                               method='sasm-ens',**config)
