"""Target fitting from already extracted, already aligned cached features."""
import numpy as np
from .optimizer import certified_fit


def fit_cached_features(features, labels, raw_weights=None, raw_biases=None,
                        method="sasm-ens", alpha=0.1, lam=10.0):
    """Return a raw-feature affine classifier and convergence diagnostics.

    features: (support_samples, feature_dim); labels: binary (samples, labels).
    raw_weights: (heads, feature_dim, labels); raw_biases: (heads, labels).
    Methods: sasm-ens, meananchor, lp. No backbone or sampling code is used.
    """
    f=np.asarray(features,dtype=float);y=np.asarray(labels,dtype=float)
    if f.ndim!=2 or y.ndim!=2 or len(f)!=len(y):raise ValueError("support shapes")
    if method not in {"sasm-ens","meananchor","lp"}:raise ValueError("unknown method")
    mu=f.mean(0);sd=f.std(0)+1e-6;x=(f-mu)/sd
    if method=="lp":
        anchors=[None];bias_priors=[None];lam=0.0
    else:
        rw=np.asarray(raw_weights,dtype=float);rb=np.asarray(raw_biases,dtype=float)
        if rw.ndim!=3 or rw.shape[1:]!=(f.shape[1],y.shape[1]) or rb.shape!=(rw.shape[0],y.shape[1]):
            raise ValueError("aligned source-head shapes")
        anchors=[sd[:,None]*w for w in rw]
        bias_priors=[mu@w+b for w,b in zip(rw,rb)]
        if method=="meananchor":
            anchors=[np.mean(anchors,axis=0)]
            bias_priors=[np.mean(bias_priors,axis=0)]
    weights=[];biases=[];diagnostics=[]
    constant=np.flatnonzero((y.sum(0)==0)|(y.sum(0)==len(y)))
    for a,bp in zip(anchors,bias_priors):
        w,b,diag=certified_fit(x,y,alpha,W_anchor=a,lam=lam)
        if len(constant):
            w[:,constant]=a[:,constant] if a is not None else 0.0
            b[constant]=bp[constant] if bp is not None else np.log((y[:,constant].sum(0)+0.5)/(len(y)-y[:,constant].sum(0)+0.5))
        weights.append(w);biases.append(b);diagnostics.append(diag)
    raw_weight=np.mean(weights,axis=0)/sd[:,None]
    raw_bias=np.mean(biases,axis=0)-mu@raw_weight
    return raw_weight,raw_bias,diagnostics
