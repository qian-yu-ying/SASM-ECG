"""Objective and certified optimizer extracted from the experiment implementation."""
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize

def objective(x, y, alpha, anchor=None, lam=0.0, lam_vec=None, b_anchor=None, lam_b=0.0):
    n, p = x.shape
    m = y.shape[1]

    def fun(t):
        w = t[:p * m].reshape(p, m)
        b = t[p * m:]
        z = x @ w + b
        v = float(np.sum(np.logaddexp(0, z) - y * z)) / n + alpha * float(np.sum(w * w)) / n
        d = (expit(z) - y) / n
        gw = x.T @ d + 2 * alpha * w / n
        gb = d.sum(0)
        if anchor is not None:
            weight = lam if lam_vec is None else np.asarray(lam_vec)
            delta = w - anchor
            v += (weight * (delta * delta)).sum() / n
            gw += 2 * weight * delta / n
        if b_anchor is not None and lam_b > 0:
            delta = b - b_anchor
            v += lam_b * (delta * delta).sum() / n
            gb += 2 * lam_b * delta / n
        return (float(v), np.r_[gw.ravel(), gb])
    return fun

def certified_fit(x, y, alpha, W_anchor=None, lam=0.0, lam_vec=None, ub=None, nonneg=False, b_anchor=None, lam_b=0.0):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n, p = x.shape
    m = y.shape[1]
    if n < 1 or not np.isfinite(x).all() or (not np.isfinite(y).all()) or (not ((y == 0) | (y == 1)).all()):
        raise ValueError('invalid fit input')
    if alpha < 0 or lam < 0 or lam_b < 0:
        raise ValueError('negative regularization')
    active = np.flatnonzero((y.sum(0) > 0) & (y.sum(0) < n))
    ma = len(active)
    wfull = np.zeros((p, m))
    bfull = np.zeros(m)
    if not ma:
        return (wfull, bfull, {'retry': False, 'nit': 0, 'max_gradient': 0.0, 'constant_columns': list(range(m))})
    ya = y[:, active]
    anc = None if W_anchor is None else W_anchor[:, active]
    la = None if lam_vec is None else np.asarray(lam_vec)[active]
    ba = None if b_anchor is None else np.asarray(b_anchor)[active]
    w0 = np.zeros((p, ma)) if anc is None else anc.copy()
    lower = np.zeros_like(w0) if nonneg else np.full_like(w0, -np.inf)
    upper = np.full_like(w0, np.inf) if ub is None else np.asarray(ub)[:, active]
    w0 = np.clip(w0, lower, upper)
    my = ya.mean(0)
    b0 = np.log(np.clip(my, 1e-08, 1 - 1e-08) / np.clip(1 - my, 1e-08, 1))
    t0 = np.r_[w0.ravel(), b0]
    fun = objective(x, ya, alpha, anc, lam, la, ba, lam_b)
    bounds = None if not nonneg and ub is None else list(zip(lower.ravel(), upper.ravel())) + [(None, None)] * ma

    def gradmax(t):
        val, g = fun(t)
        gw = g[:p * ma].reshape(p, ma).copy()
        w = t[:p * ma].reshape(p, ma)
        gw[(w <= lower + 1e-12) & (gw > 0)] = 0
        gw[(w >= upper - 1e-12) & (gw < 0)] = 0
        return (float(max(abs(gw).max(), abs(g[p * ma:]).max())), val)
    res = minimize(fun, t0, jac=True, method='L-BFGS-B', bounds=bounds, options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-07})
    g, v = gradmax(res.x)
    retry = not res.success or g > 5e-05
    if retry:
        res = minimize(fun, res.x, jac=True, method='L-BFGS-B', bounds=bounds, options={'maxiter': 6000, 'ftol': 1e-14, 'gtol': 1e-08})
        g, v = gradmax(res.x)
    if not np.isfinite(res.x).all() or not np.isfinite(v) or g > 5e-05:
        raise RuntimeError('target stationarity gate failed: ' + str({'max_gradient': g, 'message': str(res.message)}))
    wfull[:, active] = res.x[:p * ma].reshape(p, ma)
    bfull[active] = res.x[p * ma:]
    return (wfull, bfull, {'retry': retry, 'nit': int(res.nit), 'max_gradient': g, 'objective': v, 'success_flag': bool(res.success), 'constant_columns': np.setdiff1d(np.arange(m), active).tolist()})
