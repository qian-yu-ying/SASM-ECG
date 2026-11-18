"""Numerical functions retained from the archived source BCE solver.

Only pure numerical functions are exported. No private path, helper import,
network call, scheduler or encoder training entry point is included.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
EPS=1e-6
DEADLINE=float('inf')

def check_deadline():
    if time.perf_counter() > DEADLINE:
        raise TimeoutError("predeclared wall-clock cap reached; incomplete output retained")

def scaler_fit(x):
    x = np.asarray(x, dtype=np.float64)
    return x.mean(axis=0), x.std(axis=0) + EPS

def ridge_solution(x, y, mask, alpha):
    """Primal analytic solution; intercept is not penalized, even under masks."""
    n, p = x.shape
    weights = np.zeros((p, y.shape[1]), dtype=np.float64)
    bias = np.zeros(y.shape[1], dtype=np.float64)
    if np.all(mask):
        mu_x, mu_y = x.mean(axis=0), y.mean(axis=0)
        xx, yy = x - mu_x, y - mu_y
        weights = np.linalg.solve(xx.T @ xx + alpha * np.eye(p), xx.T @ yy)
        bias = mu_y - mu_x @ weights
    else:
        for j in range(y.shape[1]):
            take = mask[:, j].astype(bool)
            if not np.any(take):
                raise ValueError("a label has no observed support cells")
            xx, yy = x[take], y[take, j]
            mu_x, mu_y = xx.mean(axis=0), yy.mean()
            centered = xx - mu_x
            weights[:, j] = np.linalg.solve(centered.T @ centered + alpha * np.eye(p),
                                             centered.T @ (yy - mu_y))
            bias[j] = mu_y - mu_x @ weights[:, j]
    return weights, bias

def make_objective(x, y, mask, alpha, loss):
    p, q = x.shape[1], y.shape[1]
    scale = float(len(x))
    full = bool(np.all(mask))
    if loss == "squared" and full:
        design = np.column_stack((x, np.ones(len(x))))
        gram, cross = design.T @ design, design.T @ y
        y_sq = float(np.square(y).sum())

    def objective(flat):
        check_deadline()
        coef = flat.reshape(p + 1, q)
        w, b = coef[:-1], coef[-1]
        if loss == "squared" and full:
            loss_sum = float(np.sum(coef * (gram @ coef)) - 2 * np.sum(coef * cross) + y_sq)
            grad = 2 * (gram @ coef - cross)
        else:
            score = x @ w + b
            if loss == "bce":
                loss_sum = float(np.sum((np.logaddexp(0, score) - y * score) * mask))
                residual = (expit(score) - y) * mask
            elif loss == "squared":
                residual = (score - y) * mask
                loss_sum = float(np.square(residual).sum())
                residual *= 2
            else:
                raise ValueError(loss)
            grad = np.vstack((x.T @ residual, residual.sum(axis=0)))
        reg = float(alpha * np.square(w).sum())
        grad[:-1] += 2 * alpha * w
        return (loss_sum + reg) / scale, grad.ravel() / scale

    return objective

def projected_gradient(gradient, coef, nonnegative):
    out = gradient.reshape(coef.shape).copy()
    if nonnegative:
        # At W=0 a positive derivative satisfies the lower-bound KKT condition.
        blocked = (coef[:-1] <= 1e-10) & (out[:-1] > 0)
        out[:-1][blocked] = 0
    return out

def solve_cell(x, y, mask, alpha, method, max_iter, kkt_tol, warm=None):
    started = time.perf_counter()
    loss, sign = method.split("_")
    nonnegative = sign == "nonnegative"
    p, q = x.shape[1], y.shape[1]
    n_obs = mask.sum(axis=0)
    positives = (y * mask).sum(axis=0)
    active = (positives > 0) & (positives < n_obs)
    if np.any(n_obs == 0):
        raise ValueError("no observed cells for a target label")
    w = np.zeros((p, q))
    mean_y = positives / n_obs
    b = mean_y.copy() if loss == "squared" else np.log(np.clip(mean_y, 1e-8, 1 - 1e-8) /
                                                     np.clip(1 - mean_y, 1e-8, 1 - 1e-8))
    if not np.any(active):
        return w, b, {"converged": True, "constant_label_indices": np.flatnonzero(~active).tolist(),
                      "iterations": 0, "runtime_seconds": time.perf_counter() - started,
                      "kkt_max_abs_mean_per_record": 0.0, "negative_coef_fraction": 0.0}
    ya, ma = y[:, active], mask[:, active]
    objective = make_objective(x, ya, ma, alpha, loss)
    if method == "squared_signed":
        wa, ba = ridge_solution(x, ya, ma, alpha)
        coef = np.vstack((wa, ba))
        value, grad = objective(coef.ravel())
        max_pg = float(np.max(np.abs(grad)))
        meta = {"solver": "analytic_primal_ridge", "solver_success": True, "status": 0,
                "message": "exact centered solve, intercept unpenalized", "iterations": 0,
                "function_evaluations": 1, "kkt_max_abs_mean_per_record": max_pg,
                "converged": bool(max_pg <= kkt_tol), "objective_mean_per_record": float(value),
                "retry_performed": False}
    else:
        if warm is None:
            coef0 = np.vstack((np.zeros((p, active.sum())), b[active]))
        else:
            coef0 = np.vstack((warm[0][:, active], warm[1][active])).copy()
            if nonnegative:
                coef0[:-1] = np.maximum(coef0[:-1], 0)
        bounds = [(0.0, None)] * (p * active.sum()) + [(None, None)] * active.sum() if nonnegative else None
        result = minimize(objective, coef0.ravel(), jac=True, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": max_iter, "ftol": 1e-12, "gtol": kkt_tol / 10,
                                   "maxcor": 20, "maxls": 40})
        coef = result.x.reshape(p + 1, active.sum())
        _, grad = objective(result.x)
        pg = projected_gradient(grad, coef, nonnegative)
        max_pg = float(np.max(np.abs(pg)))
        first_iterations = int(result.nit)
        retry = not (np.isfinite(result.fun) and max_pg <= kkt_tol)
        if retry:
            result = minimize(objective, result.x, jac=True, method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": max_iter * 2, "ftol": 1e-14,
                                       "gtol": kkt_tol / 10, "maxcor": 30, "maxls": 60})
            coef = result.x.reshape(p + 1, active.sum())
            _, grad = objective(result.x)
            max_pg = float(np.max(np.abs(projected_gradient(grad, coef, nonnegative))))
        meta = {"solver": "scipy_L-BFGS-B_direct_coefficients", "solver_success": bool(result.success),
                "status": int(result.status), "message": str(result.message),
                "iterations": int(result.nit) + (first_iterations if retry else 0),
                "function_evaluations_final_attempt": int(result.nfev),
                "kkt_max_abs_mean_per_record": max_pg, "retry_performed": retry,
                "converged": bool(np.isfinite(result.fun) and max_pg <= kkt_tol),
                "objective_mean_per_record": float(result.fun)}
    w[:, active], b[active] = coef[:-1], coef[-1]
    meta.update({"objective_sum": meta["objective_mean_per_record"] * len(x),
                 "negative_coef_fraction": float(np.mean(w < -1e-10)),
                 "zero_coef_fraction": float(np.mean(np.abs(w) <= 1e-10)),
                 "weight_l2_squared": float(np.square(w).sum()),
                 "alpha": float(alpha), "loss": loss, "sign": sign,
                 "constant_label_indices": np.flatnonzero(~active).tolist(),
                 "runtime_seconds": time.perf_counter() - started})
    return w, b, meta

def average_precision(y, score):
    positives = int(y.sum())
    if positives == 0 or positives == len(y):
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    sy, ss = y[order], score[order]
    end = np.r_[np.flatnonzero(ss[1:] != ss[:-1]), len(y) - 1]
    tp = np.cumsum(sy)[end]
    return float(np.sum(tp / (end + 1) * np.diff(np.r_[0.0, tp / positives])))

def oof_ap(y, score, mask, labels):
    values = [average_precision(y[mask[:, j].astype(bool), j],
                                score[mask[:, j].astype(bool), j]) for j in labels]
    return float(np.mean(values)) if values else float("nan")

