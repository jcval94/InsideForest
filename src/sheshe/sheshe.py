import numpy as np

def _project_bounds(x, bounds):
    if bounds is None:
        return x
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    return np.clip(x, lo, hi)

def _finite_diff_grad(f, x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(len(x)):
        dx = np.zeros_like(x)
        dx[i] = eps
        grad[i] = (f(x + dx) - f(x - dx)) / (2 * eps)
    return grad

def _finite_diff_hessian(f, x, eps=1e-4):
    x = np.asarray(x, dtype=float)
    n = len(x)
    hess = np.zeros((n, n))
    fx = f(x)
    for i in range(n):
        dx_i = np.zeros_like(x)
        dx_i[i] = eps
        for j in range(i, n):
            dx_j = np.zeros_like(x)
            dx_j[j] = eps
            fpp = f(x + dx_i + dx_j)
            fpm = f(x + dx_i - dx_j)
            fmp = f(x - dx_i + dx_j)
            fmm = f(x - dx_i - dx_j)
            hess[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps * eps)
            hess[j, i] = hess[i, j]
    return hess

def gradient_ascent(f, x0, bounds, step_size=0.1, max_iter=100, tol=1e-6):
    x = _project_bounds(np.array(x0, dtype=float), bounds)
    for _ in range(max_iter):
        grad = f.grad(x) if hasattr(f, 'grad') else _finite_diff_grad(f, x)
        if np.linalg.norm(grad) < tol:
            break
        x = _project_bounds(x + step_size * grad, bounds)
    return x

def trust_region_newton(f, x0, bounds, radius=1.0, max_iter=100, tol=1e-6,
                        eta=0.15, radius_max=100.0):
    x = _project_bounds(np.array(x0, dtype=float), bounds)
    fx = f(x)
    for _ in range(max_iter):
        grad = f.grad(x) if hasattr(f, 'grad') else _finite_diff_grad(f, x)
        if np.linalg.norm(grad) < tol:
            break
        hess = f.hess(x) if hasattr(f, 'hess') else _finite_diff_hessian(f, x)
        try:
            p = -np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            p = -grad  # fall back to gradient step
        # enforce trust region radius
        if np.linalg.norm(p) > radius:
            p = p / np.linalg.norm(p) * radius
        x_trial = _project_bounds(x + p, bounds)
        fx_trial = f(x_trial)
        pred_improve = grad.dot(p) + 0.5 * p.dot(hess).dot(p)
        actual_improve = fx_trial - fx
        if pred_improve == 0:
            ratio = 0
        else:
            ratio = actual_improve / pred_improve
        if ratio > eta:
            x = x_trial
            fx = fx_trial
        if ratio < 0.25:
            radius *= 0.25
        elif ratio > 0.75 and np.linalg.norm(p) >= radius * 0.99:
            radius = min(2 * radius, radius_max)
    return x

def _find_maximum(f, x0, bounds, optim_method='gradient_ascent', **kwargs):
    if optim_method == 'gradient_ascent':
        return gradient_ascent(f, x0, bounds, **kwargs)
    elif optim_method == 'trust_region_newton':
        return trust_region_newton(f, x0, bounds, **kwargs)
    else:
        raise ValueError(f"Unknown optim_method: {optim_method}")
