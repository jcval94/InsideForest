import numpy as np
from src.sheshe.sheshe import _find_maximum

class Quadratic:
    def __init__(self, center=(1.0, 2.0)):
        self.center = np.array(center, dtype=float)
        self.evals = 0

    def __call__(self, x):
        self.evals += 1
        x = np.asarray(x)
        diff = x - self.center
        return 5.0 - np.dot(diff, diff)

    def grad(self, x):
        return -2.0 * (np.asarray(x) - self.center)

    def hess(self, x):
        return -2.0 * np.eye(len(x))


class QuadraticNoDeriv:
    """Same quadratic but without analytic derivatives."""

    def __init__(self, center=(1.0, 2.0)):
        self.center = np.array(center, dtype=float)
        self.evals = 0

    def __call__(self, x):
        self.evals += 1
        x = np.asarray(x)
        diff = x - self.center
        return 5.0 - np.dot(diff, diff)

def test_trust_region_converges_with_fewer_evals():
    q1 = QuadraticNoDeriv()
    _find_maximum(
        q1,
        [-2.0, -2.0],
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        optim_method="gradient_ascent",
        step_size=0.1,
        max_iter=100,
    )
    evals_grad = q1.evals

    q2 = Quadratic()
    _find_maximum(
        q2,
        [-2.0, -2.0],
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        optim_method="trust_region_newton",
        max_iter=10,
    )
    evals_tr = q2.evals

    assert evals_tr < evals_grad

def test_bounds_respected():
    q = Quadratic()
    bounds = [(0.0, 0.5), (0.0, 0.5)]
    x = _find_maximum(q, [0.2, 0.2], bounds=bounds,
                      optim_method='trust_region_newton', max_iter=10)
    assert np.all(x >= np.array([b[0] for b in bounds]))
    assert np.all(x <= np.array([b[1] for b in bounds]))
