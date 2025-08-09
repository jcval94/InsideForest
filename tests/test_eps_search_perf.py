import numpy as np
from time import perf_counter
from sklearn.cluster import DBSCAN
from InsideForest.regions import Regions


def _old_get_eps(data, eps_min=1e-5, eps_max=None):
    if len(data) == 1:
        return 1e-2
    if len(data) == 2:
        return 0.5
    if eps_max is None:
        eps_max = np.max(np.sqrt(np.sum((data - np.mean(data, axis=0)) ** 2, axis=1)))
        if eps_max <= 1e-10:
            eps_max = 0.1
    eps_values = np.linspace(eps_min, eps_max, num=75)
    n_groups = []
    was_multiple = False
    for eps in eps_values:
        if eps <= 0:
            continue
        labels = DBSCAN(eps=eps, min_samples=2).fit_predict(data)
        unique = np.unique(labels)
        if unique.size > 1:
            n_groups.append(unique.size)
            was_multiple = True
        elif unique.size == 1 and was_multiple:
            break
    if not n_groups:
        return (eps_min + eps_max) / 2
    mode = max(set(n_groups), key=n_groups.count)
    return eps_values[n_groups.index(mode)]


def _time(func, data, repeats=3):
    times = []
    for _ in range(repeats):
        start = perf_counter()
        func(data)
        times.append(perf_counter() - start)
    return min(times)


def test_eps_search_speedup():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(200, 2))

    old_time = _time(_old_get_eps, data)
    regions = Regions()
    new_time = _time(regions.get_eps_multiple_groups_opt, data)

    assert new_time < old_time
