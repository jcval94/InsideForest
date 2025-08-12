"""Benchmark parallel performance of get_rangos.

Trains a small random forest and runs ``Trees.get_rangos`` twice: once
sequentially (``n_jobs=1``) and once using all available cores
(``n_jobs=-1``). Runtime and result equality are recorded in
``experiments/benchmark_get_rangos.csv``.
"""

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from InsideForest.trees import Trees


def main() -> None:
    rng = np.random.RandomState(0)
    X = rng.rand(1000, 10)
    y = rng.rand(1000)
    df = pd.DataFrame(X, columns=[f"col{i}" for i in range(X.shape[1])])
    reg = RandomForestRegressor(n_estimators=1, random_state=0).fit(df, y)
    t = Trees()

    results = []
    baseline = None
    for nj in [1, -1]:
        start = time.perf_counter()
        res = t.get_rangos(reg, df, percentil=0, n_jobs=nj)
        elapsed = time.perf_counter() - start
        equal = True
        if baseline is None:
            baseline = res
        else:
            try:
                pd.testing.assert_frame_equal(
                    baseline.sort_index(axis=1), res.sort_index(axis=1)
                )
            except AssertionError:
                equal = False
        results.append({"n_jobs": nj, "seconds": elapsed, "rows": len(res), "equal": equal})

    pd.DataFrame(results).to_csv("experiments/benchmark_get_rangos.csv", index=False)


if __name__ == "__main__":
    main()
