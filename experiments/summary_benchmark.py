"""Benchmark get_summary vs get_summary_optimizado."""
import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from InsideForest.trees import Trees


def benchmark(n_samples=500, n_features=10, n_estimators=10, random_state=0):
    """Run a simple benchmark and print runtimes."""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_state,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    reg = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state).fit(df, y)
    t = Trees()
    df_full = t.get_rangos(reg, df, percentil=0, n_jobs=1)
    df_full = t.get_fro(df_full)
    df_with_target = df.copy()
    df_with_target['y'] = y

    start = time.perf_counter()
    t.get_summary(df_with_target, df_full, 'y')
    orig_time = time.perf_counter() - start

    start = time.perf_counter()
    t.get_summary_optimizado(df_with_target, df_full, 'y', n_jobs=1)
    opt_time = time.perf_counter() - start

    print(f"get_summary: {orig_time:.2f}s")
    print(f"get_summary_optimizado: {opt_time:.2f}s")
    print(f"Speedup: {orig_time/opt_time:.1f}x")
    return orig_time, opt_time


if __name__ == "__main__":
    benchmark()
