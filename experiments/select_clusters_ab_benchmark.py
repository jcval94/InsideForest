from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from InsideForest.cluster_selector import select_clusters
from InsideForest.legacy_select_clusters import select_clusters_legacy


def _build_random_case(
    rng: np.random.Generator,
    n_rows: int,
    n_features: int,
    n_rules: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columnas = [f"f{i}" for i in range(n_features)]
    df_datos = pd.DataFrame(rng.uniform(0.0, 1.0, size=(n_rows, n_features)), columns=columnas)

    reglas_rows: List[List[float]] = []
    clusters: List[float] = []
    for ridx in range(n_rules):
        use_col = rng.random(n_features) > 0.5
        if not use_col.any():
            use_col[ridx % n_features] = True

        linf = np.full(n_features, np.nan)
        lsup = np.full(n_features, np.nan)
        linf_vals = rng.uniform(0.0, 0.7, size=use_col.sum())
        lsup_vals = linf_vals + rng.uniform(0.05, 0.25, size=use_col.sum())
        lsup_vals = np.clip(lsup_vals, 0.0, 1.0)

        linf[use_col] = linf_vals
        lsup[use_col] = lsup_vals

        ponderador = float(1 + rng.integers(0, 10))
        reglas_rows.append(list(linf) + list(lsup) + [ponderador])
        clusters.append(float(ridx + 1))

    columnas_multi: List[tuple[str, str]] = []
    columnas_multi.extend(("linf", col) for col in columnas)
    columnas_multi.extend(("lsup", col) for col in columnas)
    columnas_multi.append(("metrics", "ponderador"))

    df_reglas = pd.DataFrame(
        reglas_rows,
        columns=pd.MultiIndex.from_tuples(columnas_multi),
    )
    df_reglas["cluster"] = clusters

    return df_datos, df_reglas


def _time_function(func, *args, repeat: int, **kwargs) -> tuple[float, float]:
    times: List[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)
    arr = np.array(times, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)


def run_benchmark(
    *,
    rows: Sequence[int],
    rules: Sequence[int],
    features: int,
    repeat: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: List[dict[str, float | int | str]] = []
    for n_rows in rows:
        for n_rules in rules:
            df_datos, df_reglas = _build_random_case(rng, n_rows, features, n_rules)

            legacy_result = select_clusters_legacy(
                df_datos, df_reglas, keep_all_clusters=True, fallback_cluster=-1
            )
            optimized_result = select_clusters(
                df_datos, df_reglas, keep_all_clusters=True, fallback_cluster=-1
            )

            if legacy_result[0].shape != optimized_result[0].shape:
                raise RuntimeError("Legacy and optimized cluster outputs differ in shape")
            if not np.array_equal(legacy_result[0], optimized_result[0]):
                raise RuntimeError("Legacy and optimized cluster assignments differ")
            if legacy_result[1] != optimized_result[1] or legacy_result[2] != optimized_result[2]:
                raise RuntimeError("Legacy and optimized auxiliary outputs differ")

            legacy_mean, legacy_std = _time_function(
                select_clusters_legacy,
                df_datos,
                df_reglas,
                keep_all_clusters=True,
                fallback_cluster=-1,
                repeat=repeat,
            )
            optimized_mean, optimized_std = _time_function(
                select_clusters,
                df_datos,
                df_reglas,
                keep_all_clusters=True,
                fallback_cluster=-1,
                repeat=repeat,
            )

            speedup = legacy_mean / optimized_mean if optimized_mean > 0 else math.inf
            legacy_speedup_vs_optimized = optimized_mean / legacy_mean if legacy_mean > 0 else math.inf
            records.append(
                {
                    "n_rows": n_rows,
                    "n_rules": n_rules,
                    "n_features": features,
                    "implementation": "legacy",
                    "mean_seconds": legacy_mean,
                    "std_seconds": legacy_std,
                    "speedup_vs_legacy": 1.0,
                    "speedup_vs_optimized": legacy_speedup_vs_optimized,
                }
            )
            records.append(
                {
                    "n_rows": n_rows,
                    "n_rules": n_rules,
                    "n_features": features,
                    "implementation": "optimized",
                    "mean_seconds": optimized_mean,
                    "std_seconds": optimized_std,
                    "speedup_vs_legacy": speedup,
                    "speedup_vs_optimized": 1.0,
                }
            )

    return pd.DataFrame.from_records(records)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark select_clusters implementations")
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[100, 1000, 5000],
        help="List of dataset sizes to benchmark.",
    )
    parser.add_argument(
        "--rules",
        type=int,
        nargs="+",
        default=[5, 20, 50],
        help="List of rule counts to benchmark.",
    )
    parser.add_argument(
        "--features",
        type=int,
        default=6,
        help="Number of features in the synthetic dataset.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="Number of repetitions per configuration for timing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed for the random number generator.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/select_clusters_ab_benchmark.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    df = run_benchmark(
        rows=args.rows,
        rules=args.rules,
        features=args.features,
        repeat=args.repeat,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(df)


if __name__ == "__main__":
    main()
