from __future__ import annotations

import argparse

import os
import sys

import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from InsideForest.ab_testing import (
    BenchmarkConfig,
    benchmark_select_clusters,
    save_benchmark_results,
)


def generate_random_inputs(
    *,
    n_rows: int,
    n_columns: int,
    n_rules: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    columnas = [f"col_{idx}" for idx in range(n_columns)]
    df_datos = pd.DataFrame(rng.uniform(0.0, 1.0, size=(n_rows, n_columns)), columns=columnas)

    reglas_rows: list[list[float]] = []
    clusters: list[float] = []
    for rule_idx in range(n_rules):
        use_cols = rng.random(n_columns) > 0.4
        if not use_cols.any():
            use_cols[rng.integers(0, n_columns)] = True

        linf = np.full(n_columns, np.nan)
        lsup = np.full(n_columns, np.nan)

        linf_vals = rng.uniform(0.0, 0.7, size=use_cols.sum())
        lsup_vals = np.clip(linf_vals + rng.uniform(0.05, 0.3, size=use_cols.sum()), 0.0, 1.0)

        linf[use_cols] = linf_vals
        lsup[use_cols] = lsup_vals

        ponderador = float(rule_idx + 1)
        reglas_rows.append(list(linf) + list(lsup) + [ponderador])
        clusters.append(float(rule_idx % 5))

    multi_columns = [
        ("linf", col) for col in columnas
    ] + [
        ("lsup", col) for col in columnas
    ] + [
        ("metrics", "ponderador"),
    ]

    df_reglas = pd.DataFrame(reglas_rows, columns=pd.MultiIndex.from_tuples(multi_columns))
    df_reglas["cluster"] = clusters

    return df_datos, df_reglas


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark select_clusters implementations")
    parser.add_argument("--rows", type=int, default=10_000, help="Number of data rows")
    parser.add_argument("--columns", type=int, default=8, help="Number of data columns")
    parser.add_argument("--rules", type=int, default=120, help="Number of rules")
    parser.add_argument("--runs", type=int, default=5, help="Number of benchmark runs")
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/select_clusters_ab_test.csv",
        help="Path to the CSV file with results",
    )
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    parser.add_argument(
        "--keep-all",
        action="store_true",
        help="Keep all cluster assignments during the benchmark",
    )
    parser.add_argument(
        "--fallback",
        type=float,
        default=None,
        help="Fallback cluster to use when no rule matches",
    )

    args = parser.parse_args()

    df_datos, df_reglas = generate_random_inputs(
        n_rows=args.rows,
        n_columns=args.columns,
        n_rules=args.rules,
        seed=args.seed,
    )

    config = BenchmarkConfig(
        runs=args.runs,
        keep_all_clusters=args.keep_all,
        fallback_cluster=args.fallback,
    )
    df_results = benchmark_select_clusters(df_datos, df_reglas, config=config)
    save_benchmark_results(df_results, args.output)

    print(df_results)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
