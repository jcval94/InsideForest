from __future__ import annotations

import argparse
import os
import sys
import time
import tracemalloc

import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from InsideForest.cluster_selector import select_clusters


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
        use_cols = rng.random(n_columns) > 0.35
        if not use_cols.any():
            use_cols[rng.integers(0, n_columns)] = True

        linf = np.full(n_columns, np.nan)
        lsup = np.full(n_columns, np.nan)

        linf_vals = rng.uniform(0.0, 0.7, size=use_cols.sum())
        lsup_vals = np.clip(linf_vals + rng.uniform(0.05, 0.25, size=use_cols.sum()), 0.0, 1.0)

        linf[use_cols] = linf_vals
        lsup[use_cols] = lsup_vals

        reglas_rows.append(list(linf) + list(lsup) + [float(rule_idx + 1)])
        clusters.append(float(rule_idx % 7))

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


def profile_run(df_datos: pd.DataFrame, df_reglas: pd.DataFrame, *, batch_size: int | None) -> tuple[float, float]:
    tracemalloc.start()
    t0 = time.perf_counter()
    select_clusters(
        df_datos,
        df_reglas,
        keep_all_clusters=True,
        fallback_cluster=None,
        batch_size=batch_size,
    )
    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak_bytes / (1024 ** 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark chunked vs full-matrix select_clusters")
    parser.add_argument("--sizes", type=int, nargs="+", default=[2_000, 10_000, 30_000])
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--rules", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/results/select_clusters_chunking_benchmark.csv",
    )
    args = parser.parse_args()

    rows: list[dict[str, float | int | str]] = []
    for n_rows in args.sizes:
        df_datos, df_reglas = generate_random_inputs(
            n_rows=n_rows,
            n_columns=args.columns,
            n_rules=args.rules,
            seed=args.seed + n_rows,
        )

        time_full, mem_full = profile_run(df_datos, df_reglas, batch_size=None)
        time_chunked, mem_chunked = profile_run(df_datos, df_reglas, batch_size=args.batch_size)

        rows.append(
            {
                "n_rows": n_rows,
                "n_columns": args.columns,
                "n_rules": args.rules,
                "mode": "full_matrix",
                "batch_size": -1,
                "seconds": time_full,
                "peak_memory_mb": mem_full,
            }
        )
        rows.append(
            {
                "n_rows": n_rows,
                "n_columns": args.columns,
                "n_rules": args.rules,
                "mode": "chunked",
                "batch_size": args.batch_size,
                "seconds": time_chunked,
                "peak_memory_mb": mem_chunked,
            }
        )

    df_results = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_results.to_csv(args.output, index=False)

    print(df_results)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
