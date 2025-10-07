"""AB test comparing the optimised and baseline implementations of
``Trees.get_summary_optimizado``.

The script exposes :func:`run_ab_test` so it can be reused inside the test
suite. Executing the module as a script writes the benchmark results to the
provided CSV file and prints the resulting DataFrame.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from InsideForest.trees import Trees


def _baseline_get_summary_optimizado(
    data1: pd.DataFrame,
    df_full_arboles: pd.DataFrame,
    var_obj: str,
    no_branch_lim: Optional[int] = None,
):
    """Baseline implementation of ``get_summary_optimizado``.

    This mirrors the previous version of the production code. It is preserved
    exclusively for validation and benchmarking.
    """

    agrupacion = pd.pivot_table(
        df_full_arboles,
        index=["N_regla", "N_arbol", "feature", "operador"],
        values=["rangos", "Importancia"],
        aggfunc=["min", "max", "mean"],
    )

    agrupacion_min = agrupacion["min"].reset_index()
    agrupacion_min = agrupacion_min[agrupacion_min["operador"] == "<="]
    agrupacion_max = agrupacion["max"].reset_index()
    agrupacion_max = agrupacion_max[agrupacion_max["operador"] == ">"]

    agrupacion = pd.concat([agrupacion_min, agrupacion_max]).sort_values(
        ["N_arbol", "N_regla"]
    )

    top_100_arboles = agrupacion["N_arbol"].unique()
    if no_branch_lim is not None:
        top_100_arboles = top_100_arboles[:no_branch_lim]

    X = data1.to_numpy()
    col_to_idx = {col: i for i, col in enumerate(data1.columns)}
    y = data1[var_obj].to_numpy()

    resultados = []
    for arbol_num in top_100_arboles:
        ag_arbol = agrupacion[agrupacion["N_arbol"] == arbol_num]
        reglas_info = []

        for regla_num in ag_arbol["N_regla"].unique():
            ag_regla = ag_arbol[ag_arbol["N_regla"] == regla_num]
            men_ = ag_regla[ag_regla["operador"] == "<="][["feature", "rangos"]].values
            may_ = ag_regla[ag_regla["operador"] == ">"][
                ["feature", "rangos"]
            ].values

            le_idx = (
                np.array([col_to_idx[c] for c in men_[:, 0]])
                if len(men_)
                else np.array([], dtype=int)
            )
            le_val = men_[:, 1].astype(float) if len(men_) else np.array([], dtype=float)
            gt_idx = (
                np.array([col_to_idx[c] for c in may_[:, 0]])
                if len(may_)
                else np.array([], dtype=int)
            )
            gt_val = may_[:, 1].astype(float) if len(may_) else np.array([], dtype=float)

            reglas_info.append((ag_regla.copy(), le_idx, le_val, gt_idx, gt_val))

        if not reglas_info:
            continue

        masks = []
        for _, le_idx, le_val, gt_idx, gt_val in reglas_info:
            conds = []
            if le_idx.size:
                conds.append(X[:, le_idx] <= le_val)
            if gt_idx.size:
                conds.append(X[:, gt_idx] > gt_val)
            if conds:
                mask = np.logical_and.reduce(np.concatenate(conds, axis=1), axis=1)
            else:
                mask = np.ones(X.shape[0], dtype=bool)
            masks.append(mask)

        mask_matrix = np.vstack(masks)
        n_sample = mask_matrix.sum(axis=1)
        sums = mask_matrix @ y
        ef_sample = np.divide(
            sums,
            n_sample,
            out=np.zeros_like(sums, dtype=float),
            where=n_sample > 0,
        )

        res = []
        for (df_regla, _, _, _, _), ns, ef in zip(reglas_info, n_sample, ef_sample):
            df_regla = df_regla.copy()
            df_regla["n_sample"] = ns
            df_regla["ef_sample"] = ef
            res.append(df_regla)

        resultados.append(pd.concat(res, ignore_index=True))

    if not resultados:
        return pd.DataFrame(
            columns=[
                "N_regla",
                "N_arbol",
                "feature",
                "operador",
                "rangos",
                "Importancia",
                "n_sample",
                "ef_sample",
            ]
        )

    resultado = pd.concat(resultados, ignore_index=True)
    return resultado.sort_values(by=["ef_sample", "n_sample"], ascending=False)


def _build_synthetic_regression(
    rng: np.random.Generator,
    *,
    n_samples: int,
    n_features: int,
    feature_cardinality: int,
    noise_scale: float,
) -> tuple[pd.DataFrame, pd.DataFrame, str, np.ndarray]:
    """Generate a synthetic regression dataset for the AB benchmark."""

    X = rng.integers(0, feature_cardinality, size=(n_samples, n_features)).astype(float)
    weights = rng.normal(loc=0.0, scale=1.0, size=n_features)
    noise = rng.normal(loc=0.0, scale=noise_scale, size=n_samples)
    y = X @ weights + noise

    feature_names = [f"f{i}" for i in range(n_features)]
    df_features = pd.DataFrame(X, columns=feature_names)
    target_name = "target"
    df = df_features.copy()
    df[target_name] = y
    return df, df_features, target_name, y


def _run_single_benchmark(
    *,
    run_id: int,
    run_seed: int,
    rng: np.random.Generator,
    trees: Trees,
    n_samples: int,
    n_features: int,
    feature_cardinality: int,
    n_estimators: int,
    max_depth: int,
    noise_scale: float,
) -> dict:
    """Execute one AB benchmark run and return the result rows."""

    df, df_features, target_name, y = _build_synthetic_regression(
        rng,
        n_samples=n_samples,
        n_features=n_features,
        feature_cardinality=feature_cardinality,
        noise_scale=noise_scale,
    )

    rf_seed = int(rng.integers(0, np.iinfo(np.int32).max, dtype=np.int64))
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=rf_seed,
        n_jobs=1,
    ).fit(df_features, y)

    df_full_arboles = trees.get_rangos(model, df_features, percentil=0, n_jobs=1)
    df_full_arboles = trees.get_fro(df_full_arboles)

    start = perf_counter()
    optimized = trees.get_summary_optimizado(
        df,
        df_full_arboles,
        target_name,
        n_jobs=1,
    )
    optimized_time = perf_counter() - start

    start = perf_counter()
    baseline = _baseline_get_summary_optimizado(
        df,
        df_full_arboles,
        target_name,
    )
    baseline_time = perf_counter() - start

    optimized_sorted = optimized.sort_index(axis=1).reset_index(drop=True)
    baseline_sorted = baseline.sort_index(axis=1).reset_index(drop=True)

    try:
        pd.testing.assert_frame_equal(
            optimized_sorted,
            baseline_sorted,
            check_dtype=False,
        )
        outputs_match = True
    except AssertionError:
        outputs_match = False

    if not outputs_match:
        raise AssertionError(
            "Optimised implementation does not match baseline output in run"
            f" {run_id}"
        )

    speedup = baseline_time / optimized_time if optimized_time else np.nan
    return {
        "run_id": run_id,
        "run_seed": run_seed,
        "model_seed": rf_seed,
        "baseline_duration_seconds": baseline_time,
        "optimized_duration_seconds": optimized_time,
        "baseline_n_rows": len(baseline_sorted),
        "optimized_n_rows": len(optimized_sorted),
        "matches_baseline": outputs_match,
        "speedup_vs_baseline": speedup,
        "optimized_faster": optimized_time < baseline_time,
    }


def run_ab_test(
    output_csv_path: Optional[Path] = None,
    random_state: int = 0,
    *,
    n_runs: int = 3,
    n_samples: int = 300,
    n_features: int = 6,
    feature_cardinality: int = 10,
    n_estimators: int = 20,
    max_depth: int = 6,
    noise_scale: float = 2.0,
) -> pd.DataFrame:
    """Run the AB test and optionally dump the results to ``output_csv_path``."""

    if n_runs <= 0:
        raise ValueError("n_runs must be a positive integer")

    seed_sequence = np.random.SeedSequence(random_state)
    trees = Trees()
    rows: List[dict] = []

    for run_id, child in enumerate(seed_sequence.spawn(n_runs)):
        run_seed = int(child.generate_state(1, dtype=np.uint32)[0])
        rng = np.random.default_rng(child)
        rows.append(
            _run_single_benchmark(
                run_id=run_id,
                run_seed=run_seed,
                rng=rng,
                trees=trees,
                n_samples=n_samples,
                n_features=n_features,
                feature_cardinality=feature_cardinality,
                n_estimators=n_estimators,
                max_depth=max_depth,
                noise_scale=noise_scale,
            )
        )

    results = pd.DataFrame(rows)
    results["run_id"] = results["run_id"].astype(str)
    results["run_seed"] = results["run_seed"].astype("Int64")
    results["model_seed"] = results["model_seed"].astype("Int64")
    results["baseline_n_rows"] = results["baseline_n_rows"].astype("Int64")
    results["optimized_n_rows"] = results["optimized_n_rows"].astype("Int64")
    results["n_rows_equal"] = results["baseline_n_rows"] == results["optimized_n_rows"]
    results["all_runs_match"] = results["matches_baseline"].all()
    results["optimized_faster_share"] = results["optimized_faster"].mean()

    summary_row = {
        "run_id": "aggregate",
        "run_seed": pd.NA,
        "model_seed": pd.NA,
        "baseline_duration_seconds": results["baseline_duration_seconds"].mean(),
        "optimized_duration_seconds": results["optimized_duration_seconds"].mean(),
        "baseline_n_rows": int(results["baseline_n_rows"].median()),
        "optimized_n_rows": int(results["optimized_n_rows"].median()),
        "matches_baseline": results["matches_baseline"].all(),
        "speedup_vs_baseline": results["speedup_vs_baseline"].median(),
        "optimized_faster": results["optimized_faster"].all(),
        "n_rows_equal": results["n_rows_equal"].all(),
        "all_runs_match": results["matches_baseline"].all(),
        "optimized_faster_share": results["optimized_faster"].mean(),
    }
    summary_df = pd.DataFrame([summary_row])
    summary_df = summary_df.astype(results.dtypes.to_dict(), errors="ignore")
    results = pd.concat([results, summary_df], ignore_index=True)

    if output_csv_path is not None:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_csv_path, index=False)

    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run get_summary_optimizado AB test")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments") / "get_summary_ab_results.csv",
        help="Destination CSV file for benchmark results",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed used to generate the synthetic dataset",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=3,
        help="Number of independent synthetic datasets to benchmark",
    )
    parser.add_argument("--n-samples", type=int, default=300, help="Number of synthetic samples")
    parser.add_argument("--n-features", type=int, default=6, help="Number of synthetic features")
    parser.add_argument(
        "--feature-cardinality",
        type=int,
        default=10,
        help="Upper bound (exclusive) for the discrete feature values",
    )
    parser.add_argument("--n-estimators", type=int, default=20, help="Random forest estimators")
    parser.add_argument("--max-depth", type=int, default=6, help="Random forest depth")
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=2.0,
        help="Standard deviation of the noise added to the synthetic target",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df_results = run_ab_test(
        args.output,
        random_state=args.random_state,
        n_runs=args.n_runs,
        n_samples=args.n_samples,
        n_features=args.n_features,
        feature_cardinality=args.feature_cardinality,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        noise_scale=args.noise_scale,
    )
    print(df_results)


if __name__ == "__main__":
    main()
