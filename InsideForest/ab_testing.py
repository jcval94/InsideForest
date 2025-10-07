from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional

import numpy as np
import pandas as pd

from .cluster_selector import select_clusters
from .legacy_select_clusters import select_clusters_legacy


@dataclass
class BenchmarkConfig:
    runs: int = 5
    keep_all_clusters: bool = True
    fallback_cluster: Optional[float] = None


def _results_match(
    nuevo: tuple[np.ndarray, Optional[list[list[float]]], Optional[list[list[float]]]],
    anterior: tuple[np.ndarray, Optional[list[list[float]]], Optional[list[list[float]]]],
    keep_all_clusters: bool,
) -> bool:
    if not np.array_equal(nuevo[0], anterior[0]):
        return False

    if not keep_all_clusters:
        return (
            nuevo[1] is None
            and nuevo[2] is None
            and anterior[1] is None
            and anterior[2] is None
        )

    if nuevo[1] is None or nuevo[2] is None:
        return False
    if anterior[1] is None or anterior[2] is None:
        return False

    for clusters_nuevo, clusters_anterior in zip(nuevo[1], anterior[1]):
        if list(clusters_nuevo) != list(clusters_anterior):
            return False
    for ponder_nuevo, ponder_anterior in zip(nuevo[2], anterior[2]):
        if list(ponder_nuevo) != list(ponder_anterior):
            return False

    return True


def benchmark_select_clusters(
    df_datos: pd.DataFrame,
    df_reglas: pd.DataFrame,
    *,
    config: BenchmarkConfig | None = None,
) -> pd.DataFrame:
    """Benchmark the vectorised and legacy implementations.

    Parameters
    ----------
    df_datos:
        Dataset with the observations to cluster.
    df_reglas:
        Rules definition dataframe.
    config:
        Benchmark configuration. When ``None`` a default configuration with
        five runs, ``keep_all_clusters=True`` and ``fallback_cluster=None`` is
        used.
    """

    cfg = config or BenchmarkConfig()
    runs = max(1, int(cfg.runs))

    rows: list[dict[str, object]] = []

    for run_idx in range(1, runs + 1):
        start = perf_counter()
        nuevo = select_clusters(
            df_datos,
            df_reglas,
            keep_all_clusters=cfg.keep_all_clusters,
            fallback_cluster=cfg.fallback_cluster,
        )
        vectorized_seconds = perf_counter() - start

        start = perf_counter()
        anterior = select_clusters_legacy(
            df_datos,
            df_reglas,
            keep_all_clusters=cfg.keep_all_clusters,
            fallback_cluster=cfg.fallback_cluster,
        )
        legacy_seconds = perf_counter() - start

        rows.append(
            {
                "run": run_idx,
                "vectorized_seconds": vectorized_seconds,
                "legacy_seconds": legacy_seconds,
                "speedup": legacy_seconds / vectorized_seconds if vectorized_seconds else np.nan,
                "results_match": _results_match(nuevo, anterior, cfg.keep_all_clusters),
                "faster": "vectorized" if vectorized_seconds <= legacy_seconds else "legacy",
            }
        )

    results = pd.DataFrame(rows)

    summary = pd.DataFrame(
        [
            {
                "run": "mean",
                "vectorized_seconds": results["vectorized_seconds"].mean(),
                "legacy_seconds": results["legacy_seconds"].mean(),
                "speedup": results["legacy_seconds"].mean()
                / results["vectorized_seconds"].mean()
                if results["vectorized_seconds"].mean()
                else np.nan,
                "results_match": bool(results["results_match"].all()),
                "faster": "vectorized"
                if results["vectorized_seconds"].mean()
                <= results["legacy_seconds"].mean()
                else "legacy",
            }
        ]
    )

    return pd.concat([results, summary], ignore_index=True)


def save_benchmark_results(
    df_results: pd.DataFrame,
    output_path: str,
) -> None:
    """Persist benchmark results to CSV format."""

    df_results.to_csv(output_path, index=False)


__all__ = [
    "BenchmarkConfig",
    "benchmark_select_clusters",
    "save_benchmark_results",
]
