from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import warnings


def select_clusters_legacy(
    df_datos: pd.DataFrame,
    df_reglas: pd.DataFrame,
    *,
    keep_all_clusters: bool = True,
    fallback_cluster: float | None = None,
) -> Tuple[np.ndarray, Optional[List[List[float]]], Optional[List[List[float]]]]:
    """Reference implementation of ``select_clusters`` used for regression tests."""

    n_datos = df_datos.shape[0]
    clusters_datos = np.full(n_datos, -1, dtype=float)
    ponderador_datos = np.full(n_datos, -np.inf, dtype=float)
    clusters_datos_all = [[] for _ in range(n_datos)] if keep_all_clusters else None
    ponderadores_datos_all = [[] for _ in range(n_datos)] if keep_all_clusters else None

    reglas_info = []
    for _, row in df_reglas.iterrows():
        if row[("metrics", "ponderador")] == 0:
            continue
        linf = row["linf"].dropna()
        lsup = row["lsup"].dropna()
        variables = linf.index.tolist()

        p_val = row[("metrics", "ponderador")]
        ponderador = p_val.mean() if hasattr(p_val, "__iter__") else p_val

        cluster_raw = row["cluster"]
        if hasattr(cluster_raw, "values") and len(cluster_raw.values) == 1:
            cluster_raw = float(cluster_raw.values[0])
        else:
            cluster_raw = float(cluster_raw)

        reglas_info.append(
            {
                "variables": variables,
                "linf": linf.to_dict(),
                "lsup": lsup.to_dict(),
                "ponderador": ponderador,
                "cluster": cluster_raw,
            }
        )

    for regla in reglas_info:
        variables = regla["variables"]
        linf = regla["linf"]
        lsup = regla["lsup"]
        ponderador = regla["ponderador"]
        cluster = regla["cluster"]

        missing_cols = [col for col in variables if col not in df_datos.columns]
        if missing_cols:
            raise KeyError(f"Columns not found in df_datos: {missing_cols}")
        X_datos = df_datos[variables]
        condiciones = [
            (X_datos[var].to_numpy() >= linf[var])
            & (X_datos[var].to_numpy() <= lsup[var])
            for var in variables
        ]
        if condiciones:
            cumple_regla = np.logical_and.reduce(condiciones)
        else:
            cumple_regla = np.zeros(n_datos, dtype=bool)

        if keep_all_clusters:
            indices_cumple = np.where(cumple_regla)[0]
            for i in indices_cumple:
                clusters_datos_all[i].append(cluster)
                ponderadores_datos_all[i].append(ponderador)

        actualizar = cumple_regla & (ponderador > ponderador_datos)
        clusters_datos[actualizar] = cluster
        ponderador_datos[actualizar] = ponderador

    indices_sin_cluster = np.where(clusters_datos == -1)[0]
    if len(indices_sin_cluster) > 0:
        if fallback_cluster is not None:
            clusters_datos[indices_sin_cluster] = fallback_cluster
            if keep_all_clusters:
                for i in indices_sin_cluster:
                    clusters_datos_all[i].append(fallback_cluster)
                    ponderadores_datos_all[i].append(0.0)
        else:
            warnings.warn(
                f"{len(indices_sin_cluster)} records did not match any rule.",
                UserWarning,
            )

    return clusters_datos, clusters_datos_all, ponderadores_datos_all


__all__ = ["select_clusters_legacy"]
