import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest
from InsideForest.cluster_selector import select_clusters
from InsideForest.legacy_select_clusters import select_clusters_legacy


def ab_test_select_clusters(
    df_datos: pd.DataFrame,
    df_reglas: pd.DataFrame,
    *,
    keep_all_clusters: bool = True,
    fallback_cluster: float | None = None,
) -> bool:
    """Compare the vectorised implementation against the reference version."""

    nuevo = select_clusters(
        df_datos,
        df_reglas,
        keep_all_clusters=keep_all_clusters,
        fallback_cluster=fallback_cluster,
    )
    anterior = select_clusters_legacy(
        df_datos,
        df_reglas,
        keep_all_clusters=keep_all_clusters,
        fallback_cluster=fallback_cluster,
    )

    if not np.array_equal(nuevo[0], anterior[0]):
        return False

    if keep_all_clusters:
        return nuevo[1] == anterior[1] and nuevo[2] == anterior[2]

    return nuevo[1] is None and nuevo[2] is None and anterior[1] is None and anterior[2] is None


def test_fallback_cluster_assignment():
    df_datos = pd.DataFrame({'x': [0.5, 2.0]})
    cols = pd.MultiIndex.from_tuples([
        ('linf', 'x'),
        ('lsup', 'x'),
        ('metrics', 'ponderador'),
    ])
    df_reglas = pd.DataFrame([[0.0, 1.0, 1.0]], columns=cols)
    df_reglas['cluster'] = [1.0]

    clusters, clusters_all, ponderadores_all = select_clusters(
        df_datos, df_reglas, keep_all_clusters=True, fallback_cluster=99
    )

    assert clusters[0] == 1.0
    assert clusters[1] == 99
    assert clusters_all[1] == [99]
    assert ponderadores_all[1] == [0.0]


def test_missing_column_in_rule_raises_error():
    df_datos = pd.DataFrame({'x': [0.5]})
    cols = pd.MultiIndex.from_tuples([
        ('linf', 'y'),
        ('lsup', 'y'),
        ('metrics', 'ponderador'),
    ])
    df_reglas = pd.DataFrame([[0.0, 1.0, 1.0]], columns=cols)
    df_reglas['cluster'] = [1.0]

    with pytest.raises(KeyError) as excinfo:
        select_clusters(df_datos, df_reglas)
    assert 'y' in str(excinfo.value)


def test_select_clusters_matches_reference_ab():
    rng = np.random.default_rng(1234)
    columnas = list("abc")
    df_datos = pd.DataFrame(
        rng.uniform(0.0, 1.0, size=(200, len(columnas))), columns=columnas
    )

    reglas_rows = []
    clusters = []
    for idx in range(6):
        use_col = rng.random(len(columnas)) > 0.4
        if not use_col.any():
            use_col[0] = True

        linf = np.full(len(columnas), np.nan)
        lsup = np.full(len(columnas), np.nan)
        linf_vals = rng.uniform(0.0, 0.6, size=use_col.sum())
        lsup_vals = linf_vals + rng.uniform(0.05, 0.3, size=use_col.sum())
        lsup_vals = np.clip(lsup_vals, linf_vals, 1.0)

        linf[use_col] = linf_vals
        lsup[use_col] = lsup_vals

        ponderador = float(idx + 1)
        reglas_rows.append(list(linf) + list(lsup) + [ponderador])
        clusters.append(float(idx + 1))

    columnas_multi = [
        ("linf", col) for col in columnas
    ] + [
        ("lsup", col) for col in columnas
    ] + [
        ("metrics", "ponderador")
    ]
    df_reglas = pd.DataFrame(
        reglas_rows,
        columns=pd.MultiIndex.from_tuples(columnas_multi),
    )
    df_reglas["cluster"] = clusters

    with pytest.warns(UserWarning):
        assert ab_test_select_clusters(
            df_datos,
            df_reglas,
            keep_all_clusters=True,
        )

    assert ab_test_select_clusters(
        df_datos,
        df_reglas,
        keep_all_clusters=False,
        fallback_cluster=0.0,
    )
