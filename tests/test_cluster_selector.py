import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from InsideForest.cluster_selector import select_clusters


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
