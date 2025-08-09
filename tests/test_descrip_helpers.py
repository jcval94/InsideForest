import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from InsideForest.descrip import (
    _prepare_cluster_data,
    _scale_clusters,
    _compute_inflection_points,
    _merge_outputs,
)


def _sample_frames():
    df_desc = pd.DataFrame({
        "cluster": [0, 1, 2],
        "cluster_ponderador": [0.3, 0.5, 0.2],
        "cluster_descripcion": ["a", "b", "c"],
    })
    df_cluster = pd.DataFrame({
        "cluster": [0, 0, 1, 1, 1, 2],
        "target": [1, 0, 1, 0, 1, 0],
    })
    return df_desc, df_cluster


def test_prepare_cluster_data():
    df_desc, df_cluster = _sample_frames()
    (
        sorted_desc,
        stacked,
        rate_series,
        cluster_stats,
        valuable,
    ) = _prepare_cluster_data(df_desc, df_cluster, ["target"])
    assert list(sorted_desc["cluster"]) == [1, 0, 2]
    assert stacked.shape == (2, 3)
    assert set(valuable.index) == {0, 1}
    assert cluster_stats.shape[1] == 2


def test_scale_clusters():
    df_desc, df_cluster = _sample_frames()
    _, _, _, _, valuable = _prepare_cluster_data(df_desc, df_cluster, ["target"])
    scaled = _scale_clusters(valuable)
    assert "importancia" in scaled.columns
    assert np.isclose(scaled[0].mean(), 0.0)
    assert np.isclose(scaled[1].mean(), 0.0)


def test_compute_inflection_points():
    df_desc, df_cluster = _sample_frames()
    _, _, _, cluster_stats, valuable = _prepare_cluster_data(
        df_desc, df_cluster, ["target"]
    )
    scaled = _scale_clusters(valuable)
    updated, p0, p1 = _compute_inflection_points(
        cluster_stats, scaled, 0.4, 0.5
    )
    assert "buenos" in updated.columns
    assert updated["buenos"].isin([0, 1]).all()
    assert p0 is not None and p1 is not None


def test_merge_outputs():
    df_desc, df_cluster = _sample_frames()
    (
        sorted_desc,
        stacked,
        rate_series,
        cluster_stats,
        valuable,
    ) = _prepare_cluster_data(df_desc, df_cluster, ["target"])
    scaled = _scale_clusters(valuable)
    updated, _, _ = _compute_inflection_points(
        cluster_stats, scaled, 0.4, 0.5
    )
    final_df = _merge_outputs(sorted_desc, rate_series, updated, {})
    expected_cols = {
        "cluster",
        "cluster_descripcion",
        "Probabilidad",
        "N_probabilidad",
        "Soporte",
        "buenos",
    }
    assert expected_cols.issubset(final_df.columns)
    assert "cluster_ponderador" not in final_df.columns
