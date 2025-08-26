import pandas as pd
import pytest

from InsideForest.metadata import MetaExtractor, run_experiments


def test_run_experiments_includes_intersection_stats():
    # dataset with simple target
    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'y': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'target': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    })

    # cluster descriptions with a shared rule on x
    df2 = pd.DataFrame({
        'cluster': [0, 1],
        'cluster_descripcion': [
            '0 <= x <= 5 AND 0 <= y <= 5',
            '0 <= x <= 5 AND 5 <= y <= 10',
        ],
        'cluster_ef_sample': [0.2, 1.0],
        'cluster_n_sample': [5, 1],
    })

    # minimal metadata for variables x and y
    meta_df = pd.DataFrame({
        'actionability.increase_difficulty': [1, 1],
        'actionability.decrease_difficulty': [1, 1],
    }, index=['x', 'y'])

    mx = MetaExtractor(meta_df, var_obj='target')

    result = run_experiments(mx, {'ds': df2}, data_dict={'ds': df})

    assert 'intersection_n_sample' in result.columns
    assert 'intersection_ef_sample' in result.columns
    row = result.iloc[0]
    assert row['intersection_n_sample'] == 5
    assert row['intersection_ef_sample'] == pytest.approx(0.2)
