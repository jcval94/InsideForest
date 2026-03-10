import pandas as pd
import pytest

from InsideForest.regions import Regions


def _build_clusterized_df(dimensions):
    cols = pd.MultiIndex.from_product([['linf', 'lsup'], dimensions])
    return pd.DataFrame([[0.0] * len(cols)], columns=cols)


def _build_separation_df(dimensions):
    return pd.DataFrame({'dimension': dimensions})


def test_search_original_tree_returns_matching_tree():
    regions = Regions()
    df_clusterizado = _build_clusterized_df(['x', 'y'])
    separacion_dim = [
        _build_separation_df(['a', 'b']),
        _build_separation_df(['y', 'x', 'x']),
    ]

    result = regions.search_original_tree(df_clusterizado, separacion_dim)

    pd.testing.assert_frame_equal(result, separacion_dim[1])


def test_search_original_tree_raises_when_no_match():
    regions = Regions()
    df_clusterizado = _build_clusterized_df(['x', 'y'])
    separacion_dim = [
        _build_separation_df(['a', 'b']),
        _build_separation_df(['x', 'z']),
    ]

    with pytest.raises(ValueError, match='No matching original tree found') as exc_info:
        regions.search_original_tree(df_clusterizado, separacion_dim)

    message = str(exc_info.value)
    assert "['x', 'y']" in message
    assert "['a', 'b']" in message
    assert "['x', 'z']" in message


def test_search_original_tree_raises_when_separation_is_empty():
    regions = Regions()
    df_clusterizado = _build_clusterized_df(['x', 'y'])

    with pytest.raises(ValueError, match='separacion_dim is empty') as exc_info:
        regions.search_original_tree(df_clusterizado, [])

    assert "['x', 'y']" in str(exc_info.value)
