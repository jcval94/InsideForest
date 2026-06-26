import math

import numpy as np
import pandas as pd
import pytest

from InsideForest.group_priority import (
    RegionDescriptor,
    feature_weights_from_model,
    from_multiclass_rules,
    from_traditional_regions,
    rank_region_pairs,
)


def _region(region_id, X, support, dimensions, bounds=None, source="manual"):
    support = np.asarray(support, dtype=int)
    dimensions = tuple(dimensions)
    if bounds is None:
        bounds = {dim: (-np.inf, np.inf) for dim in dimensions}
    centroid = X.iloc[support][list(dimensions)].mean(axis=0)
    return RegionDescriptor(
        id=region_id,
        bounds=bounds,
        dimensions=dimensions,
        support_indices=support,
        centroid=centroid,
        n_support=len(support),
        source=source,
        metadata={},
    )


def test_original_formula_matches_numeric_example():
    X = pd.DataFrame(
        [(0.0, 0.0)] * 10 + [(8.0 / 3.0, 8.0)] * 3,
        columns=["x", "y"],
    )
    region_a = _region("A", X, range(10), ["x", "y"])
    region_b = _region("B", X, range(5, 13), ["x", "y"])

    ranked = rank_region_pairs(
        X,
        [region_a, region_b],
        feature_weights={"x": 0.75, "y": 0.25},
        tau=2.0,
        scaler=None,
        variant="original",
    )

    expected_distance = math.sqrt(3.0)
    expected_centroid = math.exp(-expected_distance / 2.0)
    expected_intersection = 10.0 / 18.0
    expected_priority = 0.60 * expected_centroid + 0.30 * expected_intersection + 0.10

    row = ranked.iloc[0]
    assert row["centroid_distance"] == pytest.approx(expected_distance)
    assert row["centroid_similarity"] == pytest.approx(expected_centroid)
    assert row["intersection_similarity"] == pytest.approx(expected_intersection)
    assert row["priority"] == pytest.approx(expected_priority)


def test_robust_dimension_score_uses_weighted_jaccard():
    X = pd.DataFrame(
        {
            "x": [0.0, 0.1, 0.0, 2.0],
            "y": [0.0, 0.2, 3.0, 4.0],
            "z": [0.0, 3.0, 0.2, 5.0],
        }
    )
    region_a = _region("A", X, [0, 1], ["x", "y"])
    region_b = _region("B", X, [0, 2], ["x", "z"])

    ranked = rank_region_pairs(
        X,
        [region_a, region_b],
        feature_weights={"x": 0.8, "y": 0.1, "z": 0.1},
        tau=1.0,
        scaler=None,
        variant="robust",
    )

    row = ranked.iloc[0]
    assert row["dimension_jaccard"] == pytest.approx(1.0 / 3.0)
    assert row["weighted_dimension_jaccard"] == pytest.approx(0.8)
    assert row["shared_weight_mass"] == pytest.approx(0.8)
    assert row["dimension_similarity"] == pytest.approx(math.sqrt(0.8))


def test_pairs_without_shared_dimensions_are_excluded():
    X = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]})
    region_a = _region("A", X, [0, 1], ["x"])
    region_b = _region("B", X, [1, 2], ["y"])

    ranked = rank_region_pairs(X, [region_a, region_b], scaler=None)

    assert ranked.empty


def test_tau_median_is_computed_from_positive_distances():
    X = pd.DataFrame({"x": [0.0, 0.0, 2.0, 2.0, 6.0, 6.0]})
    regions = [
        _region("A", X, [0, 1], ["x"]),
        _region("B", X, [2, 3], ["x"]),
        _region("C", X, [4, 5], ["x"]),
    ]

    ranked = rank_region_pairs(X, regions, scaler=None, tau="median")

    assert set(ranked["tau"]) == {4.0}


def test_standard_scaling_handles_zero_variance_and_nans():
    X = pd.DataFrame({"x": [1.0, 1.0, 1.0, 1.0], "y": [np.nan, 0.0, np.nan, 0.0]})
    region_a = _region("A", X, [0, 1], ["x", "y"])
    region_b = _region("B", X, [2, 3], ["x", "y"])

    ranked = rank_region_pairs(X, [region_a, region_b], scaler="standard")

    assert not ranked.empty
    assert np.isfinite(ranked["centroid_distance"]).all()
    assert np.isfinite(ranked["priority"]).all()


def test_adapter_missing_bound_column_raises_key_error():
    X = pd.DataFrame({"x": [0.0, 1.0]})
    rules = pd.DataFrame(
        [
            {
                "leaf_region_id": "leaf_1",
                "lower_bounds": {"z": 0.0},
                "upper_bounds": {"z": 1.0},
            }
        ]
    )

    with pytest.raises(KeyError, match="z"):
        from_multiclass_rules(rules, X, min_support=1)


def test_traditional_adapter_uses_inclusive_bounds():
    X = pd.DataFrame({"x": [0.0, 1.0, 1.1]})
    columns = pd.MultiIndex.from_tuples(
        [("linf", "x"), ("lsup", "x"), ("metrics", "ponderador")]
    )
    df_reres = pd.DataFrame([[0.0, 1.0, 1.0]], columns=columns)

    regions = from_traditional_regions(df_reres, X, min_support=1)

    assert len(regions) == 1
    assert regions[0].support_indices.tolist() == [0, 1]


def test_multiclass_adapter_uses_open_lower_closed_upper_bounds():
    X = pd.DataFrame({"x": [0.0, 0.5, 1.0, 1.1]})
    rules = pd.DataFrame(
        [
            {
                "leaf_region_id": "leaf_1",
                "lower_bounds": {"x": 0.0},
                "upper_bounds": {"x": 1.0},
                "score": 1.0,
                "support": 2,
            }
        ]
    )

    regions = from_multiclass_rules(rules, X, min_support=1)

    assert len(regions) == 1
    assert regions[0].support_indices.tolist() == [1, 2]


def test_dice_intersection_uses_observed_membership():
    X = pd.DataFrame({"x": [0, 1, 2, 3]})
    region_a = _region("A", X, [0, 1, 2], ["x"])
    region_b = _region("B", X, [1, 2, 3], ["x"])

    ranked = rank_region_pairs(X, [region_a, region_b], scaler=None, tau=1.0)

    assert ranked.iloc[0]["n_ab"] == 2
    assert ranked.iloc[0]["intersection_similarity"] == pytest.approx(4.0 / 6.0)


def test_multiclass_adapter_deduplicates_leaf_region_id():
    X = pd.DataFrame({"x": [0.5, 0.75, 2.0]})
    rules = pd.DataFrame(
        [
            {
                "leaf_region_id": "leaf_1",
                "target_class": 0,
                "lower_bounds": {"x": 0.0},
                "upper_bounds": {"x": 1.0},
                "score": 0.5,
                "support": 2,
            },
            {
                "leaf_region_id": "leaf_1",
                "target_class": 1,
                "lower_bounds": {"x": 0.0},
                "upper_bounds": {"x": 1.0},
                "score": 0.8,
                "support": 2,
            },
        ]
    )

    deduped = from_multiclass_rules(rules, X, deduplicate=True, min_support=1)
    expanded = from_multiclass_rules(rules, X, deduplicate=False, min_support=1)

    assert len(deduped) == 1
    assert deduped[0].metadata["target_class"] == 1
    assert len(expanded) == 2


def test_rank_region_pairs_matches_naive_reference():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(12, 3)), columns=["a", "b", "c"])
    regions = [
        _region("R0", X, [0, 1, 2, 3], ["a", "b"]),
        _region("R1", X, [2, 3, 4, 5], ["a", "c"]),
        _region("R2", X, [6, 7, 8, 9], ["b", "c"]),
    ]

    ranked = rank_region_pairs(
        X,
        regions,
        feature_weights={"a": 0.5, "b": 0.3, "c": 0.2},
        scaler=None,
        tau=1.7,
        variant="robust",
    )

    by_pair = {
        frozenset((row.region_a, row.region_b)): row
        for row in ranked.itertuples(index=False)
    }
    for left, right in [(0, 1), (0, 2), (1, 2)]:
        key = frozenset((regions[left].id, regions[right].id))
        row = by_pair[key]
        shared = set(regions[left].dimensions) & set(regions[right].dimensions)
        assert shared
        n_ab = len(set(regions[left].support_indices) & set(regions[right].support_indices))
        dice = 2 * n_ab / (regions[left].n_support + regions[right].n_support)
        assert row.n_ab == n_ab
        assert row.intersection_similarity == pytest.approx(dice)


def test_feature_weights_from_model_preserves_named_feature_alignment():
    class NamedForest:
        feature_importances_ = np.array([0.8, 0.1, 0.1])
        feature_names_in_ = np.array(["x", "y", "z"])

    X = pd.DataFrame(
        {
            "z": [0.0, 0.1, 0.0, 2.0],
            "x": [0.0, 0.1, 0.0, 2.0],
            "y": [0.0, 0.2, 3.0, 4.0],
        }
    )
    region_a = _region("A", X, [0, 1], ["x", "y"])
    region_b = _region("B", X, [0, 2], ["x", "z"])

    weights = feature_weights_from_model(NamedForest())
    ranked = rank_region_pairs(
        X,
        [region_a, region_b],
        feature_weights=NamedForest(),
        scaler=None,
        tau=1.0,
    )

    assert weights["x"] == pytest.approx(0.8)
    assert weights["y"] == pytest.approx(0.1)
    assert weights["z"] == pytest.approx(0.1)
    assert ranked.iloc[0]["shared_weight_mass"] == pytest.approx(0.8)


def test_feature_weights_from_unnamed_model_falls_back_to_x_order():
    class UnnamedForest:
        feature_importances_ = np.array([0.2, 0.7, 0.1])

    X = pd.DataFrame(
        {
            "a": [0.0, 0.1, 0.0, 2.0],
            "b": [0.0, 0.1, 0.0, 2.0],
            "c": [0.0, 0.2, 3.0, 4.0],
        }
    )
    region_a = _region("A", X, [0, 1], ["b", "c"])
    region_b = _region("B", X, [0, 2], ["a", "b"])

    ranked = rank_region_pairs(
        X,
        [region_a, region_b],
        feature_weights=UnnamedForest(),
        scaler=None,
        tau=1.0,
    )

    assert ranked.iloc[0]["shared_dimensions"] == ("b",)
    assert ranked.iloc[0]["shared_weight_mass"] == pytest.approx(0.7)
