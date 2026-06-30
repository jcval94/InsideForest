import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from InsideForest.multiclass_rules import extract_multiclass_leaf_rules


def _iris_frame():
    data = load_iris()
    return pd.DataFrame(data.data, columns=data.feature_names), data.target


def test_iris_rules_include_all_class_distributions():
    X, y = _iris_frame()
    rf = RandomForestClassifier(n_estimators=12, max_depth=4, random_state=7).fit(X, y)

    rules = extract_multiclass_leaf_rules(
        rf,
        X,
        y,
        leaf_percentile=90,
        min_support=2,
        random_state=7,
    )

    assert rules["target_class"].nunique() == 3
    assert not rules.empty
    assert all(np.isclose(dist.sum(), 1.0) for dist in rules["class_distribution"])
    assert {"target_probability", "prior_probability", "lift", "score"}.issubset(rules.columns)


def test_rule_scores_are_stable_under_numeric_class_id_remapping():
    X, y = _iris_frame()
    mapping = {0: 20, 1: 10, 2: 99}
    y_remapped = np.array([mapping[label] for label in y])

    rf_base = RandomForestClassifier(n_estimators=8, max_depth=4, random_state=11).fit(X, y)
    rf_remapped = RandomForestClassifier(n_estimators=8, max_depth=4, random_state=11).fit(X, y_remapped)

    base = extract_multiclass_leaf_rules(
        rf_base,
        X,
        y,
        leaf_percentile=None,
        random_state=11,
    )
    remapped = extract_multiclass_leaf_rules(
        rf_remapped,
        X,
        y_remapped,
        leaf_percentile=None,
        random_state=11,
    )

    base_cmp = base.assign(mapped_class=base["target_class"].map(mapping))[
        ["leaf_region_id", "mapped_class", "target_probability", "score"]
    ].sort_values(["leaf_region_id", "mapped_class"]).reset_index(drop=True)
    remap_cmp = remapped[
        ["leaf_region_id", "target_class", "target_probability", "score"]
    ].rename(columns={"target_class": "mapped_class"}).sort_values(
        ["leaf_region_id", "mapped_class"]
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(base_cmp, remap_cmp, check_dtype=False)


def test_extract_rejects_unfitted_or_wrong_model():
    X, y = _iris_frame()
    with pytest.raises(ValueError):
        extract_multiclass_leaf_rules(RandomForestClassifier(), X, y)


def test_legacy_rule_parameters_warn_and_match_canonical_names():
    X, y = _iris_frame()
    rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=17).fit(
        X, y
    )

    canonical = extract_multiclass_leaf_rules(
        rf,
        X,
        y,
        leaf_percentile=90,
        low_leaf_fraction=0.0,
        max_regions_per_class=4,
        rule_score="purity_lift_coverage",
        random_state=17,
    )
    with pytest.warns(FutureWarning, match="removed in InsideForest 0.5.0"):
        legacy = extract_multiclass_leaf_rules(
            rf,
            X,
            y,
            percentil=90,
            low_frac=0.0,
            max_rules_per_class=4,
            score="purity_lift_coverage",
            random_state=17,
        )

    pd.testing.assert_frame_equal(canonical, legacy)


def test_conflicting_legacy_and_canonical_rule_parameters_raise():
    X, y = _iris_frame()
    rf = RandomForestClassifier(n_estimators=2, random_state=19).fit(X, y)

    with pytest.warns(FutureWarning), pytest.raises(TypeError, match="conflicting"):
        extract_multiclass_leaf_rules(
            rf,
            X,
            y,
            leaf_percentile=80,
            percentil=90,
        )


def test_legacy_none_percentile_still_disables_filtering():
    X, y = _iris_frame()
    rf = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=23).fit(
        X, y
    )

    canonical = extract_multiclass_leaf_rules(
        rf, X, y, leaf_percentile=None, random_state=23
    )
    with pytest.warns(FutureWarning, match="leaf_percentile"):
        legacy = extract_multiclass_leaf_rules(
            rf, X, y, percentil=None, random_state=23
        )

    pd.testing.assert_frame_equal(canonical, legacy)
