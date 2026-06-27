import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier, is_clusterer
from sklearn.datasets import load_iris, load_wine

from InsideForest.multiclass import (
    InsideForestClassRegionClusterer,
    InsideForestMulticlassClassifier,
)


def _dataset_frame(loader=load_iris, *, string_labels=False):
    data = loader()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    if string_labels:
        y = np.asarray([data.target_names[idx] for idx in y], dtype=object)
    return X, y


def _model(**overrides):
    params = {
        "rf_params": {"n_estimators": 12, "max_depth": 4},
        "leaf_percentile": 85,
        "min_support": 2,
        "random_state": 13,
    }
    params.update(overrides)
    return InsideForestClassRegionClusterer(**params)


def test_class_region_clusterer_contract_and_clone():
    model = _model()
    assert is_clusterer(model)
    assert not is_classifier(model)
    cloned = clone(model)
    assert isinstance(cloned, InsideForestClassRegionClusterer)
    assert cloned.get_params()["leaf_percentile"] == 85


def test_fit_creates_one_target_class_per_physical_leaf():
    X, y = _dataset_frame()
    model = _model().fit(X, y)

    assert model.regions_["leaf_region_id"].is_unique
    assert model.raw_regions_["leaf_region_id"].is_unique
    assert model.regions_["region_target_class"].isin(model.classes_).all()
    assert model.labels_.shape == (len(X),)
    assert model.region_metrics_ is not None

    winning = (
        model._class_rule_views_
        .sort_values(
            ["leaf_region_id", "score", "target_probability", "support", "target_class_index"],
            ascending=[True, False, False, False, True],
        )
        .drop_duplicates("leaf_region_id")
        .set_index("leaf_region_id")["target_class"]
    )
    actual = model.raw_regions_.set_index("leaf_region_id")["region_target_class"]
    pd.testing.assert_series_equal(actual.sort_index(), winning.sort_index(), check_names=False)


def test_fit_predict_predict_and_transform_are_consistent():
    X, y = _dataset_frame(load_wine)
    model = _model(random_state=17)
    fit_predict = model.fit_predict(X, y)
    predicted = model.predict(X)
    memberships = model.transform(X)
    covered = predicted != -1

    assert np.array_equal(fit_predict, predicted)
    assert np.array_equal(model.labels_, predicted)
    assert memberships.shape == (len(X), len(model.regions_))
    assert np.array_equal(memberships[covered].argmax(axis=1), predicted[covered])


def test_assign_regions_has_cluster_semantics_and_no_fallback():
    X, y = _dataset_frame()
    model = _model(random_state=19).fit(X, y)
    assigned = model.assign_regions(X.head(12))

    expected = {
        "cluster_id",
        "representative_region_id",
        "region_target_class",
        "membership_score",
        "target_probability",
        "class_distribution",
        "lift",
        "entropy",
        "class_margin",
        "matched_region_count",
        "matched_region_ids",
        "source",
    }
    assert expected == set(assigned.columns)
    assert len(assigned) == 12
    assert set(assigned["source"]).issubset({"region", "unmatched"})
    assert "predicted_class" not in assigned
    assert "confidence" not in assigned

    model.regions_ = model.regions_.iloc[0:0].copy()
    assert np.array_equal(model.predict(X.head(3)), np.full(3, -1))
    unmatched = model.assign_regions(X.head(3))
    assert set(unmatched["source"]) == {"unmatched"}
    assert (unmatched["cluster_id"] == -1).all()
    assert (unmatched["membership_score"] == 0).all()


def test_string_and_remapped_targets_do_not_change_region_scores():
    X, y = _dataset_frame()
    mapping = {0: "third", 1: "first", 2: "second"}
    y_remapped = np.asarray([mapping[value] for value in y], dtype=object)

    base = _model(random_state=23, leaf_percentile=None).fit(X, y)
    remapped = _model(random_state=23, leaf_percentile=None).fit(X, y_remapped)

    left = base.raw_regions_.assign(
        mapped_target=base.raw_regions_["region_target_class"].map(mapping)
    )[["leaf_region_id", "mapped_target", "region_score"]].sort_values("leaf_region_id")
    right = remapped.raw_regions_[
        ["leaf_region_id", "region_target_class", "region_score"]
    ].rename(columns={"region_target_class": "mapped_target"}).sort_values("leaf_region_id")
    pd.testing.assert_frame_equal(
        left.reset_index(drop=True), right.reset_index(drop=True), check_dtype=False
    )
    assert np.issubdtype(remapped.predict(X).dtype, np.integer)


def test_quality_and_class_coverage_reports():
    X, y = _dataset_frame()
    model = _model().fit(X, y)
    report = model.region_quality_report()
    evaluated = model.region_quality_report(X, y)
    coverage = model.class_coverage_report()

    assert 0 <= report["coverage"] <= 1
    assert 0 <= report["unmatched_rate"] <= 1
    assert np.isclose(model.score(X, y), evaluated["ami"])
    assert set(coverage["class_label"]) == set(model.classes_)
    assert coverage["class_coverage"].dropna().between(0, 1).all()
    assert coverage["target_precision"].dropna().between(0, 1).all()


def test_regions_for_class_and_ambiguous_regions():
    X, y = _dataset_frame()
    model = _model(ambiguity_margin=1.0).fit(X, y)
    class_regions = model.regions_for_class(model.classes_[0], top_n=2)
    ambiguous = model.ambiguous_regions(top_n=5)

    assert len(class_regions) <= 2
    assert (class_regions["region_target_class"] == model.classes_[0]).all()
    assert not ambiguous.empty
    assert ambiguous["class_margin"].le(1.0).all()


def test_save_load_preserves_regions_and_assignments(tmp_path):
    X, y = _dataset_frame()
    model = _model().fit(X, y)
    path = tmp_path / "class-regions.joblib"
    model.save(path)
    loaded = InsideForestClassRegionClusterer.load(path)

    pd.testing.assert_frame_equal(loaded.regions_, model.regions_)
    pd.testing.assert_frame_equal(loaded.region_metrics_, model.region_metrics_)
    assert np.array_equal(loaded.predict(X), model.predict(X))


def test_deprecated_multiclass_name_and_methods_remain_available():
    X, y = _dataset_frame()
    with pytest.warns(FutureWarning, match="InsideForestClassRegionClusterer"):
        model = InsideForestMulticlassClassifier(
            rf_params={"n_estimators": 6, "max_depth": 3},
            percentil=80,
            random_state=29,
        )
    model.fit(X, y)
    with pytest.warns(FutureWarning):
        explained = model.explain(top_n=2)
    with pytest.warns(FutureWarning):
        prototypes = model.prototype_regions(top_n=1)
    with pytest.warns(FutureWarning):
        ambiguous = model.confusion_regions(top_n=2)

    assert len(explained) <= 2
    assert not prototypes.empty
    assert isinstance(ambiguous, pd.DataFrame)


def test_invalid_branch_aggregation_is_rejected():
    X, y = _dataset_frame()
    with pytest.raises(ValueError, match="branch_aggregation"):
        _model(branch_aggregation="iou").fit(X, y)
