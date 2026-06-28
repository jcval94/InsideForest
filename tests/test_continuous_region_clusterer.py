import numpy as np
import pandas as pd
import pytest
import joblib
from sklearn.base import clone, is_clusterer, is_regressor
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

from InsideForest import (
    InsideForestContinuousRegionClusterer,
    InsideForestRegressor,
)


def _data(n=120):
    dataset = load_diabetes(as_frame=True)
    return dataset.data.iloc[:n].copy(), dataset.target.iloc[:n].to_numpy()


def _model(**kwargs):
    params = {
        "rf_params": {
            "n_estimators": 8,
            "max_depth": 4,
            "min_samples_leaf": 3,
            "random_state": 7,
            "n_jobs": 1,
        },
        "leaf_percentile": 80,
        "low_leaf_fraction": 0.0,
        "min_support": 2,
        "random_state": 7,
        "n_jobs": 1,
    }
    params.update(kwargs)
    return InsideForestContinuousRegionClusterer(**params)


def test_continuous_clusterer_contract_and_fit_predict_identity():
    X, y = _data()
    model = _model()
    labels = model.fit_predict(X, y)

    assert is_clusterer(model)
    assert not is_regressor(model)
    assert isinstance(clone(model), InsideForestContinuousRegionClusterer)
    assert np.array_equal(labels, model.predict(X))
    assert model.forest_ is not None
    assert model.labels_.shape == (len(X),)
    assert model.n_features_in_ == X.shape[1]
    assert model.raw_regions_["leaf_region_id"].is_unique
    assert len(model.regions_) <= len(model.raw_regions_)


def test_transform_argmax_matches_predict_for_covered_rows():
    X, y = _data()
    model = _model().fit(X, y)
    labels = model.predict(X)
    transformed = model.transform(X)
    covered = labels != -1

    assert transformed.shape == (len(X), len(model.regions_))
    assert np.array_equal(transformed[covered].argmax(axis=1), labels[covered])


def test_assignments_expose_continuous_statistics_and_unmatched_is_explicit():
    X, y = _data()
    model = _model().fit(X, y)
    assignments = model.assign_regions(X.head(10))
    required = {
        "cluster_id",
        "representative_region_id",
        "membership_score",
        "target_mean",
        "target_median",
        "target_std",
        "target_iqr",
        "target_min",
        "target_max",
        "mean_shift",
        "standardized_mean_shift",
        "dispersion_reduction",
        "matched_region_count",
        "matched_region_ids",
        "source",
    }
    assert required == set(assignments.columns)
    assert set(assignments["source"]).issubset({"region", "unmatched"})

    selected = model.regions_
    try:
        model.regions_ = selected.iloc[0:0].copy()
        unmatched = model.assign_regions(X.head(2))
        assert np.array_equal(model.predict(X.head(2)), np.array([-1, -1]))
        assert (unmatched["source"] == "unmatched").all()
        assert (unmatched["membership_score"] == 0).all()
        assert unmatched["target_mean"].isna().all()
    finally:
        model.regions_ = selected


def test_region_objective_matches_documented_formula():
    X, y = _data()
    model = _model().fit(X, y)
    raw = model.raw_regions_
    expected = (
        raw["coverage"]
        * raw["dispersion_reduction"]
        * (0.5 + 0.5 * raw["separation"])
    )
    assert np.allclose(raw["region_score"], expected)
    assert raw["dispersion_reduction"].between(0, 1).all()


def test_score_is_eta_squared_including_unmatched_and_constant_target_is_zero():
    X, y = _data()
    model = _model().fit(X, y)
    labels = model.predict(X)
    global_mean = y.mean()
    total = np.sum((y - global_mean) ** 2)
    between = sum(
        np.sum(labels == label)
        * (y[labels == label].mean() - global_mean) ** 2
        for label in np.unique(labels)
    )
    assert model.score(X, y) == pytest.approx(between / total)
    assert model.score(X, np.ones(len(X))) == 0.0
    assert model.region_quality_report()["target_variance_explained"] == pytest.approx(
        model.score(X, y)
    )


def test_target_affine_and_sign_transformations_preserve_regions():
    X, y = _data()
    baseline = _model().fit(X, y)
    shifted = _model().fit(X, 3.5 * y + 17)
    inverted = _model().fit(X, -y)

    columns = ["leaf_region_id", "description"]
    assert baseline.regions_[columns].equals(shifted.regions_[columns])
    assert baseline.regions_[columns].equals(inverted.regions_[columns])
    assert np.allclose(
        baseline.regions_["region_score"], shifted.regions_["region_score"]
    )
    assert np.allclose(
        baseline.regions_["region_score"], inverted.regions_["region_score"]
    )
    assert np.array_equal(baseline.predict(X), shifted.predict(X))
    assert np.array_equal(baseline.predict(X), inverted.predict(X))


@pytest.mark.parametrize(
    "target",
    [
        ["a", "b", "c"],
        [1.0, np.nan, 3.0],
        [1.0, np.inf, 3.0],
        [2.0, 2.0, 2.0],
    ],
)
def test_invalid_continuous_targets_are_rejected(target):
    X = pd.DataFrame({"a": [0, 1, 2], "b": [2, 1, 0]})
    with pytest.raises(ValueError):
        _model().fit(X, target)


def test_persistence_preserves_regions_statistics_and_assignments(tmp_path):
    X, y = _data()
    model = _model().fit(X, y)
    path = tmp_path / "continuous-region-clusterer.joblib"
    model.save(path)
    loaded = InsideForestContinuousRegionClusterer.load(path)

    assert np.array_equal(loaded.predict(X), model.predict(X))
    pd.testing.assert_frame_equal(loaded.regions_, model.regions_)
    pd.testing.assert_frame_equal(
        loaded.assign_regions(X.head(8)), model.assign_regions(X.head(8))
    )


def test_selection_tie_break_is_std_support_then_leaf_id():
    model = _model(leaf_percentile=None)
    frame = pd.DataFrame(
        {
            "leaf_region_id": ["c", "b", "a"],
            "region_score": [0.5, 0.5, 0.5],
            "target_std": [2.0, 1.0, 1.0],
            "support": [50, 10, 20],
        }
    )
    selected = model._select_regions(frame)
    assert selected["leaf_region_id"].tolist() == ["a", "b", "c"]
    assert selected["cluster_id"].tolist() == [0, 1, 2]


def test_legacy_regressor_warns_and_keeps_forest_r2():
    X, y = _data(80)
    with pytest.warns(FutureWarning, match="ContinuousRegionClusterer"):
        legacy = InsideForestRegressor(
            rf_params={"n_estimators": 5, "max_depth": 3, "random_state": 4}
        )
    legacy.fit(X, y)
    with pytest.warns(FutureWarning, match="legacy forest R2"):
        legacy_score = legacy.score(X, y)
    expected = legacy.forest_.score(legacy._coerce_X_predict(X), y)
    assert legacy_score == pytest.approx(expected)
    assert np.array_equal(legacy.predict(X), legacy.labels_)


def test_canonical_load_migrates_legacy_payload_and_preserves_region_ids(tmp_path):
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0, 0.0]})
    y = np.array([10.0, 12.0, 20.0, 22.0])
    forest = RandomForestRegressor(n_estimators=3, max_depth=2, random_state=2).fit(X, y)
    quality = pd.DataFrame(
        {
            "region_id": [7, 9],
            "lower_bounds": [{}, {"a": 1.5}],
            "upper_bounds": [{"a": 1.5}, {}],
            "description": ["a <= 1.5", "a > 1.5"],
            "weight": [0.8, 0.7],
            "support": [2, 2],
            "coverage": [0.5, 0.5],
            "target_mean": [11.0, 21.0],
            "target_std": [1.0, 1.0],
            "target_min": [10.0, 20.0],
            "target_max": [12.0, 22.0],
        }
    )
    payload = {
        "rf": forest,
        "rf_params": {"n_estimators": 3, "max_depth": 2, "random_state": 2},
        "feature_names_": ["a", "b"],
        "labels_": np.array([7, 7, 9, 9]),
        "region_quality_": quality,
        "region_quality_summary_": {"coverage": 1.0, "unmatched_rate": 0.0},
        "seed": 2,
    }
    path = tmp_path / "legacy-regressor.joblib"
    joblib.dump(payload, path)

    with pytest.warns(FutureWarning, match="legacy InsideForestRegressor"):
        loaded = InsideForestContinuousRegionClusterer.load(path)
    assert set(loaded.regions_["cluster_id"]) == {7, 9}
    assert np.array_equal(loaded.labels_, payload["labels_"])
    assert np.array_equal(loaded.predict(X), payload["labels_"])
