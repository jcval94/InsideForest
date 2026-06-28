import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier, is_clusterer
from sklearn.datasets import load_iris

from InsideForest import InsideForestClassifier, InsideForestRegionClusterer


def _data():
    data = load_iris()
    X = pd.DataFrame(
        data.data,
        columns=[str(column).replace(" ", "_") for column in data.feature_names],
    )
    return X, data.target


def test_region_clusterer_exposes_shared_region_api():
    X, y = _data()
    model = InsideForestRegionClusterer(
        rf_params={"n_estimators": 6, "max_depth": 3, "random_state": 7},
        no_trees_search=6,
        max_cases=len(X),
        seed=7,
    ).fit(X, y)

    assert is_clusterer(model)
    assert not is_classifier(model)
    assert isinstance(clone(model), InsideForestRegionClusterer)
    assert model.forest_ is model.rf
    assert model.regions_ is model.region_rules_
    assert model.region_metrics_ is model.region_quality_
    assert model.raw_regions_ is not None
    assert model.n_features_in_ == X.shape[1]

    transformed = model.transform(X.head(8))
    assignments = model.assign_regions(X.head(8))
    explained = model.explain_regions(top_n=3)

    assert transformed.shape == (8, len(model.region_rules_))
    assert len(assignments) == 8
    assert set(assignments["source"]).issubset({"region", "unmatched"})
    assert len(explained) <= 3
    assert "region_target_class" in explained
    assert np.isclose(model.score(X, y), model.region_quality_report(X, y)["ami"])


def test_legacy_classifier_name_keeps_forest_score():
    X, y = _data()
    with pytest.warns(FutureWarning, match="InsideForestRegionClusterer"):
        model = InsideForestClassifier(
            rf_params={"n_estimators": 5, "max_depth": 3, "random_state": 9},
            no_trees_search=5,
            max_cases=len(X),
            seed=9,
        )
    model.fit(X, y)
    with pytest.warns(FutureWarning, match="forest accuracy"):
        legacy_score = model.score(X, y)
    assert np.isclose(legacy_score, model.rf.score(X, y))


def test_region_clusterer_fit_predict_matches_predict():
    X, y = _data()
    model = InsideForestRegionClusterer(
        rf_params={"n_estimators": 5, "max_depth": 3, "random_state": 5},
        no_trees_search=5,
        max_cases=len(X),
        seed=5,
    )
    labels = model.fit_predict(X, y)
    assert np.array_equal(labels, model.predict(X))


def test_region_clusterer_persistence_restores_common_attributes(tmp_path):
    X, y = _data()
    model = InsideForestRegionClusterer(
        rf_params={"n_estimators": 5, "max_depth": 3, "random_state": 11},
        no_trees_search=5,
        max_cases=len(X),
        seed=11,
    ).fit(X, y)
    path = tmp_path / "region-clusterer.joblib"
    model.save(path)
    loaded = InsideForestRegionClusterer.load(path)

    assert np.array_equal(loaded.predict(X), model.predict(X))
    assert loaded.forest_ is loaded.rf
    assert loaded.regions_ is loaded.region_rules_
    assert loaded.region_metrics_ is loaded.region_quality_
    assert loaded.raw_regions_ is not None
