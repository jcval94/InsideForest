import os, sys
import warnings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import pytest
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

from InsideForest import InsideForestRegressor
import InsideForest.inside_forest as inside_forest_module


def test_inside_forest_fit_predict_runs():
    X = pd.DataFrame(data={"feat1": [0, 1, 2, 3], "feat2": [3, 2, 1, 0]})
    y = [0.1, 0.2, 0.3, 0.4]
    model = InsideForestRegressor(rf_params={"n_estimators": 5, "random_state": 0})
    fitted = model.fit(X=X, y=y)
    assert fitted is model
    preds = model.predict(X=X)
    assert preds.shape == (4,)
    assert np.array_equal(preds, model.labels_)


def test_predict_before_fit_raises():
    model = InsideForestRegressor()
    X = pd.DataFrame(data={"feat1": [0], "feat2": [0]})
    with pytest.raises(RuntimeError):
        model.predict(X=X)


def test_predict_missing_columns_raises():
    X = pd.DataFrame(data={"feat1": [0, 1, 2, 3], "feat2": [3, 2, 1, 0]})
    y = [0.1, 0.2, 0.3, 0.4]
    model = InsideForestRegressor(rf_params={"n_estimators": 5, "random_state": 0})
    model.fit(X=X, y=y)
    X_missing = pd.DataFrame(data={"feat1": [0.1, 0.2]})
    with pytest.raises(ValueError, match="feat2"):
        model.predict(X=X_missing)


def test_fit_accepts_df_with_target_column():
    df = pd.DataFrame(
        data={
            "feat1": [0, 1, 2, 3],
            "feat2": [3, 2, 1, 0],
            "target": [0.1, 0.2, 0.3, 0.4],
        }
    )
    model = InsideForestRegressor(rf_params={"n_estimators": 5, "random_state": 0})
    fitted = model.fit(X=df)
    assert fitted is model
    assert model.labels_.shape[0] == len(df)


def test_fit_df_missing_target_raises():
    df = pd.DataFrame(data={"feat1": [0, 1], "feat2": [1, 0]})
    model = InsideForestRegressor()
    with pytest.raises(ValueError):
        model.fit(X=df)


def test_custom_label_and_frontier_params():
    df = pd.DataFrame(
        data={
            "feat1": [0, 1, 2, 3],
            "feat2": [3, 2, 1, 0],
            "target": [0.1, 0.2, 0.3, 0.4],
        }
    )
    model = InsideForestRegressor(
        rf_params={"n_estimators": 5, "random_state": 0},
        n_clusters=2,
        include_summary_cluster=True,
        method="balance_lists_n_clusters",
        divide=3,
    )
    model.fit(X=df)
    assert model.n_clusters == 2
    assert model.include_summary_cluster is True
    assert model.method == "balance_lists_n_clusters"
    assert model.divide == 3
    assert model.labels_.shape[0] == len(df)


def test_fit_accepts_custom_rf_instance():
    X = pd.DataFrame(data={"feat1": [0, 1, 2, 3], "feat2": [3, 2, 1, 0]})
    y = [0.1, 0.2, 0.3, 0.4]
    rf = RandomForestRegressor(n_estimators=5, random_state=0)
    model = InsideForestRegressor()
    fitted = model.fit(X=X, y=y, rf=rf)
    assert fitted is model
    assert model.rf is rf
    preds = model.predict(X=X)
    assert preds.shape == (4,)
    assert np.array_equal(preds, model.labels_)


def test_fit_accepts_trained_rf_without_refitting():
    class TrackingRF(RandomForestRegressor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.fit_calls = 0

        def fit(self, X, y, **kwargs):
            self.fit_calls += 1
            return super().fit(X, y, **kwargs)

    X = pd.DataFrame(data={"feat1": [0, 1, 2, 3], "feat2": [3, 2, 1, 0]})
    y = [0.1, 0.2, 0.3, 0.4]
    rf = TrackingRF(n_estimators=5, random_state=0)
    rf.fit(X, y)
    assert rf.fit_calls == 1

    model = InsideForestRegressor()
    fitted = model.fit(X=X, y=y, rf=rf)
    assert fitted is model
    assert model.rf is rf
    assert rf.fit_calls == 1

    preds = model.predict(X=X)
    assert preds.shape == (4,)
    assert np.array_equal(preds, model.labels_)


def test_save_and_load_roundtrip(tmp_path):
    X = pd.DataFrame(data={"feat1": [0, 1, 2, 3], "feat2": [3, 2, 1, 0]})
    y = [0.1, 0.2, 0.3, 0.4]
    model = InsideForestRegressor(rf_params={"n_estimators": 5, "random_state": 0})
    model.fit(X=X, y=y)
    preds = model.predict(X=X)
    filepath = tmp_path / "reg.joblib"
    model.save(str(filepath))

    loaded = InsideForestRegressor.load(str(filepath))
    loaded_preds = loaded.predict(X=X)

    assert np.array_equal(model.labels_, loaded.labels_)
    assert np.array_equal(preds, loaded_preds)


def test_feature_importances_and_plot():
    X = pd.DataFrame(data={"feat1": [0, 1, 2, 3], "feat2": [3, 2, 1, 0]})
    y = [0.1, 0.2, 0.3, 0.4]
    model = InsideForestRegressor(rf_params={"n_estimators": 5, "random_state": 0})
    model.fit(X=X, y=y)

    importances = model.feature_importances_
    assert isinstance(importances, np.ndarray)
    assert importances.shape[0] == X.shape[1]

    import matplotlib.axes

    ax = model.plot_importances()
    assert isinstance(ax, matplotlib.axes.Axes)


def test_score_matches_rf_and_normalizes_input():
    X = pd.DataFrame(data={"feat 1": [0, 1, 2, 3], "feat 2": [3, 2, 1, 0]})
    y = [0.1, 0.2, 0.3, 0.4]
    model = InsideForestRegressor(rf_params={"n_estimators": 5, "random_state": 0})
    model.fit(X=X, y=y)

    X_messy = X[["feat 2", "feat 1"]]
    X_norm = X.copy()
    X_norm.columns = [c.replace(" ", "_") for c in X_norm.columns]
    expected = model.rf.score(X_norm, y)
    assert model.score(X_messy, y) == pytest.approx(expected)


def test_fit_respects_get_detail_flag_regressor():
    X = pd.DataFrame(data={'feat1': [0, 1, 2, 3], 'feat2': [3, 2, 1, 0]})
    y = [0.1, 0.4, 0.2, 0.3]
    model = InsideForestRegressor(rf_params={'n_estimators': 5, 'random_state': 0})
    model.fit(X=X, y=y)
    assert model.df_clusters_description_ is None

    model_detail = InsideForestRegressor(
        rf_params={'n_estimators': 5, 'random_state': 0}, get_detail=True
    )
    model_detail.fit(X=X, y=y)
    assert model_detail.df_clusters_description_ is not None


def test_regressor_feature_reduction_uses_regression_scores_for_integer_targets(monkeypatch):
    calls = {"regression": 0, "classification": 0}

    def fake_mutual_info_regression(X, y):
        calls["regression"] += 1
        return np.arange(X.shape[1], dtype=float)

    def fake_mutual_info_classif(X, y):
        calls["classification"] += 1
        return np.zeros(X.shape[1], dtype=float)

    monkeypatch.setattr(
        inside_forest_module, "mutual_info_regression", fake_mutual_info_regression
    )
    monkeypatch.setattr(
        inside_forest_module, "mutual_info_classif", fake_mutual_info_classif
    )

    X = pd.DataFrame(np.arange(120).reshape(30, 4), columns=list("abcd"))
    y = pd.Series(np.arange(30), dtype=int)
    model = InsideForestRegressor(
        auto_feature_reduce=True,
        explicit_k_features=2,
        no_trees_search=3,
        rf_params={"n_estimators": 3, "max_depth": 3, "random_state": 0, "n_jobs": 1},
        seed=0,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)

    assert calls == {"regression": 1, "classification": 0}
    assert model.feature_names_ == ["c", "d"]
    warning_text = "\n".join(str(w.message) for w in caught)
    assert "could represent a regression problem" not in warning_text
    assert "unique classes" not in warning_text


def test_regressor_auto_fast_keeps_select_clusters_default():
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 5)))
    y = pd.Series(np.linspace(0, 100, 50))
    model = InsideForestRegressor(
        auto_fast=True,
        no_trees_search=4,
        rf_params={"n_estimators": 4, "max_depth": 3, "random_state": 0, "n_jobs": 1},
        seed=0,
    )

    model.fit(X, y)

    assert model.method == "select_clusters"
    assert model._fast_params_used_["method"] == "select_clusters"


def test_predict_and_score_accept_full_numpy_after_feature_reduction():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 20))
    y = X[:, 0] * 2.0 - X[:, 3] + rng.normal(scale=0.1, size=80)
    model = InsideForestRegressor(
        auto_feature_reduce=True,
        explicit_k_features=8,
        no_trees_search=5,
        rf_params={"n_estimators": 5, "max_depth": 4, "random_state": 0, "n_jobs": 1},
        seed=0,
    )
    model.fit(X, y)

    labels = model.predict(X[:10])
    score = model.score(X, y)

    X_reduced = pd.DataFrame(X[:, model._feature_mask_], columns=model.feature_names_)
    expected_score = model.rf.score(X_reduced, y)
    assert labels.shape == (10,)
    assert score == pytest.approx(expected_score)


def test_fit_numpy_with_misaligned_series_target_uses_position():
    class TrackingRF(RandomForestRegressor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.y_seen = None

        def fit(self, X, y, **kwargs):
            self.y_seen = np.asarray(y)
            return super().fit(X, y, **kwargs)

    X = np.arange(80, dtype=float).reshape(40, 2)
    y = pd.Series(np.linspace(10.0, 20.0, 40), index=np.arange(100, 140))
    rf = TrackingRF(n_estimators=3, max_depth=3, random_state=0)
    model = InsideForestRegressor(no_trees_search=3)

    model.fit(X, y, rf=rf)

    assert np.array_equal(rf.y_seen, y.to_numpy())
    assert not np.isnan(rf.y_seen).any()


def test_regressor_predict_contract_returns_region_labels_not_continuous_values():
    X = pd.DataFrame(data={"feat1": np.arange(10), "feat2": np.arange(10)[::-1]})
    y = np.linspace(10.0, 30.0, 10)
    model = InsideForestRegressor(
        no_trees_search=4,
        rf_params={"n_estimators": 4, "max_depth": 3, "random_state": 0, "n_jobs": 1},
    )
    model.fit(X, y)

    region_labels = model.predict(X)
    continuous_values = model.rf.predict(X)

    assert np.array_equal(region_labels, model.labels_)
    assert region_labels.shape == continuous_values.shape
    assert not np.allclose(region_labels, continuous_values)


def test_diabetes_regression_smoke_has_stable_region_semantics_without_class_warnings():
    data = load_diabetes(as_frame=True)
    X = data.data.iloc[:120]
    y = data.target.iloc[:120]
    model = InsideForestRegressor(
        auto_fast=True,
        auto_feature_reduce=True,
        explicit_k_features=8,
        no_trees_search=6,
        rf_params={"n_estimators": 8, "max_depth": 4, "random_state": 0, "n_jobs": 1},
        seed=0,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)

    labels = model.predict(X.iloc[:25])
    warning_text = "\n".join(str(w.message) for w in caught)
    assert "could represent a regression problem" not in warning_text
    assert "unique classes" not in warning_text
    assert model.method == "select_clusters"
    assert np.unique(model.labels_).size >= 2
    assert np.mean(labels == -1) < 1.0
    assert np.isfinite(model.score(X, y))


def test_regressor_region_quality_uses_continuous_target_statistics():
    data = load_diabetes(as_frame=True)
    X = data.data.iloc[:90]
    y = data.target.iloc[:90]
    model = InsideForestRegressor(
        no_trees_search=5,
        rf_params={"n_estimators": 5, "max_depth": 3, "random_state": 0, "n_jobs": 1},
        seed=0,
    )

    model.fit(X, y)

    assert model.region_quality_ is not None
    assert not model.region_quality_.empty
    assert {"target_mean", "target_std", "support", "coverage"}.issubset(
        model.region_quality_.columns
    )
    assert model.region_quality_["target_mean"].notna().any()
    assert model.region_quality_["dominant_probability"].isna().all()
    report = model.region_quality_report()
    assert report["n_regions"] == len(model.region_quality_)
    assert np.isnan(report["weighted_region_purity"])

