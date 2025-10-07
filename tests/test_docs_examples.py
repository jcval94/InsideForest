import numpy as np
import pandas as pd
from pathlib import Path
import sys

from sklearn.datasets import load_diabetes, load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from InsideForest import (
    InsideForestClassifier,
    InsideForestRegressor,
    MenuClusterSelector,
    Regions,
    Trees,
    cluster_selector,
)


def _prepare_iris_dataframe():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    return iris, df


def test_quick_api_classifier_workflow():
    X, y = load_iris(return_X_y=True)
    clf = InsideForestClassifier(
        rf_params={"n_estimators": 10, "random_state": 21},
        auto_fast=True,
        get_detail=True,
    )
    clf.fit(X, y)

    cluster_labels = clf.predict(X)

    assert cluster_labels.shape == y.shape
    assert clf.df_clusters_description_ is not None
    assert not clf.df_clusters_description_.empty


def test_quick_api_regressor_workflow():
    diabetes = load_diabetes()
    X, y = diabetes.data[:200], diabetes.target[:200]

    regr = InsideForestRegressor(
        rf_params={"n_estimators": 10, "random_state": 7},
        auto_feature_reduce=True,
        get_detail=True,
    )
    regr.fit(X, y)

    predictions = regr.predict(X)

    assert predictions.shape == y.shape
    assert regr.df_clusters_description_ is not None
    assert not regr.df_clusters_description_.empty


def test_cluster_selector_menu():
    X, y = load_iris(return_X_y=True)
    clf = InsideForestClassifier(
        rf_params={"n_estimators": 5, "random_state": 5},
        auto_fast=True,
        get_detail=True,
    )
    clf.fit(X, y)

    df = pd.DataFrame(X, columns=clf.feature_names_out_)
    df["target"] = y
    clustered, descriptive, labels = clf.regions.labels(
        df=df,
        df_reres=clf.df_reres_,
        n_clusters=None,
        include_summary_cluster=False,
        method="select_clusters",
        return_dfs=True,
        var_obj="target",
    )

    records_full = clustered["clusters_list"].tolist()
    records = records_full[:30]
    labels_subset = y[:30]

    selector = MenuClusterSelector(target_K=2, seed=0, max_passes=1)
    selector.fit(records, labels_subset)
    assignments = selector.predict(records, n_clusters=2)

    assert len(assignments) == len(records)

    balanced = cluster_selector.balance_lists_n_clusters(
        records=records,
        n_clusters=2,
        seed=0,
        max_iter=20,
        restarts=1,
    )
    assert len(balanced) == len(records)


def test_model_persistence_roundtrip(tmp_path):
    X, y = load_iris(return_X_y=True)
    clf = InsideForestClassifier(
        rf_params={"n_estimators": 10, "random_state": 11},
        auto_fast=True,
        get_detail=True,
    )
    clf.fit(X, y)

    destination = tmp_path / "model.joblib"
    clf.save(destination.as_posix())

    loaded = InsideForestClassifier.load(destination.as_posix())
    np.testing.assert_array_equal(loaded.predict(X), clf.predict(X))
    assert list(loaded.feature_names_out_) == list(clf.feature_names_out_)


def test_how_it_works_flow():
    iris, df_raw = _prepare_iris_dataframe()
    indices = np.concatenate(
        [np.arange(0, 2), np.arange(50, 52), np.arange(100, 102)]
    )
    df_raw = df_raw.iloc[indices].reset_index(drop=True)
    iris_subset = iris.target[indices]
    feature_columns = [str(i) for i in range(df_raw.shape[1] - 1)]
    df = df_raw.copy()
    df.columns = feature_columns + ["target"]

    rf = RandomForestClassifier(n_estimators=3, random_state=3)
    rf.fit(df[feature_columns], iris_subset)

    trees = Trees("sklearn", n_sample_multiplier=0.005, ef_sample_multiplier=1)
    branches = trees.get_branches(
        df=df,
        var_obj="target",
        regr=rf,
        no_trees_search=1,
    )

    regions = Regions()
    priority_ranges = regions.prio_ranges(branches, df[feature_columns])

    clustered, descriptive, labels = regions.labels(
        df=df,
        df_reres=priority_ranges,
        n_clusters=None,
        include_summary_cluster=False,
        method="select_clusters",
        return_dfs=True,
        var_obj="target",
    )

    assert not descriptive.empty
    assert "cluster" in clustered.columns
    assert len(labels) == df.shape[0]


def test_classification_tutorial_flow():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        train_size=0.35,
        stratify=data.target,
        random_state=15,
    )

    clf = InsideForestClassifier(
        rf_params={"n_estimators": 10, "random_state": 19},
        auto_fast=True,
        auto_feature_reduce=True,
        get_detail=True,
    )
    clf.fit(X_train, y_train)

    segments = clf.df_clusters_description_
    predictions = clf.predict(X_test)

    assert predictions.shape[0] == X_test.shape[0]
    assert segments is not None and not segments.empty


def test_regression_tutorial_flow():
    data = load_diabetes()
    X, y = data.data, data.target

    regr = InsideForestRegressor(
        rf_params={"n_estimators": 10, "random_state": 23},
        auto_feature_reduce=True,
        auto_fast=True,
        get_detail=True,
    )
    regr.fit(X, y)

    assert regr.df_clusters_description_ is not None
    assert not regr.df_clusters_description_.empty
    np.testing.assert_equal(regr.predict(X).shape[0], X.shape[0])
