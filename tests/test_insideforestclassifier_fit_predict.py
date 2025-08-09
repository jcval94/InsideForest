import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import pytest

from InsideForest.insideforestclassifier import InsideForestClassifier


def test_insideforestclassifier_fit_predict_runs():
    X = pd.DataFrame(data={'feat1': [0, 1, 2, 3], 'feat2': [3, 2, 1, 0]})
    y = [0, 1, 0, 1]
    model = InsideForestClassifier(rf_params={'n_estimators': 5, 'random_state': 0})
    fitted = model.fit(X=X, y=y)
    assert fitted is model
    preds = model.predict(X=X)
    assert preds.shape == (4,)
    assert np.array_equal(preds, model.labels_)


def test_predict_before_fit_raises():
    model = InsideForestClassifier()
    X = pd.DataFrame(data={'feat1': [0], 'feat2': [0]})
    with pytest.raises(RuntimeError):
        model.predict(X=X)


def test_fit_accepts_df_with_target_column():
    df = pd.DataFrame(data={
        'feat1': [0, 1, 2, 3],
        'feat2': [3, 2, 1, 0],
        'target': [0, 1, 0, 1]
    })
    model = InsideForestClassifier(rf_params={'n_estimators': 5, 'random_state': 0})
    fitted = model.fit(X=df)
    assert fitted is model
    assert model.labels_.shape[0] == len(df)


def test_fit_df_missing_target_raises():
    df = pd.DataFrame(data={'feat1': [0, 1], 'feat2': [1, 0]})
    model = InsideForestClassifier()
    with pytest.raises(ValueError):
        model.fit(X=df)


def test_custom_label_and_frontier_params():
    df = pd.DataFrame(
        data={
            'feat1': [0, 1, 2, 3],
            'feat2': [3, 2, 1, 0],
            'target': [0, 1, 0, 1],
        }
    )
    model = InsideForestClassifier(
        rf_params={'n_estimators': 5, 'random_state': 0},
        n_clusters=2,
        include_summary_cluster=True,
        balanced=True,
        divide=3,
    )
    model.fit(X=df)
    assert model.n_clusters == 2
    assert model.include_summary_cluster is True
    assert model.balanced is True
    assert model.divide == 3
    assert model.labels_.shape[0] == len(df)
