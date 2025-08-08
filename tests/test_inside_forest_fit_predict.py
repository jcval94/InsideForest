import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import pytest

from InsideForest.inside_forest import InsideForest


def test_inside_forest_fit_predict_runs():
    X = pd.DataFrame({'feat1': [0, 1, 2, 3], 'feat2': [3, 2, 1, 0]})
    y = [0, 1, 0, 1]
    model = InsideForest(rf_params={'n_estimators': 5, 'random_state': 0})
    fitted = model.fit(X, y)
    assert fitted is model
    preds = model.predict(X)
    assert preds.shape == (4,)
    assert np.array_equal(preds, model.labels_)


def test_predict_before_fit_raises():
    model = InsideForest()
    X = pd.DataFrame({'feat1': [0], 'feat2': [0]})
    with pytest.raises(RuntimeError):
        model.predict(X)
