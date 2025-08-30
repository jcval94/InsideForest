import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from InsideForest import InsideForestClassifier


def test_get_params_returns_init_values():
    model = InsideForestClassifier(
        rf_params={"n_estimators": 5},
        tree_params={"lang": "python"},
        n_clusters=2,
        include_summary_cluster=True,
        method="balance_lists_n_clusters",
        divide=3,
        get_detail=True,
    )
    params = model.get_params()
    assert params["rf_params"]["n_estimators"] == 5
    assert params["tree_params"]["lang"] == "python"
    assert params["n_clusters"] == 2
    assert params["include_summary_cluster"] is True
    assert params["method"] == "balance_lists_n_clusters"
    assert params["divide"] == 3
    assert params["get_detail"] is True
    assert params["leaf_percentile"] == 96
    assert params["low_leaf_fraction"] == 0.03
    assert params["max_cases"] == 750
    assert params["seed"] == 42


def test_set_params_updates_attributes():
    model = InsideForestClassifier(rf_params={"n_estimators": 5})
    model.set_params(rf_params={"n_estimators": 3})
    assert model.rf_params["n_estimators"] == 3
    assert model.rf.get_params()["n_estimators"] == 3

    model.set_params(rf__max_depth=4)
    assert model.rf_params["max_depth"] == 4
    assert model.rf.get_params()["max_depth"] == 4

    model.set_params(n_clusters=7)
    assert model.n_clusters == 7

    model.set_params(get_detail=True)
    assert model.get_detail is True

    model.set_params(method="max_prob_clusters")
    assert model.method == "max_prob_clusters"

    model.set_params(method="menu")
    assert model.method == "menu"

    model.set_params(leaf_percentile=80)
    assert model.leaf_percentile == 80
    model.set_params(low_leaf_fraction=0.1)
    assert model.low_leaf_fraction == 0.1
    model.set_params(max_cases=100)
    assert model.max_cases == 100

    model.set_params(seed=123)
    assert model.seed == 123
    assert model.rf.get_params()["random_state"] == 123

    with pytest.raises(ValueError):
        model.set_params(unknown=1)


def test_fit_respects_max_cases():
    import numpy as np

    X = np.random.rand(100, 2)
    y = np.random.randint(0, 2, size=100)
    model = InsideForestClassifier(rf_params={"n_estimators": 1}, max_cases=50)
    model.fit(X, y)
    assert len(model.labels_) == 50


def test_sampling_is_deterministic():
    import numpy as np

    X = np.random.rand(1000, 2)
    y = np.arange(1000)
    model = InsideForestClassifier(rf_params={"n_estimators": 1}, max_cases=100)
    model.fit(X, y)
    expected = np.random.default_rng(42).choice(1000, size=100, replace=False)
    assert np.array_equal(model._sample_indices_, expected)


def test_fit_is_reproducible_with_seed():
    import numpy as np

    X = np.random.rand(1000, 5)
    y = np.random.randint(0, 2, size=1000)
    params = dict(rf_params={"n_estimators": 5, "n_jobs": 1},
                  max_cases=200,
                  n_clusters=3,
                  method="balance_lists_n_clusters",
                  seed=7)
    m1 = InsideForestClassifier(**params)
    m2 = InsideForestClassifier(**params)
    m1.fit(X, y)
    m2.fit(X, y)
    assert np.array_equal(m1.labels_, m2.labels_)
    assert np.array_equal(m1._sample_indices_, m2._sample_indices_)
