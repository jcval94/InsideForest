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

    with pytest.raises(ValueError):
        model.set_params(unknown=1)
