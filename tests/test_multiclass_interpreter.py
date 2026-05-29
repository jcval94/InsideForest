import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine

from InsideForest.multiclass import InsideForestMulticlassClassifier


def _dataset_frame(loader):
    data = loader()
    return pd.DataFrame(data.data, columns=data.feature_names), data.target


def test_interpreter_generates_rules_for_iris_and_wine():
    for loader in (load_iris, load_wine):
        X, y = _dataset_frame(loader)
        model = InsideForestMulticlassClassifier(
            rf_params={"n_estimators": 16, "max_depth": 5},
            percentil=90,
            min_support=2,
            random_state=13,
        ).fit(X, y)

        assert model.rules_["target_class"].nunique() > 2
        assert not model.prototype_regions(top_n=2).empty


def test_assign_regions_returns_complete_region_and_fallback_rows():
    X, y = _dataset_frame(load_iris)
    model = InsideForestMulticlassClassifier(
        rf_params={"n_estimators": 10, "max_depth": 4},
        percentil=95,
        random_state=17,
    ).fit(X, y)

    assigned = model.assign_regions(X.head(12))
    expected_columns = {
        "region_id",
        "predicted_class",
        "confidence",
        "score",
        "matched_region_count",
        "second_class",
        "margin",
        "is_conflict",
        "source",
    }

    assert expected_columns.issubset(assigned.columns)
    assert len(assigned) == 12
    assert assigned[["confidence", "score", "margin"]].notna().all().all()
    assert set(assigned["source"]).issubset({"region", "model_fallback"})

    model.rules_ = model.rules_.iloc[0:0].copy()
    fallback = model.assign_regions(X.head(3))

    assert set(fallback["source"]) == {"model_fallback"}
    assert fallback[["confidence", "score", "margin"]].notna().all().all()
    assert (fallback["region_id"] == -1).all()


def test_confusion_regions_flags_low_margin_regions():
    X, y = _dataset_frame(load_iris)
    model = InsideForestMulticlassClassifier(
        rf_params={"n_estimators": 8, "max_depth": 3},
        percentil=80,
        conflict_margin=1.0,
        random_state=19,
    ).fit(X, y)

    conflicts = model.confusion_regions(top_n=5)

    assert not conflicts.empty
    assert conflicts["is_conflict"].all()
    assert np.all(conflicts["margin"].to_numpy() <= 1.0)
