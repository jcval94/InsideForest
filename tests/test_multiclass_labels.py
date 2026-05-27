import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from InsideForest.multiclass_labels import get_multiclass_labels
from InsideForest.multiclass_rules import extract_multiclass_leaf_rules


def test_multiclass_labels_support_string_targets_without_target_mean():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = np.asarray([iris.target_names[idx] for idx in iris.target], dtype=object)
    rf = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=3).fit(X, y)
    rules = extract_multiclass_leaf_rules(
        rf,
        X,
        y,
        percentil=95,
        random_state=3,
    )

    labels = get_multiclass_labels(rules, X, y, class_labels=rf.classes_)

    assert not labels.empty
    assert "target_mean" not in labels.columns
    assert labels["target_class"].isin(iris.target_names).all()
    assert all(np.isclose(dist.sum(), 1.0) for dist in labels["class_distribution"] if dist.sum() > 0)
    assert labels["entropy"].ge(0).all()
