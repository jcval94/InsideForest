import os, sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from InsideForest.cluster_selector import ChimeraValuesSelector


def test_chimera_values_selector_basic():
    records_train = [["x", "y"], ["x"], ["y"], ["x"]]
    y_train = [0, 0, 1, 1]
    selector = ChimeraValuesSelector()
    selector.fit(records_train, y_train)

    records = [["x", "y"], ["y"], ["x"], ["y"]]
    result = selector.predict(records, n_labels=2)

    assert result["n_labels"] == 2
    assert len(result["labels"]) == len(records)

    for rec, lab in zip(records, result["labels"]):
        assert lab in rec

    result_auto = selector.predict(records)
    assert len(result_auto["labels"]) == len(records)
    for rec, lab in zip(records, result_auto["labels"]):
        assert lab in rec
