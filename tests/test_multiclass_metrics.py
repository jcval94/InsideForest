import math

import numpy as np
import pandas as pd
import pytest

from InsideForest.multiclass_metrics import (
    build_class_priors,
    entropy,
    jensen_shannon_divergence,
    normalize_counts,
    purity_lift_coverage_score,
    score_multiclass_rules,
)


def test_normalize_counts_entropy_and_js_divergence():
    probabilities = normalize_counts([2, 1, 1])

    assert np.allclose(probabilities, [0.5, 0.25, 0.25])
    assert entropy([1, 0, 0]) == 0.0
    assert entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert jensen_shannon_divergence(probabilities, probabilities) == pytest.approx(0.0)


def test_default_score_uses_probability_lift_and_coverage():
    score = purity_lift_coverage_score(
        target_probability=0.8,
        coverage=0.25,
        class_lift=2.0,
    )

    assert score == pytest.approx(0.8)


def test_score_multiclass_rules_uses_class_labels_not_numeric_magnitude():
    classes = np.array([20, 10, 99], dtype=object)
    priors = build_class_priors([20, 10, 99, 99], classes)
    rules = pd.DataFrame(
        [
            {
                "target_class": 20,
                "target_probability": 0.5,
                "coverage": 0.25,
                "class_distribution": np.array([0.5, 0.25, 0.25]),
            },
            {
                "target_class": 10,
                "target_probability": 0.25,
                "coverage": 0.25,
                "class_distribution": np.array([0.5, 0.25, 0.25]),
            },
        ]
    )
    rules.attrs["classes_"] = classes

    scored = score_multiclass_rules(rules, priors)

    assert set(scored["target_class"]) == {20, 10}
    assert math.isfinite(scored.loc[0, "score"])
    assert scored.loc[0, "prior_probability"] == pytest.approx(0.25)
    assert scored.loc[1, "prior_probability"] == pytest.approx(0.25)
