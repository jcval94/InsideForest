import math

import pandas as pd
import pandas.testing as pdt

from InsideForest.labels import Labels


def make_interval_dataframe():
    columns = pd.MultiIndex.from_tuples(
        [
            ("linf", "diameter"),
            ("lsup", "diameter"),
            ("linf", "altura_media"),
            ("lsup", "altura_media"),
            ("linf", "density"),
            ("lsup", "density"),
        ]
    )
    data = [
        [0, 1, 10, 12, 0.0005, 0.002],
        [2, 2, 15, 15, 0.002, 0.002],
    ]
    return pd.DataFrame(data, columns=columns)


def test_custom_round_varied_inputs():
    labels = Labels()

    assert labels.custom_round(150.4) == 150
    assert labels.custom_round(10.0000000001) == 10
    assert labels.custom_round(0.005) == "5.00e-03"
    assert labels.custom_round(0) == 0
    assert labels.custom_round(True) == 1
    assert math.isnan(labels.custom_round(float("nan")))
    assert labels.custom_round(float("inf")) == float("inf")
    assert labels.custom_round("text") == "text"
    assert labels.custom_round(None) is None


def test_get_intervals_formats_values_and_drops_height():
    labels = Labels()
    interval_df = make_interval_dataframe()

    descriptions = labels.get_intervals(interval_df)

    assert len(descriptions) == 2
    assert "diameter between 0 and 1" in descriptions[0]
    assert "density between 5.00e-04 and 2.00e-03" in descriptions[0]
    assert descriptions[1] == ""


def test_get_labels_generates_scores_and_populations():
    labels = Labels()
    columns = pd.MultiIndex.from_tuples(
        [
            ("linf", "diameter"),
            ("lsup", "diameter"),
            ("linf", "density"),
            ("lsup", "density"),
        ]
    )
    interval_data = [
        [0.0, 1.0, 0.0, 0.005],
        [1.0, 3.0, 0.0, 0.002],
        [5.0, 5.0, 0.0, 0.001],
    ]
    range_df = pd.DataFrame(interval_data, columns=columns)

    df = pd.DataFrame(
        {
            "diameter": [0.2, 0.8, 2.5],
            "density": [0.001, 0.003, 0.0005],
            "target": [1.0, 0.0, 0.5],
        }
    )

    labels_list = labels.get_labels([range_df], df, "target", max_labels=3, num_branches=2)

    assert len(labels_list) == 1
    branch = labels_list[0]
    assert len(branch) == 2

    first_description = next(desc for desc in branch if "diameter between 0" in desc)
    second_description = next(desc for desc in branch if "diameter between 1" in desc)

    first_score, first_population = branch[first_description]
    second_score, second_population = branch[second_description]

    assert first_score == (0.5, 2)
    pdt.assert_frame_equal(first_population, df.iloc[[0, 1]])

    assert second_score == (0.5, 1)
    pdt.assert_frame_equal(second_population, df.iloc[[2]])

    assert all("density" in desc for desc in branch)
