import pandas as pd
from InsideForest.models import Models


def test_get_knn_rows_success():
    df = pd.DataFrame({'feature': [0, 1, 2, 3, 4, 5],
                       'target': [0, 0, 0, 1, 1, 1]})
    models = Models()
    mis_df, rest_df = models.get_knn_rows(df, 'target', criterio_fp=False, min_obs=0)
    assert not mis_df.empty
    assert len(mis_df) + len(rest_df) == len(df)
    assert rest_df.equals(df.drop(mis_df.index))


def test_get_knn_rows_no_misclassification():
    df = pd.DataFrame({'feature': [0, 1, 2, 3, 4, 5],
                       'target': [0, 0, 0, 1, 1, 1]})
    models = Models()
    mis_df, rest_df = models.get_knn_rows(df, 'target', min_obs=10)
    assert mis_df.empty
    assert rest_df.equals(df)


def test_get_knn_rows_training_error():
    df = pd.DataFrame({'feature': ['a', 'b', 'c'],
                       'target': [0, 1, 0]})
    models = Models()
    mis_df, rest_df = models.get_knn_rows(df, 'target')
    assert mis_df.empty
    assert rest_df.equals(df)
