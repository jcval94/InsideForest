from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import logging

logger = logging.getLogger(__name__)


class Models:
  """Helper utilities for basic model analysis and tuning."""

  def get_knn_rows(self, df, target_col, criterio_fp=True, min_obs = 5):
    """Split a DataFrame based on KNN misclassifications.

    Trains a series of ``KNeighborsClassifier`` models increasing the number of
    neighbors until the amount of false positives or false negatives exceeds
    ``min_obs``. The rows that were misclassified by the final model are
    returned along with the remaining data.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing the features and the target column.
    target_col : str
        Name of the column with the target variable.
    criterio_fp : bool, default True
        When ``True`` the search stops when the number of false positives is
        greater than ``min_obs``. If ``False`` the criterion switches to false
        negatives instead.
    min_obs : int, default 5
        Minimum number of misclassified observations required to stop the
        search.

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame)
        Two DataFrames: the first with the misclassified rows (either false
        positives or false negatives) and the second with the remaining
        correctly classified observations.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' does not exist in the DataFrame")

    X = df.drop(columns=[target_col]).values
    y = df.loc[:, target_col].values
    for k in range(1,int(len(df))):
      try:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X, y)
        y_pred = knn.predict(X)
        cm = confusion_matrix(y, y_pred)
      except Exception as exc:
        logger.exception("Error training KNN: %s", exc)
        break
      tn, fp, fn, tp = cm.ravel()
      if criterio_fp:
        if fp>min_obs:
          break
      else:
        if fn>min_obs:
          break
    if fn>0:
      false_negatives = (y == 1) & (y_pred == 0)
      return df[false_negatives], df[~false_negatives]
    if fp>0:
      false_positives = (y == 0) & (y_pred == 1)
      return df[false_positives], df[~false_positives]
  
  def get_cvRF(self, X_train, y_train, param_grid):
    """Grid-search a RandomForest classifier.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix.
    y_train : array-like
        Training target vector.
    param_grid : dict
        Hyper-parameter grid passed to ``GridSearchCV``.

    Returns
    -------
    GridSearchCV or None
        Fitted ``GridSearchCV`` object using a ``RandomForestClassifier`` if the
        search succeeds, otherwise ``None`` is returned and the error is logged.
    """
    try:
      rf = RandomForestClassifier(random_state=31416)
      cv = GridSearchCV(rf, param_grid=param_grid, cv=5, n_jobs=-1)
      cv.fit(X_train, y_train)
      return cv
    except Exception as exc:
      logger.exception("Error in GridSearchCV: %s", exc)
      return None

