import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from .trees import Trees
from .regions import Regions
from .descrip import get_frontiers


class _BaseInsideForest:
    """Internal base class handling shared ``fit`` and ``predict`` logic.

    After fitting, the random forest's feature importances can be inspected
    via the :attr:`feature_importances_` property or visualized with the
    :meth:`plot_importances` method.

    Parameters
    ----------
    rf_cls : type
        Random forest estimator class to instantiate (classifier or regressor).
    rf_params : dict, optional
        Parameters passed directly to the random forest estimator.
    tree_params : dict, optional
        Parameters forwarded to :class:`Trees`.
    var_obj : str, default "target"
        Name of the column created for the target variable when building the
        internal DataFrame used for rule extraction.
    n_clusters : int or None, optional
        Desired number of clusters passed to :meth:`Regions.labels`.
    include_summary_cluster : bool, default False
        Whether to keep summary columns in the output of
        :meth:`Regions.labels`.
    method : str or None, optional
        Strategy used to consolidate final cluster labels in
        :meth:`Regions.labels`. Available options are ``"select_clusters"``
        (default), ``"balance_lists_n_clusters"``, ``"max_prob_clusters"`` and
        ``"menu"`` which leverages :class:`MenuClusterSelector`. When ``None``
        or ``"select_clusters"`` the raw output of :func:`select_clusters` is
        used.
    divide : int, default 5
        Value forwarded to :func:`get_frontiers` when computing cluster
        frontiers.
    get_detail : bool, default False
        When ``True`` :meth:`fit` computes and stores additional cluster
        details and frontiers.
    """

    def __init__(
        self,
        rf_cls,
        rf_params=None,
        tree_params=None,
        var_obj="target",
        n_clusters=None,
        include_summary_cluster=False,
        method="select_clusters",
        divide=5,
        get_detail=False,
    ):
        self.rf_params = rf_params or {}
        self.tree_params = tree_params or {}
        self.var_obj = var_obj
        self.n_clusters = n_clusters
        self.include_summary_cluster = include_summary_cluster
        self.method = method
        self.divide = divide
        self.get_detail = get_detail

        self.rf = rf_cls(**self.rf_params)
        self.trees = Trees(**self.tree_params)
        self.regions = Regions()

        # Attributes populated after fitting
        self.labels_ = None
        self.feature_names_ = None
        self.df_clusters_descript_ = None
        self.df_reres_ = None
        self.df_datos_explain_ = None
        self.frontiers_ = None

    def get_params(self, deep=True):
        """Return estimator parameters.

        Parameters
        ----------
        deep : bool, default=True
            Included for API compatibility. Has no effect in this
            implementation but mirrors scikit-learn's ``BaseEstimator``.

        Returns
        -------
        dict
            Mapping of parameter names to their values.
        """

        return {
            "rf_params": self.rf_params,
            "tree_params": self.tree_params,
            "var_obj": self.var_obj,
            "n_clusters": self.n_clusters,
            "include_summary_cluster": self.include_summary_cluster,
            "method": self.method,
            "divide": self.divide,
            "get_detail": self.get_detail,
        }

    def set_params(self, **params):
        """Set estimator parameters.

        Parameters
        ----------
        **params : dict
            Estimator parameters to update. Keys corresponding to ``rf``
            can be provided either as ``rf_params`` (a complete dict) or
            using the ``rf__<param>`` notation familiar from scikit-learn.

        Returns
        -------
        self
        """

        for key, value in params.items():
            if key == "rf_params":
                self.rf_params = value
                self.rf.set_params(**self.rf_params)
            elif key.startswith("rf__"):
                sub_key = key.split("__", 1)[1]
                self.rf_params[sub_key] = value
                self.rf.set_params(**{sub_key: value})
            elif key == "tree_params":
                self.tree_params = value
                self.trees = Trees(**self.tree_params)
            elif key.startswith("tree_params__"):
                sub_key = key.split("__", 1)[1]
                self.tree_params[sub_key] = value
                self.trees = Trees(**self.tree_params)
            elif key in {
                "var_obj",
                "n_clusters",
                "include_summary_cluster",
                "method",
                "divide",
                "get_detail",
            }:
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter '{key}'")

        return self

    def fit(self, X, y=None, rf=None):
        """Fit the internal random forest and compute cluster labels.

        Whether detailed cluster information is computed depends on the
        ``get_detail`` attribute set during initialization.

        Parameters
        ----------
        X : array-like or pandas.DataFrame
            Feature matrix or DataFrame containing the target column.
        y : array-like, optional
            Target vector. If not provided and ``X`` is a DataFrame containing
            ``var_obj``, that column will be used as the target.
        rf : estimator, optional
            Custom random forest instance to use during fitting. If ``None``,
            the estimator created during initialization is used. If the
            provided estimator is already trained, it will be used as is
            without additional fitting.

        Raises
        ------
        ValueError
            If ``y`` is ``None`` and ``X`` does not contain ``var_obj`` as a
            column.
        """

        if y is None:
            if isinstance(X, pd.DataFrame):
                if self.var_obj in X.columns:
                    y = X[self.var_obj]
                    X_df = X.drop(columns=[self.var_obj])
                else:
                    raise ValueError(
                        f"Target column '{self.var_obj}' not found in DataFrame. "
                        "Provide the target column name via 'var_obj' or pass 'y' explicitly."
                    )
            else:
                raise ValueError(
                    "When 'y' is None, 'X' must be a DataFrame containing the target column."
                )
        else:
            if isinstance(X, pd.DataFrame):
                X_df = X.copy()
            else:
                X_df = pd.DataFrame(data=X)

        # Replace spaces with underscores to keep compatibility with Trees
        X_df.columns = [str(c).replace(" ", "_") for c in X_df.columns]
        self.feature_names_ = list(X_df.columns)

        # Allow passing a custom random forest estimator
        if rf is not None:
            self.rf = rf

        # Train RandomForest only if it has not been fitted already
        try:
            check_is_fitted(self.rf)
        except NotFittedError:
            self.rf.fit(X=X_df, y=y)

        # Build DataFrame including target for region extraction
        df = X_df.copy()
        df[self.var_obj] = y

        # Extract rules and compute labels using existing utilities
        separacion_dim = self.trees.get_branches(
            df=df, var_obj=self.var_obj, regr=self.rf
        )
        df_reres = self.regions.prio_ranges(separacion_dim=separacion_dim, df=df)

        if self.get_detail:
            df_datos_clusterizados, df_clusters_descripcion, labels = self.regions.labels(
                df=df,
                df_reres=df_reres,
                n_clusters=self.n_clusters,
                include_summary_cluster=self.include_summary_cluster,
                method=self.method,
                return_dfs=True,
                var_obj=self.var_obj,
            )
            labels = pd.Series(labels).fillna(value=-1).to_numpy()
            self.labels_ = labels
            self.df_clusters_descript_ = df_clusters_descripcion
            self.df_reres_ = df_reres

            df_datos_explain, frontiers = get_frontiers(
                df_datos_descript=df_clusters_descripcion, df=df, divide=self.divide
            )
            self.df_datos_explain_ = df_datos_explain
            self.frontiers_ = frontiers
        else:
            labels = self.regions.labels(
                df=df,
                df_reres=df_reres,
                n_clusters=self.n_clusters,
                include_summary_cluster=self.include_summary_cluster,
                method=self.method,
                return_dfs=False,
                var_obj=self.var_obj,
            )
            labels = pd.Series(labels).fillna(value=-1).to_numpy()
            self.labels_ = labels
            self.df_reres_ = df_reres
            self.df_clusters_descript_ = None
            self.df_datos_explain_ = None
            self.frontiers_ = None

        return self

    @property
    def feature_importances_(self):
        """Feature importances of the fitted random forest.

        Returns
        -------
        ndarray of shape (n_features,)
            Importance of each feature as computed by the underlying random
            forest.

        Raises
        ------
        NotFittedError
            If the internal random forest has not been fitted yet.
        """

        check_is_fitted(self.rf)
        return self.rf.feature_importances_

    def plot_importances(self):
        """Plot feature importances of the underlying random forest.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object containing a bar chart of feature importances.
        """

        check_is_fitted(self.rf)

        import numpy as np
        import matplotlib.pyplot as plt

        importances = self.rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        feature_names = [self.feature_names_[i] for i in indices]

        fig, ax = plt.subplots()
        ax.bar(range(len(importances)), importances[indices])
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels(feature_names, rotation=90)
        ax.set_ylabel("Feature importance")
        ax.set_title("Feature importances")
        fig.tight_layout()
        return ax

    def predict(self, X):
        """Assign cluster labels to new data based on learned regions.

        Parameters
        ----------
        X : array-like or pandas.DataFrame
            Feature matrix. If a DataFrame is supplied, columns are
            re-ordered to match the training data.

        Returns
        -------
        ndarray of shape (n_samples,)
            Cluster label for each sample. Records that do not fall in any
            learned region receive the label ``-1``.
        """

        if self.df_reres_ is None:
            raise RuntimeError("InsideForest instance is not fitted yet")

        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
            X_df.columns = [str(c).replace(" ", "_") for c in X_df.columns]
            missing_cols = [col for col in self.feature_names_ if col not in X_df.columns]
            if missing_cols:
                missing_str = ", ".join(missing_cols)
                raise ValueError(
                    "Input data is missing required feature columns: "
                    f"{missing_str}. Ensure these columns are present in 'X' or refit the model with this feature set."
                )
            # Reorder/Subset columns to match training features
            X_df = X_df[self.feature_names_]
        else:
            try:
                X_df = pd.DataFrame(data=X, columns=self.feature_names_)
            except ValueError as err:
                raise ValueError(
                    "Input data must contain all feature columns used during fitting: "
                    f"{', '.join(self.feature_names_)}."
                ) from err

        labels = self.regions.labels(
            df=X_df,
            df_reres=self.df_reres_,
            n_clusters=self.n_clusters,
            include_summary_cluster=False,
            method=self.method,
            return_dfs=False,
            var_obj=self.var_obj,
        )
        labels = pd.Series(labels).fillna(value=-1).to_numpy()
        return labels

    def score(self, X, y):
        """Return the score of the underlying random forest on the given data.

        The metric reported depends on the type of wrapped random forest
        estimator. For classifiers this is the mean accuracy, while for
        regressors it corresponds to :math:`R^2`.

        Parameters
        ----------
        X : array-like or pandas.DataFrame
            Feature matrix. If a DataFrame is provided, columns are normalized
            and reordered to match the training data.
        y : array-like
            Ground truth target values.

        Returns
        -------
        float
            Score as computed by :meth:`sklearn.ensemble.RandomForestRegressor.score`
            or :meth:`sklearn.ensemble.RandomForestClassifier.score` depending on
            the estimator type.
        """

        check_is_fitted(self.rf)

        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
            X_df.columns = [str(c).replace(" ", "_") for c in X_df.columns]
            X_df = X_df[self.feature_names_]
        else:
            X_df = pd.DataFrame(data=X, columns=self.feature_names_)

        return self.rf.score(X_df, y)

    def save(self, filepath: str):
        """Save the fitted model and derived attributes to ``filepath``.

        Parameters
        ----------
        filepath : str
            Destination path where the model will be serialized.
        """

        payload = {
            "rf": self.rf,
            "rf_params": self.rf_params,
            "tree_params": self.tree_params,
            "var_obj": self.var_obj,
            "n_clusters": self.n_clusters,
            "include_summary_cluster": self.include_summary_cluster,
            "method": self.method,
            "divide": self.divide,
            "get_detail": self.get_detail,
            "labels_": self.labels_,
            "feature_names_": self.feature_names_,
            "df_clusters_descript_": self.df_clusters_descript_,
            "df_reres_": self.df_reres_,
            "df_datos_explain_": self.df_datos_explain_,
            "frontiers_": self.frontiers_,
            "menu_selector_": getattr(self.regions, "_menu_selector", None),
        }
        joblib.dump(payload, filepath)

    @classmethod
    def load(cls, filepath: str):
        """Load a previously saved InsideForest model from ``filepath``.

        Parameters
        ----------
        filepath : str
            Path to the serialized model file.

        Returns
        -------
        Instance of ``cls``
            Restored model with fitted attributes.
        """

        payload = joblib.load(filepath)
        model = cls(
            rf_params=payload["rf_params"],
            tree_params=payload["tree_params"],
            var_obj=payload["var_obj"],
            n_clusters=payload["n_clusters"],
            include_summary_cluster=payload["include_summary_cluster"],
            method=payload.get("method", "select_clusters"),
            divide=payload["divide"],
            get_detail=payload.get("get_detail", False),
        )
        model.rf = payload["rf"]
        model.labels_ = payload["labels_"]
        model.feature_names_ = payload["feature_names_"]
        model.df_clusters_descript_ = payload["df_clusters_descript_"]
        model.df_reres_ = payload["df_reres_"]
        model.df_datos_explain_ = payload["df_datos_explain_"]
        model.frontiers_ = payload["frontiers_"]
        if "menu_selector_" in payload:
            model.regions._menu_selector = payload["menu_selector_"]
        return model


class InsideForestClassifier(_BaseInsideForest):
    """Wrapper model that combines a ``RandomForestClassifier`` with the
    Trees/Regions utilities to provide cluster labels for the training data."""

    def __init__(
        self,
        rf_params=None,
        tree_params=None,
        var_obj="target",
        n_clusters=None,
        include_summary_cluster=False,
        method="select_clusters",
        divide=5,
        get_detail=False,
    ):
        super().__init__(
            RandomForestClassifier,
            rf_params=rf_params,
            tree_params=tree_params,
            var_obj=var_obj,
            n_clusters=n_clusters,
            include_summary_cluster=include_summary_cluster,
            method=method,
            divide=divide,
            get_detail=get_detail,
        )


class InsideForestRegressor(_BaseInsideForest):
    """Wrapper model that combines a ``RandomForestRegressor`` with the
    Trees/Regions utilities to provide cluster labels for the training data."""

    def __init__(
        self,
        rf_params=None,
        tree_params=None,
        var_obj="target",
        n_clusters=None,
        include_summary_cluster=False,
        method="select_clusters",
        divide=5,
        get_detail=False,
    ):
        super().__init__(
            RandomForestRegressor,
            rf_params=rf_params,
            tree_params=tree_params,
            var_obj=var_obj,
            n_clusters=n_clusters,
            include_summary_cluster=include_summary_cluster,
            method=method,
            divide=divide,
            get_detail=get_detail,
        )


# Backward compatibility alias
InsideForest = InsideForestClassifier

