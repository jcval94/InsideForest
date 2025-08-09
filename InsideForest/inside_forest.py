import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from .trees import Trees
from .regions import Regions
from .descrip import get_frontiers


class _BaseInsideForest:
    """Internal base class handling shared ``fit`` and ``predict`` logic.

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
    balanced : bool, default False
        Use balanced assignment of cluster labels instead of probability based
        assignment in :meth:`Regions.labels`.
    divide : int, default 5
        Value forwarded to :func:`get_frontiers` when computing cluster
        frontiers.
    """

    def __init__(
        self,
        rf_cls,
        rf_params=None,
        tree_params=None,
        var_obj="target",
        n_clusters=None,
        include_summary_cluster=False,
        balanced=False,
        divide=5,
    ):
        self.rf_params = rf_params or {}
        self.tree_params = tree_params or {}
        self.var_obj = var_obj
        self.n_clusters = n_clusters
        self.include_summary_cluster = include_summary_cluster
        self.balanced = balanced
        self.divide = divide

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

    def fit(self, X, y=None, rf=None):
        """Fit the internal random forest and compute cluster labels.

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
        df_datos_clusterizados, df_clusters_descripcion = self.regions.labels(
            df=df,
            df_reres=df_reres,
            n_clusters=self.n_clusters,
            include_summary_cluster=self.include_summary_cluster,
            balanced=self.balanced,
        )

        df_datos_clusterizados["cluster"] = df_datos_clusterizados["cluster"].fillna(
            value=-1
        )
        self.labels_ = df_datos_clusterizados["cluster"].to_numpy()
        self.df_clusters_descript_ = df_clusters_descripcion
        self.df_reres_ = df_reres

        df_datos_explain, frontiers = get_frontiers(
            df_datos_descript=df_clusters_descripcion, df=df, divide=self.divide
        )
        self.df_datos_explain_ = df_datos_explain
        self.frontiers_ = frontiers

        return self

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
            # Reorder/Subset columns to match training features
            X_df = X_df[self.feature_names_]
        else:
            X_df = pd.DataFrame(data=X, columns=self.feature_names_)

        df_clusterizado, _ = self.regions.labels(
            df=X_df,
            df_reres=self.df_reres_,
            n_clusters=self.n_clusters,
            include_summary_cluster=False,
            balanced=self.balanced,
        )
        df_clusterizado["cluster"] = df_clusterizado["cluster"].fillna(value=-1)
        return df_clusterizado["cluster"].to_numpy()


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
        balanced=False,
        divide=5,
    ):
        super().__init__(
            RandomForestClassifier,
            rf_params=rf_params,
            tree_params=tree_params,
            var_obj=var_obj,
            n_clusters=n_clusters,
            include_summary_cluster=include_summary_cluster,
            balanced=balanced,
            divide=divide,
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
        balanced=False,
        divide=5,
    ):
        super().__init__(
            RandomForestRegressor,
            rf_params=rf_params,
            tree_params=tree_params,
            var_obj=var_obj,
            n_clusters=n_clusters,
            include_summary_cluster=include_summary_cluster,
            balanced=balanced,
            divide=divide,
        )


# Backward compatibility alias
InsideForest = InsideForestClassifier

