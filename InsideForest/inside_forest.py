import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .trees import Trees
from .regions import Regions
from .descrip import get_frontiers


class InsideForest:
    """Wrapper model that combines a RandomForest and the Trees/Regions
    utilities to provide cluster labels for the training data.

    Parameters
    ----------
    rf_params : dict, optional
        Parameters passed directly to ``RandomForestClassifier``.
    tree_params : dict, optional
        Parameters passed to :class:`Trees`.
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

        # Instantiate internal helpers
        self.rf = RandomForestClassifier(**self.rf_params)
        self.trees = Trees(**self.tree_params)
        self.regions = Regions()

        # Attributes populated after fitting
        self.labels_ = None
        self.feature_names_ = None
        self.df_clusters_descript_ = None
        self.df_reres_ = None
        self.df_datos_explain_ = None
        self.frontiers_ = None

    def fit(self, X, y=None):
        """Fit the internal RandomForest and compute cluster labels.

        Parameters
        ----------
        X : array-like or pandas.DataFrame
            Feature matrix or DataFrame containing the target column.
        y : array-like, optional
            Target vector. If not provided and ``X`` is a DataFrame containing
            ``var_obj``, that column will be used as the target.

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

        # Train RandomForest
        self.rf.fit(X=X_df, y=y)

        # Build DataFrame including target for region extraction
        df = X_df.copy()
        df[self.var_obj] = y

        # Extract rules and compute labels using existing utilities
        separacion_dim = self.trees.get_branches(
            df=df, var_obj=self.var_obj, regr=self.rf
        )
        df_reres = self.regions.prio_ranges(
            separacion_dim=separacion_dim, df=df
        )
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
