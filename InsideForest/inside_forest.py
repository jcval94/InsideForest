import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .trees import Trees
from .regions import Regions


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
    """

    def __init__(self, rf_params=None, tree_params=None, var_obj="target"):
        self.rf_params = rf_params or {}
        self.tree_params = tree_params or {}
        self.var_obj = var_obj

        # Instantiate internal helpers
        self.rf = RandomForestClassifier(**self.rf_params)
        self.trees = Trees(**self.tree_params)
        self.regions = Regions()

        # Attributes populated after fitting
        self.labels_ = None
        self.feature_names_ = None
        self.df_clusters_descript_ = None
        self.df_reres_ = None

    def fit(self, X, y):
        """Fit the internal RandomForest and compute cluster labels.

        Parameters
        ----------
        X : array-like or pandas.DataFrame
            Feature matrix.
        y : array-like
            Target vector.
        """

        # Ensure DataFrame with meaningful column names
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X)

        # Replace spaces with underscores to keep compatibility with Trees
        X_df.columns = [str(c).replace(" ", "_") for c in X_df.columns]
        self.feature_names_ = list(X_df.columns)

        # Train RandomForest
        self.rf.fit(X_df, y)

        # Build DataFrame including target for region extraction
        df = X_df.copy()
        df[self.var_obj] = y

        # Extract rules and compute labels using existing utilities
        separacion_dim = self.trees.get_branches(df, self.var_obj, self.rf)
        df_reres = self.regions.prio_ranges(separacion_dim, df)
        df_datos_clusterizados, df_clusters_descripcion = self.regions.labels(
            df, df_reres, False
        )

        df_datos_clusterizados["cluster"] = df_datos_clusterizados["cluster"].fillna(-1)
        self.labels_ = df_datos_clusterizados["cluster"].to_numpy()
        self.df_clusters_descript_ = df_clusters_descripcion
        self.df_reres_ = df_reres

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
            X_df = pd.DataFrame(X, columns=self.feature_names_)

        df_clusterizado, _ = self.regions.labels(X_df, self.df_reres_, False)
        df_clusterizado["cluster"] = df_clusterizado["cluster"].fillna(-1)
        return df_clusterizado["cluster"].to_numpy()
