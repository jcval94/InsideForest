from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import joblib
try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.utils.multiclass import type_of_target

from .trees import Trees
from .regions import Regions
from .descrip import get_frontiers


# ---------- FAST helpers ----------
def _size_bucket(n: int, d: int) -> str:
    prod = n * d
    if prod <= 50_000:
        return "small"
    elif prod <= 200_000:
        return "medium"
    elif prod <= 1_000_000:
        return "large"
    else:
        return "huge"


def _choose_k_features(n: int, d: int) -> int:
    bucket = _size_bucket(n, d)
    if bucket == "small":
        k = min(d, 64)
    elif bucket == "medium":
        k = min(d, 48)
    elif bucket == "large":
        k = min(d, 32)
    else:
        k = min(d, 24)
    return max(8, k)


def _choose_fast_params(n: int, d: int) -> Dict[str, Any]:
    bucket = _size_bucket(n, d)
    if bucket == "small":
        rf_params = dict(n_estimators=80, max_depth=12, min_samples_leaf=3, n_jobs=-1, random_state=42)
        tree_params = dict(percentil=98, low_frac=0.02)
        divide = 3
        method = "menu"
    elif bucket == "medium":
        rf_params = dict(n_estimators=60, max_depth=10, min_samples_leaf=5, n_jobs=-1, random_state=42)
        tree_params = dict(percentil=99, low_frac=0.01)
        divide = 3
        method = "menu"
    elif bucket == "large":
        rf_params = dict(n_estimators=40, max_depth=9, min_samples_leaf=8, n_jobs=-1, random_state=42)
        tree_params = dict(percentil=99.5, low_frac=0.0075)
        divide = 3
        method = "menu"
    else:
        rf_params = dict(n_estimators=30, max_depth=8, min_samples_leaf=10, n_jobs=-1, random_state=42)
        tree_params = dict(percentil=99.7, low_frac=0.005)
        divide = 3
        method = "menu"

    return dict(
        rf_params=rf_params,
        tree_params=tree_params,
        divide=divide,
        method=method,
        get_detail=False,
    )


def _merge_dicts(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not override:
        return base
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


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
        (default), ``"balance_lists_n_clusters"``, ``"max_prob_clusters``,
        ``"match_class_distribution"``, ``"chimera`` and ``"menu"`` which
        leverages :class:`MenuClusterSelector`. When ``None`` or
        ``"select_clusters"`` the raw output of :func:`select_clusters` is
        used.
        divide : int, default 5
            Value forwarded to :func:`get_frontiers` when computing cluster
            frontiers.
        get_detail : bool, default False
            When ``True`` :meth:`fit` computes and stores additional cluster
            details and frontiers.
        leaf_percentile : int, default 95
            Percentile used to retain the most important leaves when extracting
            rules from trees.
        low_leaf_fraction : float, default 0.05
            Fraction of leaves below ``leaf_percentile`` to sample when
            building the rule set.
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
        leaf_percentile=95,
        low_leaf_fraction=0.05,
        auto_fast=False,
        auto_feature_reduce=False,
        explicit_k_features: Optional[int] = None,
        fast_overrides: Optional[Dict[str, Any]] = None,
    ):
        self.rf_cls = rf_cls
        self.rf_params = rf_params or {}
        self.tree_params = tree_params or {}
        self.var_obj = var_obj
        self.n_clusters = n_clusters
        self.include_summary_cluster = include_summary_cluster
        self.method = method
        self.divide = divide
        self.get_detail = get_detail
        self.leaf_percentile = leaf_percentile
        self.low_leaf_fraction = low_leaf_fraction

        # FAST knobs
        self.auto_fast = auto_fast
        self.auto_feature_reduce = auto_feature_reduce
        self.explicit_k_features = explicit_k_features
        self.fast_overrides = fast_overrides or {}

        # FAST bookkeeping
        self._feature_mask_: Optional[np.ndarray] = None
        self.feature_names_in_: Optional[List[str]] = None
        self.feature_names_out_: Optional[List[str]] = None
        self._size_bucket_: Optional[str] = None
        self._fast_params_used_: Optional[Dict[str, Any]] = None

        # Ensure tree parameters include the percentile settings
        self.tree_params.setdefault("percentil", leaf_percentile)
        self.tree_params.setdefault("low_frac", low_leaf_fraction)

        self.rf = rf_cls(**self.rf_params)
        self.trees = Trees(**self.tree_params)
        self.regions = Regions()

        # Attributes populated after fitting
        self.labels_ = None
        self.feature_names_ = None
        self.df_clusters_description_ = None
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
            "leaf_percentile": self.leaf_percentile,
            "low_leaf_fraction": self.low_leaf_fraction,
            "auto_fast": self.auto_fast,
            "auto_feature_reduce": self.auto_feature_reduce,
            "explicit_k_features": self.explicit_k_features,
            "fast_overrides": self.fast_overrides,
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
                self.tree_params.setdefault("percentil", self.leaf_percentile)
                self.tree_params.setdefault("low_frac", self.low_leaf_fraction)
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
                "leaf_percentile",
                "low_leaf_fraction",
                "auto_fast",
                "auto_feature_reduce",
                "explicit_k_features",
                "fast_overrides",
            }:
                setattr(self, key, value)
                if key == "leaf_percentile":
                    self.tree_params["percentil"] = value
                    self.trees = Trees(**self.tree_params)
                elif key == "low_leaf_fraction":
                    self.tree_params["low_frac"] = value
                    self.trees = Trees(**self.tree_params)
            else:
                raise ValueError(f"Invalid parameter '{key}'")

        return self

    def _maybe_reduce_features(self, X, y=None):
        """Optionally reduce features; preserve original column names if DataFrame."""
        if not self.auto_feature_reduce:
            if _HAS_PANDAS and isinstance(X, pd.DataFrame):
                self.feature_names_in_ = list(X.columns)
                self.feature_names_out_ = list(X.columns)
            else:
                self.feature_names_in_ = None
                self.feature_names_out_ = None
            self._feature_mask_ = None
            return X

        n, d = X.shape
        k = (
            self.explicit_k_features
            if self.explicit_k_features is not None
            else _choose_k_features(n, d)
        )
        k = min(k, d)

        is_df = _HAS_PANDAS and isinstance(X, pd.DataFrame)
        self.feature_names_in_ = list(X.columns) if is_df else None
        X_arr = X.values if is_df else np.asarray(X)

        support = None
        if y is not None:
            try:
                ytype = type_of_target(y)
            except Exception:
                ytype = None
            if ytype in {"binary", "multiclass"}:
                sel = SelectKBest(mutual_info_classif, k=k).fit(X_arr, y)
                support = sel.get_support()
            elif ytype in {"continuous", "continuous-multioutput"}:
                sel = SelectKBest(mutual_info_regression, k=k).fit(X_arr, y)
                support = sel.get_support()
        if support is None:
            variances = X_arr.var(axis=0)
            idx_sorted = np.argsort(-variances)[:k]
            support = np.zeros(X_arr.shape[1], dtype=bool)
            support[idx_sorted] = True

        self._feature_mask_ = support

        if is_df:
            cols = np.array(self.feature_names_in_)
            keep_cols = cols[support].tolist()
            self.feature_names_out_ = keep_cols
            return X[keep_cols]
        else:
            self.feature_names_out_ = None
            return X_arr[:, support]

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
                if self.var_obj in X_df.columns:
                    X_df = X_df.drop(columns=[self.var_obj])
            else:
                X_df = pd.DataFrame(data=X)

        # Replace spaces with underscores to keep compatibility with Trees
        X_df.columns = [str(c).replace(" ", "_") for c in X_df.columns]

        # 0) Feature reduction (optional)
        Xr = self._maybe_reduce_features(X_df, y)
        if _HAS_PANDAS and isinstance(Xr, pd.DataFrame):
            X_df = Xr
        else:
            X_df = pd.DataFrame(Xr)
        self.feature_names_ = list(X_df.columns)

        # 1) Fast preset (optional)
        if self.auto_fast:
            n, d = X_df.shape
            auto = _choose_fast_params(n, d)
            combined = dict(auto)

            if isinstance(self.rf_params, dict):
                combined["rf_params"] = {**auto["rf_params"], **self.rf_params}
            if isinstance(self.tree_params, dict):
                combined["tree_params"] = {**auto["tree_params"], **self.tree_params}
            if hasattr(self, "divide"):
                combined["divide"] = getattr(self, "divide", auto["divide"])
            if hasattr(self, "method"):
                combined["method"] = "menu" if y is not None else getattr(self, "method", auto["method"])
            if hasattr(self, "get_detail"):
                combined["get_detail"] = getattr(self, "get_detail", auto["get_detail"])

            combined = _merge_dicts(combined, self.fast_overrides)

            self._fast_params_used_ = combined
            self._size_bucket_ = _size_bucket(n, d)

            self.rf_params = combined.get("rf_params", self.rf_params)
            self.tree_params = combined.get("tree_params", self.tree_params)
            self.divide = combined.get("divide", self.divide)
            self.method = combined.get("method", self.method)
            self.get_detail = combined.get("get_detail", self.get_detail)

            self.rf = self.rf_cls(**self.rf_params)
            self.trees = Trees(**self.tree_params)

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
            df_datos_clusterizados, df_clusters_description, labels = self.regions.labels(
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
            self.df_clusters_description_ = df_clusters_description
            self.df_reres_ = df_reres

            df_datos_explain, frontiers = get_frontiers(
                df_descriptive=df_clusters_description, df=df, divide=self.divide
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
            self.df_clusters_description_ = None
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

        # Some label consolidation strategies require an auxiliary target
        # column even at prediction time. When the method depends on class
        # information and the incoming data lacks it (as is typical during
        # inference), we use the random forest's own predictions as a
        # surrogate target so that downstream selectors can operate.
        if self.method == "match_class_distribution" and self.var_obj not in X_df.columns:
            try:
                y_pred = self.rf.predict(X_df)
            except NotFittedError:
                raise RuntimeError("Random forest not fitted; call fit() first")
            X_df = X_df.copy()
            X_df[self.var_obj] = y_pred

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
            "leaf_percentile": self.leaf_percentile,
            "low_leaf_fraction": self.low_leaf_fraction,
            "labels_": self.labels_,
            "feature_names_": self.feature_names_,
            "df_clusters_description_": self.df_clusters_description_,
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
            leaf_percentile=payload.get("leaf_percentile", 95),
            low_leaf_fraction=payload.get("low_leaf_fraction", 0.05),
        )
        model.rf = payload["rf"]
        model.labels_ = payload["labels_"]
        model.feature_names_ = payload["feature_names_"]
        model.df_clusters_description_ = payload["df_clusters_description_"]
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
        leaf_percentile=95,
        low_leaf_fraction=0.05,
        auto_fast=False,
        auto_feature_reduce=False,
        explicit_k_features: Optional[int] = None,
        fast_overrides: Optional[Dict[str, Any]] = None,
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
            leaf_percentile=leaf_percentile,
            low_leaf_fraction=low_leaf_fraction,
            auto_fast=auto_fast,
            auto_feature_reduce=auto_feature_reduce,
            explicit_k_features=explicit_k_features,
            fast_overrides=fast_overrides,
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
        leaf_percentile=95,
        low_leaf_fraction=0.05,
        auto_fast=False,
        auto_feature_reduce=False,
        explicit_k_features: Optional[int] = None,
        fast_overrides: Optional[Dict[str, Any]] = None,
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
            leaf_percentile=leaf_percentile,
            low_leaf_fraction=low_leaf_fraction,
            auto_fast=auto_fast,
            auto_feature_reduce=auto_feature_reduce,
            explicit_k_features=explicit_k_features,
            fast_overrides=fast_overrides,
        )


# Backward compatibility alias
InsideForest = InsideForestClassifier

