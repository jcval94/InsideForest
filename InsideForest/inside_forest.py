from typing import Optional, Dict, Any, List, Tuple
import warnings

import numpy as np
import joblib
try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.utils.multiclass import type_of_target

from .trees import Trees
from .regions import Regions
from .descrip import get_frontiers
from .region_quality import (
    build_region_rule_table,
    cluster_label_quality,
    score_region_rules,
    summarize_region_quality,
)


_DEFAULT_NO_TREES_SEARCH = object()


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


class _BaseInsideForest(BaseEstimator):
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
    no_trees_search : int or None, optional, default 300
        Maximum number of trees analysed when extracting rules. When
        provided, this value is forwarded to :meth:`Trees.get_branches`.
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
        leaf_percentile : int, default 96
            Percentile used to retain the most important leaves when extracting
            rules from trees.
        low_leaf_fraction : float, default 0.03
            Fraction of leaves below ``leaf_percentile`` to sample when
            building the rule set.
        max_cases : int, default 750
            Maximum number of cases to analyze. If the input dataset contains
            more rows, a random subset of at most ``max_cases`` rows is used.
        balance_clusters : bool, default False
            When ``True`` and fitting a classification task with more than two
            classes, the underlying random forest is trained with
            ``class_weight='balanced'`` and the cluster consolidation ``method``
            defaults to ``"menu"`` to encourage a more even distribution of
            clusters across classes.
    seed : int, default 42
        Random seed controlling subsampling, the underlying random forest and
        the stochastic components of :meth:`Regions.labels`.
    """

    def __init__(
        self,
        rf_cls,
        rf_params=None,
        tree_params=None,
        no_trees_search: Optional[int] = _DEFAULT_NO_TREES_SEARCH,
        var_obj="target",
        n_clusters=None,
        include_summary_cluster=False,
        method="select_clusters",
        divide=5,
        get_detail=False,
        leaf_percentile=96,
        low_leaf_fraction=0.03,
        max_cases=750,
        balance_clusters=False,
        auto_fast=False,
        auto_feature_reduce=False,
        explicit_k_features: Optional[int] = None,
        fast_overrides: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ):
        self.rf_cls = rf_cls
        self._is_regressor = issubclass(rf_cls, RandomForestRegressor)
        self.rf_params = rf_params if rf_params is not None else {}
        self._user_rf_random_state = self.rf_params.get("random_state")
        self.seed = seed
        if self._user_rf_random_state is None:
            self.rf_params["random_state"] = seed
        self.tree_params = tree_params if tree_params is not None else {}
        if no_trees_search is _DEFAULT_NO_TREES_SEARCH:
            if "no_trees_search" in self.tree_params:
                self.no_trees_search = self.tree_params["no_trees_search"]
            else:
                self.no_trees_search = 300
        else:
            self.no_trees_search = no_trees_search
        self.var_obj = var_obj
        self.n_clusters = n_clusters
        self.include_summary_cluster = include_summary_cluster
        self.method = method
        self.divide = divide
        self.get_detail = get_detail
        self.leaf_percentile = leaf_percentile
        self.low_leaf_fraction = low_leaf_fraction
        self.max_cases = max_cases
        self.balance_clusters = balance_clusters

        # FAST knobs
        self.auto_fast = auto_fast
        self.auto_feature_reduce = auto_feature_reduce
        self.explicit_k_features = explicit_k_features
        self.fast_overrides = fast_overrides if fast_overrides is not None else {}

        # FAST bookkeeping
        self._feature_mask_: Optional[np.ndarray] = None
        self.feature_names_in_: Optional[List[str]] = None
        self.feature_names_out_: Optional[List[str]] = None
        self._size_bucket_: Optional[str] = None
        self._fast_params_used_: Optional[Dict[str, Any]] = None

        # Ensure tree parameters include the percentile settings
        self.tree_params.setdefault("percentil", leaf_percentile)
        self.tree_params.setdefault("low_frac", low_leaf_fraction)
        if self.no_trees_search is None:
            self.tree_params.pop("no_trees_search", None)
        else:
            self.tree_params["no_trees_search"] = self.no_trees_search

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
        self.region_rules_ = None
        self.region_quality_ = None
        self.region_quality_summary_ = None
        self._sample_indices_ = None
        self.forest_ = None
        self.regions_ = None
        self.region_metrics_ = None
        self.raw_regions_ = None
        self.n_features_in_ = None

    @staticmethod
    def _target_as_series(y, index):
        """Return a one-dimensional target aligned by position to ``index``."""

        arr = np.asarray(y)
        if arr.ndim > 1:
            if arr.shape[1] != 1:
                raise ValueError(
                    "InsideForest requires a one-dimensional target for rule extraction."
                )
            arr = arr[:, 0]
        return pd.Series(arr, index=index)

    def _prepare_prediction_frame(self, X):
        """Coerce prediction/score input and apply the fitted feature mask."""

        if self.feature_names_ is None:
            raise RuntimeError("InsideForest instance is not fitted yet")

        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
            X_df.columns = [str(c).replace(" ", "_") for c in X_df.columns]

            missing_cols = [col for col in self.feature_names_ if col not in X_df.columns]
            if missing_cols:
                if self._feature_mask_ is None or self.feature_names_in_ is None:
                    missing_str = ", ".join(missing_cols)
                    raise ValueError(
                        "Input data is missing required feature columns: "
                        f"{missing_str}. Ensure these columns are present in 'X' or refit the model with this feature set."
                    )

                original_missing = [
                    col for col in self.feature_names_in_ if col not in X_df.columns
                ]
                if original_missing:
                    missing_str = ", ".join(original_missing)
                    raise ValueError(
                        "Input data is missing required feature columns: "
                        f"{missing_str}. Ensure these columns are present in 'X' or refit the model with this feature set."
                    )

                X_df = X_df[self.feature_names_in_]
                X_df = X_df.loc[:, self._feature_mask_]
                X_df.columns = self.feature_names_
                return X_df

            return X_df[self.feature_names_]

        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        n_features = X_arr.shape[1]
        if n_features == len(self.feature_names_):
            return pd.DataFrame(data=X_arr, columns=self.feature_names_)

        if self._feature_mask_ is not None and n_features == len(self._feature_mask_):
            return pd.DataFrame(data=X_arr[:, self._feature_mask_], columns=self.feature_names_)

        if self._feature_mask_ is not None:
            expected = len(self._feature_mask_)
            raise ValueError(
                "Input data must contain either the original feature set "
                f"({expected} columns) or the reduced feature set "
                f"({len(self.feature_names_)} columns)."
            )

        raise ValueError(
            "Input data must contain all feature columns used during fitting: "
            f"{', '.join(self.feature_names_)}."
        )

    def _extract_region_candidates(self, df):
        """Extract raw rectangular regions from the fitted forest."""

        return self.trees.get_branches(
            df=df,
            var_obj=self.var_obj,
            regr=self.rf,
            no_trees_search=self.no_trees_search,
        )

    def _score_region_candidates(self, separacion_dim, df):
        """Prioritize raw regions before cluster consolidation."""

        return self.regions.prio_ranges(separacion_dim=separacion_dim, df=df)

    def _fit_region_labels(self, df, df_reres):
        """Consolidate prioritized regions and store compatibility outputs."""

        if self.get_detail:
            df_datos_clusterizados, df_clusters_description, labels = self.regions.labels(
                df=df,
                df_reres=df_reres,
                n_clusters=self.n_clusters,
                include_summary_cluster=self.include_summary_cluster,
                method=self.method,
                return_dfs=True,
                var_obj=self.var_obj,
                seed=self.seed,
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
                seed=self.seed,
            )
            labels = pd.Series(labels).fillna(value=-1).to_numpy()
            self.labels_ = labels
            self.df_reres_ = df_reres
            self.df_clusters_description_ = None
            self.df_datos_explain_ = None
            self.frontiers_ = None

        return self.labels_

    def _store_region_quality(self, df, y):
        """Build the region quality table and aggregate report."""

        rule_table = getattr(self.regions, "last_rule_table_", None)
        X_quality = df.drop(columns=[self.var_obj], errors="ignore")
        task = "regression" if self._is_regressor else "classification"

        self.region_rules_ = build_region_rule_table(rule_table)
        self.region_quality_ = score_region_rules(
            self.region_rules_,
            X_quality,
            y,
            task=task,
        )
        self.region_quality_summary_ = summarize_region_quality(
            self.region_quality_,
            self.labels_,
            y,
            task=task,
        )
        self.regions_ = self.region_rules_
        self.region_metrics_ = self.region_quality_
        return self.region_quality_summary_

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
            "max_cases": self.max_cases,
            "no_trees_search": self.no_trees_search,
            "balance_clusters": self.balance_clusters,
            "auto_fast": self.auto_fast,
            "auto_feature_reduce": self.auto_feature_reduce,
            "explicit_k_features": self.explicit_k_features,
            "fast_overrides": self.fast_overrides,
            "seed": self.seed,
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
                self._user_rf_random_state = self.rf_params.get("random_state")
                if self._user_rf_random_state is None:
                    self.rf_params["random_state"] = self.seed
                self.rf.set_params(**self.rf_params)
            elif key.startswith("rf__"):
                sub_key = key.split("__", 1)[1]
                self.rf_params[sub_key] = value
                if sub_key == "random_state":
                    self._user_rf_random_state = value
                self.rf.set_params(**{sub_key: value})
            elif key == "tree_params":
                self.tree_params = value
                self.tree_params.setdefault("percentil", self.leaf_percentile)
                self.tree_params.setdefault("low_frac", self.low_leaf_fraction)
                if "no_trees_search" in self.tree_params:
                    self.no_trees_search = self.tree_params["no_trees_search"]
                elif self.no_trees_search is not None:
                    self.tree_params["no_trees_search"] = self.no_trees_search
                else:
                    self.tree_params.pop("no_trees_search", None)
                self.trees = Trees(**self.tree_params)
            elif key.startswith("tree_params__"):
                sub_key = key.split("__", 1)[1]
                if sub_key == "no_trees_search" and value is None:
                    self.tree_params.pop(sub_key, None)
                else:
                    self.tree_params[sub_key] = value
                if sub_key == "no_trees_search":
                    self.no_trees_search = value
                self.trees = Trees(**self.tree_params)
            elif key == "no_trees_search":
                self.no_trees_search = value
                if value is None:
                    self.tree_params.pop("no_trees_search", None)
                else:
                    self.tree_params["no_trees_search"] = value
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
                "max_cases",
                "balance_clusters",
                "auto_fast",
                "auto_feature_reduce",
                "explicit_k_features",
                "fast_overrides",
                "seed",
            }:
                setattr(self, key, value)
                if key == "leaf_percentile":
                    self.tree_params["percentil"] = value
                    self.trees = Trees(**self.tree_params)
                elif key == "low_leaf_fraction":
                    self.tree_params["low_frac"] = value
                    self.trees = Trees(**self.tree_params)
                elif key == "seed":
                    if self._user_rf_random_state is None:
                        self.rf_params["random_state"] = value
                        self.rf.set_params(random_state=value)
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
            if self._is_regressor:
                sel = SelectKBest(mutual_info_regression, k=k).fit(X_arr, np.asarray(y))
                support = sel.get_support()
            else:
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

        # Limit number of cases analyzed using random sampling
        n_samples = len(X_df)
        if n_samples > self.max_cases:
            rng = np.random.default_rng(self.seed)
            selected = rng.choice(n_samples, size=self.max_cases, replace=False)
            X_df = X_df.iloc[selected].copy()
            if y is not None:
                if _HAS_PANDAS and isinstance(y, (pd.Series, pd.DataFrame)):
                    y = y.iloc[selected]
                else:
                    y = np.asarray(y)[selected]
            self._sample_indices_ = selected
        else:
            self._sample_indices_ = np.arange(n_samples)

        y = self._target_as_series(y, X_df.index)

        # Replace spaces with underscores to keep compatibility with Trees
        X_df.columns = [str(c).replace(" ", "_") for c in X_df.columns]
        self.n_features_in_ = int(X_df.shape[1])

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
                if self._is_regressor:
                    combined["method"] = getattr(self, "method", "select_clusters")
                else:
                    combined["method"] = "menu" if y is not None else getattr(self, "method", auto["method"])
            if hasattr(self, "get_detail"):
                combined["get_detail"] = getattr(self, "get_detail", auto["get_detail"])

            combined = _merge_dicts(combined, self.fast_overrides)

            self._fast_params_used_ = combined
            self._size_bucket_ = _size_bucket(n, d)

            self.rf_params = combined.get("rf_params", self.rf_params)
            if self._user_rf_random_state is None:
                self.rf_params["random_state"] = self.seed
            self.tree_params = combined.get("tree_params", self.tree_params)
            if "no_trees_search" in self.tree_params:
                self.no_trees_search = self.tree_params["no_trees_search"]
            elif self.no_trees_search is not None:
                self.tree_params["no_trees_search"] = self.no_trees_search
            else:
                self.tree_params.pop("no_trees_search", None)
            self.divide = combined.get("divide", self.divide)
            self.method = combined.get("method", self.method)
            self.get_detail = combined.get("get_detail", self.get_detail)

            self.rf = self.rf_cls(**self.rf_params)
            self.trees = Trees(**self.tree_params)

        # Allow passing a custom random forest estimator
        if rf is not None:
            self.rf = rf

        # Optional balancing of class distribution and cluster method
        if self.balance_clusters and y is not None and not self._is_regressor:
            try:
                ytype = type_of_target(y)
            except Exception:
                ytype = None
            if ytype in {"multiclass", "binary"}:
                if "class_weight" not in self.rf_params:
                    self.rf_params["class_weight"] = "balanced"
                    self.rf.set_params(class_weight="balanced")
                if self.method in (None, "select_clusters"):
                    self.method = "menu"

        if self._is_regressor and self.method in {
            "match_class_distribution",
            "chimera",
            "chimera_values_selector",
            "ChimeraValuesSelector",
            "menu",
            "MenuClusterSelector",
            "menu_cluster_selector",
        }:
            warnings.warn(
                f"method='{self.method}' treats the target as discrete labels. "
                "For continuous regression targets, the default 'select_clusters' "
                "usually keeps the region semantics clearer.",
                UserWarning,
                stacklevel=2,
            )

        # Train RandomForest only if it has not been fitted already
        try:
            check_is_fitted(self.rf)
        except NotFittedError:
            self.rf.fit(X=X_df, y=y)
        self.forest_ = self.rf

        # Build DataFrame including target for region extraction
        df = X_df.copy()
        df[self.var_obj] = y

        separacion_dim = self._extract_region_candidates(df)
        self.raw_regions_ = separacion_dim
        df_reres = self._score_region_candidates(separacion_dim, df)
        self._fit_region_labels(df, df_reres)
        self._store_region_quality(df, y)

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

        ``InsideForestRegressor`` keeps this same contract for compatibility:
        this method returns region labels, not continuous target estimates. For
        numeric predictions use the fitted ``rf`` estimator with the same
        feature frame used by the forest; :meth:`score` applies any automatic
        feature mask before delegating to that estimator.

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

        X_df = self._prepare_prediction_frame(X)

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
            seed=self.seed,
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

        X_df = self._prepare_prediction_frame(X)

        return self.rf.score(X_df, y)

    def region_quality_report(self) -> Dict[str, float]:
        """Return aggregate quality metrics for fitted regions and clusters."""

        if self.region_quality_summary_ is None:
            raise RuntimeError("InsideForest instance is not fitted yet")
        return dict(self.region_quality_summary_)

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
            "max_cases": self.max_cases,
            "no_trees_search": self.no_trees_search,
            "balance_clusters": self.balance_clusters,
            "auto_fast": self.auto_fast,
            "auto_feature_reduce": self.auto_feature_reduce,
            "explicit_k_features": self.explicit_k_features,
            "fast_overrides": self.fast_overrides,
            "seed": self.seed,
            "labels_": self.labels_,
            "feature_names_": self.feature_names_,
            "feature_names_in_": self.feature_names_in_,
            "feature_names_out_": self.feature_names_out_,
            "_feature_mask_": self._feature_mask_,
            "_size_bucket_": self._size_bucket_,
            "_fast_params_used_": self._fast_params_used_,
            "df_clusters_description_": self.df_clusters_description_,
            "df_reres_": self.df_reres_,
            "df_datos_explain_": self.df_datos_explain_,
            "frontiers_": self.frontiers_,
            "region_rules_": self.region_rules_,
            "region_quality_": self.region_quality_,
            "region_quality_summary_": self.region_quality_summary_,
            "raw_regions_": self.raw_regions_,
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
        init_kwargs = dict(
            rf_params=payload["rf_params"],
            tree_params=payload["tree_params"],
            var_obj=payload["var_obj"],
            n_clusters=payload["n_clusters"],
            include_summary_cluster=payload["include_summary_cluster"],
            method=payload.get("method", "select_clusters"),
            divide=payload["divide"],
            get_detail=payload.get("get_detail", False),
            leaf_percentile=payload.get("leaf_percentile", 96),
            low_leaf_fraction=payload.get("low_leaf_fraction", 0.03),
            max_cases=payload.get("max_cases", 750),
            balance_clusters=payload.get("balance_clusters", False),
            auto_fast=payload.get("auto_fast", False),
            auto_feature_reduce=payload.get("auto_feature_reduce", False),
            explicit_k_features=payload.get("explicit_k_features", None),
            fast_overrides=payload.get("fast_overrides", None),
            seed=payload.get("seed", 42),
        )
        stored_no_trees = payload.get("no_trees_search", _DEFAULT_NO_TREES_SEARCH)
        if stored_no_trees is not _DEFAULT_NO_TREES_SEARCH:
            init_kwargs["no_trees_search"] = stored_no_trees
        model = cls(**init_kwargs)
        model.rf = payload["rf"]
        model.labels_ = payload["labels_"]
        model.feature_names_ = payload["feature_names_"]
        model.feature_names_in_ = payload.get("feature_names_in_")
        model.feature_names_out_ = payload.get("feature_names_out_")
        model._feature_mask_ = payload.get("_feature_mask_")
        model._size_bucket_ = payload.get("_size_bucket_")
        model._fast_params_used_ = payload.get("_fast_params_used_")
        model.df_clusters_description_ = payload["df_clusters_description_"]
        model.df_reres_ = payload["df_reres_"]
        model.df_datos_explain_ = payload["df_datos_explain_"]
        model.frontiers_ = payload["frontiers_"]
        model.region_rules_ = payload.get("region_rules_")
        model.region_quality_ = payload.get("region_quality_")
        model.region_quality_summary_ = payload.get("region_quality_summary_")
        model.raw_regions_ = payload.get("raw_regions_")
        model.forest_ = model.rf
        model.regions_ = model.region_rules_
        model.region_metrics_ = model.region_quality_
        if model.feature_names_in_ is not None:
            model.n_features_in_ = len(model.feature_names_in_)
        elif model.feature_names_ is not None:
            model.n_features_in_ = len(model.feature_names_)
        if "menu_selector_" in payload:
            model.regions._menu_selector = payload["menu_selector_"]
        return model


class InsideForestRegionClusterer(TransformerMixin, ClusterMixin, _BaseInsideForest):
    """Supervised region clusterer backed by a random-forest branch generator."""

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        return tags

    def __init__(
        self,
        rf_params=None,
        tree_params=None,
        no_trees_search: Optional[int] = _DEFAULT_NO_TREES_SEARCH,
        var_obj="target",
        n_clusters=None,
        include_summary_cluster=False,
        method="select_clusters",
        divide=5,
        get_detail=False,
        leaf_percentile=96,
        low_leaf_fraction=0.03,
        max_cases=750,
        balance_clusters=False,
        auto_fast=False,
        auto_feature_reduce=False,
        explicit_k_features: Optional[int] = None,
        fast_overrides: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ):
        super().__init__(
            RandomForestClassifier,
            rf_params=rf_params,
            tree_params=tree_params,
            no_trees_search=no_trees_search,
            var_obj=var_obj,
            n_clusters=n_clusters,
            include_summary_cluster=include_summary_cluster,
            method=method,
            divide=divide,
            get_detail=get_detail,
            leaf_percentile=leaf_percentile,
            low_leaf_fraction=low_leaf_fraction,
            max_cases=max_cases,
            balance_clusters=balance_clusters,
            auto_fast=auto_fast,
            auto_feature_reduce=auto_feature_reduce,
            explicit_k_features=explicit_k_features,
            fast_overrides=fast_overrides,
            seed=seed,
        )

    def fit_predict(self, X, y=None, rf=None):
        """Fit the region clusterer and assign cluster IDs to ``X``."""

        self.fit(X, y=y, rf=rf)
        features = X.drop(columns=[self.var_obj], errors="ignore") if isinstance(X, pd.DataFrame) else X
        return self.predict(features)

    def transform(self, X):
        """Return hard rule-membership scores for every final region."""

        if self.region_rules_ is None:
            raise RuntimeError("InsideForestRegionClusterer is not fitted yet")
        X_df = self._prepare_prediction_frame(X)
        matches = self._region_match_matrix(X_df)
        if self.region_rules_.empty:
            return np.zeros((len(X_df), 0), dtype=float)
        weights = pd.to_numeric(
            self.region_rules_["weight"], errors="coerce"
        ).fillna(1.0).to_numpy(dtype=float)
        return matches.astype(float) * weights[None, :]

    def assign_regions(self, X):
        """Return detailed region-cluster assignments for each row."""

        if self.region_rules_ is None:
            raise RuntimeError("InsideForestRegionClusterer is not fitted yet")
        X_df = self._prepare_prediction_frame(X)
        labels = self.predict(X_df)
        matches = self._region_match_matrix(X_df)
        weights = (
            pd.to_numeric(self.region_rules_["weight"], errors="coerce")
            .fillna(1.0)
            .to_numpy(dtype=float)
        )
        region_ids = self.region_rules_["region_id"].to_numpy()
        quality_by_id = {}
        if self.region_quality_ is not None:
            quality_by_id = {
                row["region_id"]: row
                for row in self.region_quality_.to_dict("records")
            }

        rows = []
        for row_position, cluster_id in enumerate(labels):
            positions = np.flatnonzero(matches[row_position])
            matched_ids = region_ids[positions].tolist()
            if cluster_id == -1:
                rows.append(
                    {
                        "cluster_id": -1,
                        "representative_region_id": None,
                        "region_target_class": None,
                        "membership_score": 0.0,
                        "target_probability": np.nan,
                        "class_distribution": None,
                        "lift": np.nan,
                        "entropy": np.nan,
                        "class_margin": np.nan,
                        "matched_region_count": int(len(positions)),
                        "matched_region_ids": matched_ids,
                        "source": "unmatched",
                    }
                )
                continue

            chosen = quality_by_id.get(cluster_id, {})
            distribution = chosen.get("target_distribution")
            probabilities = (
                sorted(distribution.values(), reverse=True)
                if isinstance(distribution, dict)
                else []
            )
            margin = (
                float(probabilities[0] - probabilities[1])
                if len(probabilities) > 1
                else np.nan
            )
            selected_positions = np.flatnonzero(region_ids == cluster_id)
            membership = (
                float(weights[selected_positions[0]])
                if selected_positions.size
                else float(np.max(weights[positions])) if positions.size else 0.0
            )
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "representative_region_id": cluster_id,
                    "region_target_class": chosen.get("dominant_target"),
                    "membership_score": membership,
                    "target_probability": chosen.get("dominant_probability", np.nan),
                    "class_distribution": distribution,
                    "lift": chosen.get("lift", np.nan),
                    "entropy": chosen.get("entropy", np.nan),
                    "class_margin": margin,
                    "matched_region_count": int(len(positions)),
                    "matched_region_ids": matched_ids,
                    "source": "region",
                }
            )
        return pd.DataFrame(rows)

    def explain_regions(self, top_n=None):
        """Return flattened final regions with their supervised metrics."""

        if self.region_quality_ is None:
            raise RuntimeError("InsideForestRegionClusterer is not fitted yet")
        out = self.region_quality_.copy()
        out["cluster_id"] = out["region_id"]
        out["region_target_class"] = out["dominant_target"]
        out["region_score"] = out["weight"]
        out = out.sort_values(
            ["region_score", "support"], ascending=[False, False]
        )
        if top_n is not None:
            out = out.head(int(top_n))
        return out.reset_index(drop=True)

    def region_quality_report(self, X=None, y=None) -> Dict[str, float]:
        """Return stored quality or evaluate cluster assignments on ``X``."""

        if X is None:
            if y is not None:
                raise ValueError("X is required when y is provided")
            return super().region_quality_report()
        labels = self.predict(X)
        valid = labels != -1
        report = dict(super().region_quality_report())
        report["coverage"] = float(np.mean(valid)) if len(labels) else 0.0
        report["unmatched_rate"] = float(np.mean(~valid)) if len(labels) else 1.0
        report["n_clusters"] = int(len(set(labels[valid].tolist())))
        if y is not None:
            y_array = np.asarray(y)
            if y_array.ndim != 1:
                y_array = np.ravel(y_array)
            if len(y_array) != len(labels):
                raise ValueError("X and y must contain the same number of rows")
            report.update(cluster_label_quality(y_array, labels))
        return report

    def score(self, X, y):
        """Return AMI between target values and region IDs, including ``-1``."""

        labels = self.predict(X)
        y_array = np.asarray(y)
        if y_array.ndim != 1:
            y_array = np.ravel(y_array)
        if len(y_array) != len(labels):
            raise ValueError("X and y must contain the same number of rows")
        return float(adjusted_mutual_info_score(y_array, labels))

    def _region_match_matrix(self, X_df):
        matches = np.ones((len(X_df), len(self.region_rules_)), dtype=bool)
        if self.region_rules_.empty:
            return matches
        values = {
            column: X_df[column].to_numpy(dtype=float, copy=False)
            for column in X_df.columns
        }
        for position, rule in enumerate(self.region_rules_.to_dict("records")):
            for feature, lower in rule["lower_bounds"].items():
                matches[:, position] &= values[feature] >= float(lower)
            for feature, upper in rule["upper_bounds"].items():
                matches[:, position] &= values[feature] <= float(upper)
        return matches


class InsideForestClassifier(InsideForestRegionClusterer):
    """Deprecated compatibility name for :class:`InsideForestRegionClusterer`."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "InsideForestClassifier is deprecated; use InsideForestRegionClusterer",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)

    def score(self, X, y):
        """Return legacy random-forest accuracy for backward compatibility."""

        warnings.warn(
            "InsideForestClassifier.score() reports legacy forest accuracy; "
            "InsideForestRegionClusterer.score() reports AMI",
            FutureWarning,
            stacklevel=2,
        )
        return _BaseInsideForest.score(self, X, y)


class InsideForestRegressor(_BaseInsideForest):
    """Wrapper model that combines a ``RandomForestRegressor`` with the
    Trees/Regions utilities to provide interpretable region labels.

    ``predict`` returns region labels for compatibility with earlier
    InsideForest releases. Continuous target estimates are available from the
    fitted ``rf`` estimator; ``score`` reports the estimator's R2 score.
    """

    def __init__(
        self,
        rf_params=None,
        tree_params=None,
        no_trees_search: Optional[int] = _DEFAULT_NO_TREES_SEARCH,
        var_obj="target",
        n_clusters=None,
        include_summary_cluster=False,
        method="select_clusters",
        divide=5,
        get_detail=False,
        leaf_percentile=96,
        low_leaf_fraction=0.03,
        max_cases=750,
        balance_clusters=False,
        auto_fast=False,
        auto_feature_reduce=False,
        explicit_k_features: Optional[int] = None,
        fast_overrides: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ):
        super().__init__(
            RandomForestRegressor,
            rf_params=rf_params,
            tree_params=tree_params,
            no_trees_search=no_trees_search,
            var_obj=var_obj,
            n_clusters=n_clusters,
            include_summary_cluster=include_summary_cluster,
            method=method,
            divide=divide,
            get_detail=get_detail,
            leaf_percentile=leaf_percentile,
            low_leaf_fraction=low_leaf_fraction,
            max_cases=max_cases,
            balance_clusters=balance_clusters,
            auto_fast=auto_fast,
            auto_feature_reduce=auto_feature_reduce,
            explicit_k_features=explicit_k_features,
            fast_overrides=fast_overrides,
            seed=seed,
        )


# Backward compatibility alias
InsideForest = InsideForestClassifier

