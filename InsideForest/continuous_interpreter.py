"""Supervised region clustering for continuous targets."""

from __future__ import annotations

from typing import Optional
import warnings

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.tree import _tree
from sklearn.utils.validation import check_is_fitted, validate_data


ASSIGNMENT_COLUMNS = [
    "cluster_id",
    "representative_region_id",
    "membership_score",
    "target_mean",
    "target_median",
    "target_std",
    "target_iqr",
    "target_min",
    "target_max",
    "mean_shift",
    "standardized_mean_shift",
    "dispersion_reduction",
    "matched_region_count",
    "matched_region_ids",
    "source",
]


class InsideForestContinuousRegionClusterer(
    TransformerMixin, ClusterMixin, BaseEstimator
):
    """Discover interpretable regions guided by a continuous target.

    The random forest is only a branch generator. ``predict`` returns region
    cluster IDs, never continuous estimates, and unmatched observations remain
    explicitly assigned to ``-1``.
    """

    _FORMAT_VERSION = 1
    _REGION_SCORE = "dispersion_separation_coverage"

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.target_tags.required = True
        return tags

    def __init__(
        self,
        forest: Optional[RandomForestRegressor] = None,
        rf_params=None,
        leaf_percentile=95,
        low_leaf_fraction=0.05,
        min_support=1,
        max_regions=None,
        region_score="dispersion_separation_coverage",
        random_state=42,
        n_jobs=1,
        branch_aggregation="none",
        max_cases=None,
        auto_feature_reduce=False,
        explicit_k_features=None,
    ):
        self.forest = forest
        self.rf_params = rf_params
        self.leaf_percentile = leaf_percentile
        self.low_leaf_fraction = low_leaf_fraction
        self.min_support = min_support
        self.max_regions = max_regions
        self.region_score = region_score
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.branch_aggregation = branch_aggregation
        self.max_cases = max_cases
        self.auto_feature_reduce = auto_feature_reduce
        self.explicit_k_features = explicit_k_features

    def fit(self, X, y=None):
        """Fit the branch generator and learn continuous-target regions."""

        self._validate_parameters()
        if y is None:
            raise ValueError("y is required for supervised region clustering")

        y_array = self._coerce_y(y)
        validate_data(
            self,
            X,
            y_array,
            reset=True,
            dtype=None,
            accept_sparse=("csr", "csc"),
        )
        X_full = self._coerce_X_fit(X)
        if len(X_full) != len(y_array):
            raise ValueError("X and y must contain the same number of rows")

        X_sample, y_sample = self._sample_training_rows(X_full, y_array)
        X_model = self._fit_feature_reduction(X_sample, y_sample)
        self.forest_ = self._prepare_forest(X_model, y_sample)
        self.global_target_mean_ = float(np.mean(y_sample))
        self.global_target_variance_ = float(np.var(y_sample))
        self.global_target_std_ = float(np.sqrt(self.global_target_variance_))

        self.raw_regions_ = self._extract_leaf_regions(X_model, y_sample)
        self.regions_ = self._select_regions(self.raw_regions_)
        self.region_metrics_ = self.regions_.copy()

        self.labels_ = self.predict(X_full)
        self.training_assignments_ = self.assign_regions(X_full)
        self._y_fit_ = y_array.copy()
        self.region_quality_summary_ = self._quality_summary(
            self.labels_, y_array, region_metrics=self.region_metrics_
        )
        return self

    def fit_predict(self, X, y):
        """Fit and return training region cluster IDs."""

        return self.fit(X, y).labels_.copy()

    def transform(self, X) -> np.ndarray:
        """Return the sample-by-region membership score matrix."""

        self._require_fitted()
        X_df = self._coerce_X_predict(X)
        matches = self._region_match_matrix(X_df)
        if self.regions_.empty:
            return np.zeros((len(X_df), 0), dtype=float)
        scores = self.regions_["region_score"].to_numpy(dtype=float)
        return matches.astype(float) * scores[None, :]

    def predict(self, X) -> np.ndarray:
        """Assign the highest-scoring matching region or ``-1``."""

        self._require_fitted()
        X_df = self._coerce_X_predict(X)
        matches = self._region_match_matrix(X_df)
        labels = np.full(len(X_df), -1, dtype=int)
        if self.regions_.empty:
            return labels
        covered = matches.any(axis=1)
        if np.any(covered):
            memberships = matches[covered].astype(float) * self.regions_[
                "region_score"
            ].to_numpy(dtype=float)[None, :]
            best_positions = np.argmax(memberships, axis=1)
            labels[covered] = self.regions_.iloc[best_positions][
                "cluster_id"
            ].to_numpy(dtype=int)
        return labels

    def assign_regions(self, X) -> pd.DataFrame:
        """Return continuous-target metadata for every region assignment."""

        self._require_fitted()
        X_df = self._coerce_X_predict(X)
        matches = self._region_match_matrix(X_df)
        rows = []
        for row_position in range(len(X_df)):
            matched_positions = np.flatnonzero(matches[row_position])
            if matched_positions.size == 0:
                rows.append(self._unmatched_assignment())
                continue
            matched = self.regions_.iloc[matched_positions]
            best = matched.iloc[0]
            rows.append(
                {
                    "cluster_id": int(best["cluster_id"]),
                    "representative_region_id": best[
                        "representative_region_id"
                    ],
                    "membership_score": float(best["region_score"]),
                    "target_mean": float(best["target_mean"]),
                    "target_median": float(best["target_median"]),
                    "target_std": float(best["target_std"]),
                    "target_iqr": float(best["target_iqr"]),
                    "target_min": float(best["target_min"]),
                    "target_max": float(best["target_max"]),
                    "mean_shift": float(best["mean_shift"]),
                    "standardized_mean_shift": float(
                        best["standardized_mean_shift"]
                    ),
                    "dispersion_reduction": float(
                        best["dispersion_reduction"]
                    ),
                    "matched_region_count": int(len(matched)),
                    "matched_region_ids": matched["cluster_id"]
                    .astype(int)
                    .tolist(),
                    "source": "region",
                }
            )
        return pd.DataFrame(rows, columns=ASSIGNMENT_COLUMNS)

    def explain_regions(self, top_n=None) -> pd.DataFrame:
        """Return final regions in deterministic assignment order."""

        self._require_fitted()
        out = self.regions_.copy()
        if top_n is not None:
            out = out.head(int(top_n))
        return out.reset_index(drop=True)

    def region_quality_report(self, X=None, y=None) -> dict[str, float]:
        """Return coverage, compression and continuous clustering quality."""

        self._require_fitted()
        if X is None:
            if y is not None:
                raise ValueError("X is required when y is provided")
            if not hasattr(self, "region_quality_summary_"):
                raise RuntimeError(
                    "This migrated model has no training report; provide X and y"
                )
            return dict(self.region_quality_summary_)

        labels = self.predict(X)
        if y is None:
            return self._quality_summary(labels, None)
        y_array = self._coerce_y(y, require_variation=False)
        if len(y_array) != len(labels):
            raise ValueError("X and y must contain the same number of rows")
        X_df = self._coerce_X_predict(X)
        evaluated = self._evaluate_region_statistics(X_df, y_array)
        return self._quality_summary(labels, y_array, region_metrics=evaluated)

    def score(self, X, y) -> float:
        """Return target variance explained by cluster IDs, including ``-1``."""

        labels = self.predict(X)
        y_array = self._coerce_y(y, require_variation=False)
        if len(y_array) != len(labels):
            raise ValueError("X and y must contain the same number of rows")
        return _eta_squared(y_array, labels)

    @property
    def feature_importances_(self):
        """Feature importances of the branch-generating forest."""

        self._require_fitted()
        return self.forest_.feature_importances_

    def plot_importances(self):
        """Plot feature importances of the branch-generating forest."""

        self._require_fitted()
        import matplotlib.pyplot as plt

        order = np.argsort(self.feature_importances_)[::-1]
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(order)), self.feature_importances_[order])
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(np.asarray(self.feature_names_out_)[order], rotation=90)
        ax.set_ylabel("Feature importance")
        ax.set_title("Branch-generator feature importances")
        fig.tight_layout()
        return ax

    def save(self, filepath: str):
        """Persist the fitted clusterer using a versioned payload."""

        self._require_fitted()
        joblib.dump(
            {"format_version": self._FORMAT_VERSION, "estimator": self}, filepath
        )

    @classmethod
    def load(cls, filepath: str):
        """Load a canonical or migrate a historical regressor payload."""

        payload = joblib.load(filepath)
        if isinstance(payload, cls):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("estimator"), cls):
            return payload["estimator"]
        if isinstance(payload, dict) and "rf" in payload:
            return cls._from_legacy_payload(payload)
        raise TypeError("File does not contain a compatible InsideForest clusterer")

    def _validate_parameters(self):
        if self.branch_aggregation != "none":
            raise ValueError("branch_aggregation currently supports only 'none'")
        if self.region_score != self._REGION_SCORE:
            raise ValueError(
                "region_score currently supports only "
                f"{self._REGION_SCORE!r}"
            )
        if int(self.min_support) < 1:
            raise ValueError("min_support must be at least 1")
        if self.max_regions is not None and int(self.max_regions) < 1:
            raise ValueError("max_regions must be positive or None")

    @staticmethod
    def _coerce_y(y, *, require_variation=True) -> np.ndarray:
        try:
            values = np.asarray(y, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("y must contain finite numeric values") from exc
        if values.ndim != 1:
            values = np.ravel(values)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError("y must contain finite numeric values")
        if require_variation and np.unique(values).size < 2:
            raise ValueError("y must contain at least two distinct numeric values")
        return values

    def _sample_training_rows(self, X_df, y_array):
        if self.max_cases is None or len(X_df) <= int(self.max_cases):
            self._sample_indices_ = np.arange(len(X_df))
            return X_df.reset_index(drop=True), y_array.copy()
        rng = np.random.RandomState(self.random_state)
        selected = np.sort(
            rng.choice(len(X_df), size=int(self.max_cases), replace=False)
        )
        self._sample_indices_ = selected
        return X_df.iloc[selected].reset_index(drop=True), y_array[selected]

    def _fit_feature_reduction(self, X_df, y_array):
        n_features = X_df.shape[1]
        k = self.explicit_k_features
        if k is None:
            k = min(n_features, 32)
        k = max(1, min(int(k), n_features))
        if not self.auto_feature_reduce or k >= n_features:
            self._feature_mask_ = np.ones(n_features, dtype=bool)
            self.feature_names_out_ = np.asarray(X_df.columns, dtype=object)
            return X_df.copy()

        selector = SelectKBest(self._mutual_info_score, k=k)
        selector.fit(X_df, y_array)
        self._feature_mask_ = selector.get_support()
        self.feature_names_out_ = np.asarray(X_df.columns, dtype=object)[
            self._feature_mask_
        ]
        return X_df.loc[:, list(self.feature_names_out_)].copy()

    @staticmethod
    def _mutual_info_score(X, y):
        return mutual_info_regression(X, y)

    def _prepare_forest(self, X_df, y_array):
        if self.forest is None:
            params = dict(self.rf_params or {})
            params.setdefault("random_state", self.random_state)
            params.setdefault("n_jobs", self.n_jobs)
            forest = RandomForestRegressor(**params)
        else:
            if not isinstance(self.forest, RandomForestRegressor):
                raise TypeError(
                    "forest must be a sklearn.ensemble.RandomForestRegressor"
                )
            forest = self.forest
        try:
            check_is_fitted(forest)
        except NotFittedError:
            forest.fit(X_df, y_array)
        if int(forest.n_features_in_) != X_df.shape[1]:
            raise ValueError(
                "The fitted forest feature count does not match the selected X features"
            )
        return forest

    def _extract_leaf_regions(self, X_df, y_array) -> pd.DataFrame:
        X_values = X_df.to_numpy(dtype=float)

        def process_tree(tree_index, estimator):
            leaf_assignments = estimator.apply(X_values)
            leaf_to_indices = {}
            for row_index, leaf_id in enumerate(leaf_assignments):
                leaf_to_indices.setdefault(int(leaf_id), []).append(row_index)
            rows = []

            def recurse(node, conditions, lower, upper):
                if estimator.tree_.feature[node] != _tree.TREE_UNDEFINED:
                    feature_index = int(estimator.tree_.feature[node])
                    feature = X_df.columns[feature_index]
                    threshold = float(estimator.tree_.threshold[node])
                    left_upper = dict(upper)
                    left_upper[feature] = min(
                        left_upper.get(feature, np.inf), threshold
                    )
                    recurse(
                        int(estimator.tree_.children_left[node]),
                        conditions + [f"{feature} <= {threshold:.6f}"],
                        dict(lower),
                        left_upper,
                    )
                    right_lower = dict(lower)
                    right_lower[feature] = max(
                        right_lower.get(feature, -np.inf), threshold
                    )
                    recurse(
                        int(estimator.tree_.children_right[node]),
                        conditions + [f"{feature} > {threshold:.6f}"],
                        right_lower,
                        dict(upper),
                    )
                    return

                indices = leaf_to_indices.get(int(node), [])
                if len(indices) < int(self.min_support):
                    return
                values = y_array[indices]
                stats = self._target_statistics(values)
                leaf_region_id = f"tree_{tree_index}_leaf_{node}"
                rows.append(
                    {
                        "tree_index": int(tree_index),
                        "leaf_id": int(node),
                        "leaf_region_id": leaf_region_id,
                        "representative_region_id": leaf_region_id,
                        "source_region_ids": (leaf_region_id,),
                        "source_region_count": 1,
                        "branch_aggregation": "none",
                        "conditions": tuple(conditions),
                        "description": " AND ".join(conditions),
                        "lower_bounds": dict(lower),
                        "upper_bounds": dict(upper),
                        "support": int(len(indices)),
                        "coverage": float(len(indices) / len(X_df)),
                        **stats,
                    }
                )

            recurse(0, [], {}, {})
            return rows

        try:
            if self.n_jobs == 1:
                nested = [
                    process_tree(index, estimator)
                    for index, estimator in enumerate(self.forest_.estimators_)
                ]
            else:
                nested = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                    delayed(process_tree)(index, estimator)
                    for index, estimator in enumerate(self.forest_.estimators_)
                )
        except Exception:
            nested = [
                process_tree(index, estimator)
                for index, estimator in enumerate(self.forest_.estimators_)
            ]
        rows = [row for tree_rows in nested for row in tree_rows]
        if not rows:
            return self._empty_regions(include_cluster_id=False)
        return pd.DataFrame(rows)

    def _target_statistics(self, values):
        mean = float(np.mean(values))
        variance = float(np.var(values))
        std = float(np.sqrt(variance))
        q25, median, q75 = np.percentile(values, [25, 50, 75])
        shift = mean - self.global_target_mean_
        if self.global_target_variance_ > 0:
            reduction = float(
                np.clip(1.0 - variance / self.global_target_variance_, 0.0, 1.0)
            )
            standardized = float(shift / self.global_target_std_)
            separation = float(
                abs(shift) / (self.global_target_std_ + abs(shift))
            )
        else:
            reduction = standardized = separation = 0.0
        coverage = float(len(values) / len(self._sample_indices_))
        score = float(coverage * reduction * (0.5 + 0.5 * separation))
        return {
            "target_mean": mean,
            "target_median": float(median),
            "target_variance": variance,
            "target_std": std,
            "target_q25": float(q25),
            "target_q75": float(q75),
            "target_iqr": float(q75 - q25),
            "target_min": float(np.min(values)),
            "target_max": float(np.max(values)),
            "mean_shift": float(shift),
            "standardized_mean_shift": standardized,
            "dispersion_reduction": reduction,
            "separation": separation,
            "region_score": score,
        }

    def _select_regions(self, raw_regions):
        if raw_regions.empty:
            return self._empty_regions(include_cluster_id=True)
        candidates = raw_regions[raw_regions["region_score"] > 0].copy()
        if candidates.empty:
            return self._empty_regions(include_cluster_id=True)
        if self.leaf_percentile is None:
            selected = candidates
        else:
            threshold = np.percentile(
                candidates["region_score"].to_numpy(dtype=float),
                float(self.leaf_percentile),
            )
            high = candidates[candidates["region_score"] >= threshold]
            low = candidates[candidates["region_score"] < threshold]
            sample_size = int(
                len(low) * max(float(self.low_leaf_fraction), 0.0)
            )
            if sample_size:
                rng = np.random.RandomState(self.random_state)
                sampled = rng.choice(
                    low.index.to_numpy(), size=sample_size, replace=False
                )
                low = low.loc[sampled]
            else:
                low = low.iloc[0:0]
            selected = pd.concat([high, low], axis=0)

        selected = selected.sort_values(
            ["region_score", "target_std", "support", "leaf_region_id"],
            ascending=[False, True, False, True],
        )
        if self.max_regions is not None:
            selected = selected.head(int(self.max_regions))
        selected = selected.reset_index(drop=True)
        selected.insert(0, "cluster_id", np.arange(len(selected), dtype=int))
        selected.insert(1, "region_id", selected["cluster_id"])
        return selected

    def _evaluate_region_statistics(self, X_df, y_array):
        matches = self._region_match_matrix(X_df)
        rows = []
        global_mean = float(np.mean(y_array))
        global_variance = float(np.var(y_array))
        global_std = float(np.sqrt(global_variance))
        for position, region in self.regions_.iterrows():
            values = y_array[matches[:, position]]
            row = region.to_dict()
            row["support"] = int(len(values))
            row["coverage"] = float(len(values) / len(y_array))
            if len(values):
                variance = float(np.var(values))
                mean = float(np.mean(values))
                row["target_mean"] = mean
                row["target_std"] = float(np.sqrt(variance))
                row["mean_shift"] = mean - global_mean
                row["standardized_mean_shift"] = (
                    (mean - global_mean) / global_std if global_std else 0.0
                )
                row["dispersion_reduction"] = (
                    float(np.clip(1.0 - variance / global_variance, 0.0, 1.0))
                    if global_variance
                    else 0.0
                )
            rows.append(row)
        return pd.DataFrame(rows)

    def _quality_summary(self, labels, y, *, region_metrics=None):
        labels = np.asarray(labels)
        covered = labels != -1
        metrics = (
            region_metrics
            if isinstance(region_metrics, pd.DataFrame)
            else self.regions_
        )
        summary = {
            "n_raw_regions": int(len(self.raw_regions_)),
            "n_regions": int(len(self.regions_)),
            "n_clusters": int(len(set(labels[covered].tolist()))),
            "compression_ratio": float(
                1.0 - len(self.regions_) / len(self.raw_regions_)
            )
            if len(self.raw_regions_)
            else 0.0,
            "coverage": float(np.mean(covered)) if labels.size else 0.0,
            "unmatched_rate": float(np.mean(~covered)) if labels.size else 1.0,
            "target_variance_explained": np.nan,
            "target_std_reduction": np.nan,
            "weighted_target_std": _weighted_average(
                metrics.get("target_std"), metrics.get("support")
            ),
            "mean_absolute_target_shift": _weighted_average(
                metrics.get("mean_shift", pd.Series(dtype=float)).abs(),
                metrics.get("support"),
            ),
            "mean_dispersion_reduction": _weighted_average(
                metrics.get("dispersion_reduction"), metrics.get("support")
            ),
            "mean_rule_length": _safe_mean(
                self.regions_.get("conditions", pd.Series(dtype=object)).map(len)
                if "conditions" in self.regions_
                else None
            ),
        }
        if y is None or len(y) == 0:
            return summary
        y_array = np.asarray(y, dtype=float)
        summary["target_variance_explained"] = _eta_squared(y_array, labels)
        global_std = float(np.std(y_array))
        within_std = _weighted_cluster_std(y_array, labels)
        summary["target_std_reduction"] = (
            float(1.0 - within_std / global_std) if global_std else 0.0
        )
        return summary

    def _region_match_matrix(self, X_df):
        matches = np.ones((len(X_df), len(self.regions_)), dtype=bool)
        if self.regions_.empty:
            return matches
        cache = {
            column: X_df[column].to_numpy(dtype=float, copy=False)
            for column in X_df.columns
        }
        for position, region in enumerate(self.regions_.itertuples(index=False)):
            for feature, lower in region.lower_bounds.items():
                matches[:, position] &= cache[feature] > float(lower)
            for feature, upper in region.upper_bounds.items():
                matches[:, position] &= cache[feature] <= float(upper)
        return matches

    def _coerce_X_fit(self, X):
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
            self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        else:
            array = X.toarray() if sparse.issparse(X) else np.asarray(X)
            if array.ndim != 2:
                raise ValueError("X must be a 2-dimensional array or DataFrame")
            self.feature_names_in_ = np.asarray(
                [f"feature_{index}" for index in range(array.shape[1])],
                dtype=object,
            )
            X_df = pd.DataFrame(array, columns=self.feature_names_in_)
        self.n_features_in_ = int(X_df.shape[1])
        return X_df

    def _coerce_X_predict(self, X):
        if isinstance(X, pd.DataFrame):
            missing = [
                column for column in self.feature_names_in_ if column not in X.columns
            ]
            if missing:
                raise ValueError(
                    f"Input data is missing required feature columns: {missing}"
                )
            X = X.loc[:, list(self.feature_names_in_)].copy()
        validate_data(
            self,
            X,
            reset=False,
            dtype=None,
            accept_sparse=("csr", "csc"),
        )
        if isinstance(X, pd.DataFrame):
            X_df = X
        else:
            array = X.toarray() if sparse.issparse(X) else np.asarray(X)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            if array.ndim != 2 or array.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {array.shape[1] if array.ndim == 2 else 1} features, "
                    f"but {self.__class__.__name__} expects {self.n_features_in_}"
                )
            X_df = pd.DataFrame(array, columns=self.feature_names_in_)
        return self._select_model_features(X_df)

    def _select_model_features(self, X_df):
        return X_df.loc[:, list(self.feature_names_out_)].copy()

    def _require_fitted(self):
        if not hasattr(self, "forest_") or not hasattr(self, "regions_"):
            raise NotFittedError(
                "InsideForestContinuousRegionClusterer is not fitted; call fit()"
            )

    @staticmethod
    def _unmatched_assignment():
        return {
            "cluster_id": -1,
            "representative_region_id": None,
            "membership_score": 0.0,
            "target_mean": np.nan,
            "target_median": np.nan,
            "target_std": np.nan,
            "target_iqr": np.nan,
            "target_min": np.nan,
            "target_max": np.nan,
            "mean_shift": np.nan,
            "standardized_mean_shift": np.nan,
            "dispersion_reduction": np.nan,
            "matched_region_count": 0,
            "matched_region_ids": [],
            "source": "unmatched",
        }

    @staticmethod
    def _empty_regions(*, include_cluster_id):
        columns = [
            "tree_index",
            "leaf_id",
            "leaf_region_id",
            "representative_region_id",
            "source_region_ids",
            "source_region_count",
            "branch_aggregation",
            "conditions",
            "description",
            "lower_bounds",
            "upper_bounds",
            "support",
            "coverage",
            "target_mean",
            "target_median",
            "target_variance",
            "target_std",
            "target_q25",
            "target_q75",
            "target_iqr",
            "target_min",
            "target_max",
            "mean_shift",
            "standardized_mean_shift",
            "dispersion_reduction",
            "separation",
            "region_score",
        ]
        if include_cluster_id:
            columns = ["cluster_id", "region_id", *columns]
        return pd.DataFrame(columns=columns)

    @classmethod
    def _from_legacy_payload(cls, payload):
        warnings.warn(
            "Loading a legacy InsideForestRegressor artifact; training "
            "assignments may need to be recalculated",
            FutureWarning,
            stacklevel=2,
        )
        model = cls(
            forest=payload["rf"],
            rf_params=payload.get("rf_params"),
            leaf_percentile=None,
            random_state=payload.get("seed", 42),
        )
        model.forest_ = payload["rf"]
        names = payload.get("feature_names_") or [
            f"feature_{index}" for index in range(model.forest_.n_features_in_)
        ]
        model.feature_names_in_ = np.asarray(names, dtype=object)
        model.feature_names_out_ = np.asarray(names, dtype=object)
        model.n_features_in_ = len(names)
        model._feature_mask_ = np.ones(len(names), dtype=bool)
        quality = payload.get("region_quality_")
        model.regions_ = _legacy_regions_to_continuous(quality)
        model.raw_regions_ = model.regions_.copy()
        model.region_metrics_ = model.regions_.copy()
        model.labels_ = payload.get("labels_")
        model.region_quality_summary_ = payload.get(
            "region_quality_summary_", {}
        )
        model._legacy_loaded_without_training_data_ = True
        return model


def _eta_squared(y, labels) -> float:
    y_array = np.asarray(y, dtype=float)
    labels_array = np.asarray(labels)
    if y_array.size == 0:
        return 0.0
    global_mean = float(np.mean(y_array))
    total = float(np.sum((y_array - global_mean) ** 2))
    if total <= 0:
        return 0.0
    between = 0.0
    for label in np.unique(labels_array):
        values = y_array[labels_array == label]
        between += len(values) * float((np.mean(values) - global_mean) ** 2)
    return float(np.clip(between / total, 0.0, 1.0))


def _weighted_cluster_std(y, labels):
    y_array = np.asarray(y, dtype=float)
    labels_array = np.asarray(labels)
    if y_array.size == 0:
        return 0.0
    return float(
        sum(
            len(y_array[labels_array == label])
            * float(np.std(y_array[labels_array == label]))
            for label in np.unique(labels_array)
        )
        / len(y_array)
    )


def _weighted_average(values, weights):
    if values is None or weights is None:
        return np.nan
    values_array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    weights_array = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values_array) & np.isfinite(weights_array) & (weights_array > 0)
    if not np.any(valid):
        return np.nan
    return float(np.average(values_array[valid], weights=weights_array[valid]))


def _safe_mean(values):
    if values is None:
        return np.nan
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else np.nan


def _legacy_regions_to_continuous(quality):
    if not isinstance(quality, pd.DataFrame) or quality.empty:
        return InsideForestContinuousRegionClusterer._empty_regions(
            include_cluster_id=True
        )
    rows = []
    candidate_ids = quality.reset_index(drop=True).get("region_id")
    preserve_ids = False
    if candidate_ids is not None:
        numeric_ids = pd.to_numeric(candidate_ids, errors="coerce")
        preserve_ids = bool(
            numeric_ids.notna().all()
            and numeric_ids.is_unique
            and (numeric_ids >= 0).all()
            and np.allclose(numeric_ids, numeric_ids.astype(int))
        )
    for position, source in quality.reset_index(drop=True).iterrows():
        target_mean = float(source.get("target_mean", np.nan))
        target_std = float(source.get("target_std", np.nan))
        leaf_id = f"legacy_region_{position}"
        weight = float(source.get("weight", 1.0))
        if not np.isfinite(weight) or weight <= 0:
            weight = 1.0
        cluster_id = int(source["region_id"]) if preserve_ids else int(position)
        rows.append(
            {
                "cluster_id": cluster_id,
                "region_id": cluster_id,
                "tree_index": -1,
                "leaf_id": int(position),
                "leaf_region_id": leaf_id,
                "representative_region_id": source.get("region_id", leaf_id),
                "source_region_ids": (source.get("region_id", leaf_id),),
                "source_region_count": 1,
                "branch_aggregation": "legacy",
                "conditions": tuple(),
                "description": source.get("description", ""),
                "lower_bounds": source.get("lower_bounds", {}),
                "upper_bounds": source.get("upper_bounds", {}),
                "support": int(source.get("support", 0)),
                "coverage": float(source.get("coverage", np.nan)),
                "target_mean": target_mean,
                "target_median": target_mean,
                "target_variance": target_std**2,
                "target_std": target_std,
                "target_q25": np.nan,
                "target_q75": np.nan,
                "target_iqr": np.nan,
                "target_min": float(source.get("target_min", np.nan)),
                "target_max": float(source.get("target_max", np.nan)),
                "mean_shift": np.nan,
                "standardized_mean_shift": np.nan,
                "dispersion_reduction": np.nan,
                "separation": np.nan,
                "region_score": weight,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["region_score", "target_std", "support", "leaf_region_id"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)
