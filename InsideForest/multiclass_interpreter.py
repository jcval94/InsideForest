"""Class-aware supervised region clustering for InsideForest."""

from __future__ import annotations

from typing import Optional
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn import metrics
from sklearn.utils.validation import check_is_fitted, validate_data

from .multiclass_labels import get_multiclass_labels
from .multiclass_metrics import build_class_priors
from .multiclass_rules import extract_multiclass_leaf_rules


ASSIGNMENT_COLUMNS = [
    "cluster_id",
    "representative_region_id",
    "region_target_class",
    "membership_score",
    "target_probability",
    "class_distribution",
    "lift",
    "entropy",
    "class_margin",
    "matched_region_count",
    "matched_region_ids",
    "source",
]


class InsideForestClassRegionClusterer(TransformerMixin, ClusterMixin, BaseEstimator):
    """Extract class-enriched regions and assign supervised cluster IDs.

    The wrapped random forest is only a branch generator.  ``predict`` returns
    region cluster IDs and never delegates unmatched rows to the forest's class
    prediction.  Every selected physical leaf is associated with exactly one
    target class: the class that maximizes the configured rule objective.

    Parameters
    ----------
    forest : RandomForestClassifier, optional
        Optional forest to fit or reuse as the branch generator.
    rf_params : dict, optional
        Parameters used when constructing the branch-generating forest.
    leaf_percentile : float or None, default=95
        Per-target-class percentile used to retain high-scoring regions.
    low_leaf_fraction : float, default=0.05
        Fraction of regions below the percentile retained as a deterministic
        exploratory sample.
    min_support : int, default=1
        Minimum training support for a candidate leaf region.
    max_regions_per_class : int or None, default=None
        Optional maximum number of final regions associated with each class.
    rule_score : {"purity_lift_coverage"}, default="purity_lift_coverage"
        Objective used to associate and rank regions.
    random_state : int, default=42
        Random state for the forest and low-score sampling.
    n_jobs : int, default=1
        Parallelism used while extracting tree leaves.
    ambiguity_margin : float, default=0.15
        Default maximum class-probability margin for ambiguous regions.
    branch_aggregation : {"none"}, default="none"
        Region aggregation policy.  Aggregation is intentionally disabled
        until a pooling strategy passes the project's ablation criteria.
    """

    _FORMAT_VERSION = 1

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.target_tags.required = True
        return tags

    def __init__(
        self,
        forest: Optional[RandomForestClassifier] = None,
        rf_params=None,
        leaf_percentile=95,
        low_leaf_fraction=0.05,
        min_support=1,
        max_regions_per_class=None,
        rule_score="purity_lift_coverage",
        random_state=42,
        n_jobs=1,
        ambiguity_margin=0.15,
        branch_aggregation="none",
    ):
        self.forest = forest
        self.rf_params = rf_params
        self.leaf_percentile = leaf_percentile
        self.low_leaf_fraction = low_leaf_fraction
        self.min_support = min_support
        self.max_regions_per_class = max_regions_per_class
        self.rule_score = rule_score
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.ambiguity_margin = ambiguity_margin
        self.branch_aggregation = branch_aggregation

    def fit(self, X, y=None):
        """Fit the branch generator and learn class-aware region clusters."""

        if self.branch_aggregation != "none":
            raise ValueError("branch_aggregation currently supports only 'none'")
        if y is None:
            raise ValueError("y is required for supervised region clustering")

        validate_data(
            self,
            X,
            y,
            reset=True,
            dtype=None,
            accept_sparse=("csr", "csc"),
        )
        X_df = self._coerce_X_fit(X)
        y_fit = np.asarray(y)
        if y_fit.ndim != 1:
            y_fit = np.ravel(y_fit)
        y_array = y_fit.astype(object, copy=False)
        if len(y_array) != len(X_df):
            raise ValueError("X and y must contain the same number of rows")
        if pd.unique(y_array).size < 2:
            raise ValueError("At least two classes are required")

        self.forest_ = self._prepare_forest(X_df, y_fit)
        self.classes_ = np.asarray(self.forest_.classes_, dtype=object)
        self.class_priors_ = build_class_priors(y_array, self.classes_)

        # Preserve every one-vs-rest view internally so the winning target
        # class is chosen before any percentile filtering is applied.
        self._class_rule_views_ = extract_multiclass_leaf_rules(
            self.forest_,
            X_df,
            y_array,
            feature_names=list(self.feature_names_in_),
            leaf_percentile=None,
            low_leaf_fraction=0.0,
            min_support=self.min_support,
            max_regions_per_class=None,
            class_priors=self.class_priors_,
            rule_score=self.rule_score,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        self.raw_regions_ = self._collapse_physical_leaves(self._class_rule_views_)
        self.regions_ = self._select_regions(self.raw_regions_)
        self.region_metrics_ = get_multiclass_labels(
            self.regions_,
            X_df,
            y_array,
            class_labels=self.classes_,
        )
        if not self.region_metrics_.empty:
            self.region_metrics_ = self.region_metrics_.merge(
                self.regions_[["region_id", "cluster_id"]],
                on="region_id",
                how="left",
            )

        self.labels_ = self.predict(X_df)
        self.training_assignments_ = self.assign_regions(X_df)
        self._y_fit_ = y_array.copy()
        self.region_quality_summary_ = self._quality_summary(
            self.labels_,
            y_array,
            self.training_assignments_,
            region_metrics=self.region_metrics_,
        )
        return self

    def fit_predict(self, X, y):
        """Fit the clusterer and return training region cluster IDs."""

        return self.fit(X, y).labels_.copy()

    def transform(self, X) -> np.ndarray:
        """Return hard region-membership scores for every row and cluster."""

        self._require_fitted()
        X_df = self._coerce_X_predict(X)
        matches = self._region_match_matrix(X_df)
        if self.regions_.empty:
            return np.zeros((len(X_df), 0), dtype=float)
        scores = self.regions_["region_score"].to_numpy(dtype=float)
        return matches.astype(float) * scores[None, :]

    def predict(self, X) -> np.ndarray:
        """Assign the highest-scoring matching region cluster or ``-1``."""

        self._require_fitted()
        X_df = self._coerce_X_predict(X)
        matches = self._region_match_matrix(X_df)
        labels = np.full(len(X_df), -1, dtype=int)
        if self.regions_.empty:
            return labels

        memberships = matches.astype(float) * self.regions_[
            "region_score"
        ].to_numpy(dtype=float)[None, :]
        covered = matches.any(axis=1)
        if np.any(covered):
            # regions_ is ordered by score, entropy, support and cluster_id;
            # np.argmax therefore provides the documented deterministic tie.
            best_positions = np.argmax(memberships[covered], axis=1)
            labels[covered] = self.regions_.iloc[best_positions][
                "cluster_id"
            ].to_numpy(dtype=int)
        return labels

    def assign_regions(self, X) -> pd.DataFrame:
        """Return detailed supervised-region assignments without fallback."""

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
            # regions_ is already globally ordered by assignment priority.
            best = matched.iloc[0]
            rows.append(
                {
                    "cluster_id": int(best["cluster_id"]),
                    "representative_region_id": best["representative_region_id"],
                    "region_target_class": best["region_target_class"],
                    "membership_score": float(best["region_score"]),
                    "target_probability": float(best["target_probability"]),
                    "class_distribution": np.asarray(
                        best["class_distribution"], dtype=float
                    ).copy(),
                    "lift": float(best["lift"]),
                    "entropy": float(best["entropy"]),
                    "class_margin": float(best["class_margin"]),
                    "matched_region_count": int(len(matched)),
                    "matched_region_ids": matched["cluster_id"].astype(int).tolist(),
                    "source": "region",
                }
            )

        return pd.DataFrame(rows, columns=ASSIGNMENT_COLUMNS)

    def explain_regions(self, top_n=None) -> pd.DataFrame:
        """Return selected regions in assignment-priority order."""

        self._require_fitted()
        out = self.regions_.copy()
        if top_n is not None:
            out = out.head(int(top_n))
        return out.reset_index(drop=True)

    def regions_for_class(self, class_label, top_n=None) -> pd.DataFrame:
        """Return regions whose supervised objective targets one class."""

        self._require_fitted()
        self._validate_class_label(class_label)
        out = self.regions_[
            self.regions_["region_target_class"] == class_label
        ].copy()
        if top_n is not None:
            out = out.head(int(top_n))
        return out.reset_index(drop=True)

    def ambiguous_regions(self, top_n=20, margin=None) -> pd.DataFrame:
        """Return regions with the smallest top-two class margins."""

        self._require_fitted()
        threshold = self.ambiguity_margin if margin is None else float(margin)
        out = self.regions_[self.regions_["class_margin"] <= threshold].copy()
        out = out.sort_values(
            ["class_margin", "support", "region_score"],
            ascending=[True, False, False],
        )
        if top_n is not None:
            out = out.head(int(top_n))
        return out.reset_index(drop=True)

    def class_coverage_report(self) -> pd.DataFrame:
        """Summarize training coverage, precision and regions by target class."""

        self._require_fitted()
        y = np.asarray(self._y_fit_, dtype=object)
        assignments = self.training_assignments_
        rows = []
        for class_label in self.classes_:
            actual = y == class_label
            assigned_target = (
                assignments["region_target_class"].to_numpy(dtype=object)
                == class_label
            )
            correct_target = actual & assigned_target
            rows.append(
                {
                    "class_label": class_label,
                    "class_support": int(actual.sum()),
                    "region_count": int(
                        (self.regions_["region_target_class"] == class_label).sum()
                    ),
                    "class_coverage": float(correct_target.sum() / actual.sum())
                    if actual.any()
                    else np.nan,
                    "target_precision": float(
                        correct_target.sum() / assigned_target.sum()
                    )
                    if assigned_target.any()
                    else np.nan,
                    "class_prior": float(self.class_priors_.loc[class_label]),
                }
            )
        return pd.DataFrame(rows)

    def region_quality_report(self, X=None, y=None) -> dict[str, float]:
        """Return region and supervised-clustering quality metrics."""

        self._require_fitted()
        if X is None:
            if y is not None:
                raise ValueError("X is required when y is provided")
            if not hasattr(self, "region_quality_summary_"):
                raise RuntimeError(
                    "This migrated model has no training assignments; provide X and y "
                    "to recompute the quality report"
                )
            return dict(self.region_quality_summary_)

        labels = self.predict(X)
        assignments = self.assign_regions(X)
        if y is None:
            return self._quality_summary(labels, None, assignments)
        y_array = np.asarray(y)
        if y_array.ndim != 1:
            y_array = np.ravel(y_array)
        if len(y_array) != len(labels):
            raise ValueError("X and y must contain the same number of rows")
        y_object = y_array.astype(object)
        evaluated_regions = get_multiclass_labels(
            self.regions_,
            self._coerce_X_predict(X),
            y_object,
            class_labels=self.classes_,
        )
        return self._quality_summary(
            labels,
            y_object,
            assignments,
            region_metrics=evaluated_regions,
        )

    def score(self, X, y) -> float:
        """Return adjusted mutual information including unmatched cluster ``-1``."""

        labels = self.predict(X)
        y_array = np.asarray(y)
        if y_array.ndim != 1:
            y_array = np.ravel(y_array)
        if len(y_array) != len(labels):
            raise ValueError("X and y must contain the same number of rows")
        return float(metrics.adjusted_mutual_info_score(y_array, labels))

    @property
    def feature_importances_(self):
        """Feature importances of the branch-generating random forest."""

        self._require_fitted()
        return self.forest_.feature_importances_

    def plot_importances(self):
        """Plot importances of the branch-generating random forest."""

        self._require_fitted()
        import matplotlib.pyplot as plt

        order = np.argsort(self.feature_importances_)[::-1]
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(order)), self.feature_importances_[order])
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(np.asarray(self.feature_names_in_)[order], rotation=90)
        ax.set_ylabel("Feature importance")
        ax.set_title("Branch-generator feature importances")
        fig.tight_layout()
        return ax

    def save(self, filepath: str):
        """Persist the fitted clusterer using a versioned payload."""

        self._require_fitted()
        joblib.dump(
            {
                "format_version": self._FORMAT_VERSION,
                "estimator": self,
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath: str):
        """Load a clusterer persisted by :meth:`save`."""

        payload = joblib.load(filepath)
        if isinstance(payload, cls):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("estimator"), cls):
            return payload["estimator"]
        raise TypeError("File does not contain a compatible InsideForest clusterer")

    def __setstate__(self, state):
        """Migrate objects serialized under the former interpreter contract."""

        old_forest = state.get("rf_")
        old_rules = state.get("rules_")
        old_region_metrics = state.get("labels_")
        old_rule_score = state.get("score", "purity_lift_coverage")
        self.__dict__.update(state)

        # ``score`` used to be a string parameter and would shadow the AMI
        # method after unpickling.
        if isinstance(self.__dict__.get("score"), str):
            self.__dict__.pop("score", None)
        self.forest = self.__dict__.get("forest", None)
        self.leaf_percentile = self.__dict__.get(
            "leaf_percentile", self.__dict__.get("percentil", 95)
        )
        self.low_leaf_fraction = self.__dict__.get(
            "low_leaf_fraction", self.__dict__.get("low_frac", 0.05)
        )
        self.max_regions_per_class = self.__dict__.get(
            "max_regions_per_class", self.__dict__.get("max_rules_per_class")
        )
        self.rule_score = self.__dict__.get("rule_score", old_rule_score)
        self.ambiguity_margin = self.__dict__.get(
            "ambiguity_margin", self.__dict__.get("conflict_margin", 0.15)
        )
        self.branch_aggregation = self.__dict__.get("branch_aggregation", "none")
        self.random_state = self.__dict__.get("random_state", 42)
        self.n_jobs = self.__dict__.get("n_jobs", 1)
        self.min_support = self.__dict__.get("min_support", 1)
        self.rf_params = self.__dict__.get("rf_params")

        if not hasattr(self, "forest_") and old_forest is not None:
            self.forest_ = old_forest
        if hasattr(self, "feature_names_") and not hasattr(self, "feature_names_in_"):
            self.feature_names_in_ = np.asarray(self.feature_names_, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            self.feature_names_in_ = np.asarray(self.feature_names_in_, dtype=object)
        if hasattr(self, "feature_names_in_"):
            self.n_features_in_ = len(self.feature_names_in_)

        if not hasattr(self, "regions_") and isinstance(old_rules, pd.DataFrame):
            self._class_rule_views_ = old_rules.copy()
            self.raw_regions_ = self._collapse_physical_leaves(old_rules)
            self.regions_ = self._select_regions(self.raw_regions_)
            self.region_metrics_ = (
                old_region_metrics.copy()
                if isinstance(old_region_metrics, pd.DataFrame)
                else pd.DataFrame()
            )
            # Historical labels_ held a metric table, not assignments.  The
            # original training X was not serialized, so assignments cannot be
            # reconstructed safely during migration.
            self.labels_ = None
            self._legacy_loaded_without_training_labels_ = True

    # Temporary method aliases -------------------------------------------------
    def explain(self, class_label=None, top_n=None) -> pd.DataFrame:
        warnings.warn(
            "explain() is deprecated; use explain_regions() or "
            "regions_for_class(). It will be removed in InsideForest 0.5.0.",
            FutureWarning,
            stacklevel=2,
        )
        if class_label is None:
            return self.explain_regions(top_n=top_n)
        return self.regions_for_class(class_label, top_n=top_n)

    def prototype_regions(self, class_label=None, top_n=10) -> pd.DataFrame:
        warnings.warn(
            "prototype_regions() is deprecated; use regions_for_class(). "
            "It will be removed in InsideForest 0.5.0.",
            FutureWarning,
            stacklevel=2,
        )
        if class_label is not None:
            return self.regions_for_class(class_label, top_n=top_n)
        parts = [
            self.regions_for_class(label, top_n=top_n) for label in self.classes_
        ]
        return (
            pd.concat(parts, ignore_index=True)
            if parts
            else self.regions_.iloc[0:0].copy()
        )

    def confusion_regions(self, top_n=20) -> pd.DataFrame:
        warnings.warn(
            "confusion_regions() is deprecated; use ambiguous_regions(). "
            "It will be removed in InsideForest 0.5.0.",
            FutureWarning,
            stacklevel=2,
        )
        return self.ambiguous_regions(top_n=top_n)

    # Temporary attribute aliases ---------------------------------------------
    @property
    def rf_(self):
        warnings.warn(
            "rf_ is deprecated; use forest_. It will be removed in "
            "InsideForest 0.5.0.",
            FutureWarning,
            stacklevel=2,
        )
        return self.forest_

    @rf_.setter
    def rf_(self, value):
        self.forest_ = value

    @property
    def rules_(self):
        warnings.warn(
            "rules_ is deprecated; use regions_. It will be removed in "
            "InsideForest 0.5.0.",
            FutureWarning,
            stacklevel=2,
        )
        return self.regions_

    @rules_.setter
    def rules_(self, value):
        self.regions_ = value

    # Internal helpers ---------------------------------------------------------
    def _prepare_forest(self, X_df, y_fit) -> RandomForestClassifier:
        if self.forest is None:
            params = dict(self.rf_params or {})
            params.setdefault("random_state", self.random_state)
            forest = RandomForestClassifier(**params)
        else:
            if not isinstance(self.forest, RandomForestClassifier):
                raise TypeError(
                    "forest must be a sklearn.ensemble.RandomForestClassifier"
                )
            forest = self.forest
        try:
            check_is_fitted(forest)
        except NotFittedError:
            forest.fit(X_df, y_fit)
        return forest

    def _collapse_physical_leaves(self, class_views: pd.DataFrame) -> pd.DataFrame:
        if class_views.empty:
            return self._empty_regions()

        ranked = class_views.sort_values(
            [
                "leaf_region_id",
                "score",
                "target_probability",
                "support",
                "target_class_index",
            ],
            ascending=[True, False, False, False, True],
        )
        out = ranked.drop_duplicates("leaf_region_id", keep="first").copy()
        out["region_target_class"] = out["target_class"]
        out["region_score"] = out["score"].astype(float)
        out["class_margin"] = [
            _target_class_margin(distribution, int(target_index))
            for distribution, target_index in zip(
                out["class_distribution"], out["target_class_index"]
            )
        ]
        out["representative_region_id"] = out["leaf_region_id"]
        out["source_region_ids"] = out["leaf_region_id"].map(lambda value: (value,))
        out["source_region_count"] = 1
        out["branch_aggregation"] = "none"
        return out.reset_index(drop=True)

    def _select_regions(self, raw_regions: pd.DataFrame) -> pd.DataFrame:
        if raw_regions.empty:
            return raw_regions.copy()

        rng = np.random.RandomState(self.random_state)
        selected = []
        for _, group in raw_regions.groupby("region_target_class", sort=False):
            group = group.copy()
            if self.leaf_percentile is None:
                kept = group
            else:
                threshold = np.percentile(
                    group["region_score"].to_numpy(dtype=float),
                    self.leaf_percentile,
                )
                high = group[group["region_score"] >= threshold]
                low = group[group["region_score"] < threshold]
                sample_size = int(len(low) * max(float(self.low_leaf_fraction), 0.0))
                if sample_size:
                    sampled = rng.choice(
                        low.index.to_numpy(), size=sample_size, replace=False
                    )
                    low = low.loc[sampled]
                else:
                    low = low.iloc[0:0]
                kept = pd.concat([high, low], axis=0)

            kept = kept.sort_values(
                ["region_score", "entropy", "support", "leaf_region_id"],
                ascending=[False, True, False, True],
            )
            if self.max_regions_per_class is not None:
                kept = kept.head(int(self.max_regions_per_class))
            selected.append(kept)

        out = pd.concat(selected, ignore_index=True) if selected else raw_regions.iloc[0:0]
        out = out.sort_values(
            ["region_score", "entropy", "support", "leaf_region_id"],
            ascending=[False, True, False, True],
        ).reset_index(drop=True)
        out["cluster_id"] = np.arange(len(out), dtype=int)
        out["region_id"] = out["cluster_id"]
        out.attrs["classes_"] = tuple(self.classes_)
        return out

    def _region_match_matrix(self, X_df: pd.DataFrame) -> np.ndarray:
        n_samples = len(X_df)
        n_regions = len(self.regions_)
        matches = np.ones((n_samples, n_regions), dtype=bool)
        if n_regions == 0:
            return matches

        column_cache = {
            column: X_df[column].to_numpy(dtype=float, copy=False)
            for column in X_df.columns
        }
        for position, region in enumerate(self.regions_.itertuples(index=False)):
            for feature, lower in region.lower_bounds.items():
                matches[:, position] &= column_cache[feature] > float(lower)
            for feature, upper in region.upper_bounds.items():
                matches[:, position] &= column_cache[feature] <= float(upper)
        return matches

    def _quality_summary(
        self,
        labels,
        y,
        assignments,
        *,
        region_metrics=None,
    ) -> dict[str, float]:
        labels = np.asarray(labels)
        covered = labels != -1
        metric_frame = (
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
            "weighted_region_purity": _weighted_average(
                metric_frame.get("target_probability"),
                metric_frame.get("support"),
            ),
            "mean_lift": _safe_mean(metric_frame.get("lift")),
            "mean_entropy": _safe_mean(metric_frame.get("entropy")),
            "mean_rule_length": _safe_mean(
                self.regions_.get("conditions", pd.Series(dtype=object)).map(len)
                if "conditions" in self.regions_
                else None
            ),
            "assignment_target_agreement": np.nan,
            "cluster_purity": np.nan,
            "nmi": np.nan,
            "ami": np.nan,
            "ari": np.nan,
            "homogeneity": np.nan,
            "completeness": np.nan,
        }
        if y is None or len(y) == 0:
            return summary

        y_array = np.asarray(y, dtype=object)
        summary.update(_cluster_metrics(y_array, labels))
        assigned_targets = assignments["region_target_class"].to_numpy(dtype=object)
        if np.any(covered):
            summary["assignment_target_agreement"] = float(
                np.mean(assigned_targets[covered] == y_array[covered])
            )
        return summary

    def _coerce_X_fit(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
            self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        else:
            arr = X.toarray() if sparse.issparse(X) else np.asarray(X)
            if arr.ndim != 2:
                raise ValueError("X must be a 2-dimensional array or DataFrame")
            self.feature_names_in_ = np.asarray(
                [f"feature_{idx}" for idx in range(arr.shape[1])], dtype=object
            )
            X_df = pd.DataFrame(arr, columns=self.feature_names_in_)
        self.n_features_in_ = int(X_df.shape[1])
        return X_df

    def _coerce_X_predict(self, X) -> pd.DataFrame:
        validate_data(
            self,
            X,
            reset=False,
            dtype=None,
            accept_sparse=("csr", "csc"),
        )
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
            missing = [
                column for column in self.feature_names_in_ if column not in X_df.columns
            ]
            if missing:
                raise ValueError(
                    f"Input data is missing required feature columns: {missing}"
                )
            return X_df[list(self.feature_names_in_)]
        arr = X.toarray() if sparse.issparse(X) else np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {arr.shape[1] if arr.ndim == 2 else 1} features, but "
                f"{self.__class__.__name__} is expecting {self.n_features_in_} "
                "features as input"
            )
        return pd.DataFrame(arr, columns=self.feature_names_in_)

    def _validate_class_label(self, class_label) -> None:
        if class_label not in set(self.classes_):
            raise ValueError(f"Unknown class_label {class_label!r}")

    def _require_fitted(self) -> None:
        if not hasattr(self, "forest_") or not hasattr(self, "regions_"):
            raise NotFittedError(
                "InsideForestClassRegionClusterer is not fitted yet; call fit() first"
            )

    @staticmethod
    def _unmatched_assignment() -> dict:
        return {
            "cluster_id": -1,
            "representative_region_id": None,
            "region_target_class": None,
            "membership_score": 0.0,
            "target_probability": np.nan,
            "class_distribution": None,
            "lift": np.nan,
            "entropy": np.nan,
            "class_margin": np.nan,
            "matched_region_count": 0,
            "matched_region_ids": [],
            "source": "unmatched",
        }

    @staticmethod
    def _empty_regions() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "region_id",
                "cluster_id",
                "leaf_region_id",
                "representative_region_id",
                "source_region_ids",
                "source_region_count",
                "lower_bounds",
                "upper_bounds",
                "class_distribution",
                "target_class",
                "region_target_class",
                "target_probability",
                "support",
                "coverage",
                "lift",
                "entropy",
                "score",
                "region_score",
                "class_margin",
            ]
        )


class InsideForestMulticlassClassifier(InsideForestClassRegionClusterer):
    """Deprecated compatibility name for the class-region clusterer."""

    def __init__(
        self,
        rf_params=None,
        percentil=95,
        low_frac=0.05,
        min_support=1,
        max_rules_per_class=None,
        score="purity_lift_coverage",
        random_state=42,
        n_jobs=1,
        conflict_margin=0.15,
        branch_aggregation="none",
        forest=None,
        seed=None,
    ):
        warnings.warn(
            "InsideForestMulticlassClassifier is deprecated; use "
            "InsideForestClassRegionClusterer. The compatibility name will "
            "be removed in InsideForest 0.5.0.",
            FutureWarning,
            stacklevel=2,
        )
        effective_random_state = random_state if seed is None else seed
        super().__init__(
            forest=forest,
            rf_params=rf_params,
            leaf_percentile=percentil,
            low_leaf_fraction=low_frac,
            min_support=min_support,
            max_regions_per_class=max_rules_per_class,
            rule_score=score,
            random_state=effective_random_state,
            n_jobs=n_jobs,
            ambiguity_margin=conflict_margin,
            branch_aggregation=branch_aggregation,
        )
        self.percentil = percentil
        self.low_frac = low_frac
        self.max_rules_per_class = max_rules_per_class
        self.conflict_margin = conflict_margin
        self.seed = seed
        self._legacy_rule_score = score

    def fit(self, X, y, rf: Optional[RandomForestClassifier] = None):
        if rf is not None:
            self.forest = rf
        return super().fit(X, y)


def _target_class_margin(probabilities: np.ndarray, target_index: int) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    target_probability = float(probabilities[target_index])
    alternatives = np.delete(probabilities, target_index)
    second_probability = float(np.max(alternatives)) if alternatives.size else target_probability
    return target_probability - second_probability


def _cluster_metrics(y, labels) -> dict[str, float]:
    y_text = pd.Series(y).astype(str).to_numpy()
    label_text = pd.Series(labels).astype(str).to_numpy()
    contingency = metrics.cluster.contingency_matrix(y_text, label_text)
    total = contingency.sum()
    return {
        "cluster_purity": float(contingency.max(axis=0).sum() / total)
        if total
        else np.nan,
        "nmi": float(metrics.normalized_mutual_info_score(y_text, label_text)),
        "ami": float(metrics.adjusted_mutual_info_score(y_text, label_text)),
        "ari": float(metrics.adjusted_rand_score(y_text, label_text)),
        "homogeneity": float(metrics.homogeneity_score(y_text, label_text)),
        "completeness": float(metrics.completeness_score(y_text, label_text)),
    }


def _safe_mean(values) -> float:
    if values is None:
        return np.nan
    array = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    return float(np.mean(array)) if array.size else np.nan


def _weighted_average(values, weights) -> float:
    if values is None or weights is None:
        return np.nan
    value_array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy()
    weight_array = pd.to_numeric(pd.Series(weights), errors="coerce").to_numpy()
    valid = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array > 0)
    return (
        float(np.average(value_array[valid], weights=weight_array[valid]))
        if np.any(valid)
        else np.nan
    )
