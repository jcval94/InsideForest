"""Multiclass leaf rule extraction for scikit-learn forests."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import _tree

from .multiclass_metrics import build_class_priors, normalize_counts, score_multiclass_rules


def extract_multiclass_leaf_rules(
    model: RandomForestClassifier,
    X,
    y,
    feature_names: Sequence[str] | None = None,
    *,
    percentil: float | None = 95,
    low_frac: float = 0.05,
    min_support: int = 1,
    max_rules_per_class: int | None = None,
    class_priors: pd.Series | dict | Sequence[float] | None = None,
    score: str = "purity_lift_coverage",
    random_state: int = 42,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Extract one-vs-rest multiclass rule rows from a fitted forest.

    The returned DataFrame contains one row per ``(tree leaf, target class)``.
    Each row preserves the full class vector in ``class_distribution`` and
    ranks that leaf from the perspective of ``target_class``.
    """

    _validate_model(model)
    X_df = _as_dataframe(X, feature_names)
    classes = np.asarray(model.classes_, dtype=object)
    class_to_index = {label: idx for idx, label in enumerate(classes)}
    y_array = np.asarray(y)
    if y_array.ndim != 1:
        y_array = np.ravel(y_array)
    y_array = y_array.astype(object, copy=False)
    y_index = np.asarray([class_to_index[label] for label in y_array], dtype=int)
    X_values = X_df.to_numpy()
    if class_priors is None:
        priors = build_class_priors(y_array, classes)
    elif isinstance(class_priors, pd.Series):
        priors = class_priors.astype(float)
    elif isinstance(class_priors, dict):
        priors = pd.Series(class_priors, dtype=float)
    else:
        priors = pd.Series(list(class_priors), index=pd.Index(classes), dtype=float)

    def process_tree(tree_index: int, estimator) -> list[dict]:
        leaf_assignments = estimator.apply(X_values)
        leaf_to_indices = {}
        for row_index, leaf_id in enumerate(leaf_assignments):
            leaf_to_indices.setdefault(int(leaf_id), []).append(row_index)

        rows = []

        def recurse(node: int, conditions: list[str], lower: dict, upper: dict) -> None:
            if estimator.tree_.feature[node] != _tree.TREE_UNDEFINED:
                feature_idx = int(estimator.tree_.feature[node])
                feature = X_df.columns[feature_idx]
                threshold = float(estimator.tree_.threshold[node])

                left_upper = dict(upper)
                left_upper[feature] = min(left_upper.get(feature, np.inf), threshold)
                recurse(
                    int(estimator.tree_.children_left[node]),
                    conditions + [f"{feature} <= {threshold:.6f}"],
                    dict(lower),
                    left_upper,
                )

                right_lower = dict(lower)
                right_lower[feature] = max(right_lower.get(feature, -np.inf), threshold)
                recurse(
                    int(estimator.tree_.children_right[node]),
                    conditions + [f"{feature} > {threshold:.6f}"],
                    right_lower,
                    dict(upper),
                )
                return

            sample_indices = leaf_to_indices.get(int(node), [])
            support = int(len(sample_indices))
            if support < min_support:
                return

            counts = np.bincount(y_index[sample_indices], minlength=len(classes)).astype(float)
            distribution = normalize_counts(counts)
            dominant_index = int(np.argmax(distribution))
            dominant_probability = float(distribution[dominant_index])
            coverage = float(support / len(X_df)) if len(X_df) else 0.0

            base = {
                "tree_index": int(tree_index),
                "leaf_id": int(node),
                "leaf_region_id": f"tree_{tree_index}_leaf_{node}",
                "conditions": tuple(conditions),
                "description": " AND ".join(conditions),
                "lower_bounds": dict(lower),
                "upper_bounds": dict(upper),
                "class_counts": counts,
                "class_distribution": distribution,
                "support": support,
                "coverage": coverage,
                "dominant_class": classes[dominant_index],
                "dominant_class_index": dominant_index,
                "dominant_probability": dominant_probability,
            }

            for class_index, class_label in enumerate(classes):
                rows.append(
                    {
                        **base,
                        "target_class": class_label,
                        "target_class_index": int(class_index),
                        "target_probability": float(distribution[class_index]),
                    }
                )

        recurse(0, [], {}, {})
        return rows

    try:
        if n_jobs == 1:
            raw_rows = [
                row
                for tree_index, estimator in enumerate(model.estimators_)
                for row in process_tree(tree_index, estimator)
            ]
        else:
            nested = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(process_tree)(tree_index, estimator)
                for tree_index, estimator in enumerate(model.estimators_)
            )
            raw_rows = [row for rows in nested for row in rows]
    except Exception:
        raw_rows = [
            row
            for tree_index, estimator in enumerate(model.estimators_)
            for row in process_tree(tree_index, estimator)
        ]

    rule_df = pd.DataFrame(raw_rows)
    rule_df.attrs["classes_"] = tuple(classes)
    if rule_df.empty:
        return _empty_rule_frame(classes)

    scored = score_multiclass_rules(rule_df, priors, score=score)
    scored.attrs["classes_"] = tuple(classes)
    selected = _filter_by_class_score(
        scored,
        percentil=percentil,
        low_frac=low_frac,
        max_rules_per_class=max_rules_per_class,
        random_state=random_state,
    )
    selected = selected.sort_values(
        ["target_class_index", "score", "support", "tree_index", "leaf_id"],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)
    selected.insert(0, "region_id", np.arange(len(selected), dtype=int))
    selected.attrs["classes_"] = tuple(classes)
    return selected


def _filter_by_class_score(
    rule_df: pd.DataFrame,
    *,
    percentil: float | None,
    low_frac: float,
    max_rules_per_class: int | None,
    random_state: int,
) -> pd.DataFrame:
    if percentil is None:
        filtered = rule_df.copy()
    else:
        rng = np.random.RandomState(random_state)
        selected_parts = []
        for _, group in rule_df.groupby("target_class_index", sort=True):
            scores = group["score"].to_numpy(dtype=float)
            threshold = np.percentile(scores, percentil)
            high = group[group["score"] >= threshold]
            low = group[group["score"] < threshold]
            sample_size = int(len(low) * max(low_frac, 0.0))
            if sample_size > 0:
                sampled_index = rng.choice(low.index.to_numpy(), size=sample_size, replace=False)
                low = low.loc[sampled_index]
            else:
                low = low.iloc[0:0]
            high = high.copy()
            low = low.copy()
            high.attrs = {}
            low.attrs = {}
            selected_parts.append(pd.concat([high, low], axis=0))
        filtered = pd.concat(selected_parts, axis=0) if selected_parts else rule_df.iloc[0:0].copy()

    if max_rules_per_class is not None:
        limited_parts = []
        for _, group in filtered.groupby("target_class_index", sort=True):
            limited_parts.append(
                group.sort_values(["score", "support"], ascending=False).head(max_rules_per_class)
            )
        filtered = pd.concat(limited_parts, axis=0) if limited_parts else filtered

    return filtered.copy()


def _as_dataframe(X, feature_names: Sequence[str] | None = None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        out = X.copy()
        if feature_names is not None:
            out.columns = list(feature_names)
        return out

    arr = X.toarray() if sparse.issparse(X) else np.asarray(X)
    if arr.ndim != 2:
        raise ValueError("X must be a 2-dimensional array or DataFrame")
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=list(feature_names))


def _validate_model(model: RandomForestClassifier) -> None:
    if not isinstance(model, RandomForestClassifier):
        raise TypeError("model must be a fitted sklearn.ensemble.RandomForestClassifier")
    for attr in ("estimators_", "classes_"):
        if not hasattr(model, attr):
            raise ValueError("model must be fitted before extracting multiclass rules")


def _empty_rule_frame(classes: np.ndarray) -> pd.DataFrame:
    columns = [
        "region_id",
        "tree_index",
        "leaf_id",
        "leaf_region_id",
        "conditions",
        "description",
        "lower_bounds",
        "upper_bounds",
        "class_counts",
        "class_distribution",
        "support",
        "coverage",
        "dominant_class",
        "dominant_class_index",
        "dominant_probability",
        "target_class",
        "target_class_index",
        "target_probability",
        "prior_probability",
        "lift",
        "entropy",
        "js_divergence",
        "score",
    ]
    out = pd.DataFrame(columns=columns)
    out.attrs["classes_"] = tuple(classes)
    return out
