"""Quality summaries for traditional InsideForest regions and clusters."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn import metrics


REGION_RULE_COLUMNS = [
    "region_id",
    "source_index",
    "features",
    "n_features",
    "n_conditions",
    "lower_bounds",
    "upper_bounds",
    "description",
    "weight",
    "effectiveness",
    "support_estimate",
    "source_count",
]


REGION_QUALITY_COLUMNS = REGION_RULE_COLUMNS + [
    "support",
    "coverage",
    "target_counts",
    "target_distribution",
    "dominant_target",
    "dominant_probability",
    "prior_probability",
    "lift",
    "entropy",
    "target_mean",
    "target_std",
    "target_min",
    "target_max",
]


CLUSTER_METRIC_KEYS = [
    "cluster_purity",
    "nmi",
    "ami",
    "ari",
    "homogeneity",
    "completeness",
]


def build_region_rule_table(df_rules: pd.DataFrame | None) -> pd.DataFrame:
    """Return a flat rule table from the internal MultiIndex rule frame."""

    if df_rules is None or df_rules.empty:
        return pd.DataFrame(columns=REGION_RULE_COLUMNS)

    rows = []
    for source_index, (_, rule) in enumerate(df_rules.iterrows()):
        lower = _bounds_from_rule(rule, "linf")
        upper = _bounds_from_rule(rule, "lsup")
        features = tuple(sorted(set(lower) | set(upper)))
        rows.append(
            {
                "region_id": _maybe_int(_scalar(rule.get("cluster", source_index))),
                "source_index": int(source_index),
                "features": features,
                "n_features": int(len(features)),
                "n_conditions": int(sum(pd.notna(v) for v in lower.values()) + sum(pd.notna(v) for v in upper.values())),
                "lower_bounds": lower,
                "upper_bounds": upper,
                "description": _describe_bounds(features, lower, upper),
                "weight": _float_or_nan(_metric(rule, "ponderador")),
                "effectiveness": _float_or_nan(_metric(rule, "ef_sample")),
                "support_estimate": _float_or_nan(_metric(rule, "n_sample")),
                "source_count": _float_or_nan(_metric(rule, "count")),
            }
        )

    return pd.DataFrame(rows, columns=REGION_RULE_COLUMNS)


def score_region_rules(
    region_rules: pd.DataFrame | None,
    X: pd.DataFrame,
    y: Iterable[Any],
    *,
    task: str = "classification",
) -> pd.DataFrame:
    """Score flat region rules against data and target values."""

    if region_rules is None or region_rules.empty:
        return pd.DataFrame(columns=REGION_QUALITY_COLUMNS)

    X_df = _coerce_frame(X)
    y_series = pd.Series(np.asarray(y), index=X_df.index)
    task = _normalize_task(task)
    priors = _class_priors(y_series) if task == "classification" else {}
    class_labels = list(priors.keys())
    column_cache = {col: X_df[col].to_numpy(dtype=float, copy=False) for col in X_df.columns}

    rows = []
    n_samples = len(X_df)
    for rule in region_rules.to_dict("records"):
        mask = _match_rule(column_cache, n_samples, rule["lower_bounds"], rule["upper_bounds"])
        y_match = y_series[mask]
        support = int(mask.sum())

        row = dict(rule)
        row["support"] = support
        row["coverage"] = float(support / n_samples) if n_samples else math.nan

        if task == "classification":
            row.update(_classification_region_metrics(y_match, class_labels, priors))
        else:
            row.update(_regression_region_metrics(y_match))

        rows.append(row)

    return pd.DataFrame(rows, columns=REGION_QUALITY_COLUMNS)


def summarize_region_quality(
    region_quality: pd.DataFrame | None,
    labels: Iterable[Any] | None,
    y: Iterable[Any] | None,
    *,
    task: str = "classification",
) -> dict[str, float]:
    """Summarize region and cluster quality for a fitted InsideForest model."""

    task = _normalize_task(task)
    summary = {
        "n_regions": 0,
        "n_clusters": 0,
        "coverage": math.nan,
        "unmatched_rate": math.nan,
        "weighted_region_purity": math.nan,
        "mean_entropy": math.nan,
        "mean_lift": math.nan,
        "mean_rule_length": math.nan,
        **{key: math.nan for key in CLUSTER_METRIC_KEYS},
    }

    if region_quality is not None and not region_quality.empty:
        summary["n_regions"] = int(len(region_quality))
        summary["weighted_region_purity"] = _weighted_mean(
            region_quality.get("dominant_probability"),
            region_quality.get("support"),
        )
        summary["mean_entropy"] = _nanmean(region_quality.get("entropy"))
        summary["mean_lift"] = _nanmean(region_quality.get("lift"))
        summary["mean_rule_length"] = _nanmean(region_quality.get("n_conditions"))

    if labels is None:
        return summary

    labels_array = np.asarray(labels)
    if labels_array.size == 0:
        summary["coverage"] = 0.0
        summary["unmatched_rate"] = 1.0
        return summary

    valid = labels_array != -1
    summary["coverage"] = float(np.mean(valid))
    summary["unmatched_rate"] = float(np.mean(~valid))
    summary["n_clusters"] = int(len(set(labels_array[valid].tolist())))

    if task == "classification" and y is not None and np.any(valid):
        summary.update(cluster_label_quality(np.asarray(y)[valid], labels_array[valid]))

    return summary


def cluster_label_quality(y_true: Iterable[Any], labels: Iterable[Any]) -> dict[str, float]:
    """Return clustering metrics for covered samples."""

    y_true_s = pd.Series(y_true).astype(str).to_numpy()
    labels_s = pd.Series(labels).astype(str).to_numpy()
    if len(y_true_s) == 0:
        return {key: math.nan for key in CLUSTER_METRIC_KEYS}

    contingency = metrics.cluster.contingency_matrix(y_true_s, labels_s)
    total = contingency.sum()
    purity = math.nan if total == 0 else float(contingency.max(axis=0).sum() / total)
    return {
        "cluster_purity": purity,
        "nmi": float(metrics.normalized_mutual_info_score(y_true_s, labels_s)),
        "ami": float(metrics.adjusted_mutual_info_score(y_true_s, labels_s)),
        "ari": float(metrics.adjusted_rand_score(y_true_s, labels_s)),
        "homogeneity": float(metrics.homogeneity_score(y_true_s, labels_s)),
        "completeness": float(metrics.completeness_score(y_true_s, labels_s)),
    }


def _classification_region_metrics(
    y_match: pd.Series,
    class_labels: list[Any],
    priors: dict[Any, float],
) -> dict[str, Any]:
    support = int(len(y_match))
    counts = {label: int((y_match == label).sum()) for label in class_labels}
    if support == 0:
        distribution = {label: 0.0 for label in class_labels}
        return {
            "target_counts": counts,
            "target_distribution": distribution,
            "dominant_target": None,
            "dominant_probability": math.nan,
            "prior_probability": math.nan,
            "lift": math.nan,
            "entropy": math.nan,
            "target_mean": math.nan,
            "target_std": math.nan,
            "target_min": math.nan,
            "target_max": math.nan,
        }

    distribution = {label: count / support for label, count in counts.items()}
    dominant = max(distribution, key=distribution.get) if distribution else None
    dominant_probability = float(distribution[dominant]) if dominant is not None else math.nan
    prior = float(priors.get(dominant, math.nan)) if dominant is not None else math.nan
    lift_value = dominant_probability / prior if prior and prior > 0 else math.nan
    probabilities = np.asarray([value for value in distribution.values() if value > 0], dtype=float)
    entropy = float(-(probabilities * np.log2(probabilities)).sum()) if probabilities.size else math.nan

    return {
        "target_counts": counts,
        "target_distribution": distribution,
        "dominant_target": dominant,
        "dominant_probability": dominant_probability,
        "prior_probability": prior,
        "lift": lift_value,
        "entropy": entropy,
        "target_mean": math.nan,
        "target_std": math.nan,
        "target_min": math.nan,
        "target_max": math.nan,
    }


def _regression_region_metrics(y_match: pd.Series) -> dict[str, Any]:
    if len(y_match) == 0:
        mean = std = min_value = max_value = math.nan
    else:
        values = y_match.to_numpy(dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values))
        min_value = float(np.min(values))
        max_value = float(np.max(values))

    return {
        "target_counts": {},
        "target_distribution": {},
        "dominant_target": None,
        "dominant_probability": math.nan,
        "prior_probability": math.nan,
        "lift": math.nan,
        "entropy": math.nan,
        "target_mean": mean,
        "target_std": std,
        "target_min": min_value,
        "target_max": max_value,
    }


def _match_rule(
    column_cache: dict[str, np.ndarray],
    n_samples: int,
    lower_bounds: dict[str, float],
    upper_bounds: dict[str, float],
) -> np.ndarray:
    features = set(lower_bounds) | set(upper_bounds)
    if not features:
        return np.zeros(n_samples, dtype=bool)

    mask = np.ones(n_samples, dtype=bool)
    for feature in features:
        if feature not in column_cache:
            raise KeyError(f"Column not found in X: {feature}")
        values = column_cache[feature]
        lower = lower_bounds.get(feature)
        upper = upper_bounds.get(feature)
        if lower is not None and pd.notna(lower):
            mask &= values >= float(lower)
        if upper is not None and pd.notna(upper):
            mask &= values <= float(upper)
    return mask


def _bounds_from_rule(rule: pd.Series, level: str) -> dict[str, float]:
    try:
        bounds = rule[level]
    except (KeyError, TypeError):
        return {}
    if not isinstance(bounds, pd.Series):
        return {}
    out = {}
    for feature, value in bounds.items():
        if feature is None or pd.isna(value):
            continue
        out[str(feature)] = float(value)
    return out


def _metric(rule: pd.Series, name: str) -> Any:
    try:
        return rule[("metrics", name)]
    except (KeyError, TypeError):
        return math.nan


def _describe_bounds(features: Iterable[str], lower: dict[str, float], upper: dict[str, float]) -> str:
    parts = []
    for feature in features:
        has_lower = feature in lower and pd.notna(lower[feature])
        has_upper = feature in upper and pd.notna(upper[feature])
        if has_lower and has_upper:
            parts.append(f"{lower[feature]} <= {feature} <= {upper[feature]}")
        elif has_lower:
            parts.append(f"{feature} >= {lower[feature]}")
        elif has_upper:
            parts.append(f"{feature} <= {upper[feature]}")
    return " AND ".join(parts)


def _class_priors(y: pd.Series) -> dict[Any, float]:
    counts = y.value_counts(sort=False, dropna=False)
    total = float(counts.sum())
    if total == 0:
        return {}
    return {label: float(count / total) for label, count in counts.items()}


def _coerce_frame(X: pd.DataFrame) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    return pd.DataFrame(np.asarray(X))


def _normalize_task(task: str) -> str:
    task = str(task).lower()
    if task in {"classification", "classifier", "class"}:
        return "classification"
    if task in {"regression", "regressor", "continuous"}:
        return "regression"
    raise ValueError("task must be 'classification' or 'regression'")


def _scalar(value: Any) -> Any:
    if isinstance(value, pd.Series):
        if len(value) == 0:
            return math.nan
        value = value.iloc[0]
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value)
        if arr.size == 0:
            return math.nan
        value = arr.flat[0]
    return value


def _maybe_int(value: Any) -> Any:
    value = _scalar(value)
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        return value
    if float_value.is_integer():
        return int(float_value)
    return float_value


def _float_or_nan(value: Any) -> float:
    value = _scalar(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _nanmean(values: Any) -> float:
    if values is None:
        return math.nan
    arr = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    return float(np.mean(arr)) if arr.size else math.nan


def _weighted_mean(values: Any, weights: Any) -> float:
    if values is None or weights is None:
        return math.nan
    value_arr = pd.Series(values).replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    weight_arr = pd.Series(weights).replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    valid = ~np.isnan(value_arr) & ~np.isnan(weight_arr) & (weight_arr > 0)
    if not np.any(valid):
        return math.nan
    return float(np.average(value_arr[valid], weights=weight_arr[valid]))

