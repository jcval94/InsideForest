"""Label reports for multiclass InsideForest regions."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .multiclass_metrics import build_class_priors, entropy, lift, normalize_counts


def get_multiclass_labels(
    regions: pd.DataFrame,
    X,
    y,
    *,
    class_labels: Sequence | None = None,
) -> pd.DataFrame:
    """Generate multiclass label metrics for region rows.

    This function never averages target identifiers.  It summarizes each
    matching population with class counts, class distributions, purity,
    entropy and lift.
    """

    if regions.empty:
        return pd.DataFrame(
            columns=[
                "region_id",
                "target_class",
                "description",
                "support",
                "coverage",
                "dominant_class",
                "dominant_probability",
                "target_probability",
                "lift",
                "entropy",
                "class_counts",
                "class_distribution",
            ]
        )

    X_df = _as_dataframe_like(X, regions)
    y_array = np.asarray(y)
    if y_array.ndim != 1:
        y_array = np.ravel(y_array)
    y_array = y_array.astype(object, copy=False)
    classes = np.asarray(class_labels if class_labels is not None else regions.attrs.get("classes_", pd.unique(y_array)), dtype=object)
    class_to_index = {label: idx for idx, label in enumerate(classes)}
    priors = build_class_priors(y_array, classes)

    rows = []
    for region in regions.itertuples(index=False):
        mask = _region_mask(X_df, getattr(region, "lower_bounds"), getattr(region, "upper_bounds"))
        support = int(mask.sum())
        if support == 0:
            counts = np.zeros(len(classes), dtype=float)
        else:
            indices = [class_to_index[label] for label in y_array[mask]]
            counts = np.bincount(indices, minlength=len(classes)).astype(float)

        distribution = normalize_counts(counts)
        dominant_idx = int(np.argmax(distribution)) if len(distribution) else 0
        target_class = getattr(region, "target_class")
        target_idx = class_to_index[target_class]
        target_probability = float(distribution[target_idx])

        rows.append(
            {
                "region_id": getattr(region, "region_id"),
                "target_class": target_class,
                "description": getattr(region, "description"),
                "support": support,
                "coverage": float(support / len(X_df)) if len(X_df) else 0.0,
                "dominant_class": classes[dominant_idx],
                "dominant_probability": float(distribution[dominant_idx]),
                "target_probability": target_probability,
                "lift": lift(target_probability, float(priors[target_class])),
                "entropy": entropy(distribution),
                "class_counts": counts,
                "class_distribution": distribution,
            }
        )

    out = pd.DataFrame(rows)
    out.attrs["classes_"] = tuple(classes)
    return out


def _region_mask(X_df: pd.DataFrame, lower_bounds: dict, upper_bounds: dict) -> np.ndarray:
    mask = np.ones(len(X_df), dtype=bool)
    for feature, lower in lower_bounds.items():
        if feature not in X_df.columns:
            raise ValueError(f"Input data is missing required feature column: {feature}")
        mask &= X_df[feature].to_numpy() > float(lower)
    for feature, upper in upper_bounds.items():
        if feature not in X_df.columns:
            raise ValueError(f"Input data is missing required feature column: {feature}")
        mask &= X_df[feature].to_numpy() <= float(upper)
    return mask


def _as_dataframe_like(X, regions: pd.DataFrame) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()

    arr = np.asarray(X)
    names = _feature_names_from_regions(regions)
    if names and len(names) == arr.shape[1]:
        return pd.DataFrame(arr, columns=names)
    return pd.DataFrame(arr)


def _feature_names_from_regions(regions: pd.DataFrame) -> list:
    names = []
    seen = set()
    for bounds_col in ("lower_bounds", "upper_bounds"):
        if bounds_col not in regions.columns:
            continue
        for bounds in regions[bounds_col]:
            for name in bounds:
                if name not in seen:
                    seen.add(name)
                    names.append(name)

    def sort_key(name):
        text = str(name)
        prefix = "feature_"
        if text.startswith(prefix) and text[len(prefix):].isdigit():
            return (0, int(text[len(prefix):]))
        return (1, text)

    return sorted(names, key=sort_key)
