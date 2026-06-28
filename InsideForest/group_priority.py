"""Experimental ranking of interdimensional region pairs.

This module is intentionally opt-in.  It does not change the classic
InsideForest region flow or the multiclass interpreter; it provides adapters
that turn their outputs into a common descriptor and a scorer for region pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class RegionDescriptor:
    """Common description for a rule/region used by pair prioritization."""

    id: Any
    bounds: dict[Any, tuple[float, float]]
    dimensions: tuple[Any, ...]
    support_indices: np.ndarray
    centroid: pd.Series
    n_support: int
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.bounds = {
            feature: (float(lower), float(upper))
            for feature, (lower, upper) in dict(self.bounds).items()
        }
        self.dimensions = _ordered_dimensions(self.dimensions)
        self.support_indices = np.unique(
            np.asarray(self.support_indices, dtype=int)
        )
        self.n_support = int(self.n_support)
        self.metadata = dict(self.metadata or {})


PAIR_COLUMNS = [
    "region_a",
    "region_b",
    "priority",
    "centroid_similarity",
    "centroid_similarity_weighted",
    "intersection_similarity",
    "dimension_similarity",
    "centroid_distance",
    "n_a",
    "n_b",
    "n_ab",
    "shared_dimensions",
    "union_dimensions",
    "shared_weight_mass",
    "dimension_jaccard",
    "weighted_dimension_jaccard",
    "support_jaccard",
    "support_product",
    "is_same_support",
    "is_same_geometry",
    "source_a",
    "source_b",
    "tau",
    "variant",
]


def from_traditional_regions(
    df_reres,
    X,
    *,
    min_support: int = 2,
    source: str = "traditional",
) -> list[RegionDescriptor]:
    """Build descriptors from ``Regions.prio_ranges`` / ``df_reres_`` output.

    Traditional InsideForest rules use inclusive bounds, matching
    :func:`InsideForest.cluster_selector.select_clusters`.
    """

    X_df = _coerce_X_frame(X)
    frames = _coerce_region_frames(df_reres)
    descriptors: list[RegionDescriptor] = []

    for frame_index, frame in enumerate(frames):
        if frame is None or frame.empty:
            continue
        if not isinstance(frame.columns, pd.MultiIndex):
            raise ValueError("Traditional regions must use MultiIndex columns")
        top_level = set(frame.columns.get_level_values(0))
        if "linf" not in top_level or "lsup" not in top_level:
            raise ValueError("Traditional regions must include 'linf' and 'lsup'")

        for row_position, (row_index, row) in enumerate(frame.iterrows()):
            bounds = _bounds_from_multiindex_row(row)
            if not bounds:
                continue
            support = _support_from_bounds(
                X_df,
                bounds,
                lower_inclusive=True,
                upper_inclusive=True,
            )
            if support.size < min_support:
                continue
            metadata = _metrics_from_row(row)
            metadata.update(
                {
                    "frame_index": frame_index,
                    "row_index": row_index,
                    "row_position": row_position,
                    "lower_inclusive": True,
                    "upper_inclusive": True,
                }
            )
            descriptors.append(
                _build_descriptor(
                    X_df,
                    region_id=f"{source}_{frame_index}_{row_index}",
                    bounds=bounds,
                    support_indices=support,
                    source=source,
                    metadata=metadata,
                )
            )

    return descriptors


def from_multiclass_rules(
    rules: pd.DataFrame,
    X,
    *,
    deduplicate: bool = True,
    min_support: int = 2,
    source: str = "multiclass",
) -> list[RegionDescriptor]:
    """Build descriptors from ``InsideForestClassRegionClusterer.regions_``.

    Multiclass tree rules use ``x > lower`` and ``x <= upper`` because lower
    bounds come from right-branch thresholds and upper bounds from left-branch
    thresholds.
    """

    X_df = _coerce_X_frame(X)
    if rules is None or rules.empty:
        return []
    required = {"lower_bounds", "upper_bounds"}
    missing = sorted(required - set(rules.columns))
    if missing:
        raise ValueError(f"Multiclass rules are missing required columns: {missing}")

    rule_df = rules.copy()
    if deduplicate and "leaf_region_id" in rule_df.columns:
        sort_cols = [col for col in ("score", "support") if col in rule_df.columns]
        if sort_cols:
            rule_df = rule_df.sort_values(sort_cols, ascending=False)
        rule_df = rule_df.drop_duplicates("leaf_region_id", keep="first")

    descriptors: list[RegionDescriptor] = []
    for row_position, (_, row) in enumerate(rule_df.iterrows()):
        bounds = _bounds_from_dicts(row["lower_bounds"], row["upper_bounds"])
        if not bounds:
            continue
        support = _support_from_bounds(
            X_df,
            bounds,
            lower_inclusive=False,
            upper_inclusive=True,
        )
        if support.size < min_support:
            continue

        leaf_region_id = row.get("leaf_region_id", row.get("region_id", row_position))
        metadata = {
            "row_position": row_position,
            "leaf_region_id": leaf_region_id,
            "lower_inclusive": False,
            "upper_inclusive": True,
        }
        for key in (
            "region_id",
            "target_class",
            "dominant_class",
            "score",
            "support",
            "coverage",
            "lift",
            "target_probability",
        ):
            if key in row:
                metadata[key] = row[key]

        descriptors.append(
            _build_descriptor(
                X_df,
                region_id=leaf_region_id,
                bounds=bounds,
                support_indices=support,
                source=source,
                metadata=metadata,
            )
        )

    return descriptors


def feature_weights_from_model(
    model: Any,
    feature_names: Sequence[Any] | None = None,
) -> pd.Series:
    """Return normalized feature-importance weights indexed by feature name.

    The function accepts a fitted scikit-learn forest, an ``InsideForest``
    estimator with ``feature_importances_``, or wrappers exposing ``rf_``/``rf``.
    When the fitted estimator has feature names, those names are used so weights
    survive column reordering. If the estimator was fitted without names,
    ``feature_names`` is used as the positional fallback.
    """

    estimator, importances = _extract_feature_importances(model)
    names, source = _resolve_importance_feature_names(
        model,
        estimator,
        n_features=len(importances),
        fallback_feature_names=feature_names,
    )
    weights = pd.Series(importances, index=pd.Index(names), dtype=float)
    weights = _normalize_feature_weight_series(weights)
    weights.attrs["feature_names_source"] = source
    return weights


def rank_region_pairs(
    X,
    regions: Iterable[RegionDescriptor],
    *,
    feature_weights: Any = None,
    alpha: float = 0.60,
    beta: float = 0.30,
    gamma: float = 0.10,
    rho: float = 0.5,
    tau: float | str = "median",
    scaler: str | None = "standard",
    top_k: int | None = None,
    variant: str = "robust",
    min_support: int = 2,
) -> pd.DataFrame:
    """Rank pairs of regions by centroid, observed overlap and dimensions.

    ``feature_weights`` can be a mapping/Series, a positional sequence, or a
    fitted RandomForest-like model exposing ``feature_importances_``. In the
    model case, feature names are used when available.

    ``variant="original"`` uses the unweighted dimensional Jaccard and the raw
    centroid similarity in the original proposed formula. ``variant="robust"``
    uses weighted dimensional Jaccard and downweights the centroid term by the
    global mass of the shared dimensions.
    """

    X_df = _coerce_X_frame(X)
    X_scaled = _scale_frame(X_df, scaler)
    regions_list = _prepare_regions(regions, X_df, min_support=min_support)
    if len(regions_list) < 2:
        return _empty_pair_frame()

    variant = str(variant).lower()
    if variant not in {"original", "robust"}:
        raise ValueError("variant must be 'original' or 'robust'")
    if rho <= 0:
        raise ValueError("rho must be positive")
    alpha, beta, gamma = _normalize_score_weights(alpha, beta, gamma)

    feature_names = list(X_df.columns)
    feature_to_pos = {feature: pos for pos, feature in enumerate(feature_names)}
    weights = _coerce_feature_weights(feature_names, feature_weights)
    centroid_matrix = _scaled_centroid_matrix(X_scaled, regions_list, feature_names)
    dimension_sets = [set(region.dimensions) for region in regions_list]
    intersection_counts = _pair_intersection_counts(
        [region.support_indices for region in regions_list],
        n_samples=len(X_df),
    )

    rows = []
    for i, j in combinations(range(len(regions_list)), 2):
        region_a = regions_list[i]
        region_b = regions_list[j]
        shared = dimension_sets[i].intersection(dimension_sets[j])
        if not shared:
            continue
        union = dimension_sets[i].union(dimension_sets[j])
        shared_for_distance = [
            dim
            for dim in _ordered_dimensions(shared)
            if dim in feature_to_pos
            and np.isfinite(centroid_matrix[i, feature_to_pos[dim]])
            and np.isfinite(centroid_matrix[j, feature_to_pos[dim]])
        ]
        if not shared_for_distance:
            continue

        shared_idx = np.array([feature_to_pos[dim] for dim in shared_for_distance])
        union_idx = np.array([feature_to_pos[dim] for dim in union if dim in feature_to_pos])
        if union_idx.size == 0:
            continue

        distance_weights = weights[shared_idx]
        distance_weight_sum = float(distance_weights.sum())
        if distance_weight_sum <= 0:
            distance_weights = np.full(shared_idx.size, 1.0 / shared_idx.size)
        else:
            distance_weights = distance_weights / distance_weight_sum
        diff = centroid_matrix[i, shared_idx] - centroid_matrix[j, shared_idx]
        centroid_distance = float(math.sqrt(float(np.sum(distance_weights * diff * diff))))

        n_a = int(region_a.n_support)
        n_b = int(region_b.n_support)
        n_ab = int(intersection_counts.get((i, j), 0))
        intersection_similarity = (
            float(2.0 * n_ab / (n_a + n_b)) if (n_a + n_b) else 0.0
        )
        support_union = n_a + n_b - n_ab
        support_jaccard = float(n_ab / support_union) if support_union else 0.0

        dimension_jaccard = float(len(shared) / len(union)) if union else 0.0
        shared_idx_all = np.array([feature_to_pos[dim] for dim in shared if dim in feature_to_pos])
        shared_weight_mass = float(weights[shared_idx_all].sum()) if shared_idx_all.size else 0.0
        union_weight_mass = float(weights[union_idx].sum())
        weighted_dimension_jaccard = (
            shared_weight_mass / union_weight_mass
            if union_weight_mass > 0
            else dimension_jaccard
        )
        dimension_base = (
            weighted_dimension_jaccard if variant == "robust" else dimension_jaccard
        )
        dimension_similarity = float(dimension_base**rho)

        rows.append(
            {
                "region_a": region_a.id,
                "region_b": region_b.id,
                "centroid_distance": centroid_distance,
                "intersection_similarity": intersection_similarity,
                "dimension_similarity": dimension_similarity,
                "n_a": n_a,
                "n_b": n_b,
                "n_ab": n_ab,
                "shared_dimensions": tuple(_ordered_dimensions(shared)),
                "union_dimensions": tuple(_ordered_dimensions(union)),
                "shared_weight_mass": shared_weight_mass,
                "dimension_jaccard": dimension_jaccard,
                "weighted_dimension_jaccard": float(weighted_dimension_jaccard),
                "support_jaccard": support_jaccard,
                "support_product": float(n_a * n_b),
                "is_same_support": bool(n_ab == n_a == n_b),
                "is_same_geometry": _same_geometry(region_a, region_b),
                "source_a": region_a.source,
                "source_b": region_b.source,
                "variant": variant,
            }
        )

    if not rows:
        return _empty_pair_frame()

    out = pd.DataFrame(rows)
    tau_value = _resolve_tau(out["centroid_distance"].to_numpy(dtype=float), tau)
    out["tau"] = tau_value
    out["centroid_similarity"] = np.exp(
        -out["centroid_distance"].to_numpy(dtype=float) / tau_value
    )
    out["centroid_similarity_weighted"] = (
        out["centroid_similarity"] * out["shared_weight_mass"]
    )
    centroid_term = (
        out["centroid_similarity_weighted"]
        if variant == "robust"
        else out["centroid_similarity"]
    )
    out["priority"] = (
        alpha * centroid_term
        + beta * out["intersection_similarity"]
        + gamma * out["dimension_similarity"]
    )

    out = out[PAIR_COLUMNS]
    out = out.sort_values(
        [
            "priority",
            "centroid_similarity",
            "intersection_similarity",
            "dimension_similarity",
            "n_ab",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    if top_k is not None:
        out = out.head(int(top_k)).reset_index(drop=True)
    return out


def _build_descriptor(
    X_df: pd.DataFrame,
    *,
    region_id: Any,
    bounds: dict[Any, tuple[float, float]],
    support_indices: np.ndarray,
    source: str,
    metadata: Mapping[str, Any] | None = None,
) -> RegionDescriptor:
    dimensions = _ordered_dimensions(bounds.keys())
    centroid = _raw_centroid(X_df, support_indices, dimensions)
    return RegionDescriptor(
        id=region_id,
        bounds=bounds,
        dimensions=dimensions,
        support_indices=support_indices,
        centroid=centroid,
        n_support=int(len(support_indices)),
        source=source,
        metadata=dict(metadata or {}),
    )


def _coerce_region_frames(df_reres) -> list[pd.DataFrame]:
    if isinstance(df_reres, pd.DataFrame):
        return [df_reres]
    if isinstance(df_reres, (list, tuple)):
        return list(df_reres)
    raise TypeError("df_reres must be a DataFrame or a sequence of DataFrames")


def _coerce_X_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    arr = X.toarray() if sparse.issparse(X) else np.asarray(X)
    if arr.ndim != 2:
        raise ValueError("X must be a 2-dimensional array or DataFrame")
    columns = [f"feature_{idx}" for idx in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=columns)


def _scale_frame(X_df: pd.DataFrame, scaler: str | None) -> pd.DataFrame:
    X_num = X_df.apply(pd.to_numeric, errors="raise").astype(float)
    if scaler is None or str(scaler).lower() in {"none", "identity"}:
        return X_num

    scaler_name = str(scaler).lower()
    if scaler_name == "standard":
        center = X_num.mean(skipna=True).fillna(0.0)
        scale = X_num.std(ddof=0, skipna=True).replace(0.0, 1.0).fillna(1.0)
    elif scaler_name == "robust":
        center = X_num.median(skipna=True).fillna(0.0)
        q75 = X_num.quantile(0.75)
        q25 = X_num.quantile(0.25)
        scale = (q75 - q25).replace(0.0, 1.0).fillna(1.0)
    else:
        raise ValueError("scaler must be 'standard', 'robust', 'none' or None")
    return (X_num - center) / scale


def _prepare_regions(
    regions: Iterable[RegionDescriptor],
    X_df: pd.DataFrame,
    *,
    min_support: int,
) -> list[RegionDescriptor]:
    if min_support < 1:
        raise ValueError("min_support must be >= 1")
    feature_set = set(X_df.columns)
    prepared: list[RegionDescriptor] = []
    for region in list(regions):
        missing = [dim for dim in region.dimensions if dim not in feature_set]
        if missing:
            raise KeyError(f"Region {region.id!r} uses columns not found in X: {missing}")
        support = np.unique(np.asarray(region.support_indices, dtype=int))
        support = support[(support >= 0) & (support < len(X_df))]
        if support.size < min_support:
            continue
        prepared.append(
            RegionDescriptor(
                id=region.id,
                bounds=dict(region.bounds),
                dimensions=_ordered_dimensions(region.dimensions),
                support_indices=support,
                centroid=region.centroid,
                n_support=int(support.size),
                source=region.source,
                metadata=dict(region.metadata),
            )
        )
    return prepared


def _coerce_feature_weights(
    feature_names: Sequence[Any],
    feature_weights: Any,
) -> np.ndarray:
    if feature_weights is None:
        weights = np.ones(len(feature_names), dtype=float)
    elif isinstance(feature_weights, pd.Series):
        weights = _weights_from_series(feature_names, feature_weights)
    elif isinstance(feature_weights, Mapping):
        missing = [feature for feature in feature_names if feature not in feature_weights]
        if missing:
            raise KeyError(f"feature_weights is missing columns: {missing}")
        weights = np.array([feature_weights[feature] for feature in feature_names], dtype=float)
    elif _looks_like_importance_model(feature_weights):
        series = feature_weights_from_model(feature_weights)
        try:
            weights = _weights_from_series(feature_names, series)
        except KeyError:
            if (
                series.attrs.get("feature_names_source") == "generated"
                and len(series) == len(feature_names)
            ):
                weights = series.to_numpy(dtype=float)
            else:
                raise
    else:
        weights = np.asarray(list(feature_weights), dtype=float)
        if weights.shape != (len(feature_names),):
            raise ValueError("feature_weights must match the number of X columns")

    if not np.all(np.isfinite(weights)):
        raise ValueError("feature_weights must be finite")
    if np.any(weights < 0):
        raise ValueError("feature_weights must be non-negative")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("At least one feature weight must be positive")
    return weights / total


def _weights_from_series(feature_names: Sequence[Any], weights: pd.Series) -> np.ndarray:
    missing = [feature for feature in feature_names if feature not in weights.index]
    if missing:
        raise KeyError(f"feature_weights is missing columns: {missing}")
    return weights.reindex(feature_names).to_numpy(dtype=float)


def _looks_like_importance_model(value: Any) -> bool:
    if value is None:
        return False
    if hasattr(value, "feature_importances_"):
        return True
    return any(hasattr(value, attr) for attr in ("rf_", "rf", "estimator_", "model_"))


def _extract_feature_importances(model: Any) -> tuple[Any, np.ndarray]:
    candidates = []
    for attr in ("rf_", "rf", "estimator_", "model_"):
        candidate = getattr(model, attr, None)
        if candidate is not None:
            candidates.append(candidate)
    candidates.append(model)

    errors = []
    for candidate in candidates:
        try:
            importances = getattr(candidate, "feature_importances_", None)
        except Exception as exc:  # pragma: no cover - defensive for unfitted wrappers
            errors.append(exc)
            continue
        if importances is None:
            continue
        values = np.asarray(importances, dtype=float)
        if values.ndim != 1:
            raise ValueError("feature_importances_ must be one-dimensional")
        return candidate, values

    if errors:
        raise ValueError("Could not read feature_importances_ from model") from errors[-1]
    raise ValueError("model must expose feature_importances_ or an rf_/rf estimator")


def _resolve_importance_feature_names(
    model: Any,
    estimator: Any,
    *,
    n_features: int,
    fallback_feature_names: Sequence[Any] | None,
) -> tuple[list[Any], str]:
    if fallback_feature_names is not None:
        names = list(fallback_feature_names)
        if len(names) != n_features:
            raise ValueError("feature_names must match feature_importances_ length")
        return names, "explicit"

    for owner in (model, estimator):
        for attr in ("feature_names_", "feature_names_out_", "feature_names_in_"):
            names = getattr(owner, attr, None)
            if names is None:
                continue
            names = list(names)
            if len(names) == n_features:
                return names, "model"

    return [f"feature_{idx}" for idx in range(n_features)], "generated"


def _normalize_feature_weight_series(weights: pd.Series) -> pd.Series:
    values = weights.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("feature_importances_ must be finite")
    if np.any(values < 0):
        raise ValueError("feature_importances_ must be non-negative")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("At least one feature importance must be positive")
    return pd.Series(values / total, index=weights.index, dtype=float)


def _normalize_score_weights(alpha: float, beta: float, gamma: float) -> tuple[float, float, float]:
    weights = np.asarray([alpha, beta, gamma], dtype=float)
    if not np.all(np.isfinite(weights)):
        raise ValueError("alpha, beta and gamma must be finite")
    if np.any(weights < 0):
        raise ValueError("alpha, beta and gamma must be non-negative")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("At least one of alpha, beta or gamma must be positive")
    return tuple(float(value / total) for value in weights)


def _scaled_centroid_matrix(
    X_scaled: pd.DataFrame,
    regions: Sequence[RegionDescriptor],
    feature_names: Sequence[Any],
) -> np.ndarray:
    values = X_scaled.to_numpy(dtype=float)
    centroids = np.full((len(regions), len(feature_names)), np.nan, dtype=float)
    feature_to_pos = {feature: pos for pos, feature in enumerate(feature_names)}
    for region_pos, region in enumerate(regions):
        support = np.asarray(region.support_indices, dtype=int)
        if support.size == 0:
            continue
        for dim in region.dimensions:
            col_pos = feature_to_pos[dim]
            with np.errstate(invalid="ignore"):
                centroids[region_pos, col_pos] = np.nanmean(values[support, col_pos])
    return centroids


def _pair_intersection_counts(
    support_indices: Sequence[np.ndarray],
    *,
    n_samples: int,
) -> dict[tuple[int, int], int]:
    row_parts = []
    col_parts = []
    for row_index, support in enumerate(support_indices):
        support = np.asarray(support, dtype=int)
        if support.size == 0:
            continue
        row_parts.append(np.full(support.size, row_index, dtype=int))
        col_parts.append(support)
    if not row_parts:
        return {}
    rows = np.concatenate(row_parts)
    cols = np.concatenate(col_parts)
    data = np.ones(rows.size, dtype=np.int8)
    membership = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(support_indices), n_samples),
        dtype=np.int16,
    )
    intersections = (membership @ membership.T).tocoo()
    counts: dict[tuple[int, int], int] = {}
    for row, col, value in zip(intersections.row, intersections.col, intersections.data):
        if row < col:
            counts[(int(row), int(col))] = int(value)
    return counts


def _resolve_tau(distances: np.ndarray, tau: float | str) -> float:
    if isinstance(tau, str):
        tau_name = tau.lower()
        if tau_name not in {"median", "auto"}:
            raise ValueError("tau must be positive or one of {'median', 'auto'}")
        positives = distances[np.isfinite(distances) & (distances > 0)]
        if positives.size == 0:
            return 1.0
        resolved = float(np.median(positives))
    else:
        resolved = float(tau)
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError("tau must resolve to a positive finite value")
    return resolved


def _raw_centroid(
    X_df: pd.DataFrame,
    support_indices: np.ndarray,
    dimensions: Sequence[Any],
) -> pd.Series:
    if len(dimensions) == 0:
        return pd.Series(dtype=float)
    if len(support_indices) == 0:
        return pd.Series(np.nan, index=list(dimensions), dtype=float)
    subset = X_df.iloc[np.asarray(support_indices, dtype=int)][list(dimensions)]
    subset = subset.apply(pd.to_numeric, errors="raise").astype(float)
    return subset.mean(axis=0, skipna=True)


def _support_from_bounds(
    X_df: pd.DataFrame,
    bounds: Mapping[Any, tuple[float, float]],
    *,
    lower_inclusive: bool,
    upper_inclusive: bool,
) -> np.ndarray:
    missing = [feature for feature in bounds if feature not in X_df.columns]
    if missing:
        raise KeyError(f"Columns not found in X: {missing}")
    mask = np.ones(len(X_df), dtype=bool)
    for feature, (lower, upper) in bounds.items():
        lower = float(lower)
        upper = float(upper)
        if lower > upper:
            raise ValueError(f"Invalid bounds for {feature!r}: lower > upper")
        values = pd.to_numeric(X_df[feature], errors="raise").to_numpy(dtype=float)
        if lower_inclusive:
            mask &= values >= lower
        else:
            mask &= values > lower
        if upper_inclusive:
            mask &= values <= upper
        else:
            mask &= values < upper
    return np.flatnonzero(mask).astype(int)


def _bounds_from_multiindex_row(row: pd.Series) -> dict[Any, tuple[float, float]]:
    linf = row["linf"] if "linf" in row.index.get_level_values(0) else pd.Series(dtype=float)
    lsup = row["lsup"] if "lsup" in row.index.get_level_values(0) else pd.Series(dtype=float)
    dimensions = _ordered_dimensions(set(linf.index).union(set(lsup.index)))
    bounds = {}
    for dim in dimensions:
        lower = _bound_value(linf.get(dim, np.nan), default=-np.inf)
        upper = _bound_value(lsup.get(dim, np.nan), default=np.inf)
        if np.isneginf(lower) and np.isposinf(upper):
            continue
        if lower > upper:
            raise ValueError(f"Invalid bounds for {dim!r}: lower > upper")
        bounds[dim] = (lower, upper)
    return bounds


def _bounds_from_dicts(lower_bounds: Any, upper_bounds: Any) -> dict[Any, tuple[float, float]]:
    lower_dict = dict(lower_bounds or {})
    upper_dict = dict(upper_bounds or {})
    dimensions = _ordered_dimensions(set(lower_dict).union(set(upper_dict)))
    bounds = {}
    for dim in dimensions:
        lower = _bound_value(lower_dict.get(dim, np.nan), default=-np.inf)
        upper = _bound_value(upper_dict.get(dim, np.nan), default=np.inf)
        if np.isneginf(lower) and np.isposinf(upper):
            continue
        if lower > upper:
            raise ValueError(f"Invalid bounds for {dim!r}: lower > upper")
        bounds[dim] = (lower, upper)
    return bounds


def _bound_value(value: Any, *, default: float) -> float:
    try:
        if pd.isna(value):
            return float(default)
    except (TypeError, ValueError):
        pass
    return float(value)


def _metrics_from_row(row: pd.Series) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if not isinstance(row.index, pd.MultiIndex):
        return metadata
    for key, value in row.items():
        if isinstance(key, tuple) and key[0] == "metrics":
            metadata[f"metric_{key[1]}"] = value
    return metadata


def _same_geometry(region_a: RegionDescriptor, region_b: RegionDescriptor) -> bool:
    if set(region_a.dimensions) != set(region_b.dimensions):
        return False
    for dim in region_a.dimensions:
        if dim not in region_b.bounds:
            return False
        lower_a, upper_a = region_a.bounds[dim]
        lower_b, upper_b = region_b.bounds[dim]
        if not (math.isclose(lower_a, lower_b) and math.isclose(upper_a, upper_b)):
            return False
    return True


def _ordered_dimensions(dimensions: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(sorted(tuple(dimensions), key=lambda value: str(value)))


def _empty_pair_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PAIR_COLUMNS)


__all__ = [
    "RegionDescriptor",
    "feature_weights_from_model",
    "from_multiclass_rules",
    "from_traditional_regions",
    "rank_region_pairs",
]
