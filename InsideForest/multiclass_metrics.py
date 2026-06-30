"""Metrics for multiclass InsideForest interpretation.

This module intentionally does not depend on the legacy binary-oriented
``Trees``/``Labels`` internals.  Scores are computed from class probability
vectors, so numeric class identifiers are treated as labels rather than
ordered magnitudes.
"""

from __future__ import annotations

import math
import warnings
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


_EPS = np.finfo(float).eps


def normalize_counts(counts: Sequence[float]) -> np.ndarray:
    """Return a probability vector from class counts."""

    arr = np.asarray(counts, dtype=float)
    total = float(np.nansum(arr))
    if total <= 0:
        return np.zeros_like(arr, dtype=float)
    return arr / total


def entropy(probabilities: Sequence[float]) -> float:
    """Compute Shannon entropy in bits for a probability vector."""

    probs = np.asarray(probabilities, dtype=float)
    probs = probs[np.isfinite(probs) & (probs > 0)]
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def jensen_shannon_divergence(
    probabilities: Sequence[float],
    reference: Sequence[float],
) -> float:
    """Compute Jensen-Shannon divergence in bits."""

    p = normalize_counts(probabilities)
    q = normalize_counts(reference)
    if p.shape != q.shape:
        raise ValueError("probabilities and reference must have the same length")
    m = 0.5 * (p + q)
    return float(0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m))


def lift(probability: float, prior_probability: float) -> float:
    """Return class lift for one class probability against its prior."""

    if prior_probability <= 0:
        return math.inf if probability > 0 else 0.0
    return float(probability / prior_probability)


def purity_lift_coverage_score(
    target_probability: float,
    coverage: float,
    class_lift: float,
) -> float:
    """Default semantic rule score.

    ``target_probability`` is the one-vs-rest purity for the target class,
    ``coverage`` is the fraction of rows covered by the leaf, and
    ``class_lift`` measures enrichment relative to the global class prior.
    """

    safe_lift = 0.0 if not np.isfinite(class_lift) else max(math.log2(max(class_lift, _EPS)), 0.0)
    return float(target_probability * math.sqrt(max(coverage, 0.0)) * (1.0 + safe_lift))


def build_class_priors(y: Iterable, classes: Sequence | None = None) -> pd.Series:
    """Return class priors indexed by class label."""

    y_array = np.asarray(list(y) if not isinstance(y, np.ndarray) else y)
    if y_array.ndim != 1:
        y_array = np.ravel(y_array)
    y_series = pd.Series(y_array)
    if classes is None:
        labels = np.asarray(pd.unique(y_series))
    else:
        labels = np.asarray(classes)

    counts = y_series.value_counts(dropna=False)
    total = float(len(y_series))
    priors = [float(counts.get(label, 0) / total) if total else 0.0 for label in labels]
    return pd.Series(priors, index=pd.Index(labels, name="class"), dtype=float)


def score_multiclass_rules(
    rule_df: pd.DataFrame,
    class_priors: pd.Series | dict | Sequence[float],
    *,
    rule_score: str = "purity_lift_coverage",
    score: str | None = None,
) -> pd.DataFrame:
    """Add semantic multiclass metrics and score columns to rule rows.

    The input is expected to contain one row per ``(leaf, target_class)`` with
    ``class_distribution``, ``target_probability``, ``target_class`` and
    ``coverage`` columns.
    """

    if score is not None:
        warnings.warn(
            "score is deprecated; use rule_score. The legacy name will be "
            "removed in InsideForest 0.5.0.",
            FutureWarning,
            stacklevel=2,
        )
        if rule_score != "purity_lift_coverage" and rule_score != score:
            raise TypeError("Received conflicting values for rule_score and score")
        rule_score = score

    if rule_df.empty:
        return rule_df.copy()

    out = rule_df.copy()
    priors = _coerce_priors(class_priors, out)

    prior_values = []
    lift_values = []
    entropy_values = []
    js_values = []
    score_values = []

    prior_vector = np.asarray([priors[label] for label in out.attrs.get("classes_", priors.index)], dtype=float)

    for row in out.itertuples(index=False):
        target_class = getattr(row, "target_class")
        target_probability = float(getattr(row, "target_probability"))
        coverage = float(getattr(row, "coverage"))
        distribution = np.asarray(getattr(row, "class_distribution"), dtype=float)
        prior_probability = float(priors[target_class])
        class_lift = lift(target_probability, prior_probability)

        if rule_score != "purity_lift_coverage":
            raise ValueError(
                "Only rule_score='purity_lift_coverage' is supported"
            )

        prior_values.append(prior_probability)
        lift_values.append(class_lift)
        entropy_values.append(entropy(distribution))
        js_values.append(jensen_shannon_divergence(distribution, prior_vector))
        score_values.append(
            purity_lift_coverage_score(target_probability, coverage, class_lift)
        )

    out["prior_probability"] = prior_values
    out["lift"] = lift_values
    out["entropy"] = entropy_values
    out["js_divergence"] = js_values
    out["score"] = score_values
    return out


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    mask = (p > 0) & (q > 0)
    if not np.any(mask):
        return 0.0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def _coerce_priors(
    class_priors: pd.Series | dict | Sequence[float],
    rule_df: pd.DataFrame,
) -> pd.Series:
    if isinstance(class_priors, pd.Series):
        priors = class_priors.astype(float)
    elif isinstance(class_priors, dict):
        priors = pd.Series(class_priors, dtype=float)
    else:
        classes = rule_df.attrs.get("classes_")
        if classes is None:
            classes = pd.unique(rule_df["target_class"])
        priors = pd.Series(list(class_priors), index=pd.Index(classes), dtype=float)

    missing = set(pd.unique(rule_df["target_class"])) - set(priors.index)
    if missing:
        raise KeyError(f"class_priors is missing classes: {sorted(missing)}")
    return priors
