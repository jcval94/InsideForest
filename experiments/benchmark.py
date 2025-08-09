"""Benchmark supervised clustering algorithms on multiple datasets.

This script evaluates InsideForest against traditional baselines on two datasets:
- Digits (medium sized, 1,797 samples)
- Synthetic large classification dataset (10,000 samples)

For each dataset we report purity, macro F1-score, an additional
*target-divergence* metric and runtime. Baselines include KMeans and DBSCAN.
A basic sensitivity analysis is also provided for key hyperparameters: the
number of clusters ``K`` for KMeans and ``eps`` / ``min_samples`` for DBSCAN.

The script is meant to be executed as a module:

```
python -m experiments.benchmark
```
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_digits, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from InsideForest import InsideForestClassifier


@dataclass
class Result:
    """Holds evaluation metrics for a single run."""

    algorithm: str
    purity: float
    f1: float
    divergence: float
    runtime: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "algorithm": self.algorithm,
            "purity": self.purity,
            "f1": self.f1,
            "divergence": self.divergence,
            "runtime": self.runtime,
        }


def _purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute clustering purity."""

    contingency = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.max(contingency, axis=0)) / np.sum(contingency)


def _f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Align cluster labels with true labels and compute macro F1."""

    contingency = metrics.cluster.contingency_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    aligned = np.array([mapping.get(c, c) for c in y_pred])
    return metrics.f1_score(y_true, aligned, average="macro")


def _target_divergence_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted average total variation between cluster and global class distributions."""

    contingency = metrics.cluster.contingency_matrix(y_true, y_pred)
    total = contingency.sum()
    if total == 0:
        return 0.0
    global_dist = contingency.sum(axis=1) / total
    cluster_sizes = contingency.sum(axis=0)
    divergence = 0.0
    for j, size in enumerate(cluster_sizes):
        if size == 0:
            continue
        cluster_dist = contingency[:, j] / size
        divergence += (size / total) * 0.5 * np.sum(np.abs(cluster_dist - global_dist))
    return float(divergence)


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, runtime: float, name: str) -> Result:
    return Result(
        name,
        _purity_score(y_true, y_pred),
        _f1_score(y_true, y_pred),
        _target_divergence_score(y_true, y_pred),
        runtime,
    )


def _run_insideforest(X: np.ndarray, y: np.ndarray) -> Result:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.35, stratify=y, random_state=42
    )
    clf = InsideForestClassifier(rf_params={"random_state": 42})
    start = time.time()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    runtime = time.time() - start
    return _evaluate(y_test, preds, runtime, "InsideForest")


def _run_kmeans(X: np.ndarray, y: np.ndarray, k: int) -> Result:
    start = time.time()
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    preds = km.fit_predict(X)
    runtime = time.time() - start
    return _evaluate(y, preds, runtime, f"KMeans(k={k})")


def _run_dbscan(X: np.ndarray, y: np.ndarray, eps: float, min_samples: int) -> Result:
    start = time.time()
    db = DBSCAN(eps=eps, min_samples=min_samples)
    preds = db.fit_predict(X)
    runtime = time.time() - start
    return _evaluate(y, preds, runtime, f"DBSCAN(eps={eps},min={min_samples})")


def _sensitivity_kmeans(X: np.ndarray, y: np.ndarray, ks: Iterable[int]) -> pd.DataFrame:
    rows = [_run_kmeans(X, y, k).as_dict() for k in ks]
    return pd.DataFrame(rows)


def _sensitivity_dbscan(
    X: np.ndarray, y: np.ndarray, eps_values: Iterable[float], min_samples_values: Iterable[int]
) -> pd.DataFrame:
    rows = []
    for eps in eps_values:
        for min_samples in min_samples_values:
            rows.append(_run_dbscan(X, y, eps, min_samples).as_dict())
    return pd.DataFrame(rows)


def _scale(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(X)


def _format_df(df: pd.DataFrame) -> str:
    """Return a nicely formatted table for console output."""

    return df.round(3).to_string(index=False)


def benchmark_dataset(name: str, X: np.ndarray, y: np.ndarray) -> None:
    print(f"\n=== Dataset: {name} ===")
    results = []
    results.append(_run_insideforest(X, y).as_dict())
    k = len(np.unique(y))
    results.append(_run_kmeans(X, y, k).as_dict())
    results.append(_run_dbscan(X, y, eps=0.5, min_samples=5).as_dict())
    print(_format_df(pd.DataFrame(results)))

    print("\nSensitivity analysis - KMeans")
    print(_format_df(_sensitivity_kmeans(X, y, ks=[max(2, k - 1), k, k + 1])))
    print("\nSensitivity analysis - DBSCAN")
    print(
        _format_df(
            _sensitivity_dbscan(
                X, y, eps_values=[0.3, 0.5, 0.7], min_samples_values=[5, 10]
            )
        )
    )


def main() -> None:
    digits_X, digits_y = load_digits(return_X_y=True)
    digits_X = _scale(digits_X)
    benchmark_dataset("Digits", digits_X, digits_y)

    X_large, y_large = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=10,
        n_classes=5,
        random_state=42,
    )
    X_large = _scale(X_large)
    benchmark_dataset("SyntheticLarge", X_large, y_large)


if __name__ == "__main__":
    main()
