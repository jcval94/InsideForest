"""Benchmark supervised clustering algorithms on multiple datasets.

This script evaluates InsideForest against traditional baselines on two datasets:
- Digits (medium sized, 1,797 samples)
- Synthetic large classification dataset (10,000 samples)

For each dataset we report a wide range of clustering metrics including
purity, multiple F1 scores, accuracy, information-theoretic metrics,
the *target-divergence* metric and runtime. Baselines include KMeans and DBSCAN.
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
from sklearn.datasets import (
    load_digits,
    load_iris,
    load_wine,
    make_classification,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

from InsideForest import InsideForestClassifier


def _contingency(y_true: np.ndarray, y_pred: np.ndarray):
    """Return contingency matrix and label mappings."""

    true_labels, true_inv = np.unique(y_true, return_inverse=True)
    pred_labels, pred_inv = np.unique(y_pred, return_inverse=True)
    C = metrics.cluster.contingency_matrix(true_inv, pred_inv)
    return C, true_labels, pred_labels, true_inv, pred_inv


def _purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    C, *_ = _contingency(y_true, y_pred)
    return C.max(axis=0).sum() / C.sum()


def _align_by_hungarian(y_true: np.ndarray, y_pred: np.ndarray):
    C, true_labels, pred_labels, *_ = _contingency(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-C)
    mapping = {pred_labels[c]: true_labels[r] for r, c in zip(row_ind, col_ind)}
    aligned = np.array([mapping.get(p, p) for p in y_pred])
    return aligned, mapping


def _f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    aligned, _ = _align_by_hungarian(y_true, y_pred)
    labels_true = np.unique(y_true)
    return metrics.f1_score(
        y_true, aligned, average="macro", labels=labels_true, zero_division=0
    )


def _bcubed(y_true: np.ndarray, y_pred: np.ndarray):
    C, _, _, true_inv, pred_inv = _contingency(y_true, y_pred)
    C = np.asarray(C)
    class_counts = C.sum(axis=1)
    cluster_counts = C.sum(axis=0)
    inter = C[true_inv, pred_inv]
    prec_i = inter / cluster_counts[pred_inv]
    rec_i = inter / class_counts[true_inv]
    P, R = prec_i.mean(), rec_i.mean()
    F = 0.0 if (P + R) == 0 else 2 * P * R / (P + R)
    return float(P), float(R), float(F)


@dataclass
class Result:
    """Holds evaluation metrics for a single run."""

    algorithm: str
    purity: float
    macro_f1: float
    weighted_f1: float
    accuracy: float
    homogeneity: float
    completeness: float
    v_measure: float
    nmi: float
    ami: float
    ari: float
    fowlkes_mallows: float
    bcubed_precision: float
    bcubed_recall: float
    bcubed_f1: float
    divergence: float
    runtime: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "algorithm": self.algorithm,
            "purity": self.purity,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "accuracy": self.accuracy,
            "homogeneity": self.homogeneity,
            "completeness": self.completeness,
            "v_measure": self.v_measure,
            "nmi": self.nmi,
            "ami": self.ami,
            "ari": self.ari,
            "fowlkes_mallows": self.fowlkes_mallows,
            "bcubed_precision": self.bcubed_precision,
            "bcubed_recall": self.bcubed_recall,
            "bcubed_f1": self.bcubed_f1,
            "divergence": self.divergence,
            "runtime": self.runtime,
        }


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
    aligned, _ = _align_by_hungarian(y_true, y_pred)
    labels_true = np.unique(y_true)

    purity = _purity_score(y_true, y_pred)
    macro_f1 = metrics.f1_score(
        y_true, aligned, average="macro", labels=labels_true, zero_division=0
    )
    weighted_f1 = metrics.f1_score(
        y_true, aligned, average="weighted", labels=labels_true, zero_division=0
    )
    acc = metrics.accuracy_score(y_true, aligned)

    hom = metrics.homogeneity_score(y_true, y_pred)
    comp = metrics.completeness_score(y_true, y_pred)
    v = metrics.v_measure_score(y_true, y_pred)
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    ami = metrics.adjusted_mutual_info_score(y_true, y_pred)
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    fm = metrics.fowlkes_mallows_score(y_true, y_pred)
    bP, bR, bF = _bcubed(y_true, y_pred)
    div = _target_divergence_score(y_true, y_pred)

    return Result(
        name,
        purity,
        macro_f1,
        weighted_f1,
        acc,
        hom,
        comp,
        v,
        nmi,
        ami,
        ari,
        fm,
        bP,
        bR,
        bF,
        div,
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


def _load_titanic() -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess the Titanic survival dataset."""

    df = sns.load_dataset("titanic")
    df = df[[
        "survived",
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked",
    ]].dropna()
    y = df["survived"].to_numpy()
    X = pd.get_dummies(df.drop(columns=["survived"])).to_numpy()
    return X, y


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

    iris_X, iris_y = load_iris(return_X_y=True)
    iris_X = _scale(iris_X)
    benchmark_dataset("Iris", iris_X, iris_y)

    wine_X, wine_y = load_wine(return_X_y=True)
    wine_X = _scale(wine_X)
    benchmark_dataset("Wine", wine_X, wine_y)

    try:
        titanic_X, titanic_y = _load_titanic()
        titanic_X = _scale(titanic_X)
        benchmark_dataset("Titanic", titanic_X, titanic_y)
    except Exception as e:
        print(f"\nSkipping Titanic dataset: {e}")

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
