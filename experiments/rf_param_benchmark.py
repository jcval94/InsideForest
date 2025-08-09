import time
from typing import List, Dict

import itertools
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from InsideForest import InsideForestClassifier
from experiments.benchmark import _evaluate, _format_df, _run_kmeans, _run_dbscan


def _run_insideforest_with_params(
    X: np.ndarray, y: np.ndarray, rf_params: Dict, name: str, method: str
) -> Dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.35, stratify=y, random_state=42
    )
    clf = InsideForestClassifier(
        rf_params=rf_params,
        tree_params={"lang": "python", "n_sample_multiplier": 0.02, "ef_sample_multiplier": 5},
        method=method,
    )
    start = time.time()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    runtime = time.time() - start
    result = _evaluate(y_test, preds, runtime, name).as_dict()
    result["params"] = rf_params
    result["method"] = method
    return result


def _scale(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(X)


def main() -> None:
    X, y = load_iris(return_X_y=True)
    X = _scale(X)
    X, _, y, _ = train_test_split(X, y, train_size=30, stratify=y, random_state=0)

    n_estimators_vals = range(5, 105, 5)
    max_depth_vals = [2, 4, 6, 8, 10, None]
    rf_param_grid: List[Dict] = []
    for i, n in enumerate(n_estimators_vals):
        md = max_depth_vals[i % len(max_depth_vals)]
        rf_param_grid.append({"random_state": 42, "n_estimators": n, "max_depth": md})

    print("=== Iris dataset (n=30) ===")
    rows = []
    methods = [
        "select_clusters",
        "balance_lists_n_clusters",
        "max_prob_clusters",
        "menu",
    ]
    for i, params in enumerate(rf_param_grid, start=1):
        for method in methods:
            name = f"{method}_cfg{i}"
            rows.append(
                _run_insideforest_with_params(X, y, params, name, method)
            )

    # Baseline algorithms
    rows.append(_run_kmeans(X, y, 3).as_dict())
    rows.append(_run_dbscan(X, y, eps=0.5, min_samples=5).as_dict())

    df = pd.DataFrame(rows)
    metric_cols = [
        "algorithm",
        "purity",
        "macro_f1",
        "accuracy",
        "nmi",
        "ami",
        "ari",
        "bcubed_f1",
        "divergence",
        "runtime",
    ]
    df_sorted = df.sort_values("purity", ascending=False)
    print(_format_df(df_sorted[metric_cols].head(10)))
    print("\nTop 10 parameter settings:")
    for _, row in df_sorted.head(10).iterrows():
        print(f"{row['algorithm']}: {row.get('params', 'N/A')}")
    df_sorted.to_csv("rf_results.csv", index=False)


if __name__ == "__main__":
    main()
