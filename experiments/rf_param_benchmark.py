import time
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from InsideForest import InsideForestClassifier
from experiments.benchmark import (
    _evaluate,
    _format_df,
    _run_dbscan,
    _run_kmeans,
    _load_titanic,
)


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


def _load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return features and target for the requested dataset."""

    if name == "iris":
        return load_iris(return_X_y=True)
    if name == "wine":
        return load_wine(return_X_y=True)
    if name == "digits":
        return load_digits(return_X_y=True)
    if name == "titanic":
        return _load_titanic()
    if name == "breast_cancer":
        return load_breast_cancer(return_X_y=True)
    raise ValueError(f"Unknown dataset: {name}")


def main() -> None:
    datasets = ["iris", "titanic", "wine", "digits", "breast_cancer"]

    n_estimators_vals = range(5, 105, 5)
    max_depth_vals = [2, 4, 6, 8, 10, None]
    rf_param_grid: List[Dict] = []
    for i, n in enumerate(n_estimators_vals):
        md = max_depth_vals[i % len(max_depth_vals)]
        rf_param_grid.append({"random_state": 42, "n_estimators": n, "max_depth": md})

    methods = [
        "select_clusters",
        "balance_lists_n_clusters",
        "max_prob_clusters",
        "match_class_distribution",
        "chimera",
        "menu",
    ]

    all_rows = []
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

    for ds_name in datasets:
        try:
            X, y = _load_dataset(ds_name)
        except Exception as e:  # pragma: no cover - network issues
            print(f"Skipping {ds_name} dataset: {e}")
            continue
        X = _scale(X)
        X, _, y, _ = train_test_split(
            X, y, train_size=30, stratify=y, random_state=0
        )

        print(f"=== {ds_name.capitalize()} dataset (n=30) ===")
        rows: List[Dict] = []
        for i, params in enumerate(rf_param_grid, start=1):
            for method in methods:
                name = f"{ds_name}_{method}_cfg{i}"
                result = _run_insideforest_with_params(X, y, params, name, method)
                result["dataset"] = ds_name
                rows.append(result)

        # Baseline algorithms
        k = len(np.unique(y))
        km_res = _run_kmeans(X, y, k).as_dict()
        km_res["dataset"] = ds_name
        rows.append(km_res)
        db_res = _run_dbscan(X, y, eps=0.5, min_samples=5).as_dict()
        db_res["dataset"] = ds_name
        rows.append(db_res)

        df = pd.DataFrame(rows)
        df_sorted = df.sort_values("purity", ascending=False)
        print(_format_df(df_sorted[metric_cols].head(10)))
        print("\nTop 10 parameter settings:")
        for _, row in df_sorted.head(10).iterrows():
            print(f"{row['algorithm']}: {row.get('params', 'N/A')}")

        all_rows.extend(rows)

    pd.DataFrame(all_rows).to_csv("rf_results.csv", index=False)


if __name__ == "__main__":
    main()
