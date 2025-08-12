import itertools
import time
from typing import Dict, List

import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from InsideForest import InsideForestClassifier
from experiments.benchmark import _evaluate


def _prepare_data(loader):
    X, y = loader(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    return train_test_split(X, y, train_size=0.35, stratify=y, random_state=42)


def run_experiments() -> pd.DataFrame:
    datasets = {
        "iris": load_iris,
        "wine": load_wine,
    }

    param_grid = {
        "divide": [3, 5, 7],
        "leaf_percentile": [85, 90, 95],
        "low_leaf_fraction": [0.01, 0.03, 0.05],
    }

    rows: List[Dict] = []

    for ds_name, loader in datasets.items():
        X_train, X_test, y_train, y_test = _prepare_data(loader)
        for divide, leaf, low_frac in itertools.product(
            param_grid["divide"],
            param_grid["leaf_percentile"],
            param_grid["low_leaf_fraction"],
        ):
            clf = InsideForestClassifier(
                method="select_clusters",
                divide=divide,
                get_detail=False,
                leaf_percentile=leaf,
                low_leaf_fraction=low_frac,
            )
            start = time.time()
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            runtime = time.time() - start

            name = f"{ds_name}_div{divide}_leaf{leaf}_low{low_frac}"
            metrics = _evaluate(y_test, preds, runtime, name).as_dict()
            metrics.update(
                {
                    "dataset": ds_name,
                    "divide": divide,
                    "leaf_percentile": leaf,
                    "low_leaf_fraction": low_frac,
                }
            )
            rows.append(metrics)

    return pd.DataFrame(rows)


def main() -> None:
    df = run_experiments()
    cols = [
        "dataset",
        "divide",
        "leaf_percentile",
        "low_leaf_fraction",
        "purity",
        "macro_f1",
        "accuracy",
        "nmi",
        "bcubed_f1",
        "runtime",
    ]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
