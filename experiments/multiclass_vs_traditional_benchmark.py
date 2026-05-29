"""Benchmark multiclass interpretation against the traditional InsideForest API."""

from __future__ import annotations

import statistics
import sys
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, make_classification

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from InsideForest import InsideForestClassifier
from InsideForest.multiclass import InsideForestMulticlassClassifier


RESULTS_DIR = Path("experiments") / "results"
CSV_PATH = RESULTS_DIR / "multiclass_vs_traditional_benchmark.csv"
MD_PATH = RESULTS_DIR / "multiclass_vs_traditional_benchmark.md"


def _as_frame(data, feature_names):
    return pd.DataFrame(data, columns=[str(name).replace(" ", "_") for name in feature_names])


def load_datasets():
    iris = load_iris()
    wine = load_wine()
    synthetic_X, synthetic_y = make_classification(
        n_samples=450,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.4,
        random_state=42,
    )
    return {
        "iris": (_as_frame(iris.data, iris.feature_names), iris.target),
        "wine": (_as_frame(wine.data, wine.feature_names), wine.target),
        "synthetic_3class": (
            pd.DataFrame(synthetic_X, columns=[f"feature_{i}" for i in range(synthetic_X.shape[1])]),
            synthetic_y,
        ),
    }


def time_call(func):
    start = perf_counter()
    result = func()
    return result, perf_counter() - start


def majority_aligned_accuracy(labels, y):
    df = pd.DataFrame({"label": labels, "y": y})
    mapping = (
        df.groupby("label")["y"]
        .agg(lambda values: values.value_counts().idxmax())
        .to_dict()
    )
    aligned = df["label"].map(mapping).to_numpy()
    return float(np.mean(aligned == np.asarray(y)))


def summarize_runs(values):
    return float(statistics.median(values))


def benchmark_dataset(name, X, y, repeats=3):
    rf_params = {
        "n_estimators": 30,
        "max_depth": 6,
        "random_state": 42,
        "n_jobs": 1,
    }
    rows = []

    traditional_fit = []
    traditional_predict = []
    traditional_rf_accuracy = []
    traditional_assignment_accuracy = []
    traditional_regions = []
    traditional_unmatched_rate = []

    multiclass_fit = []
    multiclass_assign = []
    multiclass_rf_accuracy = []
    multiclass_assignment_accuracy = []
    multiclass_rules = []
    multiclass_fallback_rate = []

    for _ in range(repeats):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            traditional, elapsed = time_call(
                lambda: InsideForestClassifier(
                    rf_params=rf_params,
                    max_cases=len(X),
                    no_trees_search=30,
                    leaf_percentile=95,
                    low_leaf_fraction=0.05,
                ).fit(X, y)
            )
            traditional_fit.append(elapsed)

            trad_pred, elapsed = time_call(lambda: traditional.predict(X))
            traditional_predict.append(elapsed)

        traditional_rf_accuracy.append(float(traditional.rf.score(X, y)))
        traditional_assignment_accuracy.append(majority_aligned_accuracy(trad_pred, y))
        traditional_regions.append(int(sum(len(df) for df in traditional.df_reres_)))
        traditional_unmatched_rate.append(float(np.mean(np.asarray(trad_pred) == -1)))

        multiclass, elapsed = time_call(
            lambda: InsideForestMulticlassClassifier(
                rf_params=rf_params,
                percentil=95,
                low_frac=0.05,
                min_support=2,
                random_state=42,
            ).fit(X, y)
        )
        multiclass_fit.append(elapsed)

        assigned, elapsed = time_call(lambda: multiclass.assign_regions(X))
        multiclass_assign.append(elapsed)

        multiclass_rf_accuracy.append(float(multiclass.rf_.score(X, y)))
        multiclass_assignment_accuracy.append(
            float(np.mean(assigned["predicted_class"].to_numpy() == np.asarray(y)))
        )
        multiclass_rules.append(int(len(multiclass.rules_)))
        multiclass_fallback_rate.append(float(np.mean(assigned["source"] == "model_fallback")))

    rows.append(
        {
            "dataset": name,
            "method": "InsideForestClassifier_traditional",
            "fit_seconds_median": summarize_runs(traditional_fit),
            "assign_seconds_median": summarize_runs(traditional_predict),
            "rf_accuracy_median": summarize_runs(traditional_rf_accuracy),
            "assignment_accuracy_median": summarize_runs(traditional_assignment_accuracy),
            "n_rules_or_regions_median": summarize_runs(traditional_regions),
            "fallback_or_unmatched_rate_median": summarize_runs(traditional_unmatched_rate),
        }
    )
    rows.append(
        {
            "dataset": name,
            "method": "InsideForestMulticlassClassifier",
            "fit_seconds_median": summarize_runs(multiclass_fit),
            "assign_seconds_median": summarize_runs(multiclass_assign),
            "rf_accuracy_median": summarize_runs(multiclass_rf_accuracy),
            "assignment_accuracy_median": summarize_runs(multiclass_assignment_accuracy),
            "n_rules_or_regions_median": summarize_runs(multiclass_rules),
            "fallback_or_unmatched_rate_median": summarize_runs(multiclass_fallback_rate),
        }
    )
    return rows


def build_markdown(results):
    lines = [
        "# Multiclass vs Traditional Benchmark",
        "",
        "Median over 3 runs. Both methods use the same RandomForest parameters: "
        "`n_estimators=30`, `max_depth=6`, `random_state=42`, `n_jobs=1`.",
        "",
        "| Dataset | Method | Fit s | Assign/Predict s | RF acc | Assignment acc | Rules/Regions | Fallback/Unmatched |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            "| {dataset} | {method} | {fit_seconds_median:.4f} | "
            "{assign_seconds_median:.4f} | {rf_accuracy_median:.4f} | "
            "{assignment_accuracy_median:.4f} | {n_rules_or_regions_median:.0f} | "
            "{fallback_or_unmatched_rate_median:.4f} |".format(**row)
        )

    lines.extend(["", "## Ratios", ""])
    df = pd.DataFrame(results)
    for dataset, group in df.groupby("dataset", sort=False):
        traditional = group[group["method"] == "InsideForestClassifier_traditional"].iloc[0]
        multiclass = group[group["method"] == "InsideForestMulticlassClassifier"].iloc[0]
        fit_speedup = traditional["fit_seconds_median"] / multiclass["fit_seconds_median"]
        assign_ratio = multiclass["assign_seconds_median"] / traditional["assign_seconds_median"]
        lines.append(
            f"- {dataset}: multiclass fit is {fit_speedup:.2f}x the speed of traditional; "
            f"multiclass assignment takes {assign_ratio:.2f}x traditional predict time."
        )

    return "\n".join(lines) + "\n"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for name, (X, y) in load_datasets().items():
        results.extend(benchmark_dataset(name, X, y))

    df = pd.DataFrame(results)
    df.to_csv(CSV_PATH, index=False)
    MD_PATH.write_text(build_markdown(results), encoding="utf-8")
    print(df.to_string(index=False))
    print(f"\nWrote {CSV_PATH} and {MD_PATH}")


if __name__ == "__main__":
    main()
