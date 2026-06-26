"""Validate InsideForestRegressor on regression datasets.

The regressor is evaluated as a region extractor for continuous targets.  The
base RandomForestRegressor is still measured with R2/RMSE, while the region
labels are measured by coverage, unmatched rate, reduction in target spread,
and the RMSE obtained by assigning each region its train-target mean.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, make_friedman1, make_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from InsideForest import InsideForestRegressor


OUT_DIR = ROOT / "experiments" / "results" / "regression_region_validation"


@dataclass
class RegressionValidationConfig:
    profile: str = "quick"
    seeds: tuple[int, ...] = (7, 19)
    test_size: float = 0.35
    max_rows: int = 360
    rf_n_estimators: int = 50
    rf_max_depth: int | None = 6
    no_trees_search: int = 35
    max_cases: int = 320
    leaf_percentile: float = 90
    low_leaf_fraction: float = 0.12
    explicit_k_features: int = 8
    out_dir: Path = OUT_DIR

    @classmethod
    def from_profile(cls, profile: str) -> "RegressionValidationConfig":
        if profile == "full":
            return cls(
                profile="full",
                seeds=(7, 19, 31, 43, 59),
                max_rows=700,
                rf_n_estimators=60,
                rf_max_depth=8,
                no_trees_search=35,
                max_cases=650,
                leaf_percentile=90,
                low_leaf_fraction=0.12,
            )
        return cls(profile="quick")


def datasets(seed: int) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    diabetes = load_diabetes()
    friedman_X, friedman_y = make_friedman1(
        n_samples=520,
        n_features=10,
        noise=1.0,
        random_state=seed,
    )
    linear_X, linear_y = make_regression(
        n_samples=520,
        n_features=14,
        n_informative=5,
        noise=12.0,
        random_state=seed + 1,
    )
    nonlinear_X, nonlinear_y = nonlinear_signal(seed)

    return {
        "diabetes": (
            pd.DataFrame(diabetes.data, columns=diabetes.feature_names),
            diabetes.target.astype(float),
        ),
        "friedman1": (
            pd.DataFrame(friedman_X, columns=[f"f{i}" for i in range(friedman_X.shape[1])]),
            friedman_y.astype(float),
        ),
        "linear_sparse": (
            pd.DataFrame(linear_X, columns=[f"f{i}" for i in range(linear_X.shape[1])]),
            linear_y.astype(float),
        ),
        "nonlinear_signal": (nonlinear_X, nonlinear_y),
    }


def nonlinear_signal(seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(520, 12))
    y = (
        2.5 * np.sin(X[:, 0])
        + 1.8 * (X[:, 1] > 0).astype(float)
        - 1.2 * X[:, 2] * X[:, 3]
        + 0.6 * X[:, 4] ** 2
        + rng.normal(scale=0.35, size=X.shape[0])
    )
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])]), y.astype(float)


def scale_frame(X: pd.DataFrame) -> pd.DataFrame:
    values = StandardScaler().fit_transform(X.to_numpy(dtype=float))
    return pd.DataFrame(values, columns=[str(col).replace(" ", "_") for col in X.columns])


def downsample(X: pd.DataFrame, y: np.ndarray, max_rows: int, seed: int):
    if len(X) <= max_rows:
        return X.reset_index(drop=True), np.asarray(y)
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(X), size=max_rows, replace=False))
    return X.iloc[selected].reset_index(drop=True), np.asarray(y)[selected]


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def region_mean_predictions(labels_train, y_train, labels_test, y_test):
    labels_train = np.asarray(labels_train)
    labels_test = np.asarray(labels_test)
    y_train = np.asarray(y_train, dtype=float)
    y_test = np.asarray(y_test, dtype=float)

    valid_train = labels_train != -1
    global_mean = float(np.mean(y_train))
    mapping = {
        label: float(np.mean(y_train[labels_train == label]))
        for label in np.unique(labels_train[valid_train])
    }

    covered = np.array([label in mapping and label != -1 for label in labels_test], dtype=bool)
    pred_all = np.array([mapping.get(label, global_mean) for label in labels_test], dtype=float)
    pred_covered = pred_all[covered]
    y_covered = y_test[covered]
    return pred_all, pred_covered, y_covered, covered


def target_std_reduction(labels, y) -> float:
    labels = np.asarray(labels)
    y = np.asarray(y, dtype=float)
    valid = labels != -1
    if not np.any(valid):
        return math.nan
    global_std = float(np.std(y[valid]))
    if global_std == 0:
        return math.nan

    weighted = 0.0
    total = 0
    for label in np.unique(labels[valid]):
        values = y[labels == label]
        weighted += float(np.std(values)) * len(values)
        total += len(values)
    return float(1.0 - (weighted / max(total, 1)) / global_std)


def evaluate_dataset(name: str, X: pd.DataFrame, y: np.ndarray, seed: int, config: RegressionValidationConfig):
    X, y = downsample(scale_frame(X), y, config.max_rows, seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=seed,
    )

    model = InsideForestRegressor(
        auto_fast=True,
        auto_feature_reduce=True,
        explicit_k_features=min(config.explicit_k_features, X.shape[1]),
        no_trees_search=config.no_trees_search,
        max_cases=min(config.max_cases, len(X_train)),
        leaf_percentile=config.leaf_percentile,
        low_leaf_fraction=config.low_leaf_fraction,
        rf_params={
            "n_estimators": config.rf_n_estimators,
            "max_depth": config.rf_max_depth,
            "min_samples_leaf": 3,
            "random_state": seed,
            "n_jobs": 1,
        },
        seed=seed,
    )

    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start

    train_labels = model.labels_
    test_labels = model.predict(X_test)
    X_test_forest = model._prepare_prediction_frame(X_test)
    rf_pred = model.rf.predict(X_test_forest)
    mean_pred = np.full_like(y_test, float(np.mean(y_train)), dtype=float)
    region_pred_all, region_pred_covered, y_covered, known_region = region_mean_predictions(
        train_labels,
        y_train,
        test_labels,
        y_test,
    )
    rule_covered = np.asarray(test_labels) != -1
    report = model.region_quality_report()
    warning_text = "\n".join(str(w.message) for w in caught)

    return {
        "dataset": name,
        "seed": int(seed),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features_in": int(X.shape[1]),
        "n_features_used": int(len(model.feature_names_)),
        "fit_seconds": float(fit_seconds),
        "rf_r2_test": float(r2_score(y_test, rf_pred)),
        "rf_rmse_test": rmse(y_test, rf_pred),
        "baseline_mean_rmse_test": rmse(y_test, mean_pred),
        "region_mean_rmse_all_test": rmse(y_test, region_pred_all),
        "region_mean_rmse_covered_test": rmse(y_covered, region_pred_covered) if len(y_covered) else math.nan,
        "region_rmse_lift_vs_mean": float(1.0 - rmse(y_test, region_pred_all) / rmse(y_test, mean_pred)),
        "train_coverage": float(report["coverage"]),
        "train_unmatched_rate": float(report["unmatched_rate"]),
        "test_rule_coverage": float(np.mean(rule_covered)),
        "test_rule_unmatched_rate": float(np.mean(~rule_covered)),
        "test_known_region_coverage": float(np.mean(known_region)),
        "test_known_region_unmatched_rate": float(np.mean(~known_region)),
        "n_regions": int(report["n_regions"]),
        "n_train_clusters": int(report["n_clusters"]),
        "n_test_clusters": int(len(set(np.asarray(test_labels)[rule_covered].tolist()))),
        "train_target_std_reduction": target_std_reduction(train_labels, y_train),
        "test_target_std_reduction": target_std_reduction(test_labels, y_test),
        "method": model.method,
        "size_bucket": model._size_bucket_,
        "warning_count": int(len(caught)),
        "classification_warning": "could represent a regression problem" in warning_text
        or "unique classes" in warning_text,
    }


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = metrics.select_dtypes(include=[np.number]).columns.difference(["seed"])
    rows = []
    for dataset, group in metrics.groupby("dataset", sort=True):
        row = {"dataset": dataset, "runs": int(len(group))}
        for col in numeric:
            row[f"{col}_median"] = float(group[col].median())
        rows.append(row)
    return pd.DataFrame(rows)


def build_readme(metrics: pd.DataFrame, summary: pd.DataFrame, config: RegressionValidationConfig) -> str:
    lines = [
        "# Regression Region Validation",
        "",
        "This report validates `InsideForestRegressor` as an interpretable region extractor for continuous targets.",
        "`predict(X)` is evaluated as region labels; the internal random forest is evaluated separately with R2/RMSE.",
        "",
        "## Reproduction",
        "",
        "```bash",
        f"python experiments/validate_regression_regions.py --profile {config.profile}",
        "```",
        "",
        "## Decision Checks",
        "",
    ]
    lines.extend(
        [
            f"- Median RF test R2 across all runs: `{metrics['rf_r2_test'].median():.4f}`.",
            f"- Median test rule coverage: `{metrics['test_rule_coverage'].median():.4f}`.",
            f"- Median test known-region coverage: `{metrics['test_known_region_coverage'].median():.4f}`.",
            f"- Median test target spread reduction inside regions: `{metrics['test_target_std_reduction'].median():.4f}`.",
            f"- Median region-mean RMSE lift vs train-mean baseline: `{metrics['region_rmse_lift_vs_mean'].median():.4f}`.",
            f"- Classification-style target warnings observed: `{int(metrics['classification_warning'].sum())}`.",
        ]
    )
    lines.extend(
        [
            "",
            "## Dataset Summary",
            "",
            "| Dataset | Runs | RF R2 | RF RMSE | Mean RMSE | Region RMSE | Region Lift | Rule Coverage | Known Coverage | Std Reduction | Regions | Clusters | Fit s |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['dataset']} | {int(row['runs'])} | "
            f"{fmt(row.get('rf_r2_test_median'))} | {fmt(row.get('rf_rmse_test_median'))} | "
            f"{fmt(row.get('baseline_mean_rmse_test_median'))} | {fmt(row.get('region_mean_rmse_all_test_median'))} | "
            f"{fmt(row.get('region_rmse_lift_vs_mean_median'))} | {fmt(row.get('test_rule_coverage_median'))} | "
            f"{fmt(row.get('test_known_region_coverage_median'))} | {fmt(row.get('test_target_std_reduction_median'))} | {fmt(row.get('n_regions_median'))} | "
            f"{fmt(row.get('n_train_clusters_median'))} | {fmt(row.get('fit_seconds_median'))} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Positive `region_rmse_lift_vs_mean` means region labels produce better region-mean estimates than a train-mean baseline.",
            "- Positive target spread reduction means covered observations are grouped into regions with tighter target values than the overall target spread.",
            "- `test_rule_coverage` counts observations assigned to any learned rule.",
            "- `test_known_region_coverage` counts observations assigned to a region label observed during training; unknown labels use the train mean in the all-row RMSE.",
            "- The absence of classification-style target warnings confirms that regression feature selection is not routed through class-based scoring.",
            "",
            "## Raw Outputs",
            "",
            "- `metrics.csv`: one row per dataset and seed.",
            "- `summary.csv`: median metrics per dataset.",
        ]
    )
    return "\n".join(lines) + "\n"


def fmt(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    config = RegressionValidationConfig.from_profile(args.profile)
    config.out_dir = args.out_dir
    config.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in config.seeds:
        available = datasets(seed)
        selected = args.datasets or list(available)
        for name in selected:
            print(f"Evaluating {name} seed={seed}")
            X, y = available[name]
            rows.append(evaluate_dataset(name, X, y, seed, config))

    metrics = pd.DataFrame(rows)
    summary = summarize(metrics)
    metrics.to_csv(config.out_dir / "metrics.csv", index=False)
    summary.to_csv(config.out_dir / "summary.csv", index=False)
    (config.out_dir / "README.md").write_text(
        build_readme(metrics, summary, config),
        encoding="utf-8",
    )
    print(f"Wrote outputs to {config.out_dir}")


if __name__ == "__main__":
    main()
