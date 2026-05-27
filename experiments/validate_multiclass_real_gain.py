"""Validate whether the multiclass InsideForest gains are real.

This benchmark separates predictive quality, region/cluster quality and
runtime.  It intentionally reports four comparison kinds:

* ``rf_baseline``: plain RandomForestClassifier.
* ``traditional_cluster``: legacy InsideForest cluster labels aligned to y.
* ``multiclass_regions_only``: new multiclass rules on covered rows only.
* ``multiclass_with_fallback``: new multiclass rules plus RandomForest fallback.

The default profile is intentionally moderate so it can be run locally.  Use
``--profile full`` or explicit CLI overrides to run a heavier validation.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import tracemalloc
import warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_classification,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from InsideForest import InsideForestClassifier
from InsideForest.multiclass import InsideForestMulticlassClassifier


OUT_DIR = ROOT / "experiments" / "results" / "multiclass_validation"
FIG_DIR = OUT_DIR / "figures"


@dataclass
class ValidationConfig:
    profile: str = "quick"
    n_splits: int = 2
    n_repeats: int = 1
    random_state: int = 42
    max_rows: int | None = 450
    rf_n_estimators: int = 25
    rf_max_depth: int | None = 6
    traditional_no_trees_search: int = 30
    traditional_max_cases: int = 320
    traditional_leaf_percentile: float = 95
    traditional_low_leaf_fraction: float = 0.05
    multiclass_percentil: float = 95
    multiclass_low_frac: float = 0.05
    multiclass_min_support: int = 2
    out_dir: Path = OUT_DIR

    @classmethod
    def from_profile(cls, profile: str) -> "ValidationConfig":
        if profile == "full":
            return cls(
                profile="full",
                n_splits=5,
                n_repeats=2,
                max_rows=1200,
                rf_n_estimators=60,
                rf_max_depth=None,
                traditional_no_trees_search=80,
                traditional_max_cases=750,
            )
        return cls(profile="quick")


def timed_peak(func):
    tracemalloc.start()
    start = perf_counter()
    result = func()
    elapsed = perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / (1024 * 1024)


def sanitize_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        df = X.copy()
    else:
        df = pd.DataFrame(np.asarray(X))
    df.columns = [str(c).replace(" ", "_") for c in df.columns]
    return df.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)


def stratified_downsample(X: pd.DataFrame, y: np.ndarray, max_rows: int | None, random_state: int):
    if max_rows is None or len(X) <= max_rows:
        return X.reset_index(drop=True), np.asarray(y)
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    selected = []
    classes, counts = np.unique(y, return_counts=True)
    for cls, count in zip(classes, counts):
        cls_idx = np.flatnonzero(y == cls)
        n_cls = max(2, int(round(max_rows * count / len(y))))
        n_cls = min(n_cls, len(cls_idx))
        selected.extend(rng.choice(cls_idx, size=n_cls, replace=False).tolist())
    selected = np.array(sorted(set(selected)))
    if len(selected) > max_rows:
        selected = rng.choice(selected, size=max_rows, replace=False)
    return X.iloc[selected].reset_index(drop=True), y[selected]


def built_in_datasets(random_state: int) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    iris = load_iris()
    wine = load_wine()
    breast = load_breast_cancer()
    digits = load_digits()

    datasets = {
        "iris": (pd.DataFrame(iris.data, columns=iris.feature_names), iris.target),
        "wine": (pd.DataFrame(wine.data, columns=wine.feature_names), wine.target),
        "breast_cancer": (
            pd.DataFrame(breast.data, columns=breast.feature_names),
            breast.target,
        ),
        "digits": (pd.DataFrame(digits.data, columns=[f"pixel_{i}" for i in range(digits.data.shape[1])]), digits.target),
    }
    datasets.update(synthetic_datasets(random_state))
    try:
        datasets["titanic_onehot"] = load_titanic_onehot()
    except Exception as exc:
        print(f"Skipping titanic_onehot: {exc}")
    return {name: (sanitize_frame(X), np.asarray(y)) for name, (X, y) in datasets.items()}


def synthetic_datasets(random_state: int) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    rng = np.random.default_rng(random_state)
    out = {}

    specs = [
        ("synthetic_3class", 3, [1 / 3] * 3, 0.01, 1.4),
        ("synthetic_5class_overlap", 5, [0.2] * 5, 0.04, 0.8),
        ("synthetic_10class", 10, [0.1] * 10, 0.02, 1.2),
        ("synthetic_imbalance_noise", 4, [0.65, 0.2, 0.1, 0.05], 0.08, 1.0),
    ]
    for name, n_classes, weights, flip_y, class_sep in specs:
        X, y = make_classification(
            n_samples=650,
            n_features=18,
            n_informative=min(10, 2 * n_classes),
            n_redundant=3,
            n_repeated=0,
            n_classes=n_classes,
            n_clusters_per_class=1,
            weights=weights,
            flip_y=flip_y,
            class_sep=class_sep,
            random_state=random_state + n_classes,
        )
        out[name] = (pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])]), y)

    X, y = out["synthetic_3class"]
    X_irrelevant = X.copy()
    for idx in range(20):
        X_irrelevant[f"noise_{idx}"] = rng.normal(size=len(X_irrelevant))
    out["synthetic_irrelevant_features"] = (X_irrelevant, y.copy())

    permuted_y = y.copy()
    rng.shuffle(permuted_y)
    out["negative_label_permuted"] = (X.copy(), permuted_y)

    remap = {0: 20, 1: 10, 2: 99}
    out["negative_remapped_ids"] = (X.copy(), np.array([remap[int(v)] for v in y]))

    permuted_cols = list(X.columns)
    rng.shuffle(permuted_cols)
    out["negative_permuted_columns"] = (X[permuted_cols].copy(), y.copy())

    out["synthetic_high_cardinality_onehot"] = synthetic_high_cardinality(random_state)
    return out


def synthetic_high_cardinality(random_state: int):
    rng = np.random.default_rng(random_state)
    n = 700
    cat_a = rng.integers(0, 45, size=n)
    cat_b = rng.integers(0, 30, size=n)
    num_0 = rng.normal(size=n)
    num_1 = rng.normal(size=n)
    y = ((cat_a % 3) + (cat_b % 2) + (num_0 > 0).astype(int)) % 4
    raw = pd.DataFrame(
        {
            "cat_a": [f"a_{v}" for v in cat_a],
            "cat_b": [f"b_{v}" for v in cat_b],
            "num_0": num_0,
            "num_1": num_1,
        }
    )
    X = pd.get_dummies(raw, columns=["cat_a", "cat_b"], dtype=float)
    return X, y


def load_titanic_onehot():
    df = sns.load_dataset("titanic")
    cols = [
        "survived",
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked",
        "class",
        "who",
        "adult_male",
        "deck",
        "embark_town",
        "alone",
    ]
    df = df[cols].copy()
    y = df["survived"].to_numpy()
    X = df.drop(columns=["survived"])
    for col in ["adult_male", "alone"]:
        X[col] = X[col].astype("Int64")
    numeric = X.select_dtypes(include=["number", "boolean"]).columns
    X[numeric] = X[numeric].fillna(X[numeric].median(numeric_only=True))
    X = pd.get_dummies(X, dummy_na=True, dtype=float)
    return X, y


def scale_numeric(X: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    values = scaler.fit_transform(X.to_numpy(dtype=float))
    return pd.DataFrame(values, columns=X.columns)


def classification_metrics(y_true, y_pred, *, evaluated_n: int | None = None) -> dict[str, float]:
    y_true_s = pd.Series(y_true).astype(str).to_numpy()
    y_pred_s = pd.Series(y_pred).astype(str).to_numpy()
    labels = sorted(pd.unique(y_true_s))
    if len(y_true_s) == 0:
        return {
            "accuracy": math.nan,
            "balanced_accuracy": math.nan,
            "macro_f1": math.nan,
            "weighted_f1": math.nan,
            "evaluated_n": 0,
        }
    recalls = []
    for label in labels:
        mask = y_true_s == label
        recalls.append(float(np.mean(y_pred_s[mask] == label)) if np.any(mask) else math.nan)
    balanced_accuracy = float(np.nanmean(recalls)) if recalls else math.nan
    return {
        "accuracy": float(metrics.accuracy_score(y_true_s, y_pred_s)),
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": float(metrics.f1_score(y_true_s, y_pred_s, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(metrics.f1_score(y_true_s, y_pred_s, average="weighted", labels=labels, zero_division=0)),
        "evaluated_n": int(len(y_true_s) if evaluated_n is None else evaluated_n),
    }


def align_clusters_hungarian(y_true, cluster_labels):
    y_true = np.asarray(y_true)
    cluster_labels = np.asarray(cluster_labels)
    valid = cluster_labels != -1
    sentinel = "__unmatched__"
    aligned = np.array([sentinel] * len(y_true), dtype=object)
    if not np.any(valid):
        return aligned, {}

    true_values = np.unique(y_true)
    cluster_values = np.unique(cluster_labels[valid])
    matrix = np.zeros((len(true_values), len(cluster_values)), dtype=int)
    for i, cls in enumerate(true_values):
        for j, cluster in enumerate(cluster_values):
            matrix[i, j] = int(np.sum((y_true == cls) & (cluster_labels == cluster)))
    row_ind, col_ind = linear_sum_assignment(-matrix)
    mapping = {cluster_values[col]: true_values[row] for row, col in zip(row_ind, col_ind)}
    for idx, label in enumerate(cluster_labels):
        if label in mapping:
            aligned[idx] = mapping[label]
    return aligned, mapping


def purity_score(y_true, y_pred) -> float:
    if len(y_true) == 0:
        return math.nan
    contingency = metrics.cluster.contingency_matrix(y_true, y_pred)
    total = contingency.sum()
    return math.nan if total == 0 else float(contingency.max(axis=0).sum() / total)


def cluster_metrics(y_true, cluster_labels) -> dict[str, float]:
    y_true_s = pd.Series(y_true).astype(str).to_numpy()
    cluster_s = pd.Series(cluster_labels).astype(str).to_numpy()
    if len(y_true_s) == 0:
        return {
            "cluster_purity": math.nan,
            "nmi": math.nan,
            "ami": math.nan,
            "ari": math.nan,
            "homogeneity": math.nan,
            "completeness": math.nan,
        }
    return {
        "cluster_purity": purity_score(y_true_s, cluster_s),
        "nmi": float(metrics.normalized_mutual_info_score(y_true_s, cluster_s)),
        "ami": float(metrics.adjusted_mutual_info_score(y_true_s, cluster_s)),
        "ari": float(metrics.adjusted_rand_score(y_true_s, cluster_s)),
        "homogeneity": float(metrics.homogeneity_score(y_true_s, cluster_s)),
        "completeness": float(metrics.completeness_score(y_true_s, cluster_s)),
    }


def multiclass_interpretability(model: InsideForestMulticlassClassifier) -> dict[str, float]:
    rules = model.rules_
    if rules is None or rules.empty:
        return {
            "n_rules": 0,
            "weighted_region_purity": math.nan,
            "mean_entropy": math.nan,
            "mean_lift": math.nan,
            "rules_per_second": math.nan,
        }
    support = rules["support"].to_numpy(dtype=float)
    weights = support / support.sum() if support.sum() > 0 else np.ones(len(rules)) / len(rules)
    return {
        "n_rules": int(len(rules)),
        "weighted_region_purity": float(np.sum(rules["dominant_probability"].to_numpy(dtype=float) * weights)),
        "mean_entropy": float(np.nanmean(rules["entropy"].to_numpy(dtype=float))),
        "mean_lift": float(np.nanmean(rules["lift"].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float))),
        "rules_per_second": math.nan,
    }


def multiclass_feature_set(model: InsideForestMulticlassClassifier) -> set[str]:
    if model.rules_ is None or model.rules_.empty:
        return set()
    features = set()
    for bounds_col in ("lower_bounds", "upper_bounds"):
        for bounds in model.rules_[bounds_col]:
            features.update(bounds.keys())
    return features


def traditional_feature_set(model: InsideForestClassifier) -> set[str]:
    features = set()
    for df in getattr(model, "df_reres_", []) or []:
        if not hasattr(df, "columns"):
            continue
        if getattr(df.columns, "nlevels", 1) > 1:
            features.update([col[1] for col in df.columns if col[0] in {"linf", "lsup"}])
    return set(map(str, features))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return math.nan
    return float(len(a & b) / len(a | b)) if (a or b) else math.nan


def evaluate_fold(name: str, X: pd.DataFrame, y: np.ndarray, train_idx, test_idx, fold_id: int, config: ValidationConfig):
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = np.asarray(y)[train_idx]
    y_test = np.asarray(y)[test_idx]

    rf_params = {
        "n_estimators": config.rf_n_estimators,
        "max_depth": config.rf_max_depth,
        "random_state": config.random_state + fold_id,
        "n_jobs": 1,
    }

    rows = []
    confusion_payload = []

    rf, rf_fit, rf_peak = timed_peak(lambda: RandomForestClassifier(**rf_params).fit(X_train, y_train))
    rf_pred, rf_assign, rf_assign_peak = timed_peak(lambda: rf.predict(X_test))
    row = base_row(name, fold_id, "rf_baseline", rf_fit, rf_assign, rf_peak, rf_assign_peak)
    row.update(classification_metrics(y_test, rf_pred))
    row.update(empty_cluster_metrics())
    row.update(empty_interpretability())
    row["coverage"] = 1.0
    row["fallback_rate"] = 0.0
    row["feature_jaccard_with_previous_fold"] = math.nan
    rows.append(row)
    confusion_payload.append(confusion_record(name, fold_id, "rf_baseline", y_test, rf_pred))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traditional, trad_fit, trad_peak = timed_peak(
            lambda: InsideForestClassifier(
                rf_params=rf_params,
                max_cases=min(config.traditional_max_cases, len(X_train)),
                no_trees_search=config.traditional_no_trees_search,
                leaf_percentile=config.traditional_leaf_percentile,
                low_leaf_fraction=config.traditional_low_leaf_fraction,
            ).fit(X_train, y_train)
        )
        trad_pred, trad_assign, trad_assign_peak = timed_peak(lambda: traditional.predict(X_test))

    aligned, _ = align_clusters_hungarian(y_test, trad_pred)
    row = base_row(name, fold_id, "traditional_cluster", trad_fit, trad_assign, trad_peak, trad_assign_peak)
    row.update(classification_metrics(y_test, aligned))
    row.update(cluster_metrics(y_test, trad_pred))
    row.update(empty_interpretability())
    row["n_rules"] = int(sum(len(df) for df in traditional.df_reres_)) if traditional.df_reres_ else 0
    row["coverage"] = float(np.mean(np.asarray(trad_pred) != -1))
    row["fallback_rate"] = float(np.mean(np.asarray(trad_pred) == -1))
    row["_feature_set"] = traditional_feature_set(traditional)
    rows.append(row)
    confusion_payload.append(confusion_record(name, fold_id, "traditional_cluster", y_test, aligned))

    multiclass, mc_fit, mc_peak = timed_peak(
        lambda: InsideForestMulticlassClassifier(
            rf_params=rf_params,
            percentil=config.multiclass_percentil,
            low_frac=config.multiclass_low_frac,
            min_support=config.multiclass_min_support,
            random_state=config.random_state + fold_id,
        ).fit(X_train, y_train)
    )
    assignments, mc_assign, mc_assign_peak = timed_peak(lambda: multiclass.assign_regions(X_test))
    region_mask = assignments["source"].to_numpy() == "region"

    interp = multiclass_interpretability(multiclass)
    if mc_fit > 0:
        interp["rules_per_second"] = float(interp["n_rules"] / mc_fit)

    row = base_row(name, fold_id, "multiclass_regions_only", mc_fit, mc_assign, mc_peak, mc_assign_peak)
    if np.any(region_mask):
        row.update(classification_metrics(y_test[region_mask], assignments.loc[region_mask, "predicted_class"].to_numpy()))
        confusion_payload.append(
            confusion_record(
                name,
                fold_id,
                "multiclass_regions_only",
                y_test[region_mask],
                assignments.loc[region_mask, "predicted_class"].to_numpy(),
            )
        )
    else:
        row.update(classification_metrics([], [], evaluated_n=0))
    row.update(empty_cluster_metrics())
    row.update(interp)
    row["coverage"] = float(np.mean(region_mask))
    row["fallback_rate"] = float(np.mean(~region_mask))
    row["_feature_set"] = multiclass_feature_set(multiclass)
    rows.append(row)

    row = base_row(name, fold_id, "multiclass_with_fallback", mc_fit, mc_assign, mc_peak, mc_assign_peak)
    row.update(classification_metrics(y_test, assignments["predicted_class"].to_numpy()))
    row.update(empty_cluster_metrics())
    row.update(interp)
    row["coverage"] = 1.0
    row["fallback_rate"] = float(np.mean(assignments["source"] == "model_fallback"))
    row["_feature_set"] = multiclass_feature_set(multiclass)
    rows.append(row)
    confusion_payload.append(
        confusion_record(name, fold_id, "multiclass_with_fallback", y_test, assignments["predicted_class"].to_numpy())
    )

    return rows, confusion_payload


def base_row(dataset, fold_id, comparison_kind, fit_seconds, assign_seconds, fit_peak_mb, assign_peak_mb):
    return {
        "dataset": dataset,
        "fold_id": fold_id,
        "comparison_kind": comparison_kind,
        "fit_seconds": float(fit_seconds),
        "assign_seconds": float(assign_seconds),
        "fit_peak_mb": float(fit_peak_mb),
        "assign_peak_mb": float(assign_peak_mb),
    }


def empty_cluster_metrics():
    return {
        "cluster_purity": math.nan,
        "nmi": math.nan,
        "ami": math.nan,
        "ari": math.nan,
        "homogeneity": math.nan,
        "completeness": math.nan,
    }


def empty_interpretability():
    return {
        "n_rules": math.nan,
        "weighted_region_purity": math.nan,
        "mean_entropy": math.nan,
        "mean_lift": math.nan,
        "rules_per_second": math.nan,
    }


def confusion_record(dataset, fold_id, comparison_kind, y_true, y_pred):
    y_true_s = pd.Series(y_true).astype(str).to_numpy()
    y_pred_s = pd.Series(y_pred).astype(str).to_numpy()
    labels = sorted(set(y_true_s) | set(y_pred_s))
    matrix = metrics.confusion_matrix(y_true_s, y_pred_s, labels=labels)
    return {
        "dataset": dataset,
        "fold_id": int(fold_id),
        "comparison_kind": comparison_kind,
        "labels": labels,
        "matrix": matrix.tolist(),
    }


def evaluate_dataset(name: str, X: pd.DataFrame, y: np.ndarray, config: ValidationConfig):
    X, y = stratified_downsample(X, y, config.max_rows, config.random_state)
    X = scale_numeric(X)
    counts = pd.Series(y).value_counts()
    n_splits = min(config.n_splits, int(counts.min()))
    if n_splits < 2:
        print(f"Skipping {name}: not enough samples per class")
        return [], []

    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=config.n_repeats,
        random_state=config.random_state,
    )
    rows = []
    confusion = []
    prev_features: dict[str, set[str]] = {}
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        print(f"  {name} fold {fold_id + 1}/{n_splits * config.n_repeats}")
        fold_rows, fold_confusion = evaluate_fold(name, X, y, train_idx, test_idx, fold_id, config)
        for row in fold_rows:
            feature_set = row.pop("_feature_set", None)
            if feature_set is None:
                row["feature_jaccard_with_previous_fold"] = math.nan
            else:
                previous = prev_features.get(row["comparison_kind"])
                row["feature_jaccard_with_previous_fold"] = jaccard(previous, feature_set) if previous is not None else math.nan
                prev_features[row["comparison_kind"]] = feature_set
            rows.append(row)
        confusion.extend(fold_confusion)
    return rows, confusion


def bootstrap_ci(values: Iterable[float], random_state: int, n_boot: int = 500):
    arr = np.asarray([v for v in values if pd.notna(v)], dtype=float)
    if arr.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(random_state)
    med = float(np.median(arr))
    if arr.size == 1:
        return med, med, med
    boots = [np.median(rng.choice(arr, size=arr.size, replace=True)) for _ in range(n_boot)]
    return med, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def summarize_results(fold_metrics: pd.DataFrame, config: ValidationConfig) -> pd.DataFrame:
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "fit_seconds",
        "assign_seconds",
        "fit_peak_mb",
        "coverage",
        "fallback_rate",
        "cluster_purity",
        "nmi",
        "ami",
        "ari",
        "homogeneity",
        "completeness",
        "n_rules",
        "weighted_region_purity",
        "mean_entropy",
        "mean_lift",
        "rules_per_second",
        "feature_jaccard_with_previous_fold",
    ]
    rows = []
    for (dataset, kind), group in fold_metrics.groupby(["dataset", "comparison_kind"], sort=True):
        for metric in metric_cols:
            median, low, high = bootstrap_ci(group[metric], config.random_state)
            rows.append(
                {
                    "dataset": dataset,
                    "comparison_kind": kind,
                    "metric": metric,
                    "median": median,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n": int(group[metric].notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def plot_metric_distributions(fold_metrics: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric in ["balanced_accuracy", "macro_f1", "fit_seconds", "coverage"]:
        fig, ax = plt.subplots(figsize=(13, 6))
        plot_df = fold_metrics[["dataset", "comparison_kind", metric]].dropna()
        if plot_df.empty:
            continue
        labels = []
        data = []
        for kind, group in plot_df.groupby("comparison_kind", sort=True):
            labels.append(kind)
            data.append(group[metric].to_numpy(dtype=float))
        ax.boxplot(data, tick_labels=labels, vert=True)
        ax.set_title(f"{metric} distribution across datasets/folds")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_distribution.png", dpi=150)
        plt.close()

    scatter_plot(
        fold_metrics,
        x="coverage",
        y="balanced_accuracy",
        title="Coverage vs balanced accuracy",
        path=out_dir / "coverage_vs_accuracy.png",
    )
    scatter_plot(
        fold_metrics,
        x="fallback_rate",
        y="balanced_accuracy",
        title="Fallback/unmatched rate vs balanced accuracy",
        path=out_dir / "fallback_rate_vs_accuracy.png",
    )
    scatter_plot(
        fold_metrics,
        x="fit_seconds",
        y="n_rules",
        title="Runtime vs number of rules/regions",
        path=out_dir / "runtime_vs_rules.png",
    )


def scatter_plot(df, *, x, y, title, path):
    plot_df = df[[x, y, "comparison_kind"]].dropna()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    for kind, group in plot_df.groupby("comparison_kind", sort=True):
        ax.scatter(group[x], group[y], alpha=0.7, label=kind)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion_matrices(confusion_records: list[dict], out_dir: Path):
    conf_dir = out_dir / "confusion_matrices"
    conf_dir.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for record in confusion_records:
        key = (record["dataset"], record["comparison_kind"])
        labels = record["labels"]
        matrix = np.asarray(record["matrix"], dtype=int)
        if key not in grouped:
            grouped[key] = {"labels": labels, "matrix": matrix}
            continue
        old = grouped[key]
        union = sorted(set(old["labels"]) | set(labels))
        old_matrix = expand_matrix(old["matrix"], old["labels"], union)
        new_matrix = expand_matrix(matrix, labels, union)
        grouped[key] = {"labels": union, "matrix": old_matrix + new_matrix}

    serializable = []
    for (dataset, kind), payload in grouped.items():
        labels = payload["labels"]
        matrix = payload["matrix"]
        serializable.append(
            {
                "dataset": dataset,
                "comparison_kind": kind,
                "labels": labels,
                "matrix": matrix.tolist(),
            }
        )
        if len(labels) > 15:
            continue
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{dataset} - {kind}")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j]:
                    ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(conf_dir / f"{dataset}__{kind}.png", dpi=150)
        plt.close()

    (out_dir / "confusion_matrices.json").write_text(
        json.dumps(serializable, indent=2),
        encoding="utf-8",
    )


def expand_matrix(matrix, labels, union):
    out = np.zeros((len(union), len(union)), dtype=int)
    label_to_old = {label: idx for idx, label in enumerate(labels)}
    for i, true_label in enumerate(union):
        for j, pred_label in enumerate(union):
            if true_label in label_to_old and pred_label in label_to_old:
                out[i, j] = matrix[label_to_old[true_label], label_to_old[pred_label]]
    return out


def verdict_markdown(fold_metrics: pd.DataFrame, summary: pd.DataFrame, config: ValidationConfig) -> str:
    lines = [
        "# Multiclass Real Gain Validation",
        "",
        f"Profile: `{config.profile}`. Splits: {config.n_splits}, repeats: {config.n_repeats}.",
        "",
        "## Decision Checks",
        "",
    ]

    pivot_fit = fold_metrics.pivot_table(
        index=["dataset", "fold_id"],
        columns="comparison_kind",
        values="fit_seconds",
        aggfunc="median",
    )
    if {"traditional_cluster", "multiclass_with_fallback"}.issubset(pivot_fit.columns):
        ratio = pivot_fit["traditional_cluster"] / pivot_fit["multiclass_with_fallback"]
        share_fast = float(np.mean(ratio >= 2.0))
        lines.append(
            f"- Efficiency: multiclass fit was at least 2x faster in {share_fast:.1%} of dataset/fold pairs "
            f"(median ratio {np.nanmedian(ratio):.2f}x)."
        )
    else:
        share_fast = math.nan

    pivot_bal = fold_metrics.pivot_table(
        index=["dataset", "fold_id"],
        columns="comparison_kind",
        values="balanced_accuracy",
        aggfunc="median",
    )
    if {"traditional_cluster", "multiclass_regions_only", "multiclass_with_fallback", "rf_baseline"}.issubset(pivot_bal.columns):
        region_delta = pivot_bal["multiclass_regions_only"] - pivot_bal["traditional_cluster"]
        fallback_delta = pivot_bal["multiclass_with_fallback"] - pivot_bal["multiclass_regions_only"]
        rf_delta = pivot_bal["rf_baseline"] - pivot_bal["multiclass_with_fallback"]
        lines.append(
            f"- Predictive regions-only delta vs traditional cluster: median balanced-accuracy delta "
            f"{np.nanmedian(region_delta):+.4f}."
        )
        lines.append(
            f"- Fallback effect: median balanced-accuracy gain from fallback "
            f"{np.nanmedian(fallback_delta):+.4f}."
        )
        lines.append(
            f"- RF baseline comparison: median RF minus multiclass-with-fallback delta "
            f"{np.nanmedian(rf_delta):+.4f}."
        )

    permuted = fold_metrics[fold_metrics["dataset"].str.contains("label_permuted")]
    if not permuted.empty:
        permuted_bal = permuted.groupby("comparison_kind")["balanced_accuracy"].median().to_dict()
        lines.append(
            "- Negative label permutation median balanced accuracy: "
            + ", ".join(f"{k}={v:.4f}" for k, v in sorted(permuted_bal.items()))
            + "."
        )

    lines.extend(
        [
            "",
            "## How To Read This",
            "",
            "- Treat `traditional_cluster` as supervised clustering, not direct classification.",
            "- Treat `multiclass_regions_only` as the cleanest test of interpretable rules.",
            "- Treat `multiclass_with_fallback` as rules plus RandomForest backstop.",
            "- If `rf_baseline` matches or beats `multiclass_with_fallback`, predictive lift is mostly the forest, not the rule layer.",
            "",
            "## Summary Table",
            "",
            "| Dataset | Kind | Balanced Acc | Macro F1 | Fit s | Coverage | Fallback | Rules |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    summary_wide = summary.pivot_table(
        index=["dataset", "comparison_kind"],
        columns="metric",
        values="median",
        aggfunc="first",
    ).reset_index()
    for _, row in summary_wide.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['comparison_kind']} | "
            f"{fmt(row.get('balanced_accuracy'))} | {fmt(row.get('macro_f1'))} | "
            f"{fmt(row.get('fit_seconds'))} | {fmt(row.get('coverage'))} | "
            f"{fmt(row.get('fallback_rate'))} | {fmt(row.get('n_rules'))} |"
        )
    return "\n".join(lines) + "\n"


def fmt(value) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset names to run")
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument("--n-repeats", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def apply_overrides(config: ValidationConfig, args) -> ValidationConfig:
    if args.n_splits is not None:
        config.n_splits = args.n_splits
    if args.n_repeats is not None:
        config.n_repeats = args.n_repeats
    if args.max_rows is not None:
        config.max_rows = args.max_rows
    config.out_dir = args.out_dir
    return config


def main():
    args = parse_args()
    config = apply_overrides(ValidationConfig.from_profile(args.profile), args)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    (config.out_dir / "figures").mkdir(parents=True, exist_ok=True)

    all_datasets = built_in_datasets(config.random_state)
    selected_names = args.datasets or list(all_datasets.keys())

    rows = []
    confusion_records = []
    for name in selected_names:
        if name not in all_datasets:
            print(f"Skipping unknown dataset {name!r}")
            continue
        print(f"\n=== {name} ===")
        dataset_rows, dataset_confusion = evaluate_dataset(name, *all_datasets[name], config)
        rows.extend(dataset_rows)
        confusion_records.extend(dataset_confusion)

    fold_metrics = pd.DataFrame(rows)
    fold_metrics.to_csv(config.out_dir / "fold_metrics.csv", index=False)
    summary = summarize_results(fold_metrics, config)
    summary.to_csv(config.out_dir / "summary.csv", index=False)
    plot_metric_distributions(fold_metrics, config.out_dir / "figures")
    plot_confusion_matrices(confusion_records, config.out_dir)
    (config.out_dir / "summary.md").write_text(
        verdict_markdown(fold_metrics, summary, config),
        encoding="utf-8",
    )
    print(f"\nWrote validation artifacts to {config.out_dir}")


if __name__ == "__main__":
    main()
