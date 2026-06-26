"""Validate traditional InsideForest as a region/cluster extractor.

This benchmark treats ``InsideForestClassifier`` as supervised clustering, not
as a direct competitor to a RandomForest classifier.  It reports region quality,
cluster quality, coverage, stability and runtime.  The legacy selector path is
included to compare the vectorized assignment against the reference
implementation when desired.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import statistics
import sys
import time
import tracemalloc
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_classification,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import InsideForest.regions as regions_module
from InsideForest import InsideForestClassifier
from InsideForest.cluster_selector import select_clusters
from InsideForest.legacy_select_clusters import select_clusters_legacy
from InsideForest.region_quality import cluster_label_quality


OUT_DIR = ROOT / "experiments" / "results" / "region_cluster_validation"


@dataclass
class ValidationConfig:
    profile: str = "quick"
    n_splits: int = 2
    n_repeats: int = 1
    max_rows: int | None = 320
    random_state: int = 42
    rf_n_estimators: int = 20
    rf_max_depth: int | None = 6
    no_trees_search: int = 20
    max_cases: int = 320
    leaf_percentile: float = 95
    low_leaf_fraction: float = 0.05
    include_legacy_selector: bool = True
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
                no_trees_search=80,
                max_cases=750,
                include_legacy_selector=True,
            )
        return cls(profile="quick")


def timed_peak(func: Callable):
    tracemalloc.start()
    start = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / (1024 * 1024)


@contextlib.contextmanager
def selector_patch(selector):
    original = regions_module.select_clusters
    regions_module.select_clusters = selector
    try:
        yield
    finally:
        regions_module.select_clusters = original


def legacy_selector_adapter(
    df_datos,
    df_reglas,
    keep_all_clusters=True,
    fallback_cluster=None,
    batch_size=None,
    warn_unmatched=True,
):
    """Adapt the keyword-only legacy selector to Regions' call shape."""

    return select_clusters_legacy(
        df_datos,
        df_reglas,
        keep_all_clusters=keep_all_clusters,
        fallback_cluster=fallback_cluster,
        warn_unmatched=warn_unmatched,
    )


def sanitize_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        df = X.copy()
    else:
        df = pd.DataFrame(np.asarray(X))
    df.columns = [str(col).replace(" ", "_") for col in df.columns]
    return df.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)


def scale_frame(X: pd.DataFrame) -> pd.DataFrame:
    values = StandardScaler().fit_transform(X.to_numpy(dtype=float))
    return pd.DataFrame(values, columns=X.columns)


def stratified_downsample(X: pd.DataFrame, y: np.ndarray, max_rows: int | None, seed: int):
    if max_rows is None or len(X) <= max_rows:
        return X.reset_index(drop=True), np.asarray(y)
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    selected = []
    for label, count in zip(*np.unique(y, return_counts=True)):
        label_idx = np.flatnonzero(y == label)
        n_label = max(2, int(round(max_rows * count / len(y))))
        selected.extend(rng.choice(label_idx, size=min(n_label, len(label_idx)), replace=False))
    selected = np.array(sorted(set(selected)))
    if len(selected) > max_rows:
        selected = rng.choice(selected, size=max_rows, replace=False)
    return X.iloc[selected].reset_index(drop=True), y[selected]


def datasets(seed: int) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    iris = load_iris()
    wine = load_wine()
    breast = load_breast_cancer()
    digits = load_digits()
    out = {
        "iris": (pd.DataFrame(iris.data, columns=iris.feature_names), iris.target),
        "wine": (pd.DataFrame(wine.data, columns=wine.feature_names), wine.target),
        "breast_cancer": (pd.DataFrame(breast.data, columns=breast.feature_names), breast.target),
        "digits": (pd.DataFrame(digits.data, columns=[f"pixel_{i}" for i in range(digits.data.shape[1])]), digits.target),
    }
    out.update(synthetic_datasets(seed))
    return {name: (sanitize_frame(X), np.asarray(y)) for name, (X, y) in out.items()}


def synthetic_datasets(seed: int) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    rng = np.random.default_rng(seed)
    out = {}
    specs = [
        ("synthetic_3class", 3, [1 / 3] * 3, 0.01, 1.4),
        ("synthetic_5class_overlap", 5, [0.2] * 5, 0.05, 0.8),
        ("synthetic_10class", 10, [0.1] * 10, 0.02, 1.2),
        ("synthetic_imbalance_noise", 4, [0.65, 0.2, 0.1, 0.05], 0.08, 1.0),
    ]
    for name, n_classes, weights, flip_y, class_sep in specs:
        X, y = make_classification(
            n_samples=650,
            n_features=18,
            n_informative=min(10, 2 * n_classes),
            n_redundant=3,
            n_classes=n_classes,
            n_clusters_per_class=1,
            weights=weights,
            flip_y=flip_y,
            class_sep=class_sep,
            random_state=seed + n_classes,
        )
        out[name] = (pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])]), y)

    X_base, y_base = out["synthetic_3class"]
    X_noise = X_base.copy()
    for idx in range(20):
        X_noise[f"noise_{idx}"] = rng.normal(size=len(X_noise))
    out["synthetic_irrelevant_features"] = (X_noise, y_base.copy())

    y_permuted = y_base.copy()
    rng.shuffle(y_permuted)
    out["negative_label_permuted"] = (X_base.copy(), y_permuted)

    shuffled_cols = list(X_base.columns)
    rng.shuffle(shuffled_cols)
    out["negative_permuted_features"] = (X_base[shuffled_cols].copy(), y_base.copy())

    out["synthetic_high_cardinality_onehot"] = synthetic_high_cardinality(seed)
    return out


def synthetic_high_cardinality(seed: int):
    rng = np.random.default_rng(seed)
    n = 700
    raw = pd.DataFrame(
        {
            "cat_a": [f"a_{v}" for v in rng.integers(0, 45, size=n)],
            "cat_b": [f"b_{v}" for v in rng.integers(0, 30, size=n)],
            "num_0": rng.normal(size=n),
            "num_1": rng.normal(size=n),
        }
    )
    y = ((raw["cat_a"].str.extract(r"(\d+)").astype(int)[0] % 3) + (raw["num_0"] > 0).astype(int)) % 4
    X = pd.get_dummies(raw, columns=["cat_a", "cat_b"], dtype=float)
    return X, y.to_numpy()


def run_model(kind: str, selector, X_train, y_train, X_test, y_test, fold_id: int, config: ValidationConfig):
    rf_params = {
        "n_estimators": config.rf_n_estimators,
        "max_depth": config.rf_max_depth,
        "random_state": config.random_state + fold_id,
        "n_jobs": 1,
    }

    def fit_model():
        with selector_patch(selector):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return InsideForestClassifier(
                    rf_params=rf_params,
                    max_cases=min(config.max_cases, len(X_train)),
                    no_trees_search=config.no_trees_search,
                    leaf_percentile=config.leaf_percentile,
                    low_leaf_fraction=config.low_leaf_fraction,
                    seed=config.random_state + fold_id,
                ).fit(X_train, y_train)

    model, fit_seconds, fit_peak_mb = timed_peak(fit_model)
    with selector_patch(selector):
        labels, predict_seconds, predict_peak_mb = timed_peak(lambda: model.predict(X_test))

    valid = np.asarray(labels) != -1
    test_cluster = (
        cluster_label_quality(np.asarray(y_test)[valid], np.asarray(labels)[valid])
        if np.any(valid)
        else {key: math.nan for key in ["cluster_purity", "nmi", "ami", "ari", "homogeneity", "completeness"]}
    )

    report = model.region_quality_report()
    row = {
        "kind": kind,
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "fit_peak_mb": float(fit_peak_mb),
        "predict_peak_mb": float(predict_peak_mb),
        "test_coverage": float(np.mean(valid)),
        "test_unmatched_rate": float(np.mean(~valid)),
        "regions_per_second": float(report["n_regions"] / fit_seconds) if fit_seconds > 0 else math.nan,
        "feature_set": region_feature_set(model),
    }
    row.update({f"train_{key}": value for key, value in report.items()})
    row.update({f"test_{key}": value for key, value in test_cluster.items()})
    return row


def region_feature_set(model: InsideForestClassifier) -> set[str]:
    if model.region_rules_ is None or model.region_rules_.empty:
        return set()
    features = set()
    for values in model.region_rules_["features"]:
        features.update(values)
    return features


def jaccard(left: set[str] | None, right: set[str] | None) -> float:
    if left is None or right is None:
        return math.nan
    if not left and not right:
        return math.nan
    return float(len(left & right) / len(left | right))


def evaluate_dataset(name: str, X: pd.DataFrame, y: np.ndarray, config: ValidationConfig):
    X, y = stratified_downsample(X, y, config.max_rows, config.random_state)
    X = scale_frame(X)
    counts = pd.Series(y).value_counts()
    n_splits = min(config.n_splits, int(counts.min()))
    if n_splits < 2:
        print(f"Skipping {name}: not enough rows per class")
        return []

    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=config.n_repeats,
        random_state=config.random_state,
    )
    selectors = [("normal_optimized", select_clusters)]
    if config.include_legacy_selector:
        selectors.append(("normal_legacy_selector", legacy_selector_adapter))

    rows = []
    previous_features: dict[str, set[str]] = {}
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        print(f"  {name} fold {fold_id + 1}/{n_splits * config.n_repeats}")
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_train = np.asarray(y)[train_idx]
        y_test = np.asarray(y)[test_idx]
        for kind, selector in selectors:
            row = run_model(kind, selector, X_train, y_train, X_test, y_test, fold_id, config)
            feature_set = row.pop("feature_set")
            row["dataset"] = name
            row["fold_id"] = fold_id
            row["feature_jaccard_with_previous_fold"] = jaccard(previous_features.get(kind), feature_set)
            previous_features[kind] = feature_set
            rows.append(row)
    return rows


def median_ci(values: Iterable[float]):
    arr = np.asarray([value for value in values if pd.notna(value)], dtype=float)
    if arr.size == 0:
        return math.nan
    return float(np.median(arr))


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [col for col in rows.columns if col not in {"dataset", "fold_id", "kind"}]
    out = []
    for (dataset, kind), group in rows.groupby(["dataset", "kind"], sort=True):
        for metric in metric_cols:
            out.append(
                {
                    "dataset": dataset,
                    "kind": kind,
                    "metric": metric,
                    "median": median_ci(group[metric]),
                    "n": int(group[metric].notna().sum()),
                }
            )
    return pd.DataFrame(out)


def build_markdown(fold_metrics: pd.DataFrame, summary: pd.DataFrame, config: ValidationConfig) -> str:
    lines = [
        "# Traditional Region/Cluster Validation",
        "",
        f"Profile: `{config.profile}`. Splits: {config.n_splits}, repeats: {config.n_repeats}.",
        "",
        "## Decision Checks",
        "",
    ]

    pivot = fold_metrics.pivot_table(index=["dataset", "fold_id"], columns="kind", values="fit_seconds", aggfunc="median")
    if {"normal_optimized", "normal_legacy_selector"}.issubset(pivot.columns):
        ratio = pivot["normal_legacy_selector"] / pivot["normal_optimized"]
        lines.append(f"- Efficiency: optimized selector median speedup vs legacy path {np.nanmedian(ratio):.2f}x.")

    opt = fold_metrics[fold_metrics["kind"] == "normal_optimized"]
    lines.append(
        "- Region quality: median weighted purity "
        f"{opt['train_weighted_region_purity'].median():.4f}, median train coverage "
        f"{opt['train_coverage'].median():.4f}."
    )
    lines.append(
        "- Cluster quality: median test NMI "
        f"{opt['test_nmi'].median():.4f}, median test unmatched rate "
        f"{opt['test_unmatched_rate'].median():.4f}."
    )
    negative = opt[opt["dataset"].str.contains("negative_label_permuted")]
    if not negative.empty:
        lines.append(
            "- Negative label control: median weighted purity "
            f"{negative['train_weighted_region_purity'].median():.4f}, median test NMI "
            f"{negative['test_nmi'].median():.4f}."
        )

    lines.extend(
        [
            "",
            "## Summary Table",
            "",
            "| Dataset | Kind | Fit s | Predict s | Train Coverage | Weighted Purity | Test Coverage | Test NMI | Regions | Rule Len |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    wide = summary.pivot_table(index=["dataset", "kind"], columns="metric", values="median", aggfunc="first").reset_index()
    for _, row in wide.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['kind']} | {fmt(row.get('fit_seconds'))} | "
            f"{fmt(row.get('predict_seconds'))} | {fmt(row.get('train_coverage'))} | "
            f"{fmt(row.get('train_weighted_region_purity'))} | {fmt(row.get('test_coverage'))} | "
            f"{fmt(row.get('test_nmi'))} | {fmt(row.get('train_n_regions'))} | "
            f"{fmt(row.get('train_mean_rule_length'))} |"
        )
    return "\n".join(lines) + "\n"


def fmt(value) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument("--n-repeats", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--skip-legacy-selector", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    config = ValidationConfig.from_profile(args.profile)
    if args.n_splits is not None:
        config.n_splits = args.n_splits
    if args.n_repeats is not None:
        config.n_repeats = args.n_repeats
    if args.max_rows is not None:
        config.max_rows = args.max_rows
    if args.skip_legacy_selector:
        config.include_legacy_selector = False
    config.out_dir = args.out_dir
    config.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    all_datasets = datasets(config.random_state)
    selected = args.datasets or list(all_datasets)
    for name in selected:
        print(f"Evaluating {name}")
        X, y = all_datasets[name]
        rows.extend(evaluate_dataset(name, X, y, config))

    fold_metrics = pd.DataFrame(rows)
    summary = summarize(fold_metrics)
    fold_metrics.to_csv(config.out_dir / "fold_metrics.csv", index=False)
    summary.to_csv(config.out_dir / "summary.csv", index=False)
    (config.out_dir / "summary.md").write_text(build_markdown(fold_metrics, summary, config), encoding="utf-8")
    print(f"Wrote outputs to {config.out_dir}")


if __name__ == "__main__":
    main()
