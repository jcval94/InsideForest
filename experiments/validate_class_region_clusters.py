"""Validate InsideForest region clusterers without classification fallback.

The random forest is treated only as a shared branch generator configuration.
Reported outcomes are cluster/region quality, coverage, stability, runtime and
memory; classifier accuracy is intentionally excluded.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
import tracemalloc

import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_classification,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from InsideForest import (  # noqa: E402
    InsideForestClassRegionClusterer,
    InsideForestRegionClusterer,
)


OUT_DIR = ROOT / "experiments" / "results" / "class_region_cluster_validation"


def timed_peak(callable_):
    tracemalloc.start()
    start = time.perf_counter()
    result = callable_()
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, float(elapsed), float(peak / (1024 * 1024))


def datasets(seed=42):
    loaders = {
        "iris": load_iris(),
        "wine": load_wine(),
        "breast_cancer": load_breast_cancer(),
        "digits": load_digits(),
    }
    out = {
        name: (
            pd.DataFrame(
                data.data,
                columns=[str(column).replace(" ", "_") for column in data.feature_names],
            ),
            np.asarray(data.target),
        )
        for name, data in loaders.items()
    }
    X, y = make_classification(
        n_samples=650,
        n_features=18,
        n_informative=10,
        n_redundant=3,
        n_classes=5,
        n_clusters_per_class=1,
        class_sep=0.8,
        flip_y=0.05,
        random_state=seed,
    )
    out["synthetic_5class_overlap"] = (
        pd.DataFrame(X, columns=[f"feature_{idx}" for idx in range(X.shape[1])]),
        y,
    )
    return out


def downsample(X, y, max_rows, seed):
    if max_rows is None or len(X) <= max_rows:
        return X.reset_index(drop=True), np.asarray(y)
    rng = np.random.default_rng(seed)
    selected = []
    y = np.asarray(y)
    for label, count in zip(*np.unique(y, return_counts=True)):
        indices = np.flatnonzero(y == label)
        n_label = max(2, int(round(max_rows * count / len(y))))
        selected.extend(
            rng.choice(indices, size=min(n_label, len(indices)), replace=False)
        )
    selected = np.asarray(sorted(set(selected)))
    if len(selected) > max_rows:
        selected = rng.choice(selected, size=max_rows, replace=False)
    return X.iloc[selected].reset_index(drop=True), y[selected]


def feature_set(model):
    regions = model.regions_
    if regions is None or len(regions) == 0:
        return set()
    features = set()
    if "features" in regions:
        for values in regions["features"]:
            features.update(values)
    elif "lower_bounds" in regions:
        for lower, upper in zip(regions["lower_bounds"], regions["upper_bounds"]):
            features.update(lower)
            features.update(upper)
    return features


def raw_region_count(model):
    raw = model.raw_regions_
    if isinstance(raw, pd.DataFrame):
        return int(len(raw))
    if isinstance(raw, (list, tuple)):
        total = 0
        for frame in raw:
            if isinstance(frame, pd.DataFrame) and "rectangulo" in frame:
                total += int(frame["rectangulo"].nunique())
            else:
                total += len(frame)
        return total
    return 0


def jaccard(left, right):
    if left is None or right is None or not (left or right):
        return math.nan
    return float(len(left & right) / len(left | right))


def build_models(seed, n_estimators, max_depth, n_train, shared_forest):
    rf_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "random_state": seed,
        "n_jobs": 1,
    }
    return {
        "traditional_region_clusterer": InsideForestRegionClusterer(
            rf_params=dict(rf_params),
            no_trees_search=n_estimators,
            max_cases=n_train,
            seed=seed,
        ),
        "class_region_clusterer": InsideForestClassRegionClusterer(
            forest=shared_forest,
            rf_params=dict(rf_params),
            leaf_percentile=95,
            low_leaf_fraction=0.05,
            min_support=2,
            random_state=seed,
            branch_aggregation="none",
        ),
    }


def run(profile="quick", selected=None, out_dir=OUT_DIR):
    if profile == "full":
        n_splits, n_repeats, max_rows = 5, 3, 1200
        n_estimators, max_depth = 60, None
    else:
        n_splits, n_repeats, max_rows = 2, 1, 360
        n_estimators, max_depth = 20, 6

    rows = []
    previous_features = {}
    all_datasets = datasets()
    for dataset_name in selected or all_datasets:
        X, y = all_datasets[dataset_name]
        X, y = downsample(X, y, max_rows=max_rows, seed=42)
        folds = min(n_splits, int(pd.Series(y).value_counts().min()))
        splitter = RepeatedStratifiedKFold(
            n_splits=folds,
            n_repeats=n_repeats,
            random_state=42,
        )
        for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            models = build_models(
                seed=42 + fold_id,
                n_estimators=n_estimators,
                max_depth=max_depth,
                n_train=len(X_train),
                shared_forest=None,
            )
            forest_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "random_state": 42 + fold_id,
                "n_jobs": 1,
            }
            shared_forest, forest_fit_seconds, forest_peak_mb = timed_peak(
                lambda: RandomForestClassifier(**forest_params).fit(X_train, y_train)
            )
            models["class_region_clusterer"].forest = shared_forest
            for kind, model in models.items():
                model, fit_seconds, fit_peak_mb = timed_peak(
                    lambda model=model, kind=kind: model.fit(
                        X_train,
                        y_train,
                        **({"rf": shared_forest} if kind == "traditional_region_clusterer" else {}),
                    )
                )
                labels, predict_seconds, predict_peak_mb = timed_peak(
                    lambda model=model: model.predict(X_test)
                )
                report = model.region_quality_report(X_test, y_test)
                current_features = feature_set(model)
                row = {
                    "dataset": dataset_name,
                    "fold_id": fold_id,
                    "kind": kind,
                    "fit_seconds": fit_seconds,
                    "forest_fit_seconds": forest_fit_seconds,
                    "total_fit_seconds": fit_seconds + forest_fit_seconds,
                    "predict_seconds": predict_seconds,
                    "fit_peak_mb": fit_peak_mb,
                    "forest_peak_mb": forest_peak_mb,
                    "predict_peak_mb": predict_peak_mb,
                    "n_raw_regions": raw_region_count(model),
                    "n_regions": len(model.regions_),
                    "feature_jaccard_previous_fold": jaccard(
                        previous_features.get((dataset_name, kind)), current_features
                    ),
                    **report,
                }
                previous_features[(dataset_name, kind)] = current_features
                rows.append(row)
                print(
                    f"{dataset_name} fold={fold_id} {kind}: "
                    f"regions={len(model.regions_)} coverage={report['coverage']:.3f} "
                    f"AMI={report.get('ami', math.nan):.3f}"
                )

    metrics = pd.DataFrame(rows)
    numeric = metrics.select_dtypes(include=[np.number]).columns
    summary = (
        metrics.groupby(["dataset", "kind"], as_index=False)[list(numeric)]
        .median(numeric_only=True)
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.md").write_text(
        build_markdown(profile, metrics, summary), encoding="utf-8"
    )
    return metrics, summary


def build_markdown(profile, metrics, summary):
    lines = [
        "# Class-aware Region Cluster Validation",
        "",
        f"Profile: `{profile}`. Random-forest accuracy and fallback are intentionally excluded.",
        "",
        "| Dataset | Kind | Regions | Coverage | Purity | NMI | AMI | ARI | Fit s | Peak MB |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['kind']} | {fmt(row.get('n_regions'))} | "
            f"{fmt(row.get('coverage'))} | {fmt(row.get('cluster_purity'))} | "
            f"{fmt(row.get('nmi'))} | {fmt(row.get('ami'))} | {fmt(row.get('ari'))} | "
            f"{fmt(row.get('fit_seconds'))} | {fmt(row.get('fit_peak_mb'))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `predict` returns region cluster IDs; class labels are not predictions.",
            "- `-1` is retained as an unmatched cluster in external metrics.",
            "- Region compression, coverage and stability must be considered together.",
        ]
    )
    return "\n".join(lines) + "\n"


def fmt(value):
    return "" if pd.isna(value) else f"{float(value):.4f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    run(args.profile, selected=args.datasets, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
