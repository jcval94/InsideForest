"""Validate interdimensional region-pair prioritization.

The script writes fold-level metrics, an aggregate summary and a short
Markdown verdict under ``experiments/results/group_priority_validation``.
It is intentionally conservative: extraction failures are recorded as rows so
one brittle source does not hide the behavior of the other sources.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "results" / "group_priority_validation"

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from InsideForest.group_priority import (  # noqa: E402
    RegionDescriptor,
    feature_weights_from_model,
    from_multiclass_rules,
    from_traditional_regions,
    rank_region_pairs,
)
from InsideForest.inside_forest import InsideForestClassifier  # noqa: E402
from InsideForest.multiclass import InsideForestMulticlassClassifier  # noqa: E402


METHODS = (
    "robust",
    "original",
    "centroid",
    "dice",
    "dimension",
    "random",
    "support_product",
)


@dataclass
class ValidationConfig:
    profile: str
    random_state: int
    n_splits: int
    max_rows: int
    max_regions: int
    top_k: int
    out_dir: Path

    @classmethod
    def from_profile(cls, profile: str) -> "ValidationConfig":
        if profile == "full":
            return cls(
                profile=profile,
                random_state=42,
                n_splits=3,
                max_rows=350,
                max_regions=180,
                top_k=20,
                out_dir=OUT_DIR,
            )
        return cls(
            profile=profile,
            random_state=42,
            n_splits=2,
            max_rows=180,
            max_regions=80,
            top_k=10,
            out_dir=OUT_DIR,
        )


def sanitize_frame(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    out.columns = [str(col).replace(" ", "_") for col in out.columns]
    out = out.apply(pd.to_numeric, errors="raise").astype(float)
    medians = out.median(numeric_only=True).fillna(0.0)
    return out.fillna(medians)


def built_in_datasets(config: ValidationConfig) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    iris = load_iris()
    wine = load_wine()
    datasets = {
        "iris": (pd.DataFrame(iris.data, columns=iris.feature_names), iris.target),
        "wine": (pd.DataFrame(wine.data, columns=wine.feature_names), wine.target),
    }
    if config.profile == "full":
        breast = load_breast_cancer()
        digits = load_digits()
        datasets.update(
            {
                "breast_cancer": (
                    pd.DataFrame(breast.data, columns=breast.feature_names),
                    breast.target,
                ),
                "digits": (
                    pd.DataFrame(
                        digits.data,
                        columns=[f"pixel_{idx}" for idx in range(digits.data.shape[1])],
                    ),
                    digits.target,
                ),
            }
        )
        try:
            datasets["titanic_onehot"] = load_titanic_onehot()
        except Exception as exc:
            print(f"Skipping titanic_onehot: {exc}")
    return {
        name: (sanitize_frame(X), np.asarray(y))
        for name, (X, y) in datasets.items()
    }


def load_titanic_onehot() -> tuple[pd.DataFrame, np.ndarray]:
    import seaborn as sns

    df = sns.load_dataset("titanic")
    y = df["survived"].to_numpy()
    X = df.drop(columns=["survived"])
    numeric = X.select_dtypes(include=["number", "boolean"]).columns
    X[numeric] = X[numeric].fillna(X[numeric].median(numeric_only=True))
    X = pd.get_dummies(X, dummy_na=True, dtype=float)
    return X, y


def stratified_downsample(
    X: pd.DataFrame,
    y: np.ndarray,
    max_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    if len(X) <= max_rows:
        return X.reset_index(drop=True), np.asarray(y)
    rng = np.random.default_rng(random_state)
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
    return X.iloc[selected].reset_index(drop=True), np.asarray(y)[selected]


def synthetic_priority_problems(random_state: int):
    rng = np.random.default_rng(random_state)
    problems = []

    for name, n_features, noise_scale in [
        ("synthetic_interdimensional", 6, 1.0),
        ("synthetic_high_dimensional", 14, 1.2),
    ]:
        n_per_group = 120
        centers = np.array([0.0, 4.0, -4.0])
        values = []
        labels = []
        for group, center in enumerate(centers):
            block = rng.normal(0.0, 0.45, size=(n_per_group, n_features))
            block[:, :3] += center
            block[:, 3:] *= noise_scale
            values.append(block)
            labels.extend([group] * n_per_group)
        X = pd.DataFrame(
            np.vstack(values),
            columns=[f"feature_{idx}" for idx in range(n_features)],
        )
        regions = []
        specs = [
            ("g0_xy", {"feature_0": (-1.0, 1.0), "feature_1": (-1.0, 1.0)}, "g0"),
            ("g0_xz", {"feature_0": (-1.1, 1.1), "feature_2": (-1.0, 1.0)}, "g0"),
            ("g0_yz", {"feature_1": (-1.0, 1.0), "feature_2": (-1.0, 1.0)}, "g0"),
            ("g1_xy", {"feature_0": (3.0, 5.0), "feature_1": (3.0, 5.0)}, "g1"),
            ("g1_xz", {"feature_0": (2.9, 5.1), "feature_2": (3.0, 5.0)}, "g1"),
            ("g1_yz", {"feature_1": (3.0, 5.0), "feature_2": (3.0, 5.0)}, "g1"),
            ("noise_34", {"feature_3": (-0.5, 0.5), "feature_4": (-0.5, 0.5)}, None),
            ("mixed_far", {"feature_0": (-1.0, 1.0), "feature_1": (3.0, 5.0)}, None),
        ]
        if n_features > 8:
            specs.extend(
                [
                    ("noise_89", {"feature_8": (-0.5, 0.5), "feature_9": (-0.5, 0.5)}, None),
                    ("g2_xnoise", {"feature_0": (-5.0, -3.0), "feature_8": (-0.5, 0.5)}, None),
                ]
            )
        for region_id, bounds, truth_group in specs:
            descriptor = descriptor_from_bounds(
                X,
                region_id=region_id,
                bounds=bounds,
                source="synthetic",
                metadata={"truth_group": truth_group},
            )
            if descriptor.n_support >= 2:
                regions.append(descriptor)
        truth_pairs = {
            frozenset((left.id, right.id))
            for i, left in enumerate(regions)
            for right in regions[i + 1 :]
            if left.metadata.get("truth_group") is not None
            and left.metadata.get("truth_group") == right.metadata.get("truth_group")
        }
        weights = {col: 0.02 for col in X.columns}
        weights.update({"feature_0": 0.30, "feature_1": 0.25, "feature_2": 0.25})
        problems.append((name, X, np.asarray(labels), regions, truth_pairs, weights))

    return problems


def descriptor_from_bounds(
    X: pd.DataFrame,
    *,
    region_id: str,
    bounds: dict[str, tuple[float, float]],
    source: str,
    metadata: dict,
) -> RegionDescriptor:
    mask = np.ones(len(X), dtype=bool)
    for feature, (lower, upper) in bounds.items():
        values = X[feature].to_numpy(dtype=float)
        mask &= values >= lower
        mask &= values <= upper
    support = np.flatnonzero(mask)
    dimensions = tuple(bounds.keys())
    centroid = X.iloc[support][list(dimensions)].mean(axis=0)
    return RegionDescriptor(
        id=region_id,
        bounds=bounds,
        dimensions=dimensions,
        support_indices=support,
        centroid=centroid,
        n_support=len(support),
        source=source,
        metadata=metadata,
    )


def evaluate_synthetics(config: ValidationConfig) -> list[dict]:
    rows = []
    for dataset, X, y, regions, truth_pairs, weights in synthetic_priority_problems(config.random_state):
        robust = rank_region_pairs(
            X,
            regions,
            feature_weights=weights,
            variant="robust",
            scaler="standard",
            top_k=None,
        )
        original = rank_region_pairs(
            X,
            regions,
            feature_weights=weights,
            variant="original",
            scaler="standard",
            top_k=None,
        )
        for method in METHODS:
            ranked = method_ranking(
                robust,
                original,
                method,
                random_state=config.random_state,
            )
            top = ranked.head(config.top_k)
            rows.append(
                {
                    **base_row(
                        dataset=dataset,
                        source="synthetic",
                        fold_id=-1,
                        comparison_kind=method,
                        n_regions=len(regions),
                        n_pairs=len(ranked),
                        top=top,
                        rank_seconds=math.nan,
                        extract_seconds=math.nan,
                    ),
                    **truth_metrics(top, truth_pairs, config.top_k),
                    "mean_class_distribution_similarity": mean_class_distribution_similarity(
                        top,
                        {region.id: region for region in regions},
                        y,
                    ),
                    "feature_signature_jaccard_with_previous_fold": math.nan,
                    "error": "",
                }
            )
    return rows


def evaluate_real_datasets(
    datasets: dict[str, tuple[pd.DataFrame, np.ndarray]],
    config: ValidationConfig,
    selected_names: list[str],
    selected_sources: set[str],
) -> list[dict]:
    rows = []
    previous_signatures: dict[tuple[str, str, str], set[tuple]] = {}
    for dataset in selected_names:
        if dataset not in datasets:
            print(f"Skipping unknown dataset {dataset!r}")
            continue
        X, y = stratified_downsample(
            *datasets[dataset],
            max_rows=config.max_rows,
            random_state=config.random_state,
        )
        counts = pd.Series(y).value_counts()
        n_splits = min(config.n_splits, int(counts.min()))
        if n_splits < 2:
            print(f"Skipping {dataset}: not enough samples per class")
            continue
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=config.random_state,
        )
        for fold_id, (train_idx, _) in enumerate(splitter.split(X, y)):
            X_train = X.iloc[train_idx].reset_index(drop=True)
            y_train = y[train_idx]
            for source in sorted(selected_sources):
                source_start = time.perf_counter()
                try:
                    regions, weights = extract_regions_for_source(
                        source,
                        X_train,
                        y_train,
                        config,
                        seed=config.random_state + fold_id,
                    )
                    extract_seconds = time.perf_counter() - source_start
                    regions = limit_regions(regions, config.max_regions)
                    if len(regions) < 2:
                        rows.append(error_row(dataset, source, fold_id, "not enough regions"))
                        continue
                    robust, original, rank_seconds = rank_both_variants(
                        X_train,
                        regions,
                        weights,
                    )
                    region_lookup = {region.id: region for region in regions}
                    for method in METHODS:
                        ranked = method_ranking(
                            robust,
                            original,
                            method,
                            random_state=config.random_state + fold_id,
                        )
                        top = ranked.head(config.top_k)
                        signature = feature_signature(top)
                        prev_key = (dataset, source, method)
                        previous = previous_signatures.get(prev_key)
                        stability = jaccard(previous, signature) if previous is not None else math.nan
                        previous_signatures[prev_key] = signature
                        rows.append(
                            {
                                **base_row(
                                    dataset=dataset,
                                    source=source,
                                    fold_id=fold_id,
                                    comparison_kind=method,
                                    n_regions=len(regions),
                                    n_pairs=len(ranked),
                                    top=top,
                                    rank_seconds=rank_seconds,
                                    extract_seconds=extract_seconds,
                                ),
                                "precision_at_k": math.nan,
                                "ndcg_at_k": math.nan,
                                "mrr": math.nan,
                                "truth_pairs": math.nan,
                                "mean_class_distribution_similarity": mean_class_distribution_similarity(
                                    top,
                                    region_lookup,
                                    y_train,
                                ),
                                "feature_signature_jaccard_with_previous_fold": stability,
                                "error": "",
                            }
                        )
                except Exception as exc:  # pragma: no cover - diagnostic path
                    rows.append(error_row(dataset, source, fold_id, repr(exc)))
    return rows


def extract_regions_for_source(
    source: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    config: ValidationConfig,
    *,
    seed: int,
) -> tuple[list[RegionDescriptor], dict[str, float]]:
    if source == "multiclass":
        model = InsideForestMulticlassClassifier(
            rf_params={"n_estimators": 24, "max_depth": 6, "random_state": seed},
            percentil=85,
            min_support=2,
            max_rules_per_class=50,
            random_state=seed,
        ).fit(X_train, y_train)
        regions = from_multiclass_rules(model.rules_, X_train, deduplicate=True, min_support=2)
        weights = feature_weights_from_model(model)
        return regions, weights

    if source == "traditional":
        model = InsideForestClassifier(
            rf_params={"n_estimators": 16, "max_depth": 5, "random_state": seed},
            no_trees_search=16,
            get_detail=False,
            leaf_percentile=90,
            low_leaf_fraction=0.02,
            max_cases=min(config.max_rows, len(X_train)),
            seed=seed,
        ).fit(X_train, y_train)
        regions = from_traditional_regions(model.df_reres_, X_train, min_support=2)
        weights = feature_weights_from_model(model)
        return regions, weights

    raise ValueError(f"Unknown source {source!r}")


def rank_both_variants(
    X: pd.DataFrame,
    regions: list[RegionDescriptor],
    weights,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    start = time.perf_counter()
    robust = rank_region_pairs(
        X,
        regions,
        feature_weights=weights,
        variant="robust",
        scaler="standard",
        top_k=None,
    )
    original = rank_region_pairs(
        X,
        regions,
        feature_weights=weights,
        variant="original",
        scaler="standard",
        top_k=None,
    )
    return robust, original, time.perf_counter() - start


def limit_regions(regions: list[RegionDescriptor], max_regions: int) -> list[RegionDescriptor]:
    def key(region: RegionDescriptor) -> tuple[float, float, float]:
        score = float(region.metadata.get("score", region.metadata.get("metric_ponderador", 0.0)) or 0.0)
        metric_n = float(region.metadata.get("metric_n_sample", region.n_support) or region.n_support)
        return score, metric_n, float(region.n_support)

    return sorted(regions, key=key, reverse=True)[:max_regions]


def method_ranking(
    robust: pd.DataFrame,
    original: pd.DataFrame,
    method: str,
    *,
    random_state: int,
) -> pd.DataFrame:
    if method == "original":
        return original.sort_values("priority", ascending=False).reset_index(drop=True)
    ranked = robust.copy()
    if method == "robust":
        return ranked.sort_values("priority", ascending=False).reset_index(drop=True)
    if method == "centroid":
        return ranked.sort_values("centroid_similarity", ascending=False).reset_index(drop=True)
    if method == "dice":
        return ranked.sort_values("intersection_similarity", ascending=False).reset_index(drop=True)
    if method == "dimension":
        return ranked.sort_values("dimension_similarity", ascending=False).reset_index(drop=True)
    if method == "support_product":
        return ranked.sort_values("support_product", ascending=False).reset_index(drop=True)
    if method == "random":
        return ranked.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    raise ValueError(f"Unknown method {method!r}")


def base_row(
    *,
    dataset: str,
    source: str,
    fold_id: int,
    comparison_kind: str,
    n_regions: int,
    n_pairs: int,
    top: pd.DataFrame,
    rank_seconds: float,
    extract_seconds: float,
) -> dict:
    return {
        "dataset": dataset,
        "source": source,
        "fold_id": int(fold_id),
        "comparison_kind": comparison_kind,
        "n_regions": int(n_regions),
        "n_pairs": int(n_pairs),
        "top_k_observed": int(len(top)),
        "rank_seconds": float(rank_seconds) if pd.notna(rank_seconds) else math.nan,
        "extract_seconds": float(extract_seconds) if pd.notna(extract_seconds) else math.nan,
        "median_priority": safe_median(top.get("priority")),
        "median_n_ab": safe_median(top.get("n_ab")),
        "zero_overlap_rate": safe_mean(top.get("n_ab") == 0) if not top.empty else math.nan,
        "same_support_rate": safe_mean(top.get("is_same_support")) if not top.empty else math.nan,
        "same_geometry_rate": safe_mean(top.get("is_same_geometry")) if not top.empty else math.nan,
        "min_support_violation_rate": safe_mean(
            (top.get("n_a") < 2) | (top.get("n_b") < 2)
        )
        if not top.empty
        else math.nan,
        "invalid_pair_rate": safe_mean(top["shared_dimensions"].apply(len) == 0)
        if not top.empty
        else math.nan,
    }


def truth_metrics(top: pd.DataFrame, truth_pairs: set[frozenset], top_k: int) -> dict:
    if top.empty:
        return {
            "precision_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "mrr": 0.0,
            "truth_pairs": len(truth_pairs),
        }
    relevance = [
        1.0 if frozenset((row.region_a, row.region_b)) in truth_pairs else 0.0
        for row in top.head(top_k).itertuples(index=False)
    ]
    precision = float(sum(relevance) / max(top_k, 1))
    dcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevance))
    ideal_relevance = [1.0] * min(len(truth_pairs), top_k)
    ideal = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal_relevance))
    ndcg = float(dcg / ideal) if ideal else 0.0
    mrr = 0.0
    for rank, rel in enumerate(relevance, start=1):
        if rel:
            mrr = 1.0 / rank
            break
    return {
        "precision_at_k": precision,
        "ndcg_at_k": ndcg,
        "mrr": float(mrr),
        "truth_pairs": len(truth_pairs),
    }


def mean_class_distribution_similarity(
    top: pd.DataFrame,
    region_lookup: dict,
    y: np.ndarray,
) -> float:
    if top.empty:
        return math.nan
    labels = sorted(pd.unique(pd.Series(y).astype(str)))
    similarities = []
    for row in top.itertuples(index=False):
        left = region_lookup.get(row.region_a)
        right = region_lookup.get(row.region_b)
        if left is None or right is None:
            continue
        p = class_distribution(y[left.support_indices], labels)
        q = class_distribution(y[right.support_indices], labels)
        similarities.append(1.0 - jensen_shannon_divergence(p, q))
    return float(np.mean(similarities)) if similarities else math.nan


def class_distribution(values: np.ndarray, labels: list[str]) -> np.ndarray:
    as_str = pd.Series(values).astype(str)
    counts = as_str.value_counts().reindex(labels, fill_value=0).to_numpy(dtype=float)
    total = counts.sum()
    return counts / total if total else np.zeros(len(labels), dtype=float)


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    mask = (p > 0) & (q > 0)
    if not np.any(mask):
        return 0.0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def feature_signature(top: pd.DataFrame) -> set[tuple]:
    if top.empty:
        return set()
    return {
        (tuple(row.shared_dimensions), tuple(row.union_dimensions))
        for row in top.itertuples(index=False)
    }


def jaccard(left: set | None, right: set | None) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return float(len(left & right) / len(left | right))


def safe_median(values) -> float:
    if values is None:
        return math.nan
    arr = pd.Series(values).dropna()
    return float(arr.median()) if not arr.empty else math.nan


def safe_mean(values) -> float:
    if values is None:
        return math.nan
    arr = pd.Series(values).dropna()
    return float(arr.astype(float).mean()) if not arr.empty else math.nan


def error_row(dataset: str, source: str, fold_id: int, error: str) -> dict:
    row = {
        "dataset": dataset,
        "source": source,
        "fold_id": int(fold_id),
        "comparison_kind": "error",
        "n_regions": 0,
        "n_pairs": 0,
        "top_k_observed": 0,
        "rank_seconds": math.nan,
        "extract_seconds": math.nan,
        "median_priority": math.nan,
        "median_n_ab": math.nan,
        "zero_overlap_rate": math.nan,
        "same_support_rate": math.nan,
        "same_geometry_rate": math.nan,
        "min_support_violation_rate": math.nan,
        "invalid_pair_rate": math.nan,
        "precision_at_k": math.nan,
        "ndcg_at_k": math.nan,
        "mrr": math.nan,
        "truth_pairs": math.nan,
        "mean_class_distribution_similarity": math.nan,
        "feature_signature_jaccard_with_previous_fold": math.nan,
        "error": error,
    }
    return row


def summarize_results(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "precision_at_k",
        "ndcg_at_k",
        "mrr",
        "zero_overlap_rate",
        "same_support_rate",
        "same_geometry_rate",
        "min_support_violation_rate",
        "invalid_pair_rate",
        "mean_class_distribution_similarity",
        "feature_signature_jaccard_with_previous_fold",
        "rank_seconds",
        "extract_seconds",
        "n_regions",
        "n_pairs",
    ]
    rows = []
    valid = fold_metrics[fold_metrics["comparison_kind"] != "error"]
    for (dataset, source, kind), group in valid.groupby(["dataset", "source", "comparison_kind"], sort=True):
        for metric in metric_cols:
            rows.append(
                {
                    "dataset": dataset,
                    "source": source,
                    "comparison_kind": kind,
                    "metric": metric,
                    "median": safe_median(group[metric]),
                    "mean": safe_mean(group[metric]),
                    "n": int(group[metric].notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def verdict_markdown(
    fold_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    config: ValidationConfig,
) -> str:
    lines = [
        "# Group Priority Validation",
        "",
        f"Profile: `{config.profile}`. Splits: {config.n_splits}. Top-k: {config.top_k}.",
        "",
        "## Decision Checks",
        "",
    ]

    synth = fold_metrics[
        (fold_metrics["source"] == "synthetic")
        & (fold_metrics["comparison_kind"] != "error")
    ]
    if not synth.empty:
        pivot = synth.pivot_table(
            index="dataset",
            columns="comparison_kind",
            values="precision_at_k",
            aggfunc="median",
        )
        robust_wins = []
        for dataset, row in pivot.iterrows():
            robust_value = row.get("robust", math.nan)
            baseline_best = row.drop(labels=["robust", "original"], errors="ignore").max()
            robust_wins.append(bool(pd.notna(robust_value) and robust_value >= baseline_best))
        lines.append(
            f"- Synthetic Precision@{config.top_k}: robust >= best simple baseline in "
            f"{sum(robust_wins)}/{len(robust_wins)} synthetic problems."
        )

    invalid = fold_metrics["invalid_pair_rate"].dropna()
    support_bad = fold_metrics["min_support_violation_rate"].dropna()
    if not invalid.empty:
        lines.append(f"- Invalid shared-dimension pair rate median: {invalid.median():.4f}.")
    if not support_bad.empty:
        lines.append(f"- Min-support violation rate median: {support_bad.median():.4f}.")

    real_stability = fold_metrics[
        (fold_metrics["source"] != "synthetic")
        & (fold_metrics["comparison_kind"].isin(["robust", "centroid", "dice"]))
    ].pivot_table(
        index=["dataset", "source", "fold_id"],
        columns="comparison_kind",
        values="feature_signature_jaccard_with_previous_fold",
        aggfunc="median",
    )
    if {"robust", "centroid", "dice"}.issubset(real_stability.columns):
        robust_delta = real_stability["robust"] - real_stability[["centroid", "dice"]].max(axis=1)
        lines.append(
            "- Real-data top-k feature-signature stability, robust minus best centroid/Dice baseline: "
            f"median {np.nanmedian(robust_delta):+.4f}."
        )

    errors = fold_metrics[fold_metrics["comparison_kind"] == "error"]
    lines.append(f"- Extraction/ranking error rows: {len(errors)}.")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Dataset | Source | Method | Precision@K | Zero overlap | Same support | Stability | Rank s | Regions | Pairs |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    wide = summary.pivot_table(
        index=["dataset", "source", "comparison_kind"],
        columns="metric",
        values="median",
        aggfunc="first",
    ).reset_index()
    for _, row in wide.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['source']} | {row['comparison_kind']} | "
            f"{fmt(row.get('precision_at_k'))} | {fmt(row.get('zero_overlap_rate'))} | "
            f"{fmt(row.get('same_support_rate'))} | "
            f"{fmt(row.get('feature_signature_jaccard_with_previous_fold'))} | "
            f"{fmt(row.get('rank_seconds'))} | {fmt(row.get('n_regions'))} | "
            f"{fmt(row.get('n_pairs'))} |"
        )

    if not errors.empty:
        lines.extend(["", "## Errors", ""])
        for _, row in errors.head(20).iterrows():
            lines.append(
                f"- {row['dataset']} / {row['source']} fold {row['fold_id']}: `{row['error']}`"
            )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `fold_metrics.csv`: raw per-fold and synthetic-problem metrics.",
            "- `summary.csv`: grouped medians and means.",
            "- `config.json`: execution settings.",
        ]
    )
    return "\n".join(lines) + "\n"


def fmt(value) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--datasets", nargs="*", default=None, help="Real datasets to run")
    parser.add_argument(
        "--sources",
        nargs="*",
        choices=["multiclass", "traditional"],
        default=["multiclass", "traditional"],
        help="Region sources to validate on real datasets",
    )
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-regions", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def apply_overrides(config: ValidationConfig, args) -> ValidationConfig:
    if args.n_splits is not None:
        config.n_splits = args.n_splits
    if args.max_rows is not None:
        config.max_rows = args.max_rows
    if args.max_regions is not None:
        config.max_regions = args.max_regions
    if args.top_k is not None:
        config.top_k = args.top_k
    config.out_dir = args.out_dir
    return config


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    config = apply_overrides(ValidationConfig.from_profile(args.profile), args)
    config.out_dir.mkdir(parents=True, exist_ok=True)

    rows = evaluate_synthetics(config)
    datasets = built_in_datasets(config)
    selected = args.datasets or list(datasets.keys())
    rows.extend(
        evaluate_real_datasets(
            datasets,
            config,
            selected_names=selected,
            selected_sources=set(args.sources),
        )
    )

    fold_metrics = pd.DataFrame(rows)
    summary = summarize_results(fold_metrics)
    fold_metrics.to_csv(config.out_dir / "fold_metrics.csv", index=False)
    summary.to_csv(config.out_dir / "summary.csv", index=False)
    (config.out_dir / "summary.md").write_text(
        verdict_markdown(fold_metrics, summary, config),
        encoding="utf-8",
    )
    (config.out_dir / "config.json").write_text(
        json.dumps(
            {
                "profile": config.profile,
                "random_state": config.random_state,
                "n_splits": config.n_splits,
                "max_rows": config.max_rows,
                "max_regions": config.max_regions,
                "top_k": config.top_k,
                "sources": args.sources,
                "datasets": selected,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote validation artifacts to {config.out_dir}")


if __name__ == "__main__":
    main()
